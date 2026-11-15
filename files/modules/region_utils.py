import math
import logging
import weakref
from functools import wraps
from operator import index as integer_index
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import torch
from torch.nn.functional import interpolate

logger = logging.getLogger(__name__)

_mask_cache = OrderedDict()
_attention_roles = ContextVar('region_attention_roles', default=None)
_query_shape = ContextVar('region_query_shape', default=None)
MASK_CACHE_LIMIT = 128
REGION_MAX_ALIGNED_TOKENS = 2048  # LCM before optional base+region concatenation.
REGION_MAX_EXPANDED_BYTES = 512 * 1024 * 1024


class RegionAttentionBudgetError(ValueError):
    """Stop REGION-specific expansion before allocating repeated buffers."""


class RegionQueryShapeError(ValueError):
    """The query sequence cannot be mapped to a known spatial grid."""


def warn_region_once(owner, key, message):
    warnings = getattr(owner, '_region_warnings', None)
    if warnings is None:
        owner._region_warnings = warnings = set()
    if key not in warnings:
        warnings.add(key)
        logger.warning(message)


def has_external_attention_mask(mask, kwargs):
    # A token mask for the original text has no defined mapping to another
    # regional text. Preserve the entire original call instead of discarding it.
    return (mask is not None or any(kwargs.get(k) is not None for k in
            ('attn_mask', 'attention_mask')) or kwargs.get('is_causal', False))


def check_region_expansion(estimated_bytes, max_bytes):
    if estimated_bytes > max_bytes:
        raise RegionAttentionBudgetError(
            f'REGION expansion budget exceeded: estimated {estimated_bytes} bytes > {max_bytes}')


def current_region_query_shape():
    return _query_shape.get()


@contextmanager
def region_query_context(shape):
    token = _query_shape.set(tuple(shape) if shape is not None else None)
    try:
        yield
    finally:
        _query_shape.reset(token)


def _spatial_forward(original):
    @wraps(original)
    def forward(*args, **kwargs):
        x = args[0] if args else kwargs.get('x')
        shape = tuple(x.shape[-2:]) if isinstance(x, torch.Tensor) and x.ndim == 4 else None
        with region_query_context(shape):
            return original(*args, **kwargs)
    return forward


def hook_region_query_shapes(model):
    """Observe SpatialTransformer's 4D input without modifying its computation."""
    saved = []
    for module in model.modules():
        if module.__class__.__name__ != 'SpatialTransformer':
            continue
        original = module.forward
        wrapped = _spatial_forward(original)
        saved.append((module, original, wrapped))
        module.forward = wrapped
    return saved


def unhook_region_query_shapes(saved):
    for module, original, wrapped in reversed(saved):
        if module.forward is wrapped:
            module.forward = original
    saved.clear()


def resolve_region_query_shape(num_tokens, height=None, width=None, query_shape=None):
    """Use actual H/W or an exact level in the standard ceil-halving chain.

    Equal area does not imply equal topology. Never guess an arbitrary
    factorisation and never repair a flattened mask by truncating/padding it.
    """
    actual = query_shape if query_shape is not None else current_region_query_shape()
    if actual is not None:
        try:
            h, w = (integer_index(v) for v in actual)
        except (TypeError, ValueError) as exc:
            raise RegionQueryShapeError('Invalid REGION query shape') from exc
        if h <= 0 or w <= 0 or h * w != num_tokens:
            raise RegionQueryShapeError(f'REGION query shape {actual} does not match {num_tokens} tokens')
        return h, w
    if not height or not width or num_tokens <= 0:
        raise RegionQueryShapeError('REGION query dimensions and token count must be positive')
    try:
        h, w = integer_index(height), integer_index(width)
    except TypeError as exc:
        raise RegionQueryShapeError('REGION query dimensions must be integers') from exc
    while h > 0 and w > 0:
        if h * w == num_tokens:
            return h, w
        if (h, w) == (1, 1):
            break
        h, w = (h + 1) // 2, (w + 1) // 2
    raise RegionQueryShapeError(f'Unknown REGION query topology: {num_tokens} tokens from {height}x{width}')


def clear_mask_cache():
    _mask_cache.clear()


@contextmanager
def region_attention_context(roles):
    """Tag one actual UNet call; restore even if its attention raises."""
    token = _attention_roles.set(tuple(roles) if roles is not None else None)
    try:
        yield
    finally:
        _attention_roles.reset(token)


def attention_batch_layout(batch, batch_size, skip_uncond=False):
    """Return (conditional rows, unconditional rows), or None if ambiguous."""
    roles = _attention_roles.get()
    if roles is not None:
        if len(roles) != batch or any(r not in (0, 1) for r in roles):
            return None
        ncond = roles.count(0)
        if roles != (0,) * ncond + (1,) * (batch - ncond):
            return None
        return ncond, batch - ncond
    if batch == 2 * batch_size:
        return batch_size, batch_size
    if batch == batch_size and skip_uncond:
        return batch, 0
    return None


def prepare_region_data(clean_text, regions, source_prompt):
    """Resolve base before filtering, keeping original canvas provenance."""
    from modules import prompt_parser
    if not regions:
        return clean_text, []
    base = next((r.base_text for r in regions if r.base_text), None)
    if base is None:
        base = prompt_parser.extract_non_region_text(clean_text, regions, original_text=source_prompt)
    return base, [r for r in regions if r.text.strip() and abs(r.weight) > 1e-8]


def region_has_multicond(base, regions):
    """Detect extra CFG branches with the same splitter as conditioning.

    Literal/nested AND is not an extra row. Top-level AND (including without
    spaces) and the ampersand alias would break one-base/one-row-per-region.
    """
    from modules import prompt_parser
    rows, _, _ = prompt_parser.get_multicond_prompt_list([base] + [r.text for r in regions])
    return any(len(branches) != 1 for branches in rows)


def build_region_masks(regions, height, width, device, dtype, canvas_decoder):
    """Shared rect/canvas/blur policy for latent, monkey and Forge shim.

    Preserve feathering and uncovered base: only overlapping coverage > 1
    is normalised. Weights/base_ratio/temporal curves are applied afterwards.
    """
    from modules.prompt_parser import REGION_LATENT_SCALE
    decoded = {}
    masks = []
    for index, region in enumerate(regions):
        mask = None
        canvas = getattr(region, 'canvas', '')
        if canvas:
            count = getattr(region, 'canvas_count', None) or len(regions)
            palette_index = getattr(region, 'canvas_index', None)
            palette_index = index if palette_index is None else palette_index
            key = (canvas, count)
            if key not in decoded:
                try:
                    decoded[key] = canvas_decoder(canvas, count, height, width, device, dtype)
                except Exception:
                    logger.warning('REGION canvas decode failed; using rectangle masks', exc_info=True)
                    decoded[key] = None
            if decoded[key] is not None:
                mask = decoded[key][palette_index].clone()
        if mask is None:
            if region.coords_pixels:
                xs = [int(v / REGION_LATENT_SCALE) for v in (region.x1, region.x2)]
                ys = [int(v / REGION_LATENT_SCALE) for v in (region.y1, region.y2)]
            else:
                xs = [int(v * width) for v in (region.x1, region.x2)]
                ys = [int(v * height) for v in (region.y1, region.y2)]
            x1, x2 = [max(0, min(v, width)) for v in xs]
            y1, y2 = [max(0, min(v, height)) for v in ys]
            mask = torch.zeros(1, height, width, device=device, dtype=dtype)
            if x2 > x1 and y2 > y1:
                mask[:, y1:y2, x1:x2] = 1
        blur = max(0.0, min(1.0, float(getattr(region, 'blur', 0.0))))
        if blur:
            kernel = min(15, int(blur * 14) + 1)
            if kernel % 2 == 0:
                kernel += 1
            if kernel > 1:
                # Outside the image is not an uncovered region. Exclude padding
                # from the divisor so a full-canvas mask stays full at its edges.
                mask = torch.nn.functional.avg_pool2d(
                    mask.unsqueeze(0), kernel, stride=1, padding=kernel // 2,
                    count_include_pad=False,
                ).squeeze(0)
        masks.append(mask)
    if not masks:
        return torch.zeros(0, 1, height, width, device=device, dtype=dtype)
    stacked = torch.stack(masks)
    if any(getattr(r, 'overlap', 'blend') == 'layers' for r in regions):
        if any(getattr(r, 'overlap', 'blend') != 'layers' for r in regions):
            raise ValueError('REGION overlap mode must agree across blocks')
        # Geometry-only alpha occlusion, last SOURCE branch on top. A layer's
        # weight is delta strength, not opacity. Windows/curves/base_ratio are
        # applied later: an inactive layer reveals base, not hidden layers.
        remaining = torch.ones_like(stacked[0])
        effective = torch.zeros_like(stacked)
        order = sorted(range(len(regions)), key=lambda i: (getattr(regions[i], 'source_order', i), i), reverse=True)
        for i in order:
            alpha = stacked[i].clamp(0, 1)
            effective[i] = alpha * remaining
            remaining = remaining * (1 - alpha)
        return effective
    return stacked / stacked.sum(dim=0, keepdim=True).clamp_min(1.0)


def encode_region_conditioning(model, region, base, steps, seed, width, height):
    """Encode regional cross-attention; leave SDXL's global vector on the base path.

    Attention coupling replaces only crossattn inside attention layers. The
    base/negative vector still reaches UNet through ordinary conditioning; a
    regional vector is not a spatial input to these hooks.
    """
    from modules import prompt_parser
    text = region.text
    if region.mode == 'common' and base:
        text = f'{base}, {text}'
    prompts = prompt_parser.SdConditioning([text], width=width, height=height)
    schedules = prompt_parser.get_learned_conditioning(model, prompts, steps, seed=seed)
    if len(schedules) != 1 or len(schedules[0]) != 1:
        raise ValueError('REGION attention needs static conditioning; use latent for scheduled regions')
    cond = prompt_parser.reconstruct_cond_batch(schedules, 0)
    if isinstance(cond, dict):
        if 'crossattn' not in cond or set(cond) - {'crossattn', 'vector'}:
            raise ValueError(f'Unsupported REGION attention conditioning keys: {sorted(cond)}; use latent')
        cond = cond['crossattn']
    if not isinstance(cond, torch.Tensor) or cond.ndim != 3 or cond.shape[0] != 1:
        raise ValueError('Unsupported REGION cross-attention conditioning shape')
    return cond


def lcm_for_list(numbers: list[int], *, max_tokens=None) -> int:
    if not numbers or any(n <= 0 for n in numbers):
        raise ValueError('REGION token lengths must be positive')
    current = 1
    for n in numbers:
        current = current * n // math.gcd(current, n)
        if max_tokens is not None and current > max_tokens:
            raise RegionAttentionBudgetError(
                f'REGION alignment budget exceeded: LCM {current} > {max_tokens} tokens')
    return current


def region_curve_factor(step: int, total_steps: int, start_ratio: float, stop_ratio: float, curve: str) -> float:
    """Per-step multiplier for a region's weight curve over its active window.

    Progress = (step/total - start_ratio) / (stop_ratio - start_ratio),
    clamped to [0,1], then mapped through the easing curve.
    curve == "linear" (or unknown) → 1.0 (constant weight).
    """
    total = max(1, int(total_steps or 1))
    window = stop_ratio - start_ratio
    if window <= 0 or step / total < start_ratio or step / total > stop_ratio:
        return 0.0
    if not curve or curve == "linear":
        return 1.0
    t = (step / total - start_ratio) / window
    t = max(0.0, min(1.0, t))
    try:
        from modules.prompt_parser import _apply_easing
        return max(0.0, _apply_easing(t, curve))
    except Exception:
        return 1.0


def _repeat_div(value: int, iterations: int) -> int:
    for _ in range(iterations):
        value = math.ceil(value / 2)
    return value


def downsample_mask(
    mask: torch.Tensor,
    batch_size: int,
    num_tokens: int,
    shape: tuple | None = None,
    latent_h: int | None = None,
    latent_w: int | None = None,
    cache=None,
    query_shape: tuple | None = None,
) -> torch.Tensor:
    """Downsample spatial mask to match attention layer token count.

    Supports two calling conventions:
      forge: downsample_mask(mask, bs, tokens, shape=(B,C,H,W))
      monkey: downsample_mask(mask, bs, tokens, latent_h=h, latent_w=w)

    Bounded cache includes tensor identity, version, device, dtype and shape.
    Patchers pass their own cache; direct callers use a bounded shared cache.
    Inference tensors without a version counter bypass caching safely.
    query_shape/current SpatialTransformer context takes precedence over the
    latent size; otherwise only an exact ceil-halving level is accepted.
    """
    if shape is not None:
        height, width = shape[2], shape[3]
    else:
        height = latent_h
        width = latent_w
    num_conds = mask.shape[0]
    if batch_size <= 0:
        raise ValueError('REGION batch size must be positive')
    size = resolve_region_query_shape(num_tokens, height, width, query_shape)
    cache = _mask_cache if cache is None else cache
    try:
        version = mask._version
    except RuntimeError:
        version = None
    cache_key = (id(mask), version, mask.device, mask.dtype, tuple(mask.shape),
                 size, num_tokens, num_conds, batch_size)
    entry = cache.get(cache_key) if version is not None else None
    if entry is not None and entry[0]() is mask:
        return entry[1]

    if mask.dim() == 3:
        m = mask.unsqueeze(1)
    else:
        m = mask
    down = interpolate(m, size=size, mode="bilinear")
    flat = down.reshape(num_conds, -1)
    result = flat.view(num_conds, num_tokens, 1).repeat_interleave(batch_size, dim=0)
    if version is not None:
        if len(cache) >= MASK_CACHE_LIMIT:
            del cache[next(iter(cache))]
        cache[cache_key] = (weakref.ref(mask), result)
    return result
