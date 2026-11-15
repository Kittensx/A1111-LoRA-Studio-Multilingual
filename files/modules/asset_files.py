"""Small, dependency-free disk identity/header helpers shared by model assets."""
import json
import os

MAX_SAFETENSORS_HEADER = 100_000_000
EMBEDDING_EXTENSIONS = frozenset(('.pt', '.bin', '.safetensors', '.png', '.webp', '.jxl', '.avif'))


def file_signature(filename):
    if not filename:
        return None
    path = os.path.normcase(os.path.realpath(filename))
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return None
    return (path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def read_safetensors_header(filename):
    """Bound allocation before reading JSON; this does not validate tensor data."""
    with open(filename, 'rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('INVALID_FILE: safetensors header needs eight length bytes')
        length = int.from_bytes(prefix, 'little')
        if length < 2 or length > MAX_SAFETENSORS_HEADER:
            raise ValueError(f'INVALID_FILE: safetensors header length {length} is outside the safety limit')
        if length > os.fstat(stream.fileno()).st_size - 8:
            raise ValueError('INVALID_FILE: truncated safetensors header')
        data = json.loads(stream.read(length))
    if not isinstance(data, dict):
        raise ValueError('INVALID_FILE: safetensors header must be an object')
    return data


def embedding_files(directory):
    """Deterministic traversal shared by watcher/loader; follow links once only."""
    visited = set()
    for root, dirs, names in os.walk(directory, followlinks=True):
        identity = os.path.normcase(os.path.realpath(root))
        if identity in visited:
            dirs[:] = []
            continue
        visited.add(identity)
        dirs[:] = sorted(dirs, key=lambda name: (name.casefold(), name))
        for name in sorted(names, key=lambda name: (name.casefold(), name)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in EMBEDDING_EXTENSIONS or stem.lower().endswith('.preview'):
                continue
            yield os.path.join(root, name)


def preflight_embedding(filename, expected_shapes):
    """Reject non-TI safetensors before allocating their potentially huge payload."""
    header = read_safetensors_header(filename)
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    dual = 'clip_l' in tensors or 'clip_g' in tensors
    if dual:
        if not all(key in tensors for key in ('clip_l', 'clip_g')):
            raise ValueError('INCOMPATIBLE_ARCH: dual embedding needs clip_l and clip_g')
        selected = {key: tensors[key] for key in ('clip_l', 'clip_g')}
    elif len(tensors) == 1:
        selected = tensors
        if len(expected_shapes) > 1:
            raise ValueError('INCOMPATIBLE_ARCH: this model requires separate encoder vectors')
    else:
        raise ValueError('INCOMPATIBLE_ARCH: safetensors is not a supported textual inversion payload')
    shapes = {}
    for key, value in selected.items():
        shape = value.get('shape') if isinstance(value, dict) else None
        if not isinstance(shape, list) or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError('INVALID_FILE: invalid embedding tensor shape')
        if not dual and len(shape) == 1:
            shape = [1, shape[0]]
        if len(shape) != 2 or value.get('dtype') not in ('F16', 'BF16', 'F32', 'F64'):
            raise ValueError('INVALID_FILE: embedding must be a floating-point vector/matrix')
        shapes[key] = shape
    if dual and shapes['clip_l'][0] != shapes['clip_g'][0]:
        raise ValueError('UNSUPPORTED_DUAL_ROWS: clip_l and clip_g must have equal vector counts')
    for key, width in expected_shapes.items():
        shape = shapes.get(key) if dual else next(iter(shapes.values()))
        if shape is None or shape[1] != width:
            raise ValueError(f'WRONG_DUAL_WIDTH: encoder {key} requires width {width}, got {shape}')
    return shapes
