import os
import collections
from dataclasses import dataclass

from modules import paths, shared, devices, script_callbacks, sd_models, extra_networks, lowvram, sd_hijack, hashes

import glob
from copy import deepcopy


vae_path = os.path.abspath(os.path.join(paths.models_path, "VAE"))
vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}
vae_dict = {}


base_vae = None
loaded_vae_file = None
checkpoint_info = None

checkpoints_loaded = collections.OrderedDict()


def get_loaded_vae_name():
    if loaded_vae_file is None:
        return None

    return os.path.basename(loaded_vae_file)


def get_loaded_vae_hash():
    if loaded_vae_file is None:
        return None

    sha256 = hashes.sha256(loaded_vae_file, 'vae')

    return sha256[0:10] if sha256 else None


def get_base_vae(model):
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info and model:
        return base_vae
    return None


def store_base_vae(model):
    global base_vae, checkpoint_info
    if checkpoint_info != model.sd_checkpoint_info:
        assert not loaded_vae_file, "Trying to store non-base VAE!"
        base_vae = deepcopy(model.first_stage_model.state_dict())
        checkpoint_info = model.sd_checkpoint_info


def delete_base_vae():
    global base_vae, checkpoint_info
    base_vae = None
    checkpoint_info = None


def restore_base_vae(model):
    global loaded_vae_file
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info:
        print("Restoring base VAE")
        _load_vae_dict(model, base_vae)
        loaded_vae_file = None
    delete_base_vae()


def get_filename(filepath):
    return os.path.basename(filepath)


def refresh_vae_list():
    from modules.asset_files import file_signature
    vae_dict.clear()

    paths = [
        os.path.join(sd_models.model_path, '**/*.vae.ckpt'),
        os.path.join(sd_models.model_path, '**/*.vae.pt'),
        os.path.join(sd_models.model_path, '**/*.vae.safetensors'),
        os.path.join(vae_path, '**/*.ckpt'),
        os.path.join(vae_path, '**/*.pt'),
        os.path.join(vae_path, '**/*.safetensors'),
    ]

    if shared.cmd_opts.ckpt_dir is not None and os.path.isdir(shared.cmd_opts.ckpt_dir):
        paths += [
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.ckpt'),
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.pt'),
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.safetensors'),
        ]

    if shared.cmd_opts.vae_dir is not None and os.path.isdir(shared.cmd_opts.vae_dir):
        paths += [
            os.path.join(shared.cmd_opts.vae_dir, '**/*.ckpt'),
            os.path.join(shared.cmd_opts.vae_dir, '**/*.pt'),
            os.path.join(shared.cmd_opts.vae_dir, '**/*.safetensors'),
        ]

    candidates = []
    for path in paths:
        candidates += glob.iglob(path, recursive=True)

    unique = {os.path.normcase(os.path.realpath(path)): os.path.abspath(path) for path in sorted(candidates)}
    groups = collections.defaultdict(list)
    for filepath in unique.values():
        groups[get_filename(filepath)].append(filepath)
    entries = {}
    for name, files in groups.items():
        for filepath in files:
            label = name if len(files) == 1 else f'{name} [{os.path.dirname(filepath)}]'
            entries[label] = filepath
        if len(files) > 1:
            print(f'Duplicate VAE name {name!r}: select a path-qualified entry; no file was hidden.')
    vae_dict.update(sorted(entries.items(), key=lambda item: (shared.natural_sort_key(item[0]), item[0])))
    for path, entry in list(checkpoints_loaded.items()):
        if not isinstance(entry, tuple) or len(entry) != 2 or entry[0] != file_signature(path):
            del checkpoints_loaded[path]


def find_vae_near_checkpoint(checkpoint_file):
    # Exact sidecar in the checkpoint directory, never an arbitrary prefix in
    # another directory. Ambiguous sidecars require an explicit user selection.
    stem = os.path.splitext(os.path.abspath(checkpoint_file))[0]
    matches = [stem + '.vae.' + ext for ext in ('safetensors', 'pt', 'ckpt')
               if os.path.isfile(stem + '.vae.' + ext)]
    if len(matches) > 1:
        print(f'Ambiguous VAE sidecars for {checkpoint_file}: choose one explicitly in settings.')
        return None
    return matches[0] if matches else None


@dataclass
class VaeResolution:
    vae: str = None
    source: str = None
    resolved: bool = True

    def tuple(self):
        return self.vae, self.source


def is_automatic():
    return shared.opts.sd_vae in {"Automatic", "auto"}  # "auto" for people with old config


def resolve_vae_from_setting() -> VaeResolution:
    if shared.opts.sd_vae == "None":
        return VaeResolution()

    vae_from_options = vae_dict.get(shared.opts.sd_vae, None)
    if vae_from_options is not None:
        return VaeResolution(vae_from_options, 'specified in settings')

    if not is_automatic():
        print(f"Couldn't find VAE named {shared.opts.sd_vae}; using None instead")

    return VaeResolution(resolved=False)


def resolve_vae_from_user_metadata(checkpoint_file) -> VaeResolution:
    metadata = extra_networks.get_user_metadata(checkpoint_file)
    vae_metadata = metadata.get("vae", None)
    if vae_metadata is not None and vae_metadata != "Automatic":
        if vae_metadata == "None":
            return VaeResolution()

        vae_from_metadata = vae_dict.get(vae_metadata, None)
        if vae_from_metadata is not None:
            return VaeResolution(vae_from_metadata, "from user metadata")

    return VaeResolution(resolved=False)


def resolve_vae_near_checkpoint(checkpoint_file) -> VaeResolution:
    vae_near_checkpoint = find_vae_near_checkpoint(checkpoint_file)
    if vae_near_checkpoint is not None and (not shared.opts.sd_vae_overrides_per_model_preferences or is_automatic()):
        return VaeResolution(vae_near_checkpoint, 'found near the checkpoint')

    return VaeResolution(resolved=False)


def resolve_vae(checkpoint_file) -> VaeResolution:
    if shared.cmd_opts.vae_path is not None:
        return VaeResolution(shared.cmd_opts.vae_path, 'from commandline argument')

    if shared.opts.sd_vae_overrides_per_model_preferences and not is_automatic():
        return resolve_vae_from_setting()

    res = resolve_vae_from_user_metadata(checkpoint_file)
    if res.resolved:
        return res

    res = resolve_vae_near_checkpoint(checkpoint_file)
    if res.resolved:
        return res

    res = resolve_vae_from_setting()

    return res


def load_vae_dict(filename, map_location):
    vae_ckpt = sd_models.read_state_dict(filename, map_location=map_location)
    vae_dict_1 = {k: v for k, v in vae_ckpt.items() if k[0:4] != "loss" and k not in vae_ignore_keys}
    return vae_dict_1


def load_vae(model, vae_file=None, vae_source="from unknown source", force_reload=False):
    from modules.asset_files import file_signature
    global vae_dict, base_vae, loaded_vae_file
    # save_settings = False

    cache_enabled = shared.opts.sd_vae_checkpoint_cache > 0

    if vae_file:
        signature = file_signature(vae_file)
        if signature is None:
            raise FileNotFoundError(f"VAE {vae_source} doesn't exist: {vae_file}")
        entry = checkpoints_loaded.get(vae_file)
        if cache_enabled and not force_reload and isinstance(entry, tuple) and len(entry) == 2 and entry[0] == signature:
            # use vae checkpoint cache
            print(f"Loading VAE weights {vae_source}: cached {get_filename(vae_file)}")
            store_base_vae(model)
            _load_vae_dict(model, entry[1])
            checkpoints_loaded.move_to_end(vae_file)
        else:
            assert os.path.isfile(vae_file), f"VAE {vae_source} doesn't exist: {vae_file}"
            print(f"Loading VAE weights {vae_source}: {vae_file}")
            store_base_vae(model)

            vae_dict_1 = load_vae_dict(vae_file, map_location=shared.weight_load_location)
            if file_signature(vae_file) != signature:
                raise RuntimeError(f'VAE changed while reading: {vae_file}; retry after the write completes')
            _load_vae_dict(model, vae_dict_1)

            if cache_enabled:
                # cache newly loaded vae
                checkpoints_loaded[vae_file] = (signature, vae_dict_1.copy())

        # clean up cache if limit is reached
        if cache_enabled:
            while len(checkpoints_loaded) > shared.opts.sd_vae_checkpoint_cache + 1: # we need to count the current model
                checkpoints_loaded.popitem(last=False)  # LRU

        # If vae used is not in dict, update it
        # It will be removed on refresh though
        vae_opt = get_filename(vae_file)
        if vae_opt in vae_dict and os.path.realpath(vae_dict[vae_opt]) != os.path.realpath(vae_file):
            vae_opt = f'{vae_opt} [{os.path.dirname(os.path.abspath(vae_file))}]'
        if vae_opt not in vae_dict:
            vae_dict[vae_opt] = vae_file

    elif loaded_vae_file:
        restore_base_vae(model)

    loaded_vae_file = vae_file
    model.base_vae = base_vae
    model.loaded_vae_file = loaded_vae_file
    model.loaded_vae_signature = file_signature(vae_file)


# don't call this from outside
def _load_vae_dict(model, vae_dict_1):
    model.first_stage_model.load_state_dict(vae_dict_1)
    model.first_stage_model.to(devices.dtype_vae)


def clear_loaded_vae():
    global loaded_vae_file
    loaded_vae_file = None


unspecified = object()


def reload_vae_weights(sd_model=None, vae_file=unspecified, force_reload=False):
    from modules.asset_files import file_signature
    if not sd_model:
        sd_model = shared.sd_model

    checkpoint_info = sd_model.sd_checkpoint_info
    checkpoint_file = checkpoint_info.filename

    if vae_file == unspecified:
        vae_file, vae_source = resolve_vae(checkpoint_file).tuple()
    else:
        vae_source = "from function argument"

    if not force_reload and loaded_vae_file == vae_file and getattr(sd_model, 'loaded_vae_signature', None) == file_signature(vae_file):
        return

    try:
        if sd_model.lowvram:
            lowvram.send_everything_to_cpu()
        else:
            sd_model.to(devices.cpu)

        sd_hijack.model_hijack.undo_hijack(sd_model)

        if force_reload:
            load_vae(sd_model, vae_file, vae_source, force_reload=True)
        else:
            load_vae(sd_model, vae_file, vae_source)

        sd_hijack.model_hijack.hijack(sd_model)

        if not sd_model.lowvram:
            sd_model.to(devices.device)

        script_callbacks.model_loaded_callback(sd_model)
    except Exception:
        # This path is also called directly from settings, not only through
        # checkpoint reuse. Partially loaded weights/unhijacked models must not
        # remain available to generation or a subsequent cache hit.
        sd_models.discard_failed_model(sd_model)
        clear_loaded_vae()
        raise

    print("VAE weights loaded.")
    return sd_model
