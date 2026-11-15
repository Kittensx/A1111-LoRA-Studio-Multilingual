from collections import namedtuple

import torch
from modules import devices, shared

module_in_gpu = None
cpu = torch.device("cpu")

ModuleWithParent = namedtuple('ModuleWithParent', ['module', 'parent'], defaults=['None'])

def send_everything_to_cpu():
    global module_in_gpu

    if module_in_gpu is not None:
        module_in_gpu.to(cpu)

    module_in_gpu = None


def is_needed(sd_model):
    return shared.cmd_opts.lowvram or shared.cmd_opts.medvram or shared.cmd_opts.medvram_sdxl and hasattr(sd_model, 'conditioner')


def apply(sd_model):
    enable = is_needed(sd_model)

    if enable:
        setup_for_low_vram(sd_model, not shared.cmd_opts.lowvram)
    else:
        sd_model.lowvram = False
    shared.parallel_processing_allowed = not enable


def setup_for_low_vram(sd_model, use_medvram):
    if getattr(sd_model, 'lowvram', False):
        return

    parents = {}
    handles = []
    method_slots = []

    def register(module):
        handles.append(module.register_forward_pre_hook(send_me_to_gpu))

    def replace_method(module, name, replacement):
        method_slots.append((module, name, name in vars(module), vars(module).get(name)))
        setattr(module, name, replacement)

    def transfer_without(target, fields):
        stored = []
        try:
            for obj, field in fields:
                # An optional field which never existed must not appear as None.
                present = hasattr(obj, field)
                stored.append((obj, field, present, getattr(obj, field, None)))
                setattr(obj, field, None)
            target.to(devices.device)
        finally:
            for obj, field, present, value in reversed(stored):
                if present:
                    setattr(obj, field, value)
                else:
                    delattr(obj, field)

    def send_me_to_gpu(module, _):
        """send this module to GPU; send whatever tracked module was previous in GPU to CPU;
        we add this as forward_pre_hook to a lot of modules and this way all but one of them will
        be in CPU
        """
        global module_in_gpu

        module = parents.get(module, module)

        if module_in_gpu == module:
            return

        if module_in_gpu is not None:
            module_in_gpu.to(cpu)

        module.to(devices.device)
        module_in_gpu = module

    # see below for register_forward_pre_hook;
    # first_stage_model does not use forward(), it uses encode/decode, so register_forward_pre_hook is
    # useless here, and we just replace those methods

    first_stage_model = sd_model.first_stage_model
    first_stage_model_encode = sd_model.first_stage_model.encode
    first_stage_model_decode = sd_model.first_stage_model.decode

    def first_stage_model_encode_wrap(x):
        send_me_to_gpu(first_stage_model, None)
        return first_stage_model_encode(x)

    def first_stage_model_decode_wrap(z):
        send_me_to_gpu(first_stage_model, None)
        return first_stage_model_decode(z)

    to_remain_in_cpu = [
        (sd_model, 'first_stage_model'),
        (sd_model, 'depth_model'),
        (sd_model, 'embedder'),
        (sd_model, 'model'),
    ]

    is_sdxl = hasattr(sd_model, 'conditioner')
    is_sd2 = not is_sdxl and hasattr(sd_model.cond_stage_model, 'model')

    if hasattr(sd_model, 'medvram_fields'):
        to_remain_in_cpu = sd_model.medvram_fields()
    elif is_sdxl:
        to_remain_in_cpu.append((sd_model, 'conditioner'))
    elif is_sd2:
        to_remain_in_cpu.append((sd_model.cond_stage_model, 'model'))
    else:
        to_remain_in_cpu.append((sd_model.cond_stage_model, 'transformer'))

    # remove several big modules: cond, first_stage, depth/embedder (if applicable), and unet from the model
    transfer_without(sd_model, to_remain_in_cpu)

    # register hooks for those the first three models
    try:
        if hasattr(sd_model, "cond_stage_model") and hasattr(sd_model.cond_stage_model, "medvram_modules"):
            for module in sd_model.cond_stage_model.medvram_modules():
                if isinstance(module, ModuleWithParent):
                    parent = module.parent
                    module = module.module
                else:
                    parent = None
                if module:
                    register(module)
                    if parent:
                        parents[module] = parent
        elif is_sdxl:
            register(sd_model.conditioner)
        elif is_sd2:
            register(sd_model.cond_stage_model.model)
            register(sd_model.cond_stage_model.model.token_embedding)
            parents[sd_model.cond_stage_model.model] = sd_model.cond_stage_model
            parents[sd_model.cond_stage_model.model.token_embedding] = sd_model.cond_stage_model
        else:
            register(sd_model.cond_stage_model.transformer)
            parents[sd_model.cond_stage_model.transformer] = sd_model.cond_stage_model

        register(sd_model.first_stage_model)
        replace_method(sd_model.first_stage_model, 'encode', first_stage_model_encode_wrap)
        replace_method(sd_model.first_stage_model, 'decode', first_stage_model_decode_wrap)
        if getattr(sd_model, 'depth_model', None) is not None:
            register(sd_model.depth_model)
        if getattr(sd_model, 'embedder', None) is not None:
            register(sd_model.embedder)

        if use_medvram:
            register(sd_model.model)
        else:
            diff_model = sd_model.model.diffusion_model
            transfer_without(sd_model.model, [(diff_model, name) for name in
                             ('input_blocks', 'middle_block', 'output_blocks', 'time_embed')])
            register(diff_model.time_embed)
            for block in diff_model.input_blocks:
                register(block)
            register(diff_model.middle_block)
            for block in diff_model.output_blocks:
                register(block)
    except BaseException:
        for handle in reversed(handles):
            handle.remove()
        for module, name, owned, value in reversed(method_slots):
            if owned:
                setattr(module, name, value)
            else:
                delattr(module, name)
        raise
    sd_model.lowvram = True


def is_enabled(sd_model):
    return sd_model.lowvram
