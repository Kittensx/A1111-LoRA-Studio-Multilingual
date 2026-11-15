import torch

import networks
from modules import patches


class LoraPatches:
    def __init__(self):
        self._installed = []
        try:
            for name in ('Linear', 'Embedding', 'Conv2d', 'GroupNorm', 'LayerNorm', 'MultiheadAttention'):
                module_type = getattr(torch.nn, name)
                for method, field in (('forward', 'forward'), ('load_state_dict', '_load_from_state_dict')):
                    attribute = f'{name}_{method}'
                    original = patches.patch(__name__, module_type, field, getattr(networks, f'network_{attribute}'))
                    self._installed.append((module_type, field, attribute))
                    setattr(self, attribute, original)
        except BaseException:
            # A late conflict must not strand the earlier hooks when this
            # constructor never returns its original-method holder.
            for module_type, field, attribute in reversed(self._installed):
                try:
                    patches.undo(__name__, module_type, field)
                    setattr(self, attribute, None)
                except Exception:
                    import logging
                    logging.exception("Could not roll back LoRA hook %s.%s", module_type.__name__, field)
            raise

    def undo(self):
        # Check ownership for the entire set first. A foreign wrapper on one
        # target must not leave this holder only partially uninstalled.
        patches.undo_many(__name__, [(module_type, field) for module_type, field, _ in reversed(self._installed)])
        for _, _, attribute in self._installed:
            setattr(self, attribute, None)
        self._installed.clear()

