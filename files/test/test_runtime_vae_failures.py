"""VAE reload lifecycle tests using production functions and service doubles."""
import unittest
from types import SimpleNamespace as NS

import test_runtime_stability as harness


class VaeFailureTests(unittest.TestCase):
    def fixture(self, fail_at=None, lowvram=False):
        info = NS(filename='checkpoint')
        events = []
        primary = RuntimeError('injected ' + str(fail_at))
        def event(name):
            events.append(name)
            if name == fail_at:
                raise primary
        model = NS(sd_checkpoint_info=info, lowvram=lowvram, to=lambda device: event(device))
        other = object()
        data = NS(sd_model=model, loaded_sd_models=[model, other], was_loaded_at_least_once=True)
        registry = harness.selected('modules/sd_models.py', {'discard_failed_model'}, model_data=data)
        sentinel = object()
        mod = harness.selected('modules/sd_vae.py', {'reload_vae_weights', 'clear_loaded_vae'},
            unspecified=sentinel, loaded_vae_file='old.vae', sd_models=registry,
            shared=NS(sd_model=model), devices=NS(cpu='cpu', device='gpu'),
            lowvram=NS(send_everything_to_cpu=lambda: event('lowvram')),
            sd_hijack=NS(model_hijack=NS(undo_hijack=lambda model: event('undo'), hijack=lambda model: event('hijack'))),
            script_callbacks=NS(model_loaded_callback=lambda model: event('callback')),
            resolve_vae=lambda *a: NS(tuple=lambda: ('new.vae', 'test')))
        def load(*args):
            mod.loaded_vae_file = 'new.vae'  # emulate partial publication
            event('weights')
        mod.load_vae = load
        return mod, model, other, data, events, primary

    def test_failed_mutation_quarantines_model_and_keeps_original_error(self):
        for stage in ('cpu', 'undo', 'weights', 'hijack', 'gpu', 'callback', 'lowvram'):
            with self.subTest(stage=stage):
                mod, model, other, data, _, error = self.fixture(stage, lowvram=stage == 'lowvram')
                with self.assertRaises(RuntimeError) as caught:
                    mod.reload_vae_weights(model)
                self.assertIs(caught.exception, error)
                self.assertIsNone(data.sd_model)
                self.assertEqual(data.loaded_sd_models, [other])
                self.assertIsNone(mod.loaded_vae_file)

    def test_resolve_failure_does_not_discard_untouched_model(self):
        mod, model, _, data, events, error = self.fixture('resolve')
        def fail(*a):
            raise error
        mod.resolve_vae = fail
        with self.assertRaises(RuntimeError) as caught:
            mod.reload_vae_weights(model)
        self.assertIs(caught.exception, error)
        self.assertIs(data.sd_model, model)
        self.assertEqual(events, [])
        self.assertEqual(mod.loaded_vae_file, 'old.vae')

    def test_success_and_noop_keep_valid_model(self):
        for lowvram in (True, False):
            with self.subTest(lowvram=lowvram):
                mod, model, _, data, events, _ = self.fixture(lowvram=lowvram)
                self.assertIs(mod.reload_vae_weights(model), model)
                self.assertIs(data.sd_model, model)
                self.assertEqual(events, ['lowvram', 'undo', 'weights', 'hijack', 'callback'] if lowvram else
                                 ['cpu', 'undo', 'weights', 'hijack', 'gpu', 'callback'])
                events.clear()
                mod.reload_vae_weights(model)
                self.assertEqual(events, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
