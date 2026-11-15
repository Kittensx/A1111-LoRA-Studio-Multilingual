import os
import copy
import hashlib
import logging
import tempfile
from collections import namedtuple
from contextlib import closing

import torch
import tqdm
import html
import datetime
import csv
import safetensors.torch

import numpy as np
from PIL import Image, PngImagePlugin

from modules import shared, devices, sd_hijack, sd_models, images, sd_samplers, sd_hijack_checkpoint, errors, hashes, cache
import modules.textual_inversion.dataset
from modules.textual_inversion.learn_schedule import LearnRateScheduler

from modules.textual_inversion.image_embedding import embedding_to_b64, embedding_from_b64, insert_image_data_embed, extract_image_data_embed, caption_image_overlay
from modules.textual_inversion.saving_settings import save_settings_to_file


TextualInversionTemplate = namedtuple("TextualInversionTemplate", ["name", "path"])
textual_inversion_templates = {}


def list_textual_inversion_templates():
    textual_inversion_templates.clear()

    for root, _, fns in os.walk(shared.cmd_opts.textual_inversion_templates_dir):
        for fn in fns:
            path = os.path.join(root, fn)

            textual_inversion_templates[fn] = TextualInversionTemplate(fn, path)

    return textual_inversion_templates


def embedding_file_hash(filename):
    digest = hashlib.sha256()
    with open(filename, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_embedding_files(filename, embedding_data, optimizer_data=None):
    """Stage all serialization before atomically replacing individual files.

    Two filesystem paths cannot be committed atomically together. A strong
    sidecar binding prevents using the old optimizer after a partial commit.
    """
    filename = os.path.abspath(os.fspath(filename))
    staged = []

    def stage(destination, data):
        with tempfile.NamedTemporaryFile(mode='w+b', dir=os.path.dirname(filename),
                                         prefix='.' + os.path.basename(destination) + '.', suffix='.tmp', delete=False) as output:
            staged.append((output.name, destination))
            torch.save(data, output)
            output.flush()
            os.fsync(output.fileno())
        return output.name

    try:
        embedding_temp = stage(filename, embedding_data)
        if optimizer_data is not None:
            stage(filename + '.optim', dict(optimizer_data, embedding_sha256=embedding_file_hash(embedding_temp)))
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass  # successfully committed, or already removed
            except OSError:
                logging.exception("Could not remove temporary embedding file %s", temporary)


def load_embedding_optimizer(filename, embedding):
    sidecar = f'{filename}.optim'
    if not os.path.exists(sidecar):
        return None
    saved = torch.load(sidecar, map_location='cpu')
    if not isinstance(saved, dict) or saved.get('hash') != embedding.checksum():
        return None
    if 'embedding_sha256' in saved:
        if saved['embedding_sha256'] != embedding_file_hash(filename):
            return None
    else:
        # Only genuinely legacy main files may use checksum-only sidecars.
        # New main + old sidecar is possible after an interrupted two-file save.
        main = torch.load(filename, map_location='cpu')
        if not isinstance(main, dict) or 'optimizer_state_binding' in main:
            return None
    return saved.get('optimizer_state_dict')


class Embedding:
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.shape = None
        self.vectors = 0
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.optimizer_state_dict = None
        self.filename = None
        self.hash = None
        self.shorthash = None

    def save(self, filename):
        embedding_data = {
            "name": self.name,
            "step": self.step,
            "sd_checkpoint": self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
            "optimizer_state_binding": "sha256",
        }
        if isinstance(self.vec, dict):
            embedding_data.update(self.validate_sdxl(self.vec))
        else:
            # Preserve the legacy single-encoder format and optimizer checksum.
            embedding_data.update(string_to_token={"*": 265}, string_to_param={"*": self.vec})

        optimizer_saved_dict = None
        if shared.opts.save_optimizer_state and self.optimizer_state_dict is not None:
            optimizer_saved_dict = {
                'hash': self.checksum(),
                'optimizer_state_dict': self.optimizer_state_dict,
            }
        save_embedding_files(filename, embedding_data, optimizer_saved_dict)

    def checksum(self):
        if self.cached_checksum is not None:
            return self.cached_checksum

        if isinstance(self.vec, dict):
            digest = hashlib.sha256()
            for key, value in sorted(self.validate_sdxl(self.vec).items()):
                # Match create_embedding_from_data's canonical float32 dtype.
                value = value.detach().to(device='cpu', dtype=torch.float32).contiguous()
                digest.update(f'{key}:{tuple(value.shape)}:'.encode())
                digest.update(bytes(value.view(torch.uint8).reshape(-1).tolist()))
            self.cached_checksum = digest.hexdigest()
            return self.cached_checksum

        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        self.cached_checksum = f'{const_hash(self.vec.reshape(-1) * 100) & 0xffff:04x}'
        return self.cached_checksum

    @staticmethod
    def validate_tensor(value, label):
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or not value.is_floating_point() or min(value.shape) < 1:
            raise ValueError(f'{label}: expected a nonempty floating-point matrix')
        if value.device.type == 'meta' or not torch.isfinite(value).all().item():
            raise ValueError(f'{label}: embedding vectors must be finite (no NaN/Inf/meta)')
        return value

    @staticmethod
    def encoder_shapes(model):
        """Read the actual TI consumers, not a hardcoded architecture width sum."""
        conditioner = getattr(model, 'cond_stage_model', None)
        shapes = {}
        if conditioner is None or not hasattr(conditioner, 'modules'):
            return shapes
        for layer in conditioner.modules():
            key = getattr(layer, 'textual_inversion_key', None)
            if key is None:
                continue
            # SD3's native Embedding exposes wrapped as a bound forward method.
            # SD1/2/XL use a wrapper whose wrapped is the actual Embedding.
            width = getattr(layer, 'embedding_dim', None)
            if width is None:
                width = getattr(getattr(layer, 'wrapped', None), 'embedding_dim', None)
            if width is None:
                continue
            if key in shapes and shapes[key] != width:
                raise ValueError(f'Ambiguous embedding width for encoder {key}')
            shapes[key] = width
        return shapes

    @staticmethod
    def validate_model(vec, shapes, name):
        if shapes is None:
            return  # Format-only import; a later registration still checks the model.
        if isinstance(vec, dict):
            if not shapes:
                raise ValueError(f'Embedding {name!r}: cannot determine per-encoder widths for this model')
            for key, width in shapes.items():
                value = vec.get(key)
                actual = None if value is None else value.shape[1]
                if actual != width:
                    raise ValueError(f'Embedding {name!r}, encoder {key}: expected width {width}, got {actual}')
        elif len(shapes) > 1:
            raise ValueError(f'Embedding {name!r}: this model requires separate encoder vectors ({", ".join(shapes)})')
        elif shapes:
            key, width = next(iter(shapes.items()))
            if vec.shape[1] != width:
                raise ValueError(f'Embedding {name!r}, encoder {key}: expected width {width}, got {vec.shape[1]}')

    @staticmethod
    def validate_sdxl(data):
        if not all(key in data for key in ('clip_l', 'clip_g')):
            raise ValueError("SDXL embedding requires both clip_l and clip_g tensors")
        vectors = {key: data[key] for key in ('clip_l', 'clip_g')}
        for key, value in vectors.items():
            Embedding.validate_tensor(value, f'SDXL embedding {key}')
        if vectors['clip_l'].shape[0] != vectors['clip_g'].shape[0]:
            raise ValueError("SDXL embedding clip_l and clip_g must have equal vector counts")
        return vectors

    def set_hash(self, v):
        self.hash = v
        self.shorthash = self.hash[0:12]


class DirWithTextualInversionEmbeddings:
    def __init__(self, path):
        self.path = path
        self.mtime = None

    def has_changed(self):
        return self.mtime != self.snapshot()

    def snapshot(self):
        from modules.asset_files import embedding_files, file_signature
        entries = []
        for path in embedding_files(self.path):
            try:
                entries.append((os.path.relpath(path, self.path), file_signature(path)))
            except OSError:
                entries.append((os.path.relpath(path, self.path), None))
        return tuple(sorted(entries))

    def update(self):
        self.mtime = self.snapshot()


class EmbeddingDatabase:
    def __init__(self):
        self.ids_lookup = {}
        self.word_embeddings = {}
        self.skipped_embeddings = {}
        self.expected_shape = -1
        self.embedding_dirs = {}
        self.previously_displayed_embeddings = ()
        self.image_embedding_cache = cache.cache('image-embedding')
        self.revision = 0
        self.model_signature = None
        self.file_errors = {}
        self._reported_file_errors = {}

    def current_model_signature(self):
        model = shared.sd_model
        conditioner = getattr(model, 'cond_stage_model', None)
        return (id(model), id(conditioner), tuple(sorted(Embedding.encoder_shapes(model).items())))

    def ensure_model_compatibility(self):
        if self.model_signature != self.current_model_signature():
            self.load_textual_inversion_embeddings(force_reload=True)

    def add_embedding_dir(self, path):
        self.embedding_dirs[path] = DirWithTextualInversionEmbeddings(path)

    def clear_embedding_dirs(self):
        self.embedding_dirs.clear()

    def register_embedding(self, embedding, model):
        return self.register_embedding_by_name(embedding, model, embedding.name)

    def register_embedding_by_name(self, embedding, model, name):
        if embedding is not None:
            Embedding.validate_model(embedding.vec, Embedding.encoder_shapes(model), name)
        self.revision += 1
        ids = model.cond_stage_model.tokenize([name])[0]
        first_id = ids[0]
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []
        if name in self.word_embeddings:
            # remove old one from the lookup list
            lookup = [x for x in self.ids_lookup[first_id] if x[1].name!=name]
        else:
            lookup = self.ids_lookup[first_id]
        if embedding is not None:
            lookup += [(ids, embedding)]
        self.ids_lookup[first_id] = sorted(lookup, key=lambda x: len(x[0]), reverse=True)
        if embedding is None:
            # unregister embedding with specified name
            if name in self.word_embeddings:
                del self.word_embeddings[name]
            if len(self.ids_lookup[first_id])==0:
                del self.ids_lookup[first_id]
            return None
        self.word_embeddings[name] = embedding
        return embedding

    def get_expected_shape(self):
        devices.torch_npu_set_device()
        vec = shared.sd_model.cond_stage_model.encode_embedding_init_text(",", 1)
        return vec.shape[1]

    def read_embedding_from_image(self, path, name):
        from modules.asset_files import file_signature
        try:
            signature = list(file_signature(path) or ())

            if (cache_embedding := self.image_embedding_cache.get(path)) and signature == cache_embedding.get('signature'):
                # cache will only be used if the file has not been modified time matches
                return cache_embedding.get('data', None), cache_embedding.get('name', None)

            with Image.open(path) as embed_image:
                if hasattr(embed_image, 'text') and 'sd-ti-embedding' in embed_image.text:
                    data = embedding_from_b64(embed_image.text['sd-ti-embedding'])
                    name = data.get('name', name)
                else:
                    data = extract_image_data_embed(embed_image)
                    if data:
                        name = data.get('name', name)

            if data is None or shared.opts.textual_inversion_image_embedding_data_cache:
                # data of image embeddings only will be cached if the option textual_inversion_image_embedding_data_cache is enabled
                # results of images that are not embeddings will allways be cached to reduce unnecessary future disk reads
                self.image_embedding_cache[path] = {'data': data, 'name': None if data is None else name, 'signature': signature}

            return data, name
        except Exception:
            errors.report(f"Error loading embedding {path}", exc_info=True)
        return None, None

    def load_from_file(self, path, filename):
        name, ext = os.path.splitext(filename)
        ext = ext.upper()

        if ext in ['.PNG', '.WEBP', '.JXL', '.AVIF']:
            _, second_ext = os.path.splitext(name)
            if second_ext.upper() == '.PREVIEW':
                return

            data, name = self.read_embedding_from_image(path, name)
            if data is None:
                return

        elif ext in ['.BIN', '.PT']:
            data = torch.load(path, map_location="cpu")
        elif ext in ['.SAFETENSORS']:
            from modules.asset_files import preflight_embedding
            shapes = preflight_embedding(path, Embedding.encoder_shapes(shared.sd_model))
            with safetensors.safe_open(path, framework='pt', device='cpu') as handle:
                data = {key: handle.get_tensor(key) for key in shapes}
        else:
            return

        if data is not None:
            embedding = create_embedding_from_data(data, name, filename=filename, filepath=path,
                                                   expected_shapes=Embedding.encoder_shapes(shared.sd_model))

            if self.expected_shape == -1 or self.expected_shape == embedding.shape:
                previous = self.word_embeddings.get(embedding.name)
                if previous is not None and previous.filename != embedding.filename:
                    raise ValueError(f'DUPLICATE_NAME: {embedding.name!r} already loaded from {previous.filename}; skipped {path}')
                self.register_embedding(embedding, shared.sd_model)
            else:
                self.skipped_embeddings[name] = embedding
        else:
            print(f"Unable to load Textual inversion embedding due to data issue: '{name}'.")

    def record_file_error(self, path, error):
        from modules.asset_files import file_signature
        message = str(error)
        status = message.partition(':')[0]
        if status not in {'INCOMPATIBLE_ARCH', 'WRONG_DUAL_WIDTH', 'UNSUPPORTED_DUAL_ROWS', 'INVALID_FILE', 'DUPLICATE_NAME'}:
            status = ('UNSUPPORTED_DUAL_ROWS' if 'equal vector counts' in message else
                      'INCOMPATIBLE_ARCH' if 'separate encoder vectors' in message else
                      'WRONG_DUAL_WIDTH' if 'expected width' in message else 'INVALID_FILE')
        self.file_errors[path] = {'status': status, 'reason': message}
        try:
            signature = file_signature(path)
        except OSError:
            signature = None
        fingerprint = (signature, self.expected_shape, status, message)
        if self._reported_file_errors.get(path) != fingerprint:
            logging.warning('Skipping embedding %s [%s]: %s', path, status, message)
            self._reported_file_errors[path] = fingerprint

    def load_from_dir(self, embdir):
        from modules.asset_files import embedding_files
        if not os.path.isdir(embdir.path):
            return

        for fullfn in embedding_files(embdir.path):
            try:
                if os.stat(fullfn).st_size == 0:
                    raise ValueError('INVALID_FILE: empty embedding file')
                self.load_from_file(fullfn, os.path.basename(fullfn))
            except (ValueError, OSError) as error:
                self.record_file_error(fullfn, error)
            except Exception as error:
                self.record_file_error(fullfn, error)
                errors.report(f"Error loading embedding {fullfn}", exc_info=True)

    def load_textual_inversion_embeddings(self, force_reload=False):
        if force_reload:
            self.image_embedding_cache.clear()  # explicit refresh also bypasses image cache
        signature = self.current_model_signature()
        force_reload = force_reload or signature != self.model_signature
        if not force_reload:
            need_reload = False
            for embdir in self.embedding_dirs.values():
                if embdir.has_changed():
                    need_reload = True
                    break

            if not need_reload:
                return

        expected_shape = self.get_expected_shape()
        self.revision += 1
        self.model_signature = None  # retry if a non-file-level failure aborts the scan
        self.ids_lookup.clear()
        self.word_embeddings.clear()
        self.skipped_embeddings.clear()
        self.file_errors.clear()
        self.expected_shape = expected_shape

        for embdir in self.embedding_dirs.values():
            snapshot = embdir.snapshot()
            self.load_from_dir(embdir)
            # A file changing during the scan must trigger another scan.
            embdir.mtime = snapshot

        # re-sort word_embeddings because load_from_dir may not load in alphabetic order.
        # using a temporary copy so we don't reinitialize self.word_embeddings in case other objects have a reference to it.
        sorted_word_embeddings = {e.name: e for e in sorted(self.word_embeddings.values(), key=lambda e: e.name.lower())}
        self.word_embeddings.clear()
        self.word_embeddings.update(sorted_word_embeddings)
        self.model_signature = signature
        self._reported_file_errors = {path: value for path, value in self._reported_file_errors.items() if path in self.file_errors}

        displayed_embeddings = (tuple(self.word_embeddings.keys()), tuple(self.skipped_embeddings.keys()))
        if shared.opts.textual_inversion_print_at_load and self.previously_displayed_embeddings != displayed_embeddings:
            self.previously_displayed_embeddings = displayed_embeddings
            print(f"Textual inversion embeddings loaded({len(self.word_embeddings)}): {', '.join(self.word_embeddings.keys())}")
            if self.skipped_embeddings:
                print(f"Textual inversion embeddings skipped({len(self.skipped_embeddings)}): {', '.join(self.skipped_embeddings.keys())}")

    def find_embedding_at_position(self, tokens, offset):
        token = tokens[offset]
        possible_matches = self.ids_lookup.get(token, None)

        if possible_matches is None:
            return None, None

        for ids, embedding in possible_matches:
            if tokens[offset:offset + len(ids)] == ids:
                return embedding, len(ids)

        return None, None


def create_embedding(name, num_vectors_per_token, overwrite_old, init_text='*'):
    cond_model = shared.sd_model.cond_stage_model
    shapes = Embedding.encoder_shapes(shared.sd_model)
    if not isinstance(num_vectors_per_token, int) or isinstance(num_vectors_per_token, bool) or num_vectors_per_token < 1:
        raise ValueError('Embedding vector count must be a positive integer')

    with devices.autocast():
        cond_model([""])  # will send cond model to GPU if lowvram/medvram is active

    #cond_model expects at least some text, so we provide '*' as backup.
    def initialize(encoder):
        embedded = encoder.encode_embedding_init_text(init_text or '*', num_vectors_per_token)
        Embedding.validate_tensor(embedded, 'Embedding initialization')
        result = torch.zeros((num_vectors_per_token, embedded.shape[1]), device=devices.device)
        if init_text:
            for i in range(num_vectors_per_token):
                result[i] = embedded[i * int(embedded.shape[0]) // num_vectors_per_token]
        return result

    if len(shapes) > 1:
        # Obtain each encoder's own initialized vectors. Never guess a split of
        # an arbitrary 2048-wide tensor or imported single-encoder embedding.
        vec = {}
        for encoder in getattr(cond_model, 'embedders', ()):
            if not hasattr(encoder, 'encode_embedding_init_text'):
                continue
            keys = {getattr(layer, 'textual_inversion_key', None) for layer in encoder.modules()}
            keys.discard(None)
            if len(keys) != 1 or next(iter(keys)) in vec:
                raise ValueError('Cannot uniquely identify the encoder for SDXL embedding initialization')
            vec[next(iter(keys))] = initialize(encoder)
        Embedding.validate_sdxl(vec)
    else:
        vec = initialize(cond_model)
    Embedding.validate_model(vec, shapes, name)

    # Remove illegal characters from name.
    name = "".join( x for x in name if (x.isalnum() or x in "._- "))
    fn = os.path.join(shared.cmd_opts.embeddings_dir, f"{name}.pt")
    if not overwrite_old:
        assert not os.path.exists(fn), f"file {fn} already exists"

    embedding = Embedding(vec, name)
    embedding.step = 0
    embedding.save(fn)

    return fn


def create_embedding_from_data(data, name, filename='unknown embedding file', filepath=None, expected_shapes=None):
    if not isinstance(data, dict) or not data:
        raise ValueError(f'Embedding {name!r} ({filename}): expected a nonempty dictionary')
    if 'string_to_param' in data:  # textual inversion embeddings
        param_dict = data['string_to_param']
        param_dict = getattr(param_dict, '_parameters', param_dict)  # fix for torch 1.12.1 loading saved file from torch 1.11
        if len(param_dict) != 1:
            raise ValueError(f"Embedding {name!r}: expected one string_to_param term, got {len(param_dict)}. Multiple vectors for one term are supported; export separate files for multiple terms.")
        emb = next(iter(param_dict.items()))[1]
        vec = Embedding.validate_tensor(emb, f'Embedding {name!r}').detach().to(device='cpu', dtype=torch.float32)
        shape = vec.shape[-1]
        vectors = vec.shape[0]
    elif 'clip_g' in data or 'clip_l' in data:  # SDXL embedding (both parts required)
        # Metadata such as name/step/checkpoint is not an encoder tensor.
        vec = {k: v.detach().to(device='cpu', dtype=torch.float32) for k, v in Embedding.validate_sdxl(data).items()}
        shape = data['clip_g'].shape[-1] + data['clip_l'].shape[-1]
        vectors = data['clip_g'].shape[0]
    elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:  # diffuser concepts
        assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

        emb = next(iter(data.values()))
        if len(emb.shape) == 1:
            emb = emb.unsqueeze(0)
        vec = Embedding.validate_tensor(emb, f'Embedding {name!r}').detach().to(device='cpu', dtype=torch.float32)
        shape = vec.shape[-1]
        vectors = vec.shape[0]
    else:
        raise Exception(f"Couldn't identify {filename} as neither textual inversion embedding nor diffuser concept.")

    # Validate the float32 representation too: finite float64 can overflow on
    # conversion. Reject before allocating on the generation device/registering.
    if isinstance(vec, dict):
        Embedding.validate_sdxl(vec)
    else:
        Embedding.validate_tensor(vec, f'Embedding {name!r}')
    Embedding.validate_model(vec, expected_shapes, name)
    vec = {key: value.to(devices.device) for key, value in vec.items()} if isinstance(vec, dict) else vec.to(devices.device)
    embedding = Embedding(vec, name)
    embedding.step = data.get('step', None)
    embedding.sd_checkpoint = data.get('sd_checkpoint', None)
    embedding.sd_checkpoint_name = data.get('sd_checkpoint_name', None)
    embedding.vectors = vectors
    embedding.shape = shape

    if filepath:
        embedding.filename = filepath
        embedding.set_hash(hashes.sha256(filepath, "textual_inversion/" + name) or '')

    return embedding


def write_loss(log_directory, filename, step, epoch_len, values):
    if shared.opts.training_write_csv_every == 0:
        return

    if step % shared.opts.training_write_csv_every != 0:
        return
    write_csv_header = False if os.path.exists(os.path.join(log_directory, filename)) else True

    with open(os.path.join(log_directory, filename), "a+", newline='') as fout:
        csv_writer = csv.DictWriter(fout, fieldnames=["step", "epoch", "epoch_step", *(values.keys())])

        if write_csv_header:
            csv_writer.writeheader()

        epoch = (step - 1) // epoch_len
        epoch_step = (step - 1) % epoch_len

        csv_writer.writerow({
            "step": step,
            "epoch": epoch,
            "epoch_step": epoch_step,
            **values,
        })

def tensorboard_setup(log_directory):
    from torch.utils.tensorboard import SummaryWriter
    os.makedirs(os.path.join(log_directory, "tensorboard"), exist_ok=True)
    return SummaryWriter(
            log_dir=os.path.join(log_directory, "tensorboard"),
            flush_secs=shared.opts.training_tensorboard_flush_every)

def tensorboard_add(tensorboard_writer, loss, global_step, step, learn_rate, epoch_num):
    tensorboard_add_scaler(tensorboard_writer, "Loss/train", loss, global_step)
    tensorboard_add_scaler(tensorboard_writer, f"Loss/train/epoch-{epoch_num}", loss, step)
    tensorboard_add_scaler(tensorboard_writer, "Learn rate/train", learn_rate, global_step)
    tensorboard_add_scaler(tensorboard_writer, f"Learn rate/train/epoch-{epoch_num}", learn_rate, step)

def tensorboard_add_scaler(tensorboard_writer, tag, value, step):
    tensorboard_writer.add_scalar(tag=tag,
        scalar_value=value, global_step=step)

def tensorboard_add_image(tensorboard_writer, tag, pil_image, step):
    # Convert a pil image to a torch tensor
    img_tensor = torch.as_tensor(np.array(pil_image, copy=True))
    img_tensor = img_tensor.view(pil_image.size[1], pil_image.size[0],
        len(pil_image.getbands()))
    img_tensor = img_tensor.permute((2, 0, 1))

    tensorboard_writer.add_image(tag, img_tensor, global_step=step)

def validate_train_inputs(model_name, learn_rate, batch_size, gradient_step, data_root, template_file, template_filename, steps, save_model_every, create_image_every, log_directory, name="embedding"):
    assert model_name, f"{name} not selected"
    assert learn_rate, "Learning rate is empty or 0"
    assert isinstance(batch_size, int), "Batch size must be integer"
    assert batch_size > 0, "Batch size must be positive"
    assert isinstance(gradient_step, int), "Gradient accumulation step must be integer"
    assert gradient_step > 0, "Gradient accumulation step must be positive"
    assert data_root, "Dataset directory is empty"
    assert os.path.isdir(data_root), "Dataset directory doesn't exist"
    assert os.listdir(data_root), "Dataset directory is empty"
    assert template_filename, "Prompt template file not selected"
    assert template_file, f"Prompt template file {template_filename} not found"
    assert os.path.isfile(template_file.path), f"Prompt template file {template_filename} doesn't exist"
    assert steps, "Max steps is empty or 0"
    assert isinstance(steps, int), "Max steps must be integer"
    assert steps > 0, "Max steps must be positive"
    assert isinstance(save_model_every, int), "Save {name} must be integer"
    assert save_model_every >= 0, "Save {name} must be positive or 0"
    assert isinstance(create_image_every, int), "Create image must be integer"
    assert create_image_every >= 0, "Create image must be positive or 0"
    if save_model_every or create_image_every:
        assert log_directory, "Log directory is empty"


def train_embedding(id_task, embedding_name, learn_rate, batch_size, gradient_step, data_root, log_directory, training_width, training_height, varsize, steps, clip_grad_mode, clip_grad_value, shuffle_tags, tag_drop_out, latent_sampling_method, use_weight, create_image_every, save_embedding_every, template_filename, save_image_with_stored_embedding, preview_from_txt2img, preview_prompt, preview_negative_prompt, preview_steps, preview_sampler_name, preview_cfg_scale, preview_seed, preview_width, preview_height):
    from modules import processing

    save_embedding_every = save_embedding_every or 0
    create_image_every = create_image_every or 0
    template_file = textual_inversion_templates.get(template_filename, None)
    validate_train_inputs(embedding_name, learn_rate, batch_size, gradient_step, data_root, template_file, template_filename, steps, save_embedding_every, create_image_every, log_directory, name="embedding")
    template_file = template_file.path

    hijack = sd_hijack.model_hijack
    embedding = hijack.embedding_db.word_embeddings[embedding_name]
    if isinstance(embedding.vec, dict):
        raise ValueError("SDXL Textual Inversion training is not supported by this single-encoder trainer. Loading and saving SDXL embeddings is supported.")

    shared.state.job = "train-embedding"
    shared.state.textinfo = "Initializing textual inversion training..."
    shared.state.job_count = steps

    filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')

    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), embedding_name)
    unload = shared.opts.unload_models_when_training

    if save_embedding_every > 0:
        embedding_dir = os.path.join(log_directory, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
    else:
        embedding_dir = None

    if create_image_every > 0:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    if create_image_every > 0 and save_image_with_stored_embedding:
        images_embeds_dir = os.path.join(log_directory, "image_embeddings")
        os.makedirs(images_embeds_dir, exist_ok=True)
    else:
        images_embeds_dir = None

    checkpoint = sd_models.select_checkpoint()

    initial_step = embedding.step or 0
    if initial_step >= steps:
        shared.state.textinfo = "Model has already been trained beyond specified max steps"
        return embedding, filename

    scheduler = LearnRateScheduler(learn_rate, steps, initial_step)
    clip_grad = torch.nn.utils.clip_grad_value_ if clip_grad_mode == "value" else \
        torch.nn.utils.clip_grad_norm_ if clip_grad_mode == "norm" else \
        None
    if clip_grad:
        clip_grad_sched = LearnRateScheduler(clip_grad_value, steps, initial_step, verbose=False)
    # dataset loading may take a while, so input validations and early returns should be done before this
    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    old_parallel_processing_allowed = shared.parallel_processing_allowed

    tensorboard_writer = None
    if shared.opts.training_enable_tensorboard:
        try:
            tensorboard_writer = tensorboard_setup(log_directory)
        except ImportError:
            errors.report("Error initializing tensorboard", exc_info=True)

    pin_memory = shared.opts.pin_memory

    ds = modules.textual_inversion.dataset.PersonalizedBase(data_root=data_root, width=training_width, height=training_height, repeats=shared.opts.training_image_repeats_per_epoch, placeholder_token=embedding_name, model=shared.sd_model, cond_model=shared.sd_model.cond_stage_model, device=devices.device, template_file=template_file, batch_size=batch_size, gradient_step=gradient_step, shuffle_tags=shuffle_tags, tag_drop_out=tag_drop_out, latent_sampling_method=latent_sampling_method, varsize=varsize, use_weight=use_weight)

    if shared.opts.save_training_settings_to_txt:
        save_settings_to_file(log_directory, {**dict(model_name=checkpoint.model_name, model_hash=checkpoint.shorthash, num_of_dataset_images=len(ds), num_vectors_per_token=len(embedding.vec)), **locals()})

    latent_sampling_method = ds.latent_sampling_method

    dl = modules.textual_inversion.dataset.PersonalizedDataLoader(ds, latent_sampling_method=latent_sampling_method, batch_size=ds.batch_size, pin_memory=pin_memory)

    if unload:
        shared.parallel_processing_allowed = False
        shared.sd_model.first_stage_model.to(devices.cpu)

    embedding.vec.requires_grad = True
    optimizer = torch.optim.AdamW([embedding.vec], lr=scheduler.learn_rate, weight_decay=0.0)
    if shared.opts.save_optimizer_state:
        optimizer_state_dict = load_embedding_optimizer(filename, embedding)

        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
            print("Loaded existing optimizer from checkpoint")
        else:
            print("No saved optimizer exists in checkpoint")

    scaler = torch.cuda.amp.GradScaler()

    batch_size = ds.batch_size
    gradient_step = ds.gradient_step
    # n steps = batch_size * gradient_step * n image processed
    steps_per_epoch = len(ds) // batch_size // gradient_step
    max_steps_per_epoch = len(ds) // batch_size - (len(ds) // batch_size) % gradient_step
    loss_step = 0
    _loss_step = 0 #internal

    last_saved_file = "<none>"
    last_saved_image = "<none>"
    forced_filename = "<none>"
    embedding_yet_to_be_embedded = False

    is_training_inpainting_model = shared.sd_model.model.conditioning_key in {'hybrid', 'concat'}
    img_c = None

    pbar = tqdm.tqdm(total=steps - initial_step)
    try:
        sd_hijack_checkpoint.add()

        for _ in range((steps-initial_step) * gradient_step):
            if scheduler.finished:
                break
            if shared.state.interrupted:
                break
            for j, batch in enumerate(dl):
                # works as a drop_last=True for gradient accumulation
                if j == max_steps_per_epoch:
                    break
                scheduler.apply(optimizer, embedding.step)
                if scheduler.finished:
                    break
                if shared.state.interrupted:
                    break

                if clip_grad:
                    clip_grad_sched.step(embedding.step)

                with devices.autocast():
                    x = batch.latent_sample.to(devices.device, non_blocking=pin_memory)
                    if use_weight:
                        w = batch.weight.to(devices.device, non_blocking=pin_memory)
                    c = shared.sd_model.cond_stage_model(batch.cond_text)

                    if is_training_inpainting_model:
                        if img_c is None:
                            img_c = processing.txt2img_image_conditioning(shared.sd_model, c, training_width, training_height)

                        cond = {"c_concat": [img_c], "c_crossattn": [c]}
                    else:
                        cond = c

                    if use_weight:
                        loss = shared.sd_model.weighted_forward(x, cond, w)[0] / gradient_step
                        del w
                    else:
                        loss = shared.sd_model.forward(x, cond)[0] / gradient_step
                    del x

                    _loss_step += loss.item()
                scaler.scale(loss).backward()

                # go back until we reach gradient accumulation steps
                if (j + 1) % gradient_step != 0:
                    continue

                if clip_grad:
                    clip_grad(embedding.vec, clip_grad_sched.learn_rate)

                scaler.step(optimizer)
                scaler.update()
                embedding.step += 1
                pbar.update()
                optimizer.zero_grad(set_to_none=True)
                loss_step = _loss_step
                _loss_step = 0

                steps_done = embedding.step

                epoch_num = embedding.step // steps_per_epoch
                epoch_step = embedding.step % steps_per_epoch

                description = f"Training textual inversion [Epoch {epoch_num}: {epoch_step+1}/{steps_per_epoch}] loss: {loss_step:.7f}"
                pbar.set_description(description)
                if embedding_dir is not None and steps_done % save_embedding_every == 0:
                    # Before saving, change name to match current checkpoint.
                    embedding_name_every = f'{embedding_name}-{steps_done}'
                    last_saved_file = os.path.join(embedding_dir, f'{embedding_name_every}.pt')
                    save_embedding(embedding, optimizer, checkpoint, embedding_name_every, last_saved_file, remove_cached_checksum=True)
                    embedding_yet_to_be_embedded = True

                write_loss(log_directory, "textual_inversion_loss.csv", embedding.step, steps_per_epoch, {
                    "loss": f"{loss_step:.7f}",
                    "learn_rate": scheduler.learn_rate
                })

                if images_dir is not None and steps_done % create_image_every == 0:
                    forced_filename = f'{embedding_name}-{steps_done}'
                    last_saved_image = os.path.join(images_dir, forced_filename)

                    shared.sd_model.first_stage_model.to(devices.device)

                    p = processing.StableDiffusionProcessingTxt2Img(
                        sd_model=shared.sd_model,
                        do_not_save_grid=True,
                        do_not_save_samples=True,
                        do_not_reload_embeddings=True,
                    )

                    if preview_from_txt2img:
                        p.prompt = preview_prompt
                        p.negative_prompt = preview_negative_prompt
                        p.steps = preview_steps
                        p.sampler_name = sd_samplers.samplers_map[preview_sampler_name.lower()]
                        p.cfg_scale = preview_cfg_scale
                        p.seed = preview_seed
                        p.width = preview_width
                        p.height = preview_height
                    else:
                        p.prompt = batch.cond_text[0]
                        p.steps = 20
                        p.width = training_width
                        p.height = training_height

                    preview_text = p.prompt

                    with closing(p):
                        processed = processing.process_images(p)
                        image = processed.images[0] if len(processed.images) > 0 else None

                    if unload:
                        shared.sd_model.first_stage_model.to(devices.cpu)

                    if image is not None:
                        shared.state.assign_current_image(image)

                        last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt, shared.opts.samples_format, processed.infotexts[0], p=p, forced_filename=forced_filename, save_to_dirs=False)
                        last_saved_image += f", prompt: {preview_text}"

                        if tensorboard_writer and shared.opts.training_tensorboard_save_images:
                            tensorboard_add_image(tensorboard_writer, f"Validation at epoch {epoch_num}", image, embedding.step)

                    if image is not None and save_image_with_stored_embedding and os.path.exists(last_saved_file) and embedding_yet_to_be_embedded:

                        last_saved_image_chunks = os.path.join(images_embeds_dir, f'{embedding_name}-{steps_done}.png')

                        info = PngImagePlugin.PngInfo()
                        data = torch.load(last_saved_file)
                        info.add_text("sd-ti-embedding", embedding_to_b64(data))

                        title = f"<{data.get('name', '???')}>"

                        try:
                            vectorSize = list(data['string_to_param'].values())[0].shape[0]
                        except Exception:
                            vectorSize = '?'

                        checkpoint = sd_models.select_checkpoint()
                        footer_left = checkpoint.model_name
                        footer_mid = f'[{checkpoint.shorthash}]'
                        footer_right = f'{vectorSize}v {steps_done}s'

                        captioned_image = caption_image_overlay(image, title, footer_left, footer_mid, footer_right)
                        captioned_image = insert_image_data_embed(captioned_image, data)

                        captioned_image.save(last_saved_image_chunks, "PNG", pnginfo=info)
                        embedding_yet_to_be_embedded = False

                shared.state.job_no = embedding.step

                shared.state.textinfo = f"""
<p>
Loss: {loss_step:.7f}<br/>
Step: {steps_done}<br/>
Last prompt: {html.escape(batch.cond_text[0])}<br/>
Last saved embedding: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""
        filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')
        save_embedding(embedding, optimizer, checkpoint, embedding_name, filename, remove_cached_checksum=True)
    except Exception:
        errors.report("Error training embedding", exc_info=True)
    finally:
        pbar.leave = False
        pbar.close()
        shared.sd_model.first_stage_model.to(devices.device)
        shared.parallel_processing_allowed = old_parallel_processing_allowed
        sd_hijack_checkpoint.remove()

    return embedding, filename


def save_embedding(embedding, optimizer, checkpoint, embedding_name, filename, remove_cached_checksum=True):
    snapshot = copy.copy(embedding)
    snapshot.vec = ({key: value.detach().clone() for key, value in embedding.vec.items()}
                    if isinstance(embedding.vec, dict) else embedding.vec.detach().clone())
    snapshot.sd_checkpoint = checkpoint.shorthash
    snapshot.sd_checkpoint_name = checkpoint.model_name
    if remove_cached_checksum:
        snapshot.cached_checksum = None
    snapshot.name = embedding_name
    snapshot.optimizer_state_dict = optimizer.state_dict()
    snapshot.save(filename)
