import hashlib

from modules import shared, errors
import modules.cache

dump_cache = modules.cache.dump_cache
cache = modules.cache.cache


def calculate_sha256(filename):
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def sha256_from_cache(filename, title, use_addnet_hash=False):
    from modules.asset_files import file_signature
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")
    signature = file_signature(filename)
    if signature is None:
        return None

    entry = hashes.get(title)
    if not isinstance(entry, dict) or entry.get('file_signature_v1') != list(signature):
        return None

    return entry.get('sha256')


def sha256(filename, title, use_addnet_hash=False):
    from modules.asset_files import file_signature
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")

    sha256_value = sha256_from_cache(filename, title, use_addnet_hash)
    if sha256_value is not None:
        return sha256_value

    if shared.cmd_opts.no_hashing:
        return None

    signature = file_signature(filename)
    if signature is None:
        raise FileNotFoundError(filename)
    print(f"Calculating sha256 for {filename}: ", end='')
    if use_addnet_hash:
        with open(filename, "rb") as file:
            sha256_value = addnet_hash_safetensors(file)
    else:
        sha256_value = calculate_sha256(filename)
    if file_signature(filename) != signature:
        raise RuntimeError(f'File changed while hashing: {filename}; retry after the write completes')
    print(f"{sha256_value}")

    hashes[title] = {
        "mtime": signature[4] / 1_000_000_000,
        "file_signature_v1": list(signature),
        "sha256": sha256_value,
    }

    dump_cache()

    return sha256_value


def addnet_hash_safetensors(b):
    """kohya-ss hash for safetensors from https://github.com/kohya-ss/sd-scripts/blob/main/library/train_util.py"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def partial_hash_from_cache(filename, *, ignore_cache: bool = False, digits: int = 8):
    """old hash that only looks at a small part of the file and is prone to collisions
    kept for compatibility, don't use this for new things
    """
    from modules.asset_files import file_signature
    try:
        filename = str(filename)
        signature = file_signature(filename)
        if signature is None:
            return 'NOFILE'
        hashes = cache('partial-hash')
        cache_entry = hashes.get(filename, {})
        cache_hash = cache_entry.get("hash", None)
        if cache_entry.get('file_signature_v1') == list(signature) and cache_hash and not ignore_cache:
            return cache_hash[0:digits]

        with open(filename, 'rb') as file:
            m = hashlib.sha256()
            file.seek(0x100000)
            m.update(file.read(0x10000))
            partial_hash = m.hexdigest()
            if file_signature(filename) != signature:
                raise RuntimeError(f'File changed while hashing: {filename}')
            hashes[filename] = {'mtime': signature[4] / 1_000_000_000, 'file_signature_v1': list(signature), 'hash': partial_hash}
            return partial_hash[0:digits]

    except FileNotFoundError:
        pass
    except Exception:
        errors.report(f'Error calculating partial hash for {filename}', exc_info=True)
    return 'NOFILE'
