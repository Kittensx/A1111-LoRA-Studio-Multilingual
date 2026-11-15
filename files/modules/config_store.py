"""Process-local serialized config writes with atomic file replacement.

No WebUI/Torch imports. The lock coordinates Options.save/set_and_save callers;
external processes and direct mutations of Options.data are not transactional.
"""
import json
import logging
import os
import tempfile
import threading


config_lock = threading.RLock()


def atomic_save(filename, data):
    # Serialize before touching disk; a bad value must leave the old file intact.
    payload = json.dumps(data, indent=4, ensure_ascii=False)
    filename = os.path.realpath(os.path.abspath(os.fspath(filename)))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', dir=os.path.dirname(filename),
                                         prefix='.' + os.path.basename(filename) + '.', suffix='.tmp', delete=False) as output:
            temporary = output.name
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        # Preserve existing permission bits; a new config keeps mkstemp's mode.
        if os.path.exists(filename):
            import stat
            os.chmod(temporary, stat.S_IMODE(os.stat(filename).st_mode))
        os.replace(temporary, filename)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            except OSError:
                logging.exception('Could not remove staged config file %s', temporary)
