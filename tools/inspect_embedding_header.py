"""Read safetensors headers only; never unpickle .pt or load tensor payloads.

python tools/inspect_embedding_header.py /content/A1111/embeddings
Optional: --clip-l 768 --clip-g 1280 to check those expected encoder widths.
Header compatibility is not tensor-data integrity or proof of training lineage.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'files'))
from modules.asset_files import embedding_files, preflight_embedding  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--clip-l', type=int)
    parser.add_argument('--clip-g', type=int)
    args = parser.parse_args()
    if not args.path.exists():
        parser.error('Path does not exist')
    expected = {key: value for key, value in (('clip_l', args.clip_l), ('clip_g', args.clip_g)) if value is not None}
    if any(value < 1 for value in expected.values()):
        parser.error('Encoder width must be positive')
    files = [str(args.path)] if args.path.is_file() else embedding_files(args.path)
    for name in files:
        if Path(name).suffix.lower() != '.safetensors':
            continue
        try:
            result = dict(path=name, status='HEADER_COMPATIBLE_ONLY', shapes=preflight_embedding(name, expected))
        except (ValueError, OSError) as error:
            result = dict(path=name, status='REJECTED', reason=str(error))
        print(json.dumps(result, ensure_ascii=False))
    print('Only headers inspected. No tensor data loaded, files modified, or .pt unpickled.')


if __name__ == '__main__':
    main()
