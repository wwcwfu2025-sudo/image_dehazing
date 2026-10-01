"""Verify the release file hashes using only the Python standard library."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'FILE_SHA256_2026100109.json').read_text(encoding='utf-8'))
    errors = []
    for relative, expected in manifest['files'].items():
        path = (root / relative).resolve()
        if root not in path.parents:
            errors.append(relative + ': unsafe path')
            continue
        if not path.is_file():
            errors.append(relative + ': missing')
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        if actual != expected['sha256'] or path.stat().st_size != expected['bytes']:
            errors.append(relative + ': mismatch')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Verified {} files. SHA-256 detects changes; this is not a cryptographic publisher signature.'.format(len(manifest['files'])))


if __name__ == '__main__':
    main()

# Generated: 2026-10-01 09:00 +08:00
