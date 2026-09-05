#!/usr/bin/env python3
"""Retrospective nonlinear-query experiment; never mutate parent evidence."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import traceback
import zipfile


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def inventory(request: dict, output: Path) -> None:
    root = Path(request['parent_cache_root']).resolve(strict=True)
    parent_id = str(request['parent_run_id'])
    candidates = [p for p in root.iterdir() if p.is_dir() and parent_id in p.name]
    if not candidates:
        candidates = [p for p in root.glob('*/*') if p.is_dir() and parent_id in p.name]
    if not candidates:
        write(output / 'inventory.json', {'status': 'parent-path-not-found', 'root': str(root), 'children': sorted(p.name for p in root.iterdir())})
        raise FileNotFoundError('Completed parent run directory was not found')
    records = []
    for parent in sorted(candidates):
        for path in sorted(parent.rglob('*')):
            if not path.is_file() or path.suffix not in ('.json', '.npz'):
                continue
            record = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
            if path.suffix == '.npz':
                arrays = {}
                with zipfile.ZipFile(path) as archive:
                    for name in archive.namelist():
                        with archive.open(name) as f:
                            magic = f.read(8)
                            if magic[:6] != b'\x93NUMPY':
                                raise ValueError('Non-NPY member in NPZ')
                            nbytes = 2 if magic[6] == 1 else 4
                            length = struct.unpack('<H' if nbytes == 2 else '<I', f.read(nbytes))[0]
                            arrays[name] = ast.literal_eval(f.read(length).decode('latin1').strip())
                record['arrays'] = arrays
            elif path.stat().st_size < 50000:
                payload = json.loads(path.read_text())
                if isinstance(payload, dict):
                    record['metadata'] = {k: v for k, v in payload.items() if k not in ('trajectories', 'bayesian_distributions', 'results', 'runtime', 'source_gate')}
            records.append(record)
    write(output / 'inventory.json', {'status': 'complete', 'parent_roots': [str(p) for p in candidates], 'records': records, 'payload_use': 'JSON metadata and NPZ headers only; no trajectory arrays', 'run_id': os.environ.get('GITHUB_RUN_ID'), 'commit': os.environ.get('GITHUB_SHA')})
    print(json.dumps({'inventory_status': 'complete', 'files': len(records)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    request = json.loads(args.request.read_text())
    try:
        if request['phase'] != 'inventory':
            raise ValueError('Only inventory is implemented at this revision')
        inventory(request, args.output)
    except Exception as exc:
        write(args.output / 'failure.json', {'status': 'technical-failure', 'error': repr(exc), 'traceback': traceback.format_exc(), 'phase': request.get('phase')})
        raise


if __name__ == '__main__':
    main()
