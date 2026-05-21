#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

TOKENIZER_FILES = [
    'tokenizer.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'tokenizer.model',
    'vocab.json',
    'merges.txt',
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Audit tokenizer comparability across HF checkpoints.')
    p.add_argument('--models-root', type=Path, default=Path('models/hf'))
    p.add_argument('--out-csv', type=Path, required=True)
    p.add_argument('--out-json', type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    file_hash_index: dict[str, dict[str, list[str]]] = {name: {} for name in TOKENIZER_FILES}

    for model_dir in sorted(p for p in args.models_root.iterdir() if p.is_dir() and not p.name.startswith('_tmp_')):
        if not ((model_dir / 'config.json').exists() or (model_dir / 'tokenizer_config.json').exists()):
            continue
        row = {'model_name': model_dir.name}
        cfg_path = model_dir / 'tokenizer_config.json'
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
            row['tokenizer_class'] = cfg.get('tokenizer_class', '')
            row['model_max_length'] = cfg.get('model_max_length', '')
            row['padding_side'] = cfg.get('padding_side', '')
        else:
            row['tokenizer_class'] = ''
            row['model_max_length'] = ''
            row['padding_side'] = ''

        st_path = model_dir / 'special_tokens_map.json'
        if st_path.exists():
            st = json.loads(st_path.read_text(encoding='utf-8'))
            row['bos_token'] = str(st.get('bos_token', ''))
            row['eos_token'] = str(st.get('eos_token', ''))
            row['unk_token'] = str(st.get('unk_token', ''))
            row['pad_token'] = str(st.get('pad_token', ''))
        else:
            row['bos_token'] = row['eos_token'] = row['unk_token'] = row['pad_token'] = ''

        cfg_model = model_dir / 'config.json'
        if cfg_model.exists():
            model_cfg = json.loads(cfg_model.read_text(encoding='utf-8'))
            row['vocab_size'] = model_cfg.get('vocab_size', '')
        else:
            row['vocab_size'] = ''

        for fname in TOKENIZER_FILES:
            fpath = model_dir / fname
            digest = sha256(fpath) if fpath.exists() else ''
            row[f'{fname}_sha256'] = digest
            if digest:
                file_hash_index[fname].setdefault(digest, []).append(model_dir.name)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values('model_name').reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    summary = {
        'models_root': str(args.models_root),
        'n_models': int(len(df)),
        'per_file_unique_hashes': {fname: len(digests) for fname, digests in file_hash_index.items()},
        'hash_groups': file_hash_index,
    }
    args.out_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Wrote: {args.out_csv}')
    print(f'Wrote: {args.out_json}')


if __name__ == '__main__':
    main()
