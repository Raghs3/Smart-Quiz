"""
One-time script: converts DeepMind Mathematics Dataset text files to questions.csv.

Usage:
    python prepare_data.py <deepmind_folder> [--max-per-level 500] [--out questions.csv]

<deepmind_folder> must contain train-easy/, train-medium/, train-hard/ subdirs.
Each .txt file alternates question/answer lines:
    What is 2 + 2?
    4
    What is 3 - 1?
    2
"""

import argparse
import csv
import os
import random
import sys


SPLIT_LEVELS = {
    'train-easy':   [0.1, 0.2, 0.3],
    'train-medium': [0.4, 0.5, 0.6, 0.7],
    'train-hard':   [0.8, 0.9, 1.0],
}


def parse_txt(path: str) -> list:
    """Read alternating question/answer lines from a DeepMind .txt file."""
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f if l.strip()]
    for i in range(0, len(lines) - 1, 2):
        q = lines[i].strip()
        a = lines[i + 1].strip()
        if q and a:
            pairs.append((q, a))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Convert DeepMind Mathematics Dataset to questions.csv'
    )
    parser.add_argument('folder', help='Path to DeepMind generated dataset folder')
    parser.add_argument('--max-per-level', type=int, default=500,
                        help='Max questions per difficulty level (default 500)')
    parser.add_argument('--out', default='questions.csv', help='Output CSV path')
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f'ERROR: {args.folder} is not a directory.', file=sys.stderr)
        sys.exit(1)

    by_level: dict = {}

    for split, levels in SPLIT_LEVELS.items():
        split_dir = os.path.join(args.folder, split)
        if not os.path.isdir(split_dir):
            print(f'WARNING: {split_dir} not found, skipping.')
            continue

        txt_files = sorted(f for f in os.listdir(split_dir) if f.endswith('.txt'))
        if not txt_files:
            print(f'WARNING: No .txt files in {split_dir}.')
            continue

        for file_idx, fname in enumerate(txt_files):
            level = levels[file_idx % len(levels)]
            pairs = parse_txt(os.path.join(split_dir, fname))
            by_level.setdefault(level, []).extend(pairs)

    rows = []
    for level in sorted(by_level.keys()):
        pairs = by_level[level]
        random.shuffle(pairs)
        pairs = pairs[:args.max_per_level]
        for q, a in pairs:
            rows.append({'level': f'{level:.1f}', 'question': q, 'answer': a})

    if not rows:
        print('ERROR: No questions extracted. Check your dataset folder.', file=sys.stderr)
        sys.exit(1)

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['level', 'question', 'answer'])
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote {len(rows)} questions to {args.out}')
    level_counts: dict = {}
    for r in rows:
        level_counts[r['level']] = level_counts.get(r['level'], 0) + 1
    for lvl in sorted(level_counts):
        print(f'  Level {lvl}: {level_counts[lvl]} questions')


if __name__ == '__main__':
    main()
