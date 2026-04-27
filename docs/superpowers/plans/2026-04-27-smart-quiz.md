# Smart Quiz Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an adaptive math quiz Streamlit app with a custom deep neural network (5→16→8→4→1), per-user login, GMAT-style baseline test, and a results dashboard.

**Architecture:** A custom `DeepNN` class (NumPy only) predicts next difficulty from a 5-feature vector per round. Quiz logic and auth live in separate modules. A one-time preprocessing script converts the DeepMind dataset to `questions.csv`. The Streamlit app routes between Login → Baseline → Quiz → Results using `st.session_state`.

**Tech Stack:** Python 3, NumPy, Streamlit, matplotlib, pytest. Venv at `.venv/`. All ML math is hand-written — no scikit-learn, TensorFlow, or PyTorch.

---

## File Map

| File | Responsibility |
|---|---|
| `nn.py` | `DeepNN` class: forward, backward, save, load |
| `auth.py` | SHA-256 password hashing, `users.json` read/write, register/login |
| `quiz.py` | Question loading, 5-feature encoding, difficulty adjustment, baseline scoring, pretraining |
| `app.py` | Streamlit entry point — all 4 pages via session state routing |
| `prepare_data.py` | One-time script: DeepMind text files → `questions.csv` |
| `tests/test_nn.py` | Unit tests for DeepNN |
| `tests/test_auth.py` | Unit tests for auth functions |
| `tests/test_quiz.py` | Unit tests for quiz logic |
| `models/` | Per-user `.npz` files created at runtime |
| `users.json` | Created by auth.py on first register |

---

## Task 1: Install Dependencies

**Files:**
- No new files

- [ ] **Step 1: Install streamlit, pytest, matplotlib into the venv**

```bash
.venv/Scripts/pip.exe install streamlit pytest matplotlib
```

Expected output includes lines like:
```
Successfully installed streamlit-... pytest-... matplotlib-...
```

- [ ] **Step 2: Verify installs**

```bash
.venv/Scripts/python.exe -c "import streamlit, pytest, matplotlib, numpy; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Create tests directory and models directory**

```bash
mkdir -p tests models
touch tests/__init__.py
```

- [ ] **Step 4: Commit**

```bash
git add tests/ models/
git commit -m "chore: add tests dir, models dir, install deps"
```

---

## Task 2: nn.py — DeepNN Class

**Files:**
- Create: `nn.py`

- [ ] **Step 1: Create `nn.py` with the DeepNN class**

```python
import numpy as np

LAYER_SIZES = [5, 16, 8, 4, 1]


class DeepNN:
    def __init__(self, layer_sizes=None):
        sizes = layer_sizes or LAYER_SIZES
        self.layer_sizes = list(sizes)
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            scale = 1.0 / self.layer_sizes[i] ** 0.5
            W = np.random.uniform(-scale, scale, (self.layer_sizes[i + 1], self.layer_sizes[i]))
            b = np.zeros((self.layer_sizes[i + 1], 1))
            self.weights.append(W)
            self.biases.append(b)
        self._acts = []

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _sig_d(self, a):
        return a * (1.0 - a)

    def forward(self, x):
        a = np.asarray(x, dtype=float).reshape(-1, 1)
        self._acts = [a]
        for W, b in zip(self.weights, self.biases):
            a = self._sigmoid(W @ a + b)
            self._acts.append(a)
        return self._acts[-1].ravel()

    def backward(self, y, lr=0.05):
        target = np.array([[float(np.clip(y, 0.0, 1.0))]])
        delta = self._acts[-1] - target
        for i in reversed(range(len(self.weights))):
            dW = delta @ self._acts[i].T
            db = delta.copy()
            if i > 0:
                delta = (self.weights[i].T @ delta) * self._sig_d(self._acts[i])
            self.weights[i] -= lr * dW
            self.biases[i] -= lr * db

    def save(self, path):
        arrays = {f'W{i}': W for i, W in enumerate(self.weights)}
        arrays.update({f'b{i}': b for i, b in enumerate(self.biases)})
        arrays['layer_sizes'] = np.array(self.layer_sizes)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        layer_sizes = data['layer_sizes'].tolist()
        nn = cls(layer_sizes)
        nn.weights = [data[f'W{i}'] for i in range(len(layer_sizes) - 1)]
        nn.biases = [data[f'b{i}'] for i in range(len(layer_sizes) - 1)]
        return nn
```

- [ ] **Step 2: Commit**

```bash
git add nn.py
git commit -m "feat: add DeepNN class (5->16->8->4->1) with forward/backward/save/load"
```

---

## Task 3: tests/test_nn.py — Neural Network Tests

**Files:**
- Create: `tests/test_nn.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
import os
import tempfile
import pytest
from nn import DeepNN, LAYER_SIZES


def test_forward_output_shape():
    nn = DeepNN()
    out = nn.forward(np.zeros(5))
    assert out.shape == (1,)


def test_forward_output_range():
    nn = DeepNN()
    for _ in range(20):
        x = np.random.uniform(0, 1, size=5)
        out = nn.forward(x)
        assert 0.0 <= out[0] <= 1.0


def test_backward_changes_weights():
    np.random.seed(0)
    nn = DeepNN()
    W0_before = nn.weights[0].copy()
    x = np.array([1.0, 0.5, 0.6, 0.2, 0.4])
    nn.forward(x)
    nn.backward(0.8)
    assert not np.allclose(nn.weights[0], W0_before)


def test_backward_reduces_loss():
    np.random.seed(42)
    nn = DeepNN()
    x = np.array([1.0, 0.3, 0.5, 0.4, 0.3])
    target = 0.7
    losses = []
    for _ in range(100):
        out = nn.forward(x)
        losses.append((out[0] - target) ** 2)
        nn.backward(target, lr=0.1)
    assert losses[-1] < losses[0]


def test_save_load_roundtrip():
    np.random.seed(7)
    nn = DeepNN()
    x = np.array([1.0, 0.5, 0.6, 0.2, 0.4])
    out_before = nn.forward(x)
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        path = f.name
    try:
        nn.save(path)
        nn2 = DeepNN.load(path)
        out_after = nn2.forward(x)
        assert np.allclose(out_before, out_after)
        assert nn2.layer_sizes == LAYER_SIZES
    finally:
        os.unlink(path)


def test_custom_layer_sizes():
    nn = DeepNN([3, 8, 1])
    out = nn.forward(np.ones(3))
    assert out.shape == (1,)
```

- [ ] **Step 2: Run tests — expect FAIL (nn.py not found)**

```bash
.venv/Scripts/python.exe -m pytest tests/test_nn.py -v
```

Expected: tests fail with `ModuleNotFoundError` or similar if nn.py wasn't created yet — but since we created it in Task 2, they should actually PASS here. Verify all 6 pass.

Expected output:
```
tests/test_nn.py::test_forward_output_shape PASSED
tests/test_nn.py::test_forward_output_range PASSED
tests/test_nn.py::test_backward_changes_weights PASSED
tests/test_nn.py::test_backward_reduces_loss PASSED
tests/test_nn.py::test_save_load_roundtrip PASSED
tests/test_nn.py::test_custom_layer_sizes PASSED
6 passed
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_nn.py
git commit -m "test: add DeepNN unit tests (forward, backward, save/load)"
```

---

## Task 4: auth.py — User Authentication

**Files:**
- Create: `auth.py`

- [ ] **Step 1: Create `auth.py`**

```python
import hashlib
import json
import os

USERS_FILE = 'users.json'


def _hash(password: str) -> str:
    return 'sha256:' + hashlib.sha256(password.encode('utf-8')).hexdigest()


def _load() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save(data: dict) -> None:
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def register(username: str, password: str) -> tuple[bool, str]:
    """Register a new user. Returns (True, '') or (False, error_message)."""
    username = username.strip()
    if not username or not password:
        return False, 'Username and password are required.'
    users = _load()
    if username in users:
        return False, f'Username "{username}" already exists.'
    users[username] = _hash(password)
    _save(users)
    return True, ''


def login(username: str, password: str) -> bool:
    """Return True if credentials match, False otherwise."""
    users = _load()
    stored = users.get(username.strip())
    if stored is None:
        return False
    return stored == _hash(password)
```

- [ ] **Step 2: Commit**

```bash
git add auth.py
git commit -m "feat: add auth module with SHA-256 hashing, register, login"
```

---

## Task 5: tests/test_auth.py — Auth Tests

**Files:**
- Create: `tests/test_auth.py`

- [ ] **Step 1: Write tests**

```python
import os
import json
import pytest
import auth


@pytest.fixture(autouse=True)
def tmp_users(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, 'USERS_FILE', str(tmp_path / 'users.json'))


def test_register_success():
    ok, msg = auth.register('alice', 'secret')
    assert ok is True
    assert msg == ''


def test_register_creates_file():
    auth.register('alice', 'secret')
    assert os.path.exists(auth.USERS_FILE)


def test_register_hashes_password():
    auth.register('alice', 'secret')
    with open(auth.USERS_FILE) as f:
        data = json.load(f)
    assert 'secret' not in data['alice']
    assert data['alice'].startswith('sha256:')


def test_register_duplicate_fails():
    auth.register('alice', 'secret')
    ok, msg = auth.register('alice', 'other')
    assert ok is False
    assert 'already exists' in msg


def test_register_empty_username_fails():
    ok, msg = auth.register('', 'secret')
    assert ok is False


def test_login_correct():
    auth.register('bob', 'pass123')
    assert auth.login('bob', 'pass123') is True


def test_login_wrong_password():
    auth.register('bob', 'pass123')
    assert auth.login('bob', 'wrong') is False


def test_login_nonexistent_user():
    assert auth.login('nobody', 'anything') is False
```

- [ ] **Step 2: Run tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_auth.py -v
```

Expected: all 8 pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_auth.py
git commit -m "test: add auth unit tests (register, login, hashing)"
```

---

## Task 6: quiz.py — Quiz Logic

**Files:**
- Create: `quiz.py`

- [ ] **Step 1: Create `quiz.py`**

```python
import csv
import os
import random
import numpy as np

QUESTIONS_CSV = 'questions.csv'
LEVEL_STEP = 0.1
MAX_RECENT = 10
DIFF_STEP = 0.08

FALLBACK = {
    0.1: [("What is 2+2?", "4"), ("What is 9-4?", "5"), ("What is 7+1?", "8")],
    0.2: [("What is 12-5?", "7"), ("What is 6+9?", "15"), ("What is 3*4?", "12")],
    0.3: [("What is 15/3?", "5"), ("What is 14+19?", "33"), ("What is 18-7?", "11")],
    0.4: [("What is 12*12?", "144"), ("What is 81/9?", "9"), ("What is 25+37?", "62")],
    0.5: [("Solve for x: x + 7 = 15", "8"), ("What is the square root of 169?", "13"), ("If 4x = 36 then what is x?", "9")],
    0.6: [("What is 2^5?", "32"), ("If 3x = 27 then what is x?", "9"), ("Solve: 2x + 5 = 17", "6")],
    0.7: [("What is the derivative of x^2?", "2x"), ("Integrate x dx", "0.5x^2"), ("Derivative of x^3?", "3x^2")],
    0.8: [("Derivative of sin(x)?", "cos(x)"), ("Integrate 2x dx", "x^2"), ("Derivative of cos(x)?", "-sin(x)")],
    0.9: [("What is the derivative of ln(x)?", "1/x"), ("What is the derivative of e^x?", "e^x"), ("Integrate e^x dx", "e^x")],
    1.0: [("What is lim(x->0) sin(x)/x?", "1"), ("What is lim(x->inf) 1/x?", "0"), ("Integral of 1/x dx", "ln|x|")],
}


def load_questions(path=None) -> dict:
    """Return dict: level (float) -> list of (question, answer) tuples."""
    p = path or QUESTIONS_CSV
    if not os.path.exists(p):
        return FALLBACK
    bank = {}
    with open(p, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                lvl = round(float(np.clip(float(row['level']), 0.1, 1.0)), 1)
            except (ValueError, KeyError, TypeError):
                continue
            q = (row.get('question') or '').strip()
            a = (row.get('answer') or '').strip()
            if q and a:
                bank.setdefault(lvl, []).append((q, a))
    return bank or FALLBACK


def sample_question(bank: dict, difficulty: float, recent: list) -> tuple:
    """Return (level, question, answer), preferring fresh questions near difficulty."""
    target = round(round(float(np.clip(difficulty, 0.1, 1.0)) / LEVEL_STEP) * LEVEL_STEP, 1)
    levels = sorted(bank.keys(), key=lambda l: abs(l - target))
    for lvl in levels:
        fresh = [(q, a) for q, a in bank[lvl] if q not in recent]
        if fresh:
            q, a = random.choice(fresh)
            return lvl, q, a
    q, a = random.choice(bank[levels[0]])
    return levels[0], q, a


def normalize(s: str) -> str:
    return s.strip().lower().replace(' ', '')


def encode_features(correct: bool, current_difficulty: float,
                    rolling_accuracy: float, streak: int,
                    question_level: float) -> np.ndarray:
    """Build the 5-feature input vector for DeepNN."""
    streak_norm = float(np.clip(streak / 5.0, -1.0, 1.0))
    return np.array([
        float(correct),
        float(current_difficulty),
        float(rolling_accuracy),
        streak_norm,
        float(question_level),
    ])


def adjust_difficulty(old_diff: float, predicted_diff: float,
                      question_level: float, correct: bool) -> float:
    """Return new difficulty clamped to [0.1, 1.0]."""
    if correct:
        candidate = min(0.6 * predicted_diff + 0.4 * question_level + DIFF_STEP, 1.0)
        new = max(old_diff, candidate)
    else:
        candidate = max(0.6 * predicted_diff + 0.4 * question_level - DIFF_STEP, 0.0)
        new = min(old_diff, candidate)
    return float(np.clip(new, 0.1, 1.0))


def compute_rolling_accuracy(history: list, window: int = 5) -> float:
    """Mean correctness over last `window` rounds."""
    if not history:
        return 0.5
    recent = history[-window:]
    return sum(1 for h in recent if h['correct']) / len(recent)


def baseline_score(answers: list) -> float:
    """
    answers: list of (level, correct) for 10 baseline questions.
    Returns weighted difficulty score in [0.1, 1.0].
    """
    total_weight = sum(lvl for lvl, _ in answers)
    if total_weight == 0:
        return 0.3
    score = sum(lvl * int(c) for lvl, c in answers) / total_weight
    return float(np.clip(score, 0.1, 1.0))


def pretrain(nn, bank: dict, epochs: int = 2000, lr: float = 0.05) -> None:
    """Warm up the NN with synthetic quiz sessions."""
    difficulty = 0.3
    streak = 0
    history = []
    for _ in range(epochs):
        lvl, _, _ = sample_question(bank, difficulty, [])
        p_correct = max(0.1, 1.0 - lvl * 0.85)
        correct = random.random() < p_correct
        rolling_acc = compute_rolling_accuracy(history)
        streak = (max(0, streak) + 1) if correct else (min(0, streak) - 1)
        x = encode_features(correct, difficulty, rolling_acc, streak, lvl)
        out = nn.forward(x)
        nn.backward(lvl, lr=lr)
        predicted = float(np.clip(out[0], 0.0, 1.0))
        difficulty = adjust_difficulty(difficulty, predicted, lvl, correct)
        history.append({'correct': correct})
        if len(history) > 5:
            history.pop(0)
```

- [ ] **Step 2: Commit**

```bash
git add quiz.py
git commit -m "feat: add quiz logic (question loading, feature encoding, difficulty adjustment, baseline scoring, pretraining)"
```

---

## Task 7: tests/test_quiz.py — Quiz Logic Tests

**Files:**
- Create: `tests/test_quiz.py`

- [ ] **Step 1: Write tests**

```python
import numpy as np
import pytest
import quiz


def test_normalize():
    assert quiz.normalize('  2X ') == '2x'
    assert quiz.normalize('cos(x)') == 'cos(x)'


def test_encode_features_shape():
    x = quiz.encode_features(True, 0.5, 0.6, 2, 0.4)
    assert x.shape == (5,)


def test_encode_features_correct_true():
    x = quiz.encode_features(True, 0.5, 0.6, 2, 0.4)
    assert x[0] == 1.0


def test_encode_features_correct_false():
    x = quiz.encode_features(False, 0.5, 0.6, -3, 0.7)
    assert x[0] == 0.0


def test_encode_features_streak_clamp():
    x = quiz.encode_features(True, 0.5, 0.5, 10, 0.5)
    assert x[3] == 1.0
    x2 = quiz.encode_features(False, 0.5, 0.5, -10, 0.5)
    assert x2[3] == -1.0


def test_adjust_difficulty_correct_increases():
    new = quiz.adjust_difficulty(0.5, 0.5, 0.5, True)
    assert new >= 0.5


def test_adjust_difficulty_wrong_decreases():
    new = quiz.adjust_difficulty(0.5, 0.5, 0.5, False)
    assert new <= 0.5


def test_adjust_difficulty_clamp():
    new = quiz.adjust_difficulty(1.0, 1.0, 1.0, True)
    assert new <= 1.0
    new2 = quiz.adjust_difficulty(0.1, 0.0, 0.1, False)
    assert new2 >= 0.1


def test_baseline_score_all_correct():
    answers = [(l, True) for l in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    score = quiz.baseline_score(answers)
    assert score == pytest.approx(1.0)


def test_baseline_score_all_wrong():
    answers = [(l, False) for l in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    score = quiz.baseline_score(answers)
    assert score == pytest.approx(0.1)


def test_baseline_score_partial():
    answers = [(0.1, True), (0.2, False), (0.3, True), (0.4, False), (0.5, True),
               (0.6, False), (0.7, True), (0.8, False), (0.9, True), (1.0, False)]
    score = quiz.baseline_score(answers)
    assert 0.1 <= score <= 1.0


def test_compute_rolling_accuracy_empty():
    assert quiz.compute_rolling_accuracy([]) == 0.5


def test_compute_rolling_accuracy_all_correct():
    history = [{'correct': True}] * 5
    assert quiz.compute_rolling_accuracy(history) == 1.0


def test_compute_rolling_accuracy_window():
    history = [{'correct': False}] * 10 + [{'correct': True}] * 5
    acc = quiz.compute_rolling_accuracy(history, window=5)
    assert acc == 1.0


def test_load_questions_fallback(tmp_path):
    bank = quiz.load_questions(str(tmp_path / 'nonexistent.csv'))
    assert 0.1 in bank
    assert len(bank[0.1]) > 0


def test_sample_question_returns_tuple():
    bank = quiz.FALLBACK
    lvl, q, a = quiz.sample_question(bank, 0.5, [])
    assert isinstance(lvl, float)
    assert isinstance(q, str)
    assert isinstance(a, str)


def test_sample_question_avoids_recent():
    bank = {0.5: [("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3")]}
    _, q, _ = quiz.sample_question(bank, 0.5, ["Q1", "Q2"])
    assert q == "Q3"
```

- [ ] **Step 2: Run tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_quiz.py -v
```

Expected: all 17 pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_quiz.py
git commit -m "test: add quiz logic unit tests"
```

---

## Task 8: prepare_data.py — DeepMind Dataset Preprocessor

**Files:**
- Create: `prepare_data.py`

**Note:** Before running this script, download/generate the DeepMind Mathematics Dataset:
```bash
git clone https://github.com/google-deepmind/mathematics_dataset.git data/deepmind_repo
cd data/deepmind_repo
pip install -e .
python -m mathematics_dataset.generate --dataset_folder=../deepmind_generated
cd ../..
```
The generated folder will contain `train-easy/`, `train-medium/`, `train-hard/` subdirectories, each with `.txt` files.

Each `.txt` file alternates question / answer lines:
```
What is 2 + 2?
4
What is 3 - 1?
2
```

- [ ] **Step 1: Create `prepare_data.py`**

```python
"""
One-time script: converts DeepMind Mathematics Dataset text files to questions.csv.

Usage:
    python prepare_data.py <deepmind_folder> [--max-per-level 500] [--out questions.csv]

<deepmind_folder> must contain train-easy/, train-medium/, train-hard/ subdirs.
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


def parse_txt(path: str) -> list[tuple[str, str]]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='Path to DeepMind generated dataset folder')
    parser.add_argument('--max-per-level', type=int, default=500,
                        help='Max questions per difficulty level (default 500)')
    parser.add_argument('--out', default='questions.csv', help='Output CSV path')
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f'ERROR: {args.folder} is not a directory.', file=sys.stderr)
        sys.exit(1)

    # Collect (level, question, answer) across all splits
    by_level: dict[float, list[tuple[str, str]]] = {}

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

    # Shuffle and cap each level
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
    level_counts = {}
    for r in rows:
        level_counts[r['level']] = level_counts.get(r['level'], 0) + 1
    for lvl in sorted(level_counts):
        print(f'  Level {lvl}: {level_counts[lvl]} questions')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the script (after downloading the DeepMind dataset)**

If the DeepMind dataset folder is at `data/deepmind_generated`:
```bash
.venv/Scripts/python.exe prepare_data.py data/deepmind_generated --max-per-level 300
```

Expected output:
```
Wrote NNNN questions to questions.csv
  Level 0.1: 300 questions
  Level 0.2: 300 questions
  ...
```

If you cannot run the generator, skip this step — the app falls back to the existing `questions.csv` (30 questions).

- [ ] **Step 3: Commit**

```bash
git add prepare_data.py
git commit -m "feat: add DeepMind dataset preprocessor (prepare_data.py)"
```

---

## Task 9: app.py — Login/Register + Baseline Test Pages

**Files:**
- Create: `app.py`

- [ ] **Step 1: Create `app.py` with session state init, login page, and baseline page**

```python
import os
import random
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import auth
import quiz as qz
from nn import DeepNN

MODELS_DIR = 'models'
LAYER_SIZES = [5, 16, 8, 4, 1]
TOTAL_ROUNDS = 20
BASELINE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Session state defaults ────────────────────────────────────────────────────

def _init_state():
    defaults = {
        'phase': 'login',
        'user': None,
        'nn': None,
        'difficulty': 0.1,
        'history': [],
        'streak': 0,
        'recent_questions': [],
        'quiz_round': 0,
        'rounds': TOTAL_ROUNDS,
        'last_feedback': None,
        'baseline_idx': 0,
        'baseline_answers': [],
        'baseline_questions': [],
        'bank': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _model_path(username: str) -> str:
    return os.path.join(MODELS_DIR, f'{username}.npz')


def _load_user_model(username: str):
    """Load user NN and baseline result. Returns (nn, start_difficulty, has_baseline)."""
    path = _model_path(username)
    if not os.path.exists(path):
        return None, None, False
    try:
        data = np.load(path, allow_pickle=False)
        nn = DeepNN.load(path)
        has_baseline = 'baseline_difficulty' in data.files
        start_diff = float(data['baseline_difficulty']) if has_baseline else None
        return nn, start_diff, has_baseline
    except Exception:
        return None, None, False


def _save_user_model(username: str, nn, baseline_difficulty: float):
    path = _model_path(username)
    arrays = {f'W{i}': W for i, W in enumerate(nn.weights)}
    arrays.update({f'b{i}': b for i, b in enumerate(nn.biases)})
    arrays['layer_sizes'] = np.array(nn.layer_sizes)
    arrays['baseline_difficulty'] = np.array(baseline_difficulty)
    np.savez(path, **arrays)


def _get_bank():
    if st.session_state.bank is None:
        st.session_state.bank = qz.load_questions()
    return st.session_state.bank


def _logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    _init_state()

# ── Page: Login / Register ────────────────────────────────────────────────────

def page_login():
    st.title('Smart Quiz')
    st.subheader('Login or Register')

    with st.form('login_form'):
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        col1, col2 = st.columns(2)
        submitted_login = col1.form_submit_button('Login')
        submitted_register = col2.form_submit_button('Register')

    if submitted_login:
        if auth.login(username, password):
            bank = _get_bank()
            nn, start_diff, has_baseline = _load_user_model(username)
            if nn is None:
                nn = DeepNN(LAYER_SIZES)
                with st.spinner('Running pretraining (first login)...'):
                    qz.pretrain(nn, bank)
                _save_user_model(username, nn, 0.3)
                has_baseline = False

            st.session_state.user = username
            st.session_state.nn = nn

            if has_baseline:
                st.session_state.difficulty = start_diff
                st.session_state.phase = 'quiz'
            else:
                # Pick one question per baseline level
                baseline_qs = []
                for lvl in BASELINE_LEVELS:
                    lvl_pool = bank.get(lvl, [])
                    if lvl_pool:
                        baseline_qs.append((lvl, *random.choice(lvl_pool)))
                    else:
                        baseline_qs.append((lvl, f'What is {int(lvl*10)}+0?', str(int(lvl*10))))
                st.session_state.baseline_questions = baseline_qs
                st.session_state.baseline_idx = 0
                st.session_state.baseline_answers = []
                st.session_state.phase = 'baseline'
            st.rerun()
        else:
            st.error('Incorrect username or password.')

    if submitted_register:
        ok, msg = auth.register(username, password)
        if ok:
            st.success('Account created! Please log in.')
        else:
            st.error(msg)

# ── Page: Baseline Test ───────────────────────────────────────────────────────

def page_baseline():
    idx = st.session_state.baseline_idx
    questions = st.session_state.baseline_questions
    total = len(questions)

    st.title('Baseline Assessment')
    st.caption('GMAT-style — answer all questions to set your starting difficulty.')
    st.progress(idx / total, text=f'Question {idx + 1} of {total}')

    if idx >= total:
        # Score and transition
        score = qz.baseline_score(
            [(lvl, c) for lvl, c in st.session_state.baseline_answers]
        )
        _save_user_model(st.session_state.user, st.session_state.nn, score)
        st.session_state.difficulty = score
        st.success(f'Baseline complete! Starting difficulty set to **{score:.2f}**')
        if st.button('Start Quiz'):
            st.session_state.phase = 'quiz'
            st.rerun()
        return

    lvl, q, ans = questions[idx]
    st.markdown(f'### {q}')
    st.caption(f'Difficulty level: {lvl:.1f}')

    with st.form(f'baseline_form_{idx}'):
        user_ans = st.text_input('Your answer')
        submitted = st.form_submit_button('Submit')

    if submitted:
        correct = qz.normalize(user_ans) == qz.normalize(ans)
        st.session_state.baseline_answers.append((lvl, correct))
        st.session_state.baseline_idx += 1
        if correct:
            st.success('Correct!')
        else:
            st.error(f'Incorrect. Answer: {ans}')
        st.rerun()
```

- [ ] **Step 2: Verify app launches (login page only at this point)**

```bash
.venv/Scripts/python.exe -m streamlit run app.py
```

Open the URL shown (usually `http://localhost:8501`). Verify:
- Login form appears
- Registering a new user shows "Account created!"
- Logging in with wrong password shows an error

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit app with login/register and baseline test pages"
```

---

## Task 10: app.py — Adaptive Quiz Page + Results Dashboard

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Append quiz and results page functions to `app.py`, then add the router**

Add these functions after `page_baseline()` and before `if __name__` at the bottom of `app.py`:

```python
# ── Page: Adaptive Quiz ───────────────────────────────────────────────────────

def page_quiz():
    user = st.session_state.user
    nn = st.session_state.nn
    bank = _get_bank()

    # Sidebar
    with st.sidebar:
        st.header(f'Player: {user}')
        st.metric('Difficulty', f"{st.session_state.difficulty:.2f}")
        history = st.session_state.history
        if history:
            rolling_acc = qz.compute_rolling_accuracy(history)
            st.metric('Rolling Accuracy', f"{rolling_acc * 100:.0f}%")
        streak = st.session_state.streak
        streak_label = f'+{streak}' if streak > 0 else str(streak)
        st.metric('Streak', streak_label)
        st.session_state.rounds = st.slider(
            'Total Rounds', min_value=5, max_value=50,
            value=st.session_state.rounds, step=5
        )
        if st.button('End Quiz'):
            _save_user_model(user, nn, st.session_state.difficulty)
            st.session_state.phase = 'results'
            st.rerun()

    rnd = st.session_state.quiz_round
    rounds = st.session_state.rounds

    if rnd >= rounds:
        _save_user_model(user, nn, st.session_state.difficulty)
        st.session_state.phase = 'results'
        st.rerun()
        return

    st.title('Adaptive Quiz')
    st.caption(f'Round {rnd + 1} / {rounds}')

    # Show feedback from previous round
    fb = st.session_state.last_feedback
    if fb:
        if fb['correct']:
            st.success(f"Correct! Difficulty → {fb['new_diff']:.2f}")
        else:
            st.error(f"Incorrect. Answer was: {fb['answer']}. Difficulty → {fb['new_diff']:.2f}")

    # Sample question if not already set for this round
    recent = st.session_state.recent_questions
    diff = st.session_state.difficulty
    lvl, q, ans = qz.sample_question(bank, diff, recent)

    st.markdown(f'## {q}')

    with st.form(f'quiz_form_{rnd}'):
        user_ans = st.text_input('Your answer', key=f'ans_{rnd}')
        submitted = st.form_submit_button('Submit Answer')

    if submitted:
        correct = qz.normalize(user_ans) == qz.normalize(ans)

        # Update recent list
        recent.append(q)
        if len(recent) > qz.MAX_RECENT:
            recent.pop(0)

        # Update streak
        streak = st.session_state.streak
        streak = (max(0, streak) + 1) if correct else (min(0, streak) - 1)
        st.session_state.streak = streak

        # Compute rolling accuracy
        rolling_acc = qz.compute_rolling_accuracy(st.session_state.history)

        # NN forward + backward
        x = qz.encode_features(correct, diff, rolling_acc, streak, lvl)
        out = nn.forward(x)
        nn.backward(diff, lr=0.05)
        predicted = float(np.clip(out[0], 0.0, 1.0))

        # Adjust difficulty
        new_diff = qz.adjust_difficulty(diff, predicted, lvl, correct)
        st.session_state.difficulty = new_diff

        # Record history
        st.session_state.history.append({
            'round': rnd + 1,
            'question': q,
            'question_level': lvl,
            'correct': correct,
            'predicted_diff': predicted,
            'difficulty_before': diff,
            'difficulty_after': new_diff,
        })

        st.session_state.last_feedback = {
            'correct': correct,
            'answer': ans,
            'new_diff': new_diff,
        }
        st.session_state.quiz_round += 1
        st.rerun()


# ── Page: Results Dashboard ───────────────────────────────────────────────────

def page_results():
    st.title('Quiz Results')
    history = st.session_state.history

    if not history:
        st.info('No rounds played yet.')
        if st.button('Back to Quiz'):
            st.session_state.phase = 'quiz'
            st.rerun()
        return

    # Summary metrics
    accuracies = [1 if h['correct'] else 0 for h in history]
    start_diff = history[0]['difficulty_before']
    end_diff = history[-1]['difficulty_after']
    total_acc = sum(accuracies) / len(accuracies)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Rounds Played', len(history))
    c2.metric('Accuracy', f'{total_acc * 100:.1f}%')
    c3.metric('Start Difficulty', f'{start_diff:.2f}')
    c4.metric('End Difficulty', f'{end_diff:.2f}')

    # Charts
    rounds = np.array([h['round'] for h in history])
    diff_before = np.array([h['difficulty_before'] for h in history])
    diff_after = np.array([h['difficulty_after'] for h in history])
    q_levels = np.array([h['question_level'] for h in history])
    predicted = np.array([h['predicted_diff'] for h in history])
    correct_arr = np.array(accuracies, dtype=float)

    window = min(5, len(correct_arr))
    rolling = np.convolve(correct_arr, np.ones(window) / window, mode='same')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(rounds, diff_before, marker='o', label='Before')
    axes[0, 0].plot(rounds, diff_after, marker='o', label='After')
    axes[0, 0].plot(rounds, q_levels, linestyle='--', alpha=0.6, label='Question level')
    axes[0, 0].set_title('Difficulty Trajectory')
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(rounds, rolling, color='green', label=f'Rolling (window={window})')
    axes[0, 1].scatter(rounds, correct_arr, alpha=0.4, label='Per-round')
    axes[0, 1].set_title('Accuracy Trend')
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].scatter(q_levels, predicted, alpha=0.7)
    lo, hi = min(q_levels.min(), predicted.min()), max(q_levels.max(), predicted.max())
    axes[1, 0].plot([lo, hi], [lo, hi], 'r--')
    axes[1, 0].set_title('Predicted vs Actual Difficulty')
    axes[1, 0].set_xlabel('Question level')
    axes[1, 0].set_ylabel('NN predicted')
    axes[1, 0].grid(alpha=0.3)

    unique_lvls, counts = np.unique(q_levels, return_counts=True)
    axes[1, 1].bar([str(l) for l in unique_lvls], counts, color='purple', alpha=0.8)
    axes[1, 1].set_title('Question Level Usage')
    axes[1, 1].set_xlabel('Level')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)
    if col1.button('Start New Quiz'):
        st.session_state.history = []
        st.session_state.quiz_round = 0
        st.session_state.streak = 0
        st.session_state.recent_questions = []
        st.session_state.last_feedback = None
        st.session_state.phase = 'quiz'
        st.rerun()
    if col2.button('Logout'):
        _logout()
        st.rerun()


# ── Router ────────────────────────────────────────────────────────────────────

PAGES = {
    'login': page_login,
    'baseline': page_baseline,
    'quiz': page_quiz,
    'results': page_results,
}

PAGES[st.session_state.phase]()
```

- [ ] **Step 2: Run the full app and test the golden path**

```bash
.venv/Scripts/python.exe -m streamlit run app.py
```

Test this sequence manually:
1. Register a new user → should trigger pretraining spinner
2. Log in → should show Baseline Test (10 questions)
3. Answer all 10 baseline questions → should show starting difficulty
4. Click "Start Quiz" → should show the quiz
5. Answer 3–4 questions → verify difficulty changes in sidebar
6. Click "End Quiz" → should show Results Dashboard with charts
7. Logout → should return to login page
8. Log back in → should skip baseline test, go straight to quiz at saved difficulty

- [ ] **Step 3: Run all unit tests to confirm nothing regressed**

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add adaptive quiz and results dashboard pages to Streamlit app"
```

---

## Self-Review

### Spec Coverage Check

| Spec Requirement | Task |
|---|---|
| 5→16→8→4→1 neural network | Task 2 |
| 5-feature input vector | Task 6 (encode_features) |
| Online training (one backward per round) | Task 10 (page_quiz) |
| MSE-based gradient descent | Task 2 (backward) |
| Pretraining 2000 epochs | Task 6 (pretrain) |
| Save/load weights per user | Task 9 (_save_user_model, _load_user_model) |
| GMAT baseline test (10 questions, one per level) | Task 9 (page_baseline) |
| Weighted baseline scoring | Task 6 (baseline_score) |
| Baseline stored in .npz, shown once | Task 9 + Task 10 |
| SHA-256 password hashing | Task 4 |
| users.json auth | Task 4 |
| Per-user models/ directory | Task 9 |
| Streamlit routing (login→baseline→quiz→results) | Task 10 (PAGES router) |
| Difficulty gauge, accuracy, streak in sidebar | Task 10 (page_quiz sidebar) |
| 4 result charts (trajectory, accuracy, scatter, heatmap) | Task 10 (page_results) |
| Anti-repetition (last 10 questions) | Task 6 (sample_question) |
| DeepMind dataset preprocessor | Task 8 |
| Fallback to existing questions.csv | Task 6 (load_questions) |

All spec requirements are covered. ✓

### Type / Method Name Consistency Check

- `DeepNN.load` is a classmethod in Task 2 and called as `DeepNN.load(path)` in Task 9 ✓
- `qz.encode_features(correct, current_difficulty, rolling_accuracy, streak, question_level)` signature matches usage in Task 10 ✓
- `qz.adjust_difficulty(old_diff, predicted_diff, question_level, correct)` matches usage in Task 10 ✓
- `qz.baseline_score(answers)` where answers is `list[(level, correct)]` matches Task 9 ✓
- `auth.login(username, password)` and `auth.register(username, password)` match Task 9 usage ✓
- `_save_user_model(username, nn, baseline_difficulty)` defined and called consistently ✓
