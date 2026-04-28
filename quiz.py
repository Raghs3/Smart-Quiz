import csv
import io
import os
import random
import re
import numpy as np

DATASETS_DIR = 'datasets'
UPLOADS_DIR = os.path.join(DATASETS_DIR, 'uploads')
QUESTIONS_CSV = os.path.join(DATASETS_DIR, 'questions.csv')
REQUIRED_COLUMNS = {'level', 'question', 'answer'}
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


def _read_bank(reader) -> dict:
    bank = {}
    for row in reader:
        try:
            lvl = round(float(np.clip(float(row['level']), 0.1, 1.0)), 1)
        except (ValueError, KeyError, TypeError):
            continue
        q = (row.get('question') or '').strip()
        a = (row.get('answer') or '').strip()
        if q and a:
            bank.setdefault(lvl, []).append((q, a))
    return bank


def load_questions(path=None) -> dict:
    """Return dict: level (float) -> list of (question, answer) tuples."""
    p = path or QUESTIONS_CSV
    if not os.path.exists(p):
        return FALLBACK
    try:
        with open(p, newline='', encoding='utf-8') as f:
            return _read_bank(csv.DictReader(f)) or FALLBACK
    except (OSError, UnicodeDecodeError):
        return FALLBACK


def dataset_slug(path: str, datasets_dir: str = DATASETS_DIR) -> str:
    """Return a filesystem-safe slug for a dataset path."""
    dataset_abs = os.path.abspath(path)
    root_abs = os.path.abspath(datasets_dir)
    try:
        label = os.path.splitext(os.path.relpath(dataset_abs, root_abs))[0]
    except ValueError:
        label = os.path.splitext(os.path.basename(path))[0]
    label = label.replace(os.sep, '__')
    if os.altsep:
        label = label.replace(os.altsep, '__')
    slug = re.sub(r'[^A-Za-z0-9_-]+', '_', label).strip('_').lower()
    return slug or 'dataset'


def list_datasets(datasets_dir: str = DATASETS_DIR) -> list:
    """Return available dataset CSVs from datasets/ and datasets/uploads/."""
    roots = [datasets_dir, os.path.join(datasets_dir, 'uploads')]
    datasets = []
    seen = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for fname in sorted(os.listdir(root)):
            if not fname.lower().endswith('.csv'):
                continue
            path = os.path.join(root, fname)
            key = os.path.abspath(path)
            if key in seen or not os.path.isfile(path):
                continue
            seen.add(key)
            datasets.append({
                'name': fname,
                'path': path,
                'slug': dataset_slug(path, datasets_dir),
            })
    return datasets


def _validate_reader(reader) -> tuple:
    fieldnames = set(reader.fieldnames or [])
    missing = sorted(REQUIRED_COLUMNS - fieldnames)
    if missing:
        return False, f"Missing required column(s): {', '.join(missing)}"
    if not _read_bank(reader):
        return False, 'Dataset has no valid question rows.'
    return True, ''


def validate_dataset(path: str) -> tuple:
    """Validate that a CSV file has the columns and rows needed by the quiz."""
    try:
        with open(path, newline='', encoding='utf-8') as f:
            return _validate_reader(csv.DictReader(f))
    except OSError as exc:
        return False, str(exc)
    except UnicodeDecodeError:
        return False, 'Dataset must be UTF-8 encoded.'


def validate_dataset_bytes(data: bytes) -> tuple:
    """Validate an uploaded dataset before saving it."""
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        return False, 'Dataset must be UTF-8 encoded.'
    return _validate_reader(csv.DictReader(io.StringIO(text)))


def safe_dataset_filename(filename: str) -> str:
    """Return a safe CSV filename for an uploaded dataset."""
    base = os.path.basename(filename or '').strip()
    name, ext = os.path.splitext(base)
    name = re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_')
    if not name:
        name = 'uploaded_dataset'
    return f'{name}{ext.lower() or ".csv"}'


def save_uploaded_dataset(filename: str, data: bytes,
                          uploads_dir: str = UPLOADS_DIR) -> tuple:
    """Validate and save an uploaded CSV. Returns (path, error_message)."""
    safe_name = safe_dataset_filename(filename)
    if not safe_name.lower().endswith('.csv'):
        return None, 'Dataset upload must be a CSV file.'
    ok, msg = validate_dataset_bytes(data)
    if not ok:
        return None, msg

    os.makedirs(uploads_dir, exist_ok=True)
    stem, ext = os.path.splitext(safe_name)
    path = os.path.join(uploads_dir, safe_name)
    counter = 2
    while os.path.exists(path):
        path = os.path.join(uploads_dir, f'{stem}_{counter}{ext}')
        counter += 1
    with open(path, 'wb') as f:
        f.write(data)
    return path, ''


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
