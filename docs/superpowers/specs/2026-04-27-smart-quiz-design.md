# Smart Quiz — Design Spec
**Date:** 2026-04-27
**Project:** VIT SY MFAI Course Project

---

## Overview

An adaptive math quiz application that uses a custom multi-layer neural network (built from scratch with NumPy) to dynamically adjust question difficulty based on user performance. Includes a GMAT-style baseline test for new users, per-user login with persistent model weights, and a Streamlit UI.

---

## Constraints

- No pre-built AI/ML model libraries (no scikit-learn, TensorFlow, PyTorch, etc.)
- NumPy and pandas are allowed
- All neural network math implemented manually
- UI via Streamlit
- Python venv at `D:\Coding\College\SY\SEM2\VIT_SY_MFAI_CP\mine\.venv`

---

## System Flow

```
Login / Register
    ↓
[First login only] GMAT-Style Baseline Test (10 questions)
    ↓ computes starting difficulty
Adaptive Quiz (deep NN adjusts difficulty each round)
    ↓
Results Dashboard (charts + summary)
```

---

## 1. Neural Network Architecture

**Architecture:** `5 → 16 → 8 → 4 → 1`

### Input Features (5)

| Feature | Range | Description |
|---|---|---|
| `correct` | {0, 1} | Whether the current answer was correct |
| `current_difficulty` | [0.0, 1.0] | Difficulty target before this round |
| `rolling_accuracy` | [0.0, 1.0] | Mean correctness over last 5 rounds |
| `streak` | [-1.0, 1.0] | Consecutive streak capped at ±5, divided by 5. Positive = correct run, negative = wrong run |
| `question_level` | [0.0, 1.0] | Actual difficulty level of the question served |

### Layers

- **Hidden 1:** 16 neurons, sigmoid activation
- **Hidden 2:** 8 neurons, sigmoid activation
- **Hidden 3:** 4 neurons, sigmoid activation
- **Output:** 1 neuron, sigmoid → predicted next difficulty ∈ [0.0, 1.0]

### Training

- **Loss:** Mean squared error (MSE)
- **Optimizer:** Gradient descent, learning rate = 0.05
- **Mode:** Online (one backward pass per quiz round)
- **Pretraining:** 2000 synthetic epochs on first run to warm up weights before the user plays

### Persistence

- Weights stored per user in `models/<username>.npz`
- Loaded on login, saved on quiz end or "End Quiz" button

---

## 2. GMAT-Style Baseline Test

Shown only on first login for a new user.

- 10 fixed questions, one from each difficulty level: 0.1, 0.2, …, 1.0
- Questions selected randomly from the question bank at each level
- Presented in order from easiest (0.1) to hardest (1.0)
- **Scoring:** weighted sum — harder correct answers contribute more
  - Weight for level `l`: `w(l) = l` (linear weight)
  - `score = sum(w(l) * correct(l)) / sum(w(l))` → normalized to [0.0, 1.0]
- Resulting score becomes the user's starting difficulty for the adaptive quiz
- Baseline result stored in the user's `.npz` so the test is never repeated

---

## 3. Adaptive Quiz

- Default 20 rounds (configurable via Streamlit sidebar)
- Each round: serve question at current difficulty level, accept answer, run one NN forward + backward pass
- Difficulty adjustment after each round:
  - Correct: `difficulty = max(old, 0.6 * predicted + 0.4 * question_level + step)`
  - Wrong: `difficulty = min(old, 0.6 * predicted + 0.4 * question_level - step)`
  - `step = 0.08`
- Anti-repetition: tracks last 10 questions, avoids repeats when possible
- Difficulty clamped to [0.1, 1.0]

---

## 4. User Login & Per-User Persistence

### Auth

- Accounts stored in `users.json`
- Passwords hashed with SHA-256 (no plain text)
- Register and login on the same Streamlit page
- `st.session_state` tracks the active user within a session

### File Structure

```
users.json                  ← account registry (username + password hash)
models/
  <username>.npz            ← NN weights + baseline result per user
questions.csv               ← full question bank (level, question, answer)
```

---

## 5. Question Bank (DeepMind Dataset)

### Source

Google DeepMind Mathematics Dataset — open source, school-level math, explicit difficulty splits (easy / medium / hard), short exact answers.

### Preprocessing (`prepare_data.py`)

One-time script run before the app:
1. Reads DeepMind text files from `train-easy/`, `train-medium/`, `train-hard/`
2. Maps difficulty splits to numeric levels:
   - easy → 0.1, 0.2, 0.3
   - medium → 0.4, 0.5, 0.6, 0.7
   - hard → 0.8, 0.9, 1.0
3. Writes `questions.csv` with columns: `level, question, answer`

After preprocessing, the app has no runtime dependency on the DeepMind repo.

---

## 6. Streamlit UI

### Pages (single-file app with `st.session_state` routing)

**Login / Register**
- Username + password fields
- "Login" and "Register" buttons
- Error messages for wrong credentials or duplicate usernames

**Baseline Test** (new users only)
- Progress bar (1/10 … 10/10)
- Question displayed prominently
- Text input for answer
- "Submit" button advances to next question
- Final screen shows computed starting difficulty before entering quiz

**Adaptive Quiz**
- Question text (large font)
- Text input + "Submit Answer" button
- Sidebar: difficulty gauge (progress bar), current accuracy, streak indicator
- After each answer: coloured feedback ("Correct!" / "Incorrect") + new difficulty shown
- "End Quiz" button → saves model → goes to results

**Results Dashboard**
- Summary card: starting difficulty, ending difficulty, total accuracy, rounds played
- Charts (matplotlib figures embedded via `st.pyplot`):
  - Difficulty trajectory (before/after per round + question level)
  - Rolling accuracy trend
  - Question-level usage bar chart
  - Predicted vs actual difficulty scatter
- "Start New Quiz" button (resets round state, keeps weights)
- "Logout" button

---

## 7. File Structure

```
mine/
├── app.py                  ← Streamlit entry point
├── prepare_data.py         ← one-time DeepMind dataset preprocessor
├── nn.py                   ← DeepNN class (5→16→8→4→1, forward, backward)
├── quiz.py                 ← quiz logic, feature encoding, difficulty adjustment
├── auth.py                 ← login, register, password hashing
├── questions.csv           ← preprocessed question bank
├── users.json              ← user accounts
├── models/                 ← per-user weight files
│   └── <username>.npz
├── docs/superpowers/specs/
│   └── 2026-04-27-smart-quiz-design.md
└── .venv/
```

---

## 8. Migration from Existing Files

The existing `main.py` and `smart_quiz_vectorized.ipynb` are replaced by the new modular structure. Their logic is preserved and distributed as follows:
- `SimpleNN` class → `nn.py` (upgraded to 5→16→8→4→1)
- Quiz loop logic → `quiz.py`
- CSV loading, question sampling → `quiz.py`
- Pretraining, save/load → `nn.py` + `quiz.py`
- The notebook (`smart_quiz_vectorized.ipynb`) is kept as-is for reference/demo purposes but is not part of the running app.

---

## 9. Out of Scope

- No online/cloud storage — everything is local files
- No multiple choice questions — exact string matching only
- No time limits per question (can be added later)
- No admin panel for managing questions
