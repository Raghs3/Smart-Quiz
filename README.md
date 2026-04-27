# Smart Quiz

An adaptive mathematics quiz application that adjusts question difficulty in real time based on your performance, powered by a custom deep neural network built from scratch with NumPy.

---

## How It Works

The app follows three phases after login:

1. **Baseline Assessment** — 10 GMAT-style questions spanning difficulty 0.1–1.0 to estimate your starting level.
2. **Adaptive Quiz** — Questions are selected near your current difficulty. After each answer, a 4-layer neural network predicts the next difficulty score using correctness, rolling accuracy, streak, and question level as inputs. Difficulty is monotonically enforced (correct → harder, wrong → easier).
3. **Results Dashboard** — Charts showing difficulty trajectory, accuracy trend, predicted vs actual difficulty, and question level distribution.

User models are saved per-account (`models/<username>.npz`) so progress persists across sessions.

---

## Project Structure

```
├── app.py            # Streamlit app — login, baseline, quiz, results pages
├── nn.py             # DeepNN: 4-layer [5→16→8→4→1] numpy neural network
├── quiz.py           # Question loading, feature encoding, difficulty adjustment
├── auth.py           # User registration and login (SHA-256, JSON storage)
├── questions.csv     # Question bank (level, question, answer)
├── models/           # Per-user trained model weights (.npz) — gitignored
├── users.json        # User credentials — gitignored
├── prepare_data.py   # One-time utility: converts DeepMind dataset → questions.csv
└── tests/
    ├── test_auth.py
    ├── test_nn.py
    └── test_quiz.py
```

---

## Neural Network

`nn.py` implements `DeepNN`, a fully connected network with sigmoid activations:

- **Input (5 features):** correctness, current difficulty, rolling accuracy (last 5 rounds), streak (normalised), question level
- **Hidden layers:** 16 → 8 → 4 neurons
- **Output (1 neuron):** predicted next difficulty in [0, 1]
- **Training:** online backpropagation after every answer (lr = 0.05)
- **Pretraining:** 2000 synthetic rounds on first login to avoid cold-start random behaviour

---

## Setup

**Requirements:** Python 3.10+

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install streamlit numpy matplotlib
```

**Run the app:**

```bash
streamlit run app.py
```

---

## Question Bank

`questions.csv` ships with 200 hand-written questions (20 per difficulty level 0.1–1.0), covering arithmetic, algebra, calculus, and limits.

To replace it with the [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset):

```bash
python prepare_data.py <deepmind_folder> --max-per-level 500 --out questions.csv
```

---

## Tests

```bash
pytest tests/
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Neural network | NumPy (from scratch) |
| Persistence | `.npz` weight files, JSON user store |
| Tests | pytest |
