# Smart Quiz

An adaptive mathematics quiz application that adjusts question difficulty in real time based on your performance, powered by a custom deep neural network built from scratch with NumPy.

## How It Works

The app follows four phases after login:

1. **Dataset Selection** - choose a bundled CSV question bank or upload a new one.
2. **Baseline Assessment** - 10 GMAT-style questions spanning difficulty 0.1-1.0 to estimate your starting level for that dataset.
3. **Adaptive Quiz** - questions are selected near your current difficulty. After each answer, a 4-layer neural network predicts the next difficulty score using correctness, rolling accuracy, streak, and question level as inputs. Difficulty is monotonically enforced.
4. **Results Dashboard** - charts showing difficulty trajectory, accuracy trend, predicted vs actual difficulty, and question level distribution.

User models are saved per account and dataset (`models/<username>__<dataset>.npz`) so progress does not mix across question banks.

## Project Structure

```text
├── app.py                 # Streamlit app: login, dataset selection, baseline, quiz, results
├── nn.py                  # DeepNN: 4-layer [5->16->8->4->1] numpy neural network
├── quiz.py                # Dataset loading/validation, question sampling, adaptive helpers
├── auth.py                # User registration and login (SHA-256, JSON storage)
├── datasets/
│   ├── questions.csv      # Starter question bank
│   ├── questions1.csv     # Larger generated question bank
│   └── uploads/           # Runtime CSV uploads (gitignored except .gitkeep)
├── models/                # Per-user, per-dataset model weights (.npz) - gitignored
├── users.json             # User credentials - gitignored
├── prepare_data.py        # Converts DeepMind dataset output to a quiz CSV
└── tests/
    ├── test_auth.py
    ├── test_nn.py
    └── test_quiz.py
```

## Dataset Format

Datasets are UTF-8 CSV files with these required columns:

```csv
level,question,answer
0.1,What is 2+2?,4
```

`level` is clamped to the 0.1-1.0 difficulty range. Invalid rows are ignored; a dataset must contain at least one valid row to be selectable or uploaded.

## Neural Network

`nn.py` implements `DeepNN`, a fully connected network with sigmoid activations:

- **Input (5 features):** correctness, current difficulty, rolling accuracy, streak, question level
- **Hidden layers:** 16 -> 8 -> 4 neurons
- **Output (1 neuron):** predicted next difficulty in [0, 1]
- **Training:** online backpropagation after every answer
- **Pretraining:** synthetic rounds per selected dataset on first use

## Setup

**Requirements:** Python 3.10+

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Generate a Dataset

To convert the DeepMind Mathematics Dataset into the default dataset path:

```bash
python prepare_data.py <deepmind_folder> --max-per-level 500 --out datasets/questions.csv
```

## Tests

```bash
pytest tests/
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit |
| Neural network | NumPy |
| Persistence | `.npz` weight files, JSON user store |
| Tests | pytest |
