import random
import os
import csv
import numpy as np

# Simple Neural Network with 2 layers (input-hidden-output)
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases randomly
        self.W1 = np.random.uniform(-1, 1, size=(hidden_size, input_size))
        self.b1 = np.random.uniform(-1, 1, size=(hidden_size, 1))
        self.W2 = np.random.uniform(-1, 1, size=(output_size, hidden_size))
        self.b2 = np.random.uniform(-1, 1, size=(output_size, 1))

    def sigmoid(self, x):
        # Sigmoid activation
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, s):
        # Derivative of sigmoid
        return s * (1 - s)

    def forward(self, x):
        # Forward pass
        x_col = np.asarray(x, dtype=float).reshape(-1, 1)
        self.last_x = x_col
        self.z1 = self.W1 @ x_col + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2.ravel()

    def backward(self, x, y, lr=0.1):
        # Backward pass (gradient descent)
        _ = np.asarray(x, dtype=float).reshape(-1, 1)
        y_col = np.asarray(y, dtype=float).reshape(-1, 1)

        dz2 = self.a2 - y_col
        dW2 = dz2 @ self.a1.T
        db2 = dz2

        dz1 = (self.W2.T @ dz2) * self.sigmoid_deriv(self.a1)
        dW1 = dz1 @ self.last_x.T
        db1 = dz1

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

MODEL_PATH = "smart_quiz_model.npz"
QUESTIONS_CSV_PATH = "questions.csv"
LEVEL_STEP = 0.1

# Fallback question pools if CSV is missing/invalid.
# Format: level -> list[(question, expected_answer)]
DEFAULT_LEVEL_QUESTIONS = {
    0.1: [
        ("What is 2+2?", "4"),
        ("What is 9-4?", "5"),
        ("What is 7+1?", "8"),
    ],
    0.2: [
        ("What is 12-5?", "7"),
        ("What is 6+9?", "15"),
        ("What is 3*4?", "12"),
    ],
    0.3: [
        ("What is 15/3?", "5"),
        ("What is 14+19?", "33"),
        ("What is 18-7?", "11"),
    ],
    0.4: [
        ("What is 12*12?", "144"),
        ("What is 81/9?", "9"),
        ("What is 25+37?", "62"),
    ],
    0.5: [
        ("Solve for x: x + 7 = 15", "8"),
        ("What is the square root of 169?", "13"),
        ("If 4x = 36, what is x?", "9"),
    ],
    0.6: [
        ("What is 2^5?", "32"),
        ("If 3x = 27, what is x?", "9"),
        ("Solve: 2x + 5 = 17", "6"),
    ],
    0.7: [
        ("What is the derivative of x^2?", "2x"),
        ("Integrate x dx", "0.5x^2"),
        ("Derivative of x^3?", "3x^2"),
    ],
    0.8: [
        ("Derivative of sin(x)?", "cos(x)"),
        ("Integrate 2x dx", "x^2"),
        ("Derivative of cos(x)?", "-sin(x)"),
    ],
    0.9: [
        ("What is the derivative of ln(x)?", "1/x"),
        ("What is the derivative of e^x?", "e^x"),
        ("Integrate e^x dx", "e^x"),
    ],
    1.0: [
        ("What is lim(x->0) sin(x)/x?", "1"),
        ("What is lim(x->∞) 1/x?", "0"),
        ("Integral of 1/x dx", "ln|x|"),
    ],
}

def resolve_data_path(file_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, file_name)

def load_level_questions_from_csv(csv_path, fallback):
    level_questions = {}

    if not os.path.exists(csv_path):
        print(f"Question CSV not found at {csv_path}. Using fallback questions.")
        return fallback

    with open(csv_path, mode="r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_fields = {"level", "question", "answer"}
        if not required_fields.issubset(set(reader.fieldnames or [])):
            print("CSV headers must include: level, question, answer. Using fallback questions.")
            return fallback

        for row in reader:
            try:
                level = float(row["level"])
            except (TypeError, ValueError):
                continue

            level = round(float(np.clip(level, 0.1, 1.0)), 1)
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question or not answer:
                continue

            level_questions.setdefault(level, []).append((question, answer))

    if not level_questions:
        print("CSV had no valid question rows. Using fallback questions.")
        return fallback

    return level_questions

LEVEL_QUESTIONS = load_level_questions_from_csv(
    resolve_data_path(QUESTIONS_CSV_PATH),
    DEFAULT_LEVEL_QUESTIONS,
)

def encode_input(correct):
    # Encode user answer: [correct, incorrect]
    return np.array([1.0, 0.0]) if correct else np.array([0.0, 1.0])

def encode_output(difficulty):
    # Continuous target in [0.0, 1.0]
    return np.array([float(np.clip(difficulty, 0.0, 1.0))])

def decode_output(output):
    # Decode NN output to difficulty score in [0.0, 1.0]
    return float(np.clip(output[0], 0.0, 1.0))

def difficulty_to_level(difficulty):
    clipped = float(np.clip(difficulty, 0.1, 1.0))
    rounded = round(clipped / LEVEL_STEP) * LEVEL_STEP
    return round(float(np.clip(rounded, 0.1, 1.0)), 1)

def get_question_for_difficulty(difficulty, recent_questions=None):
    # Prefer current level; fallback to nearest levels; avoid very recent repeats.
    recent_questions = recent_questions or []
    target_level = difficulty_to_level(difficulty)
    levels_by_distance = sorted(LEVEL_QUESTIONS.keys(), key=lambda lvl: abs(lvl - target_level))

    for level in levels_by_distance:
        pool = LEVEL_QUESTIONS[level]
        fresh_pool = [item for item in pool if item[0] not in recent_questions]
        if fresh_pool:
            q, ans = random.choice(fresh_pool)
            return level, q, ans

    # All have been recently used; allow repeats.
    q, ans = random.choice(LEVEL_QUESTIONS[target_level])
    return target_level, q, ans

def normalize_answer(value):
    return value.strip().lower().replace(" ", "")

def save_model(nn, path=MODEL_PATH):
    np.savez(resolve_data_path(path), W1=nn.W1, b1=nn.b1, W2=nn.W2, b2=nn.b2)

def load_model(nn, path=MODEL_PATH):
    resolved_path = resolve_data_path(path)
    if not os.path.exists(resolved_path):
        return False
    data = np.load(resolved_path)
    nn.W1 = data["W1"]
    nn.b1 = data["b1"]
    nn.W2 = data["W2"]
    nn.b2 = data["b2"]
    return True

def pretrain_model(nn, epochs=2000, lr=0.05):
    # Synthetic warm-up so quiz doesn't start from random behavior.
    difficulty = 0.3
    for _ in range(epochs):
        level, _, _ = get_question_for_difficulty(difficulty)
        p_correct = max(0.1, 1.0 - (level * 0.85))
        correct = random.random() < p_correct

        inp = encode_input(correct)
        out = nn.forward(inp)
        nn.backward(inp, encode_output(level), lr=lr)

        predicted_diff = decode_output(out)
        if correct:
            candidate = min((0.6 * predicted_diff) + (0.4 * level) + 0.04, 1.0)
            difficulty = max(difficulty, candidate)
        else:
            candidate = max((0.6 * predicted_diff) + (0.4 * level) - 0.04, 0.0)
            difficulty = min(difficulty, candidate)

def main():
    nn = SimpleNN(2, 8, 1)

    if load_model(nn):
        print(f"Loaded existing model from {MODEL_PATH}")
    else:
        print("No saved model found. Running pretraining...")
        pretrain_model(nn, epochs=2000, lr=0.05)
        save_model(nn)
        print(f"Pretraining complete. Saved model to {MODEL_PATH}")

    difficulty = 0.10
    step = 0.08
    rounds = 20
    recent_questions = []
    max_recent = 10

    print("Smart Quiz Demo. Answer the questions!")
    for round_number in range(rounds):
        old_difficulty = difficulty
        question_difficulty, q, ans = get_question_for_difficulty(difficulty, recent_questions)
        recent_questions.append(q)
        if len(recent_questions) > max_recent:
            recent_questions.pop(0)

        print(f"\nRound {round_number + 1}/{rounds} | Current difficulty target: {difficulty:.2f}")
        user_ans = input(f"Q: {q} ")
        correct = normalize_answer(user_ans) == normalize_answer(ans)

        inp = encode_input(correct)
        out = nn.forward(inp)

        # Train NN
        nn.backward(inp, encode_output(difficulty))

        # Predict next difficulty
        predicted_diff = decode_output(out)

        # Adjust difficulty score: enforce direction to match correctness.
        if correct:
            candidate = min((0.6 * predicted_diff) + (0.4 * question_difficulty) + step, 1.0)
            difficulty = max(old_difficulty, candidate)
        else:
            candidate = max((0.6 * predicted_diff) + (0.4 * question_difficulty) - step, 0.0)
            difficulty = min(old_difficulty, candidate)

        if difficulty > old_difficulty:
            print(f"Correct! Difficulty increased to {difficulty:.2f}.")
        elif difficulty < old_difficulty:
            print(f"Incorrect. Difficulty decreased to {difficulty:.2f}.")
        else:
            print(f"Difficulty unchanged at {difficulty:.2f}.")

    print("Quiz finished.")
    save_model(nn)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main() 





























    