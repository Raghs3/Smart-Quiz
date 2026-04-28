import os
import random
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import auth
import quiz as qz
from nn import DeepNN, LAYER_SIZES

MODELS_DIR = 'models'
TOTAL_ROUNDS = 20
BASELINE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(qz.DATASETS_DIR, exist_ok=True)
os.makedirs(qz.UPLOADS_DIR, exist_ok=True)


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
        'selected_dataset_path': None,
        'selected_dataset_name': None,
        'selected_dataset_slug': None,
        'current_question': None,
        'current_round': -1,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


def _safe_model_part(value: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9_-]+', '_', value or '').strip('_').lower()
    return safe or 'user'


def _legacy_model_path(username: str) -> str:
    return os.path.join(MODELS_DIR, f'{username}.npz')


def _model_path(username: str, dataset_path: str = None) -> str:
    if dataset_path is None:
        return _legacy_model_path(username)
    return os.path.join(
        MODELS_DIR,
        f'{_safe_model_part(username)}__{qz.dataset_slug(dataset_path)}.npz',
    )


def _is_default_dataset(dataset_path: str) -> bool:
    return qz.dataset_slug(dataset_path) == 'questions'


def _load_user_model(username: str, dataset_path: str):
    """Load user NN and baseline result. Returns (nn, start_difficulty, has_baseline)."""
    path = _model_path(username, dataset_path)
    if not os.path.exists(path) and _is_default_dataset(dataset_path):
        path = _legacy_model_path(username)
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


def _save_user_model(username: str, nn, baseline_difficulty=None):
    path = _model_path(username, st.session_state.selected_dataset_path)
    arrays = {f'W{i}': W for i, W in enumerate(nn.weights)}
    arrays.update({f'b{i}': b for i, b in enumerate(nn.biases)})
    arrays['layer_sizes'] = np.array(nn.layer_sizes)
    if baseline_difficulty is not None:
        arrays['baseline_difficulty'] = np.array(baseline_difficulty)
    np.savez(path, **arrays)


def _get_bank():
    if st.session_state.bank is None:
        st.session_state.bank = qz.load_questions(st.session_state.selected_dataset_path)
    return st.session_state.bank


def _reset_quiz_state():
    st.session_state.difficulty = 0.1
    st.session_state.history = []
    st.session_state.streak = 0
    st.session_state.recent_questions = []
    st.session_state.quiz_round = 0
    st.session_state.rounds = TOTAL_ROUNDS
    st.session_state.last_feedback = None
    st.session_state.baseline_idx = 0
    st.session_state.baseline_answers = []
    st.session_state.baseline_questions = []
    st.session_state.current_question = None
    st.session_state.current_round = -1


def _build_baseline_questions(bank: dict) -> list:
    baseline_qs = []
    for lvl in BASELINE_LEVELS:
        lvl_pool = bank.get(lvl, [])
        if lvl_pool:
            baseline_qs.append((lvl, *random.choice(lvl_pool)))
        else:
            baseline_qs.append((lvl, f'What is {int(lvl * 10)}+0?', str(int(lvl * 10))))
    return baseline_qs


def _activate_dataset(dataset: dict):
    path = dataset['path']
    ok, msg = qz.validate_dataset(path)
    if not ok:
        st.error(msg)
        return

    _reset_quiz_state()
    st.session_state.selected_dataset_path = path
    st.session_state.selected_dataset_name = dataset['name']
    st.session_state.selected_dataset_slug = dataset['slug']
    bank = qz.load_questions(path)
    st.session_state.bank = bank

    nn, start_diff, has_baseline = _load_user_model(st.session_state.user, path)
    if nn is None:
        nn = DeepNN(LAYER_SIZES)
        with st.spinner('Running pretraining for this dataset...'):
            qz.pretrain(nn, bank)
        _save_user_model(st.session_state.user, nn)
        has_baseline = False

    st.session_state.nn = nn
    if has_baseline:
        st.session_state.difficulty = start_diff
        st.session_state.phase = 'quiz'
    else:
        st.session_state.baseline_questions = _build_baseline_questions(bank)
        st.session_state.phase = 'baseline'
    st.rerun()


def _logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    _init_state()


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
            st.session_state.user = username
            st.session_state.phase = 'dataset_select'
            st.rerun()
        else:
            st.error('Incorrect username or password.')

    if submitted_register:
        ok, msg = auth.register(username, password)
        if ok:
            st.success('Account created! Please log in.')
        else:
            st.error(msg)


def page_dataset_select():
    if not st.session_state.user:
        st.session_state.phase = 'login'
        st.rerun()

    st.title('Choose Dataset')
    st.caption('Select a question bank before starting your baseline or quiz.')

    uploaded = st.file_uploader('Upload CSV dataset', type=['csv'])
    if uploaded is not None and st.button('Save Uploaded Dataset'):
        saved_path, error = qz.save_uploaded_dataset(uploaded.name, uploaded.getvalue())
        if error:
            st.error(error)
        else:
            st.success(f'Saved {os.path.basename(saved_path)}')

    datasets = qz.list_datasets()
    if not datasets:
        st.warning('No datasets found. Upload a CSV with level, question, and answer columns.')
        return

    selected = st.selectbox(
        'Available datasets',
        datasets,
        format_func=lambda d: f"{d['name']} ({d['slug']})",
    )

    col1, col2 = st.columns(2)
    if col1.button('Continue'):
        _activate_dataset(selected)
    if col2.button('Logout'):
        _logout()
        st.rerun()


def page_baseline():
    idx = st.session_state.baseline_idx
    questions = st.session_state.baseline_questions
    total = len(questions)

    st.title('Baseline Assessment')
    st.caption('GMAT-style - answer all questions to set your starting difficulty.')
    st.progress(idx / total, text=f'Question {idx + 1} of {total}')

    if idx >= total:
        score = qz.baseline_score(
            [(lvl, correct) for lvl, correct in st.session_state.baseline_answers]
        )
        _save_user_model(st.session_state.user, st.session_state.nn, score)
        st.session_state.difficulty = score
        st.success(f'Baseline complete! Starting difficulty set to **{score:.2f}**')
        if st.button('Start Quiz'):
            st.session_state.phase = 'quiz'
            st.rerun()
        return

    lvl, question, answer = questions[idx]
    st.markdown(f'### {question}')
    st.caption(f'Difficulty level: {lvl:.1f}')

    with st.form(f'baseline_form_{idx}'):
        user_ans = st.text_input('Your answer')
        submitted = st.form_submit_button('Submit')

    if submitted:
        correct = qz.normalize(user_ans) == qz.normalize(answer)
        st.session_state.baseline_answers.append((lvl, correct))
        st.session_state.baseline_idx += 1
        if correct:
            st.success('Correct!')
        else:
            st.error(f'Incorrect. Answer: {answer}')
        st.rerun()


def page_quiz():
    user = st.session_state.user
    nn = st.session_state.nn
    bank = _get_bank()

    with st.sidebar:
        st.header(f'Player: {user}')
        st.caption(f"Dataset: {st.session_state.selected_dataset_name}")
        st.metric('Difficulty', f"{st.session_state.difficulty:.2f}")
        history = st.session_state.history
        if history:
            rolling_acc = qz.compute_rolling_accuracy(history)
            st.metric('Rolling Accuracy', f"{rolling_acc * 100:.0f}%")
        streak = st.session_state.streak
        streak_label = f'+{streak}' if streak > 0 else str(streak)
        st.metric('Streak', streak_label)
        st.session_state.rounds = st.slider(
            'Total Rounds',
            min_value=5,
            max_value=50,
            value=st.session_state.rounds,
            step=5,
        )
        if st.button('Change Dataset'):
            st.session_state.phase = 'dataset_select'
            st.rerun()
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

    feedback = st.session_state.last_feedback
    if feedback:
        if feedback['correct']:
            st.success(f"Correct! Difficulty -> {feedback['new_diff']:.2f}")
        else:
            st.error(
                f"Incorrect. Answer was: {feedback['answer']}. "
                f"Difficulty -> {feedback['new_diff']:.2f}"
            )

    recent = st.session_state.recent_questions
    diff = st.session_state.difficulty

    if st.session_state.current_round != rnd:
        lvl, question, answer = qz.sample_question(bank, diff, recent)
        st.session_state.current_question = (lvl, question, answer)
        st.session_state.current_round = rnd
    else:
        lvl, question, answer = st.session_state.current_question

    st.markdown(f'## {question}')

    with st.form(f'quiz_form_{rnd}'):
        user_ans = st.text_input('Your answer', key=f'ans_{rnd}')
        submitted = st.form_submit_button('Submit Answer')

    if submitted:
        correct = qz.normalize(user_ans) == qz.normalize(answer)

        recent.append(question)
        if len(recent) > qz.MAX_RECENT:
            recent.pop(0)

        streak = st.session_state.streak
        streak = (max(0, streak) + 1) if correct else (min(0, streak) - 1)
        st.session_state.streak = streak

        rolling_acc = qz.compute_rolling_accuracy(st.session_state.history)

        x = qz.encode_features(correct, diff, rolling_acc, streak, lvl)
        out = nn.forward(x)
        nn.backward(diff, lr=0.05)
        predicted = float(np.clip(out[0], 0.0, 1.0))

        new_diff = qz.adjust_difficulty(diff, predicted, lvl, correct)
        st.session_state.difficulty = new_diff

        st.session_state.history.append({
            'round': rnd + 1,
            'question': question,
            'question_level': lvl,
            'correct': correct,
            'predicted_diff': predicted,
            'difficulty_before': diff,
            'difficulty_after': new_diff,
        })

        st.session_state.last_feedback = {
            'correct': correct,
            'answer': answer,
            'new_diff': new_diff,
        }
        st.session_state.quiz_round += 1
        st.rerun()


def page_results():
    st.title('Quiz Results')
    st.caption(f"Dataset: {st.session_state.selected_dataset_name}")
    history = st.session_state.history

    if not history:
        st.info('No rounds played yet.')
        if st.button('Back to Quiz'):
            st.session_state.phase = 'quiz'
            st.rerun()
        return

    accuracies = [1 if h['correct'] else 0 for h in history]
    start_diff = history[0]['difficulty_before']
    end_diff = history[-1]['difficulty_after']
    total_acc = sum(accuracies) / len(accuracies)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Rounds Played', len(history))
    c2.metric('Accuracy', f'{total_acc * 100:.1f}%')
    c3.metric('Start Difficulty', f'{start_diff:.2f}')
    c4.metric('End Difficulty', f'{end_diff:.2f}')

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
    lo = min(float(q_levels.min()), float(predicted.min()))
    hi = max(float(q_levels.max()), float(predicted.max()))
    axes[1, 0].plot([lo, hi], [lo, hi], 'r--')
    axes[1, 0].set_title('Predicted vs Actual Difficulty')
    axes[1, 0].set_xlabel('Question level')
    axes[1, 0].set_ylabel('NN predicted')
    axes[1, 0].grid(alpha=0.3)

    unique_lvls, counts = np.unique(q_levels, return_counts=True)
    axes[1, 1].bar([str(lvl) for lvl in unique_lvls], counts, color='purple', alpha=0.8)
    axes[1, 1].set_title('Question Level Usage')
    axes[1, 1].set_xlabel('Level')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    col1, col2, col3 = st.columns(3)
    if col1.button('Start New Quiz'):
        st.session_state.history = []
        st.session_state.quiz_round = 0
        st.session_state.streak = 0
        st.session_state.recent_questions = []
        st.session_state.last_feedback = None
        st.session_state.current_question = None
        st.session_state.current_round = -1
        st.session_state.phase = 'quiz'
        st.rerun()
    if col2.button('Change Dataset'):
        st.session_state.phase = 'dataset_select'
        st.rerun()
    if col3.button('Logout'):
        _logout()
        st.rerun()


PAGES = {
    'login': page_login,
    'dataset_select': page_dataset_select,
    'baseline': page_baseline,
    'quiz': page_quiz,
    'results': page_results,
}

PAGES[st.session_state.phase]()
