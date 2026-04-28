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


def test_load_questions_invalid_file_fallback(tmp_path):
    path = tmp_path / 'bad.csv'
    path.write_text('difficulty,prompt,result\n0.1,Q,A\n', encoding='utf-8')
    bank = quiz.load_questions(str(path))
    assert bank == quiz.FALLBACK


def test_validate_dataset_success(tmp_path):
    path = tmp_path / 'good.csv'
    path.write_text('level,question,answer\n0.1,What is 1+1?,2\n', encoding='utf-8')
    ok, msg = quiz.validate_dataset(str(path))
    assert ok is True
    assert msg == ''


def test_validate_dataset_missing_columns(tmp_path):
    path = tmp_path / 'bad.csv'
    path.write_text('level,question\n0.1,What is 1+1?\n', encoding='utf-8')
    ok, msg = quiz.validate_dataset(str(path))
    assert ok is False
    assert 'answer' in msg


def test_list_datasets_includes_root_and_uploads(tmp_path):
    root = tmp_path / 'datasets'
    uploads = root / 'uploads'
    uploads.mkdir(parents=True)
    (root / 'questions.csv').write_text('level,question,answer\n0.1,Q,A\n', encoding='utf-8')
    (uploads / 'custom.csv').write_text('level,question,answer\n0.1,Q,A\n', encoding='utf-8')

    datasets = quiz.list_datasets(str(root))
    slugs = {d['slug'] for d in datasets}

    assert slugs == {'questions', 'uploads__custom'}


def test_save_uploaded_dataset_validates_before_saving(tmp_path):
    path, error = quiz.save_uploaded_dataset(
        'bad.csv',
        b'level,question\n0.1,Q\n',
        str(tmp_path),
    )
    assert path is None
    assert 'answer' in error
    assert list(tmp_path.iterdir()) == []


def test_save_uploaded_dataset_success(tmp_path):
    path, error = quiz.save_uploaded_dataset(
        'My Dataset.csv',
        b'level,question,answer\n0.1,Q,A\n',
        str(tmp_path),
    )
    assert error == ''
    assert path.endswith('My_Dataset.csv')
    assert (tmp_path / 'My_Dataset.csv').exists()


def test_dataset_slug_uses_relative_upload_path(tmp_path):
    root = tmp_path / 'datasets'
    path = root / 'uploads' / 'My Dataset.csv'
    slug = quiz.dataset_slug(str(path), str(root))
    assert slug == 'uploads__my_dataset'


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
