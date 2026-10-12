"""Логика рекомендаций для приложения.

Читает таблицу с ответами учеников (Excel по шаблону) и подбирает каждому ученику
задачи из его слабых тем с шансом решить около 70%. Шанс считает логистическая
регрессия, обученная в блокноте recsys_tutor.ipynb на данных ASSISTments:
её параметры лежат в model.json.
"""
import json

import numpy as np
import pandas as pd

FEATURES = ["user_acc", "skill_acc", "skill_count_log", "problem_acc"]
COLUMNS = ["user_id", "date", "problem_id", "skill_name", "correct"]


def load_model(path="model.json"):
    """Свободный член и веса логистической регрессии из блокнота."""
    with open(path, encoding="utf-8") as f:
        model = json.load(f)
    coef = np.array([model["coef"][name] for name in FEATURES])
    return model["intercept"], coef


def predict(intercept, coef, table):
    """Шанс решить задачу: та же формула, что у LogisticRegression."""
    z = intercept + table[FEATURES].to_numpy(dtype=float) @ coef
    return 1 / (1 + np.exp(-z))


def _as_text(column):
    """Номера вроде 12.0 превращаем в «12», всё остальное — в обычный текст."""
    def convert(value):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return column.map(convert).str.strip()


def read_answers(file):
    """Лист «Ответы»: первые пять столбцов — ученик, дата, задача, тема, решил ли."""
    raw = pd.read_excel(file, sheet_name="Ответы")
    df = raw.iloc[:, :5].copy()
    df.columns = COLUMNS
    df = df.dropna(subset=["user_id", "problem_id", "skill_name", "correct"])
    for col in ["user_id", "problem_id", "skill_name"]:
        df[col] = _as_text(df[col])
    df = df[df["user_id"].str.upper() != "ПРИМЕР"]

    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df[df["correct"].isin([0, 1])].copy()
    df["correct"] = df["correct"].astype(int)

    # порядок ответов: по дате, а внутри одной даты — как записано в таблице
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["row"] = np.arange(len(df))
    return df.sort_values(["date", "row"], kind="stable").reset_index(drop=True)


def student_topics(student):
    """Темы ученика от слабых к сильным; доля верных со сглаживанием (верных + 1) / (всего + 2)."""
    topics = student.groupby("skill_name")["correct"].agg(["sum", "count"])
    topics["acc"] = (topics["sum"] + 1) / (topics["count"] + 2)
    return topics.sort_values("acc")


def problem_acc(df, m=5):
    """Доля верных ответов на каждой задаче у всех учеников, с m воображаемыми средними ответами."""
    mean_all = df["correct"].mean()
    stats = df.groupby("problem_id")["correct"].agg(["sum", "count"])
    return (stats["sum"] + m * mean_all) / (stats["count"] + m), mean_all


def recommend(df, student_id, intercept, coef, n=5, target=0.7, n_topics=3, min_answers=3):
    """Нерешённые задачи из самых слабых тем ученика, шанс которых ближе всего к target."""
    student = df[df["user_id"] == student_id]
    solved = set(student["problem_id"])

    topics = student_topics(student)
    eligible = topics[topics["count"] >= min_answers]
    if eligible.empty:  # у ученика пока мало ответов по каждой теме
        eligible = topics
    weak = eligible.head(n_topics)

    tasks = df[["problem_id", "skill_name"]].drop_duplicates("problem_id")
    cand = tasks[tasks["skill_name"].isin(weak.index) & ~tasks["problem_id"].isin(solved)].copy()
    if cand.empty:
        return cand.assign(chance=pd.Series(dtype=float))

    acc, mean_all = problem_acc(df)
    cand["user_acc"] = (student["correct"].sum() + 1) / (len(student) + 2)
    cand["skill_acc"] = cand["skill_name"].map(weak["acc"])
    cand["skill_count_log"] = np.log1p(cand["skill_name"].map(weak["count"]))
    cand["problem_acc"] = cand["problem_id"].map(acc).fillna(mean_all)

    cand["chance"] = predict(intercept, coef, cand)
    cand["distance"] = (cand["chance"] - target).abs()
    return cand.sort_values("distance").head(n).reset_index(drop=True)
