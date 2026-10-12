"""Страница на Streamlit: загружаешь таблицу с ответами учеников и получаешь рекомендации."""
import streamlit as st

from recommender import load_model, read_answers, recommend, student_topics

REPO = "https://github.com/polinkabaranova/recsys-tutor"

st.set_page_config(page_title="Подбор задач для ученика", page_icon="📘")
st.title("Подбор задач для ученика")
st.write(
    "Загрузите таблицу с ответами учеников, заполненную по шаблону. "
    "Для каждого ученика система подберёт задачи из его слабых тем, "
    "которые он решит примерно с шансом 70%: не слишком легко и не слишком трудно."
)

with open("template_answers.xlsx", "rb") as f:
    st.download_button("Скачать шаблон таблицы", f, file_name="Шаблон_ответы_учеников.xlsx")

st.caption("В таблице — только коды учеников (У1, У2…), без имён. Приложение не сохраняет загруженный файл.")

file = st.file_uploader("Таблица с ответами учеников (.xlsx)", type=["xlsx"])
use_demo = st.checkbox("Нет своей таблицы? Показать на демо-данных (придуманные ученики)")

source = file if file is not None else ("demo_answers.xlsx" if use_demo else None)
if source is None:
    st.info("Загрузите таблицу или включите демо-данные.")
    st.stop()

try:
    df = read_answers(source)
except Exception as error:
    st.error(f"Не получилось прочитать таблицу. Проверьте, что она заполнена по шаблону (лист «Ответы»). Ошибка: {error}")
    st.stop()

if df.empty:
    st.warning("В таблице не нашлось ответов. Заполните лист «Ответы», начиная с третьей строки.")
    st.stop()

if source == "demo_answers.xlsx":
    st.info("Это придуманные данные для демонстрации, а не настоящие ученики.")

intercept, coef = load_model()

c1, c2, c3 = st.columns(3)
c1.metric("Учеников", df["user_id"].nunique())
c2.metric("Ответов", len(df))
c3.metric("Задач", df["problem_id"].nunique())

student_id = st.selectbox("Ученик", sorted(df["user_id"].unique()))
student = df[df["user_id"] == student_id]
st.write(f"Ответов: **{len(student)}**, решено с первой попытки: **{student['correct'].mean():.0%}**")
if len(student) < 20:
    st.warning("У ученика меньше 20 ответов, поэтому рекомендации будут менее точными.")

st.subheader("Темы ученика: от слабых к сильным")
topics = student_topics(student).reset_index()
topics_view = topics.assign(share=(topics["sum"] / topics["count"]).map("{:.0%}".format))
topics_view = topics_view[["skill_name", "count", "sum", "share"]]
topics_view.columns = ["Тема", "Ответов", "Решено", "Доля решённых"]
st.dataframe(topics_view, hide_index=True)

with st.expander("Настройки подбора"):
    n = st.slider("Сколько задач предложить", 1, 10, 5)
    target = st.slider("Целевой шанс решить", 0.5, 0.9, 0.7, 0.05)

st.subheader("Рекомендованные задачи")
rec = recommend(df, student_id, intercept, coef, n=n, target=target)
if rec.empty:
    st.info(
        "В слабых темах ученика нет задач, которые он ещё не решал. "
        "Рекомендовать можно только задачи, которые уже есть в таблице у других учеников."
    )
else:
    rec_view = rec[["problem_id", "skill_name"]].copy()
    rec_view["Шанс решить"] = rec["chance"].map("{:.0%}".format)
    rec_view.columns = ["Задача", "Тема", "Шанс решить"]
    st.dataframe(rec_view, hide_index=True)
    st.download_button(
        "Скачать рекомендации (CSV)",
        rec_view.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"рекомендации_{student_id}.csv",
        mime="text/csv",
    )

st.caption(
    "Шанс считает логистическая регрессия, обученная на открытых данных ASSISTments "
    "(около 275 тысяч ответов американских школьников). Это подсказка, а не приговор: "
    f"решение остаётся за репетитором. Код и подробности: {REPO}"
)
