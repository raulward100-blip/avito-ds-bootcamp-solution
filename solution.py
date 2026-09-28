import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi

# Функция очистки текста
# Приводим текст к единому виду: нижний регистр, убираем
# знаки препинания и лишние пробелы. Так «Автоподбор» и
# «автоподбор» будут считаться одинаковыми.
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    for ch in [",", ".", "!", "?", ";", ":", "-", "(", ")", "\"", "'"]:
        text = text.replace(ch, " ")
    return " ".join(text.split())

# Шаг 1. Загружаем данные
print("=" * 50)
print("ШАГ 1: Загрузка данных")
print("=" * 50)

train = pd.read_parquet("train.parquet")
queries = pd.read_parquet("benchmark_queries.parquet")
items = pd.read_parquet("benchmark_items.parquet")

print(f"Загружено: train={len(train)}, queries={len(queries)}, items={len(items)}")

# Шаг 2. Готовим текст
# Для запроса склеиваем search_query и search_infm_params_text.
# Для объявления склеиваем заголовок, описание и параметры.
# Чем больше текста, тем точнее поиск.
print("=" * 50)
print("ШАГ 2: Очистка текста")
print("=" * 50)

queries["query_text"] = (
    queries["search_query"].fillna("") + " " +
    queries["search_infm_params_text"].fillna("")
).apply(clean_text)

items["item_text"] = (
    items["item_title_raw"].fillna("") + " " +
    items["item_description_raw"].fillna("") + " " +
    items["item_infm_params_text"].fillna("")
).apply(clean_text)

print("Текст очищен")

# Шаг 3. Строим BM25-индекс=
print("=" * 50)
print("ШАГ 3: Токенизация и BM25")
print("=" * 50)

tokenized_corpus = [doc.split() for doc in items["item_text"]]
bm25 = BM25Okapi(tokenized_corpus)

print("BM25 создан")

# Шаг 4. Ищем кандидатов для каждого запроса
# Для каждого запроса:
# 1. Считаем BM25-скоры для всех объявлений.
# 2. Берём топ-200.
# 3. Фильтруем по категории и локации.
# 4. Финальный ранкинг и топ-50.
print("=" * 50)
print("ШАГ 4: Поиск")
print("=" * 50)

predictions = []
total = len(queries)

for i, query in enumerate(queries.itertuples()):
    tokenized_query = query.query_text.split()

    # BM25-скоры для всех объявлений
    scores = bm25.get_scores(tokenized_query)

    # Берём топ-200 кандидатов
    top_indices = np.argsort(scores)[-200:][::-1]
    combined = list(top_indices)

    # Фильтр по категории: оставляем только объявления из той же категории, что и запрос
    cat_mask = items.iloc[combined]["item_category_id"] == query.search_category
    if cat_mask.sum() > 0:
        combined = [combined[j] for j in range(len(combined)) if cat_mask.iloc[j]]

    # Фильтр по локации: оставляем только объявления из той же локации
    if hasattr(query, "search_location_id") and pd.notna(query.search_location_id):
        loc_mask = items.iloc[combined]["item_location_id"] == query.search_location_id
        if loc_mask.sum() > 0:
            combined = [combined[j] for j in range(len(combined)) if loc_mask.iloc[j]]

    # Финальный ранкинг и топ-50
    final_scores = [scores[idx] for idx in combined]
    final_top = [combined[j] for j in np.argsort(final_scores)[-50:][::-1]]

    predictions.append(items.iloc[final_top]["item_id"].astype(str).tolist())

    if (i + 1) % 100 == 0 or (i + 1) == total:
        print(f"[ПРОГРЕСС] {i + 1} из {total}")

# Шаг 5. Сохраняем результат
print("=" * 50)
print("ШАГ 5: Сохранение")
print("=" * 50)

answer = pd.DataFrame({
    "query_id": queries["query_id"].astype(str),
    "answer": [" ".join(top50) for top50 in predictions]
})

answer.to_csv("answer.csv", index=False)

print("Готово. Файл answer.csv сохранён.")
print(f"Всего строк: {len(answer)}")
