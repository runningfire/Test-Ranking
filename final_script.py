import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
import argparse


# Загрузим токенизатор NLTK
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')


def extract_text_pdfminer(pdf_path):
    """ Извлекает текст с помощью pdfminer.six """
    return extract_text(pdf_path)


def split_text_custom(text):
    """
    Делим текст по \n\n, внутри каждого блока убираем \n,
    затем дополнительно делим по \x0c и возвращаем список чистых абзацев.
    """
    # 1) Делим по \n\n
    raw_blocks = text.split("\n\n")

    final_paragraphs = []

    for block in raw_blocks:
        # 2) Убираем одинарные переносы строк (заменяем \n на пробел)
        block = block.replace("\n", " ")

        # 3) Делим по \x0c и добавляем в общий список
        sub_blocks = block.split("\x0c")
        for sb in sub_blocks:
            sb = sb.strip()
            if sb:  # Пропускаем пустые строки
                final_paragraphs.append(sb)

    return final_paragraphs


def show_paragraph_lengths(paragraphs):
    """
    Выводит длину каждого абзаца и возвращает максимальную длину.
    """
    max_len = 0
    for i, para in enumerate(paragraphs, start=1):
        length = len(para)
        print(f"Абзац {i}, длина: {length}")
        if length > max_len:
            max_len = length
    return max_len


def merge_heading_paragraphs(paragraphs):
    """
    Ищет абзацы, которые начинаются с:
      - цифр
      - далее 0+ групп (.цифры)
      - опциональная точка
      - пробел(ы)
    и объединяет их с последующим абзацем в один.
    """
    # Пример: "1. Introduction" или "1.2 Something" или "3.4.5. Conclusion"
    # Объяснение:
    #  ^                 # начало строки
    #  [0-9]+           # одна или более цифр
    #  (?:\.[0-9]+)*    # 0 или более групп из: точка + цифры
    #  (?:\.)?          # опциональная точка (например, 1.2.3.)
    #  \s+              # пробел(ы)
    heading_pattern = re.compile(r'^[0-9]+(?:\.[0-9]+)*(?:\.)?\s+')

    merged_paragraphs = []
    i = 0
    n = len(paragraphs)

    while i < n:
        current_para = paragraphs[i].strip()

        # Проверяем, что есть следующий абзац
        # и текущий начинается с "цифра.(цифра).(цифра) ... "
        if i < n - 1 and heading_pattern.match(current_para):
            # Склеиваем с следующим абзацем
            next_para = paragraphs[i+1].strip()
            merged = current_para + " " + next_para
            merged_paragraphs.append(merged)
            i += 2  # пропускаем следующий, т.к. уже склеили
        else:
            # Иначе оставляем как есть
            merged_paragraphs.append(current_para)
            i += 1

    return merged_paragraphs



def chunk_paragraphs(paragraphs, max_tokens=512, overlap=0):
    """
    Разбивает список абзацев на чанки (окна) так,
    чтобы суммарное число токенов в чанке не превышало max_tokens.

    Параметры:
      paragraphs: список строк (абзацев), например merged.
      max_tokens: максимальное число 'токенов' (здесь считаем по словам).
      overlap: кол-во слов, которые повторяем в начале следующего чанка
               (скользящее окно). По умолчанию 0 — без перекрытия.

    Возвращает:
      chunks: список чанков, где каждый чанк — это строка (склеенные абзацы).
    """
    chunks = []
    current_chunk = []
    current_token_count = 0

    for para in paragraphs:
        # Подсчитаем кол-во слов (токенов) в абзаце
        para_tokens = word_tokenize(para)
        para_len = len(para_tokens)

        # Если абзац один сам по себе длиннее max_tokens,
        # придётся разбить его внутри
        if para_len > max_tokens:
            # Сразу заливаем всё внутрь, разбивая на подчанки
            start_idx = 0
            while start_idx < para_len:
                end_idx = start_idx + (max_tokens - overlap)
                sub_tokens = para_tokens[start_idx:end_idx]

                if overlap > 0 and end_idx < para_len:
                    # Добавим перекрытие в конец текущего чанка
                    # (sub_tokens + часть следующих)
                    # Но чаще делают overlap при переходе к следующему чанку.
                    pass

                # Формируем строку из суб-токенов
                sub_chunk_text = " ".join(sub_tokens)

                # Если есть уже накопленные абзацы в current_chunk,
                # сначала добавим их, чтобы сохранить логику склейки
                if current_chunk:
                    # Склеим всё в один текст
                    prev_text = " ".join(current_chunk)
                    combined_text = prev_text + " " + sub_chunk_text
                else:
                    combined_text = sub_chunk_text

                # Сохраняем итоговый чанк
                chunks.append(combined_text.strip())

                # Очищаем current_chunk, так как мы «закрыли» его
                current_chunk = []
                current_token_count = 0

                start_idx = end_idx - overlap  # сдвигаемся с учётом overlap

            continue  # переходим к следующему абзацу

        # Проверяем, влезает ли этот абзац в текущий чанк
        if current_token_count + para_len <= max_tokens:
            # Просто добавляем абзац в текущий чанк
            current_chunk.append(para)
            current_token_count += para_len
        else:
            # Текущий абзац не влезает — закрываем предыдущий чанк
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())

            # Начинаем новый чанк с этого абзаца
            current_chunk = [para]
            current_token_count = para_len

    # Если что-то осталось незакрытым в current_chunk — добавим
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    # Если нужен overlap между чанками, можно усложнить логику,
    # повторяя последние overlap слов одного чанка в начале следующего.
    # Здесь показан базовый подход.

    return chunks



def embed_chunks(chunks, model):
    """
    Возвращает список векторов-эмбеддингов для каждого чанка.
    """
    return model.encode(chunks, convert_to_numpy=True)

def search_chunks(query, chunk_embeddings, chunks, model, top_k=3):
    """
    Ищет топ-K самых релевантных чанков к запросу.

    Параметры:
      query: строка (запрос пользователя)
      chunk_embeddings: np.array, эмбеддинги чанков
      chunks: список самих чанков (текст)
      model: модель SentenceTransformer
      top_k: сколько результатов возвращать

    Возвращает список кортежей (текст_чанка, сходство).
    """
    # Получаем эмбеддинг запроса
    query_emb = model.encode([query], convert_to_numpy=True)
    # Считаем косинусную похожесть со всеми чанками
    similarities = cosine_similarity(query_emb, chunk_embeddings)[0]
    # Сортируем индексы по убыванию сходства
    top_indices = similarities.argsort()[-top_k:][::-1]
    # Собираем результаты
    results = [(chunks[idx], similarities[idx]) for idx in top_indices]
    return results

def cosine_similarity(vec1, vec2):
    # Если vec1 имеет форму (1, dim), а vec2 — (n, dim), то транспонируем vec2
    if vec1.ndim == 2:
        dot_product = np.dot(vec1, vec2.T)
        norm1 = np.linalg.norm(vec1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(vec2, axis=1)
        return dot_product / (norm1 * norm2)
    else:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2, axis=1)
        return dot_product / (norm1 * norm2)


def main(query):
    pdf_path = "gigagan_cvpr2023_original1.pdf"
    text = extract_text_pdfminer(pdf_path)
    paragraphs = split_text_custom(text)

    
    merged = merge_heading_paragraphs(paragraphs)
    chunks = chunk_paragraphs(merged, max_tokens=256, overlap=10)

    print(f"\nВсего чанков: {len(chunks)}")

    
    model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    # Получаем эмбеддинги для чанков
    chunk_embeddings = embed_chunks(chunks, model)

    # Выполняем поиск по запросу
    top_results = search_chunks(query, chunk_embeddings, chunks, model, top_k=3)
    print("\n🔎 Запрос:", query)
    print("Топ-3 релевантных чанка:\n")
    for i, (text_chunk, score) in enumerate(top_results, start=1):
        print(f"{i}. Сходство = {score:.4f}")
        print(f"Текст чанка:\n{text_chunk}\n{'-'*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Поиск по PDF документу")
    parser.add_argument('--query', type=str, help="Запрос для поиска по PDF", required=True)
    args = parser.parse_args()
    main(args.query)
