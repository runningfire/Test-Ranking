import os
import re
import argparse
import nltk
import numpy as np
from PIL import Image, ImageTk

import faiss
import torch

from nltk.tokenize import word_tokenize
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, util
import tkinter as tk

# ==============
# Загрузка ресурсов NLTK
# ==============
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# =====================
# Константы, пути
# =====================
PDF_PATH = "gigagan_cvpr2023_original1.pdf"
IMAGES_DIR = "/home/alex/Projects/Test Ranking/Test-Ranking/images"
MAX_TOKENS = 256   # ограничение на размер чанка
OVERLAP = 10       # перекрытие чанков по словам
TOP_K_TEXT = 3     # сколько текстовых чанков ищем
TOP_K_IMAGES = 3   # сколько изображений для каждого чанка

# =====================
# Функции для PDF
# =====================
def extract_text_pdfminer(pdf_path):
    """Извлекает текст из PDF с помощью pdfminer.six"""
    return extract_text(pdf_path)

def split_text_custom(text):
    """
    Делим текст по \n\n, внутри каждого блока убираем \n,
    затем дополнительно делим по \x0c и возвращаем список чистых абзацев.
    """
    raw_blocks = text.split("\n\n")
    final_paragraphs = []

    for block in raw_blocks:
        block = block.replace("\n", " ")
        sub_blocks = block.split("\x0c")
        for sb in sub_blocks:
            sb = sb.strip()
            if sb:
                final_paragraphs.append(sb)
    return final_paragraphs

def merge_heading_paragraphs(paragraphs):
    """
    Ищет абзацы, которые начинаются с цифр и объединяет их с последующим абзацем.
    """
    heading_pattern = re.compile(r'^[0-9]+(?:\.[0-9]+)*(?:\.)?\s+')
    merged_paragraphs = []
    i = 0
    n = len(paragraphs)

    while i < n:
        current_para = paragraphs[i].strip()
        if i < n - 1 and heading_pattern.match(current_para):
            next_para = paragraphs[i+1].strip()
            merged = current_para + " " + next_para
            merged_paragraphs.append(merged)
            i += 2
        else:
            merged_paragraphs.append(current_para)
            i += 1

    return merged_paragraphs

def chunk_paragraphs(paragraphs, max_tokens=512, overlap=0):
    """
    Разбивает список абзацев на чанки так, чтобы суммарное число токенов не превышало max_tokens.
    overlap - количество слов, которые повторяются в начале следующего чанка.
    """
    chunks = []
    current_chunk = []
    current_token_count = 0

    for para in paragraphs:
        para_tokens = word_tokenize(para)
        para_len = len(para_tokens)

        if para_len > max_tokens:
            start_idx = 0
            while start_idx < para_len:
                end_idx = start_idx + (max_tokens - overlap)
                sub_tokens = para_tokens[start_idx:end_idx]
                sub_chunk_text = " ".join(sub_tokens)
                if current_chunk:
                    prev_text = " ".join(current_chunk)
                    combined_text = prev_text + " " + sub_chunk_text
                else:
                    combined_text = sub_chunk_text
                chunks.append(combined_text.strip())
                current_chunk = []
                current_token_count = 0
                start_idx = end_idx - overlap
            continue

        if current_token_count + para_len <= max_tokens:
            current_chunk.append(para)
            current_token_count += para_len
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())
            current_chunk = [para]
            current_token_count = para_len

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks

def embed_chunks(chunks, model):
    """Вычисляет эмбеддинги для списка текстовых чанков."""
    return model.encode(chunks, convert_to_numpy=True)

def cosine_similarity(vec1, vec2):
    """
    Вычисляет косинусное сходство между vec1 (N x dim) и vec2 (M x dim).
    Возвращает матрицу сходства размера N x M.
    """
    if vec1.ndim == 1:
        vec1 = vec1.reshape(1, -1)
    if vec2.ndim == 1:
        vec2 = vec2.reshape(1, -1)
    dot_product = np.dot(vec1, vec2.T)
    norm1 = np.linalg.norm(vec1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(vec2, axis=1, keepdims=True)
    return dot_product / (norm1 * norm2.T)

def search_chunks(query, chunk_embeddings, chunks, model, top_k=3):
    """
    Ищет топ-K наиболее релевантных текстовых чанков для заданного запроса.
    Возвращает список кортежей (текст_чанка, сходство).
    """
    query_emb = model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, chunk_embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    return [(chunks[idx], sims[idx]) for idx in top_indices]

# =====================
# Функции для изображений и FAISS
# =====================
def get_all_image_paths(root_dir):
    """
    Рекурсивно собирает пути ко всем изображениям из заданной директории.
    """
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
    image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(valid_exts):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
    return image_paths

def build_image_faiss_index(image_paths, clip_model):
    """
    Вычисляет эмбеддинги для изображений, строит FAISS-индекс и возвращает (index, id_to_path).
    """
    embeddings = []
    valid_paths = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            emb = clip_model.encode(img, convert_to_tensor=True)
            embeddings.append(emb)
            valid_paths.append(path)
        except Exception as e:
            print(f"Error with image {path}: {e}")

    if not embeddings:
        return None, {}

    embeddings = torch.stack(embeddings, dim=0)
    embeddings_np = embeddings.cpu().numpy().astype('float32')

    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    # Опционально можно нормализовать: faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)
    id_to_path = {i: p for i, p in enumerate(valid_paths)}
    return index, id_to_path

def encode_text_clip(text, clip_model):
    """
    Получает эмбеддинг текста через CLIP-модель.
    """
    emb = clip_model.encode([text], convert_to_tensor=True)
    return emb.cpu().numpy().astype('float32')

def search_images_for_chunk_faiss(chunk_text, clip_model, faiss_index, id_to_path, top_k=3):
    """
    Для заданного текстового чанка ищет топ-K похожих изображений через FAISS.
    """
    text_emb_np = encode_text_clip(chunk_text, clip_model)
    D, I = faiss_index.search(text_emb_np, top_k)
    D = D[0]
    I = I[0]
    results = []
    for rank, idx in enumerate(I):
        score = D[rank]
        img_path = id_to_path[idx]
        results.append((img_path, score))
    return results

def show_image_popup(image_path, window_title="Image"):
    """
    Открывает всплывающее окно с изображением и заголовком window_title.
    """
    popup = tk.Toplevel()
    popup.title(window_title)
    img = Image.open(image_path).convert('RGB')
    tk_img = ImageTk.PhotoImage(img)
    label = tk.Label(popup, image=tk_img)
    label.image = tk_img  # чтобы избежать сборки мусора
    label.pack()

# =====================
# Основная логика
# =====================
def main():
    parser = argparse.ArgumentParser(description="Поиск по PDF документу + (опционально) поиск изображений через FAISS")
    parser.add_argument('--query', type=str, required=True, help="Запрос для поиска по PDF")
    parser.add_argument('--with_images', action='store_true',
                        help="Если указать, дополнительно выполнить поиск похожих изображений через FAISS и открыть их.")
    args = parser.parse_args()

    # 1) Извлекаем текст из PDF
    text = extract_text_pdfminer(PDF_PATH)
    paragraphs = split_text_custom(text)
    merged = merge_heading_paragraphs(paragraphs)
    chunks = chunk_paragraphs(merged, max_tokens=MAX_TOKENS, overlap=OVERLAP)
    print(f"\nВсего чанков: {len(chunks)}")

    # 2) Текстовый поиск по PDF через mpnet
    text_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    chunk_embeddings = embed_chunks(chunks, text_model)
    top_text_results = search_chunks(args.query, chunk_embeddings, chunks, text_model, top_k=TOP_K_TEXT)

    print(f"\n🔎 Запрос: {args.query}")
    print(f"Топ-{TOP_K_TEXT} релевантных текстовых чанков:\n")
    for i, (text_chunk, score) in enumerate(top_text_results, start=1):
        print(f"{i}. Сходство = {score:.4f}")
        print(f"Текст чанка:\n{text_chunk}\n{'-'*50}")

    # 3) Если указан флаг --with_images, выполняем поиск изображений
    if args.with_images:
        print("\n--with_images: выполняется поиск изображений через FAISS --")
        # Загружаем CLIP-модель
        clip_model = SentenceTransformer("jinaai/jina-clip-v1", trust_remote_code=True)
        # Собираем все пути к изображениям (без random.sample)
        image_paths = get_all_image_paths(IMAGES_DIR)
        print(f"Найдено {len(image_paths)} изображений в {IMAGES_DIR}")
        faiss_index, id_to_path = build_image_faiss_index(image_paths, clip_model)
        if faiss_index is None:
            print("Не удалось построить FAISS-индекс для изображений.")
            return

        # Создаём главное окно TK для всплывающих окон
        root = tk.Tk()
        root.withdraw()

        # Для каждого найденного текстового чанка ищем топ-K изображений
        for text_idx, (text_chunk, text_score) in enumerate(top_text_results, start=1):
            print(f"\nЧанк №{text_idx}, сходство с запросом = {text_score:.4f}")
            print(f"Текст:\n{text_chunk}\n")
            img_results = search_images_for_chunk_faiss(text_chunk, clip_model, faiss_index, id_to_path, top_k=TOP_K_IMAGES)
            print(f"Топ-{TOP_K_IMAGES} изображений для этого чанка (через FAISS):")
            for img_idx, (img_path, img_score) in enumerate(img_results, start=1):
                print(f"{{text_index: {text_idx}, img_index: {img_idx}}}: Сходство = {img_score:.4f} -> {img_path}")
                # Открываем всплывающее окно с изображением
                popup_title = f"text_index: {text_idx}, img_index: {img_idx}"
                show_image_popup(img_path, popup_title)
            print("-"*50)

        root.mainloop()  # запускаем цикл TK

if __name__ == "__main__":
    main()
