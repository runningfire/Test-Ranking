# Используем официальный образ Python 3.11
FROM python:3.11

# Обновляем пакеты и устанавливаем tkinter
RUN apt-get update && apt-get install -y python3-tk

# (Опционально) можно установить дополнительные системные библиотеки, если понадобятся для Pillow, FAISS и т.д.
# RUN apt-get install -y libgl1-mesa-glx

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем необходимые файлы внутрь контейнера
COPY requirements.txt /app/requirements.txt
COPY gigagan_cvpr2023_original1.pdf /app/gigagan_cvpr2023_original1.pdf
COPY final_script.py /app/final_script.py

# Устанавливаем зависимости через pip (убедись, что в requirements.txt прописаны необходимые пакеты, например:
# faiss-cpu, pillow, sentence-transformers, pdfminer.six, nltk, faiss, и т.д.)
RUN pip install --no-cache-dir -r /app/requirements.txt

# Указываем команду запуска через ENTRYPOINT, чтобы можно было передавать параметры
ENTRYPOINT ["python", "/app/final_script.py"]

