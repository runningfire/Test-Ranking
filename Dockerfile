
# Используем официальный образ Python
FROM python:3.11

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем необходимые файлы внутрь контейнера
COPY requirements.txt /app/requirements.txt
COPY gigagan_cvpr2023_original1.pdf /app/gigagan_cvpr2023_original1.pdf
COPY final_script.py /app/final_script.py

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r /app/requirements.txt

# Указываем команду запуска через ENTRYPOINT, чтобы можно было передавать параметры
ENTRYPOINT ["python", "/app/final_script.py"]

