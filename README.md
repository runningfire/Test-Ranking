# Test Ranking

Этот проект предназначен для обработки текста и поиска релевантных фрагментов в документах PDF.  

## 🔧 Требования

Перед началом работы убедитесь, что у вас установлен **Docker**.  
[Установить Docker](https://docs.docker.com/get-docker/)

## 🚀 Сборка Docker-образа

Откройте терминал в папке с `Dockerfile` и выполните команду (имя вместо test-ranking можно любое):

```bash
docker build -t test-ranking:latest .
```

Эта команда создаст образ с именем `test-ranking:latest`, который затем можно будет использовать для запуска контейнера.

---

## ▶️ Запуск контейнера

После успешной сборки контейнер можно запустить с обязательным аргументом `--query`, который задаёт поисковый запрос:
(имя контейнера по желанию)
```bash
docker run --name test-ranking-container test-ranking:latest --query "GigaGAN discriminator layers"
```

🔹 **Обязательный параметр**:  
- `--query "ваш_запрос"` – строка запроса для поиска в PDF.

🔹 **Дополнительные параметры** (по желанию):  
- `--name test-ranking-container` – имя контейнера для удобного управления.

Контейнер обработает PDF, выполнит поиск и выведет результаты в консоль.

---

## ⏹ Остановка контейнера

Чтобы остановить работающий контейнер, выполните:

```bash
docker stop test-ranking-container
```

---

## 🗑 Удаление контейнера

После остановки контейнер можно удалить:

```bash
docker rm test-ranking-container
```

---

## 📝 Дополнительная информация

Этот проект использует `argparse` для обработки аргументов командной строки. Если `--query` не передан, скрипт не запустится.

Пример запроса:
```bash
docker run --name test-ranking-container test-ranking:latest --query "GigaGAN discriminator layers"
```

![checkasdsa](https://github.com/user-attachments/assets/30893d23-dc25-4ef4-bd2f-e389c06d753e)

