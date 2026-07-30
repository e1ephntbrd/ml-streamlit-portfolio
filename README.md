# 📊 Multi-Topic ML Portfolio: RecSys & Time Series Analysis (Streamlit App)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://ml-portfolio-e1ephntbrd.streamlit.app/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Інтерактивний веб-додаток на **Streamlit**, що об'єднує рішення двох ключових напрямків машинного навчання: **Рекомендаційні системи (RecSys)** та **Аналіз часових рядів (Time Series Analysis)**. Проєкт підтримує повний цикл від дослідницького аналізу даних (EDA) до порівняння результатів інференсу між різними модельними архітектурами.

---

## 🌟 Основні модулі проєкту (Key Modules)

### 📚 1. Book Recommendation System (RecSys)
* **Exploratory Data Analysis (EDA):** Аналіз розподілу оцінок (*Positivity Bias*), розрідженості даних (*Long Tail Analysis*), а також co-occurrence матриці жанрів.
* **Vector Space Model (VSM):** Базовий підхід на основі косинусної подібності канонічних жанрів.
* **Two-Tower Architecture (User & Item Towers):** Двовежева нейромережева модель для відображення користувачів та книг у спільному $D$-вимірному просторі ембеддінгів.
* **Neural Collaborative Filtering (NCF):** Нелінійний підхід до колаборативної фільтрації.
* **t-SNE Visualizer:** 2D-проєкція векторизованого простору книг для аналізу якісної кластеризації за жанрами.

### 📈 2. Time Series Analysis & Forecasting
* **Time Series EDA:** Аналіз трендів, сезонності, автокореляції (ACF / PACF) та стаціонарності часових рядів.
* **Feature Engineering:** Генерація лагових фіч (lags), ковзних середніх (rolling metrics) та часових ембеддінгів.
* **Model Suite:** Порівняння традиційних статистичних/градієнтних моделей та нейромережевих підходів для прогнозування та класифікації часових рядів.
* **Interactive Inference:** Візуалізація прогнозних інтервалів та метрик якості (RMSE, MAE, MAPE).

---

## 🏗️ Структура проєкту (Project Structure)

```text
.
├── data/                    # Датасети (books.csv, time_series_data.csv, etc.)
├── models/                  # Натреновані ваги моделей (RecSys & Time Series)
├── pages/
│   ├── recsys_page.py       # Модуль сторінки RecSys & EDA
│   └── time_series_page.py  # Модуль сторінки Time Series
├── app.py                   # Головна точка входу Streamlit
├── Dockerfile               # Конфігурація Docker-образу
├── docker-compose.yml       # Конфігурація контейнерів
├── Makefile                 # Команди швидкого управління проєктом
├── requirements.txt         # Залежності Python
└── README.md                # Документація проєкту
```

---

## 🚀 Швидкий старт (Quick Start)

Найпростіший спосіб запустити проєкт — використати **Docker** та **Makefile**.

### Передумови (Prerequisites)
Переконайтеся, що у вас встановлені:
* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/)
* `make` (утиліта CLI)

---

### 🛠️ Інструкція з запуску

1. **Клонуйте репозиторій:**
   ```bash
   git clone https://github.com/your-username/ml-streamlit-portfolio.git
   cd ml-streamlit-portfolio
   ```

2. **Зберіть Docker-образ (Build):**
   ```bash
   make build
   ```

3. **Запустіть контейнери (Up):**
   ```bash
   make up
   ```

4. **Відкрийте додаток у браузері:**
   * 🌧️ **Streamlit App:** http://localhost:8501
   * 📝 **JupyterLab Environment:** http://localhost:8888

---

## 🧰 Команди Makefile

Для зручності розробки в проєкт додано `Makefile`. Ви можете переглянути всі доступні target-команди за допомогою виклику `make help`:

| Команда | Опис |
| :--- | :--- |
| `make build` | Збірка Docker-образу (виконується після клонування проєкту) |
| `make up` | Запуск контейнерів у фоновому режимі (Streamlit + JupyterLab) |
| `make down` | Зупинка та видалення контейнерів |
| `make restart` | Перезапуск всіх сервісів (`down` + `up`) |
| `make logs` | Перегляд логів контейнера в реальному часі |
| `make shell` | Вхід всередину CLI-оболонки головного контейнера (`python_ds_app`) |

---

## 🧪 Локальне розгортання (без Docker)

Якщо ви бажаєте запустити додаток безпосередньо у своєму Python-середовищі:

```bash
# 1. Створіть та активуйте virtualenv
python -m venv venv
source venv/bin/activate  # Для Linux/macOS
# venv\Scripts\activate   # Для Windows

# 2. Встановіть залежності
pip install -r requirements.txt

# 3. Запустіть Streamlit
streamlit run app.py
```

---

## 💡 Ключові висновки (Key Insights)

* **RecSys:** Через ефект *Long Tail* у каталозі, Two-Tower та NCF моделі потребують якісного негативного семплінгу, щоб запобігти домінуванню топових бестселерів.
* **Time Series:** Врахування сезонності та правильне розбиття за часом (Time-based Split) запобігає витоку даних (data leakage) при виборі гіперпараметрів моделей прогнозування.