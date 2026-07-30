import streamlit as st
from views.recsys_page import render_recsys_page
from views.timeseries_page import render_timeseries_page

st.set_page_config(page_title="ML Portfolio Showcase", layout="wide")


# -----------------------------------------------------------------------------
# ГОЛОВНА СТОРІНКА / LANDING PAGE
# -----------------------------------------------------------------------------
def render_home_page():
    st.markdown("<br>", unsafe_allow_html=True)

    # Заголовок по центру
    st.markdown(
        "<h1 style='text-align: center;'>🚀 ML Portfolio Showcase</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 1.2rem;'>"
        "Оберіть проєкт для перегляду інтерактивного дашборду"
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Дві колонки по центру екрана для вибору проєктів
    _, col1, col2, _ = st.columns([1, 3, 3, 1], gap="large")

    # --- КАРТКА 1: RECSYS ---
    with col1:
        with st.container(border=True):
            st.markdown("### 📚 RecSys Showcase")
            st.caption("Goodbooks-10k Recommendation Engine")
            st.write(
                "Інтерактивна рекомендаційна система книг на основі датасету Goodbooks-10k. "
                "Поєднує Collaborative Filtering та Matrix Factorization (SVD / Implicit ALS) "
                "для виявлення прихованих паттернів вподобань, прогнозування оцінок та генерації персоналізованих Top-K добірок."
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # Нативне посилання, яке ініціює новий HTTP-запит до /recsys
            st.link_button(
                "Перейти до RecSys ➔",
                url="/recsys",
                use_container_width=True,
                type="primary"
            )

    # --- КАРТКА 2: TIME SERIES ---
    with col2:
        with st.container(border=True):
            st.markdown("### 📈 Time Series Forecasting")
            st.caption("Store Sales Analysis & Prediction")
            st.write(
                "Аналіз та прогнозування часових рядів продажів мережі магазинів із можливістю гнучкої фільтрації. "
                "Побудовано на ML-моделях (XGBoost, Prophet, ARIMA) для виявлення сезонності, трендів та сплесків попиту. "
                "Дозволяє інтерактивно досліджувати динаміку продажів за датами, родинами товарів та конкретними SKU."
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # Нативне посилання, яке ініціює новий HTTP-запит до /times
            st.link_button(
                "Перейти до Time Series ➔",
                url="/ts",
                use_container_width=True,
                type="primary"
            )


# -----------------------------------------------------------------------------
# ОГОЛОШЕННЯ СТОРІНОК ТА РОУТИНГ
# -----------------------------------------------------------------------------
home_page = st.Page(
    render_home_page,
    title="Головне меню",
    icon="🏠",
    url_path="",  # Корневий URL (http://localhost:8501/)
    default=True
)

recsys_page = st.Page(
    render_recsys_page,
    title="RecSys (Goodbooks-10k)",
    icon="📚",
    url_path="recsys"
)

ts_page = st.Page(
    render_timeseries_page,
    title="Store Sales Forecasting",
    icon="📈",
    url_path="ts"
)

# Ховаємо дефолтний бічний список сторінок Streamlit,
# щоб користувач навігувався ТІЛЬКИ через наші кнопки/сторінки:
pg = st.navigation(
    {"Меню": [home_page, recsys_page, ts_page]},
    position="hidden"  # <--- Приховує стандартне радіо-меню Streamlit
)

pg.run()