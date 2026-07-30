import pandas as pd
import numpy as np
import streamlit as st


# ------------------------------------------------------------------------------
# ЗАВАНТАЖЕННЯ ТА КЕШУВАННЯ ДАНИХ
# ------------------------------------------------------------------------------
@st.cache_data
def load_timeseries_data():
    """Завантаження або генерація даних для часових рядів."""
    try:
        df = pd.read_csv('data/train.csv.zip')
    except Exception:
        try:
            df = pd.read_csv('data/train.csv')
        except Exception:
            # Генерація тестового датасету
            dates = pd.date_range(start='2013-01-01', end='2017-12-31', freq='D')
            stores = list(range(1, 11))
            items = list(range(1, 51))

            grid = pd.MultiIndex.from_product([dates, stores, items],
                                              names=['date', 'store', 'item']).to_frame().reset_index(drop=True)
            np.random.seed(42)
            grid['sales'] = np.random.poisson(lam=30, size=len(grid)) + np.random.randint(5, 15, size=len(grid))
            df = grid

    df['date'] = pd.to_datetime(df['date'])
    return df
