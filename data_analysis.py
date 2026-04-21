import pandas as pd
import numpy as np
from datetime import datetime
import logging
from mlxtend.frequent_patterns import apriori, association_rules

class DataQualityEvaluator:
    """
    Метрики качества данных:
    - completeness: пропуски
    - validity: логические проверки
    - timeliness: временные интервалы
    - uniqueness: дубликаты
    - accuracy: ограничения
    """
    @staticmethod
    def completeness(df):
        df_bin = df.isna()

        cols_nan_ratio = df_bin.sum(axis=0) / df.shape[0]
        rows_nan_ratio = df_bin.sum(axis=1) / df.shape[1]

        return {
            "full": float(df_bin.sum().sum() / np.prod(df.shape)),
            "cols_max": float(cols_nan_ratio.max()),
            "rows_max": float(rows_nan_ratio.max())
        }

    @staticmethod
    def validity(df):
        checks = {}

        for col in ["PREMIUM", "INSURED_VALUE", "CLAIM_PAID", "SEATS_NUM"]:
            if col in df.columns:
                checks[col] = bool(not ((df[col] < 0).any()))

        return checks

    @staticmethod
    def timeliness(df):
        if "INSR_BEGIN" not in df.columns:
            return {"delta_time_max": None}

        dates = pd.to_datetime(df["INSR_BEGIN"], format='%d-%b-%y', errors="coerce")
        dates = dates.dropna().sort_values().unique()

        if len(dates) < 2:
            return {"delta_time_max": 0}

        max_time_lag = np.diff(dates).max()
        return {"delta_time_max": int(max_time_lag / np.timedelta64(1, 'D'))}

    @staticmethod
    def uniqueness(df):
        return {
            "duplicates": int(df.duplicated().sum())
        }

    @staticmethod
    def accuracy(df):
        checks = {}

        try:
            begin = pd.to_datetime(df["INSR_BEGIN"], format='%d-%b-%y', errors="coerce")
            end = pd.to_datetime(df["INSR_END"], format='%d-%b-%y', errors="coerce")
            checks["date_order"] = float((end >= begin).mean())
        except:
            checks["date_order"] = None

        current_year = datetime.now().year

        if "PROD_YEAR" in df.columns:
            checks["prod_year_valid"] = float(
                df["PROD_YEAR"].between(1900, current_year).mean()
            )

        checks["premium_positive"] = float((df["PREMIUM"] >= 0).mean())
        checks["insured_value_positive"] = float((df["INSURED_VALUE"] >= 0).mean())

        return checks


def evaluate_data_quality(df):
    results = {}
    methods = {
        name: func for name, func in vars(DataQualityEvaluator).items()
        if callable(func)
    }

    for name, func in methods.items():
        res = func(df)
        results.update({f"{name}_{k}": v for k, v in res.items()})

    logging.info("Data quality оценка завершена")
    return results


def clean_data(df):
    """удаление пропусков в данных"""
    df = df.copy()
    # удаление дублей
    df = df.drop_duplicates()
    # фильтр аномалий
    if 'SEATS_NUM' in df.columns:
        df = df[df['SEATS_NUM'] <= 100]
    # числовые признаки
    numeric_cols = [
        'PREMIUM', 'INSURED_VALUE', 'SEATS_NUM',
        'CARRYING_CAPACITY', 'CCM_TON', 'PROD_YEAR'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    df['CLAIM_PAID'] = pd.to_numeric(df['CLAIM_PAID'], errors='coerce').fillna(0)

    logging.info("Очистка данных завершена")
    return df

# Межквартильный размах (IQR)
def calculate_outliers(df, columns):
    report = {}

    for col in columns:
        Q1 = np.quantile(df[col], 0.25)
        Q3 = np.quantile(df[col], 0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        report[col] = {
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "outliers_count": int(len(outliers)),
            "outliers_ratio": float(len(outliers) / len(df))
        }

    return report

# Feature Engineering
def feature_engineering(df):
    """извлечение временных признаков"""
    df = df.copy()

    df['INSR_BEGIN'] = pd.to_datetime(df['INSR_BEGIN'], format='%d-%b-%y', errors="coerce")
    df['INSR_END'] = pd.to_datetime(df['INSR_END'], format='%d-%b-%y', errors="coerce")

    df['INSR_DURATION_DAYS'] = (df['INSR_END'] - df['INSR_BEGIN']).dt.days
    df['INSR_BEGIN_YEAR'] = df['INSR_BEGIN'].dt.year
    df['INSR_BEGIN_MONTH'] = df['INSR_BEGIN'].dt.month

    df = df.drop(columns=[col for col in ['INSR_BEGIN', 'INSR_END', 'OBJECT_ID'] if col in df.columns])

    logging.info("Feature engineering завершен")
    return df

# Correlation
def calculate_correlations(df):
    numeric_cols = [
        'PREMIUM', 'INSURED_VALUE', 'SEATS_NUM',
        'CARRYING_CAPACITY', 'CCM_TON', 'CLAIM_PAID', 'PROD_YEAR'
    ]

    corr = df[numeric_cols].corr()
    logging.info("Корреляции рассчитаны")
    return corr

def run_apriori(df):
    df_bin = df.copy()
    df_bin['HIGH_PREMIUM'] = (df_bin['PREMIUM'] > df_bin['PREMIUM'].median()).astype(bool)
    df_bin['HIGH_CLAIM'] = (df_bin['CLAIM_PAID'] > 0).astype(bool)
    df_bin['NEW_CAR'] = (df_bin['PROD_YEAR'] > 2015).astype(bool)

    cols = ['HIGH_PREMIUM', 'HIGH_CLAIM', 'NEW_CAR']
    df_bin = df_bin[cols]

    freq = apriori(df_bin, min_support=0.1, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.5)

    return rules.head(5).to_dict(orient="records")

def run_analysis(df):
    logging.info("Старт анализа данных")

    df_clean = clean_data(df) 
    quality = evaluate_data_quality(df)

    outliers = calculate_outliers(
        df_clean,
        ['PREMIUM', 'INSURED_VALUE', 'SEATS_NUM',
         'CARRYING_CAPACITY', 'CCM_TON', 'PROD_YEAR']
    )

    df_features = feature_engineering(df_clean)
    rules = run_apriori(df_clean)
    corr = calculate_correlations(df_features)
            
    logging.info("Анализ завершен")

    return {
        "quality": quality,
        "outliers": outliers,
        "correlation": corr.to_dict(),
        "rules": rules
    }