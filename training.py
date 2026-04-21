import pandas as pd
import numpy as np
import os
import json
import joblib
import logging
import glob

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report, roc_auc_score, f1_score
from data_analysis import clean_data, feature_engineering

RANDOM_STATE = 42

def load_all_data(data_dir="data"):
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    files = [f for f in all_files if "full_train.csv" not in f]

    if not files:
        raise ValueError("Нет файлов в папке data")

    df_list = []
    for file in files:
        df_temp = pd.read_csv(file, low_memory=False)
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)
    logging.info(f"Загружено исходных файлов: {len(files)}, строк: {df.shape[0]}")

    return df

# Обработка пропусков
def handle_missing_values(df):
    df = df.copy()
    df["EFFECTIVE_YR"] = df["EFFECTIVE_YR"].fillna("unknown")
    df['CLAIM_PAID'] = df['CLAIM_PAID'].apply(lambda x: 'Yes' if x > 0 else 'No')

    logging.info("Пропуски обработаны")
    return df

# Препроцессор
def build_preprocessor():
    ohe_columns = ['SEX', 'TYPE_VEHICLE', 'MAKE', 'USAGE', 'EFFECTIVE_YR']

    num_columns = [
        'PREMIUM', 'INSURED_VALUE', 'SEATS_NUM',
        'CARRYING_CAPACITY', 'CCM_TON', 'PROD_YEAR',
        'INSR_DURATION_DAYS', 'INSR_BEGIN_YEAR', 'INSR_BEGIN_MONTH'
    ]

    ohe_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        [
            ('ohe', ohe_pipe, ohe_columns),
            ('num', num_pipe, num_columns)
        ],
        remainder='passthrough'
    )

    return preprocessor

# Обучение моделей
def train_models(X_train, y_train, preprocessor):

    pipeline_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(
            n_estimators=50,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])

    pipeline_mlp = Pipeline([
        ('preprocessor', preprocessor),
        ('model', MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=100,
            random_state=RANDOM_STATE
        ))
    ])

    pipeline_rf.fit(X_train, y_train)
    pipeline_mlp.fit(X_train, y_train)

    logging.info("Модели обучены")

    return pipeline_rf, pipeline_mlp

# Оценка моделей
def evaluate_model(pipeline, X_test, y_test, model_name="Model"):

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"\n{model_name}")
    print(classification_report(y_test, y_pred, zero_division=0))

    roc_auc = roc_auc_score(y_test.map({'No': 0, 'Yes': 1}), y_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")

    return {
        "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
        "roc_auc": roc_auc
    }

# Сохранение модели
def save_model_version(pipeline, metrics, model_type):

    os.makedirs("models", exist_ok=True)

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/{model_type}_{version}.pkl"

    joblib.dump(pipeline, model_path)

    metadata = {
        "version": version,
        "metrics": metrics,
        "model_path": model_path
    }

    with open(f"models/log_{model_type}.json", "a") as f:
        f.write(json.dumps(metadata) + "\n")

    print(f"Модель {model_type} сохранена: {model_path}")

# Основной pipeline
def run_training_pipeline(df, iterations):

    logging.info("Старт подготовки данных")

    df = handle_missing_values(df)

    # Загружаем все сырые данные из папки
    full_df = load_all_data("data")
    full_df = clean_data(full_df)
    full_df = feature_engineering(full_df)
    full_df = handle_missing_values(full_df)

    full_df = pd.concat([full_df, df], ignore_index=True)

    # убираем дубликаты
    full_df = full_df.drop_duplicates()

    # сохраняем объединённый датасет
    os.makedirs("data", exist_ok=True)
    full_df.to_csv("data/full_train.csv", index=False)

    df = full_df

    X = df.drop(['CLAIM_PAID'], axis=1)
    y = df['CLAIM_PAID']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    preprocessor = build_preprocessor()

    pipeline_rf, pipeline_mlp = train_models(X_train, y_train, preprocessor)

    rf_metrics = evaluate_model(pipeline_rf, X_test, y_test, "RandomForest")
    mlp_metrics = evaluate_model(pipeline_mlp, X_test, y_test, "MLP")

    save_model_version(pipeline_rf, rf_metrics, "RF")
    save_model_version(pipeline_mlp, mlp_metrics, "MLP")

    final_model = pipeline_rf if rf_metrics['roc_auc'] > mlp_metrics['roc_auc'] else pipeline_mlp
    joblib.dump(final_model, 'models/final_model.pkl')

    logging.info("Pipeline завершен")
    return final_model
