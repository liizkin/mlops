import pandas as pd
import time
import sqlite3
import logging
import json
import os

import logging
logger = logging.getLogger(__name__)

BATCH_SIZE = 5120
DB_PATH = "data/motor_data.db"
LOG_PATH = "logs/motor_data.log"
META_PATH = "reports/metadata.json"


os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

def create_batches_stream(df, batch_size, interval=0.1):
    """создание батчей"""
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        time.sleep(interval)
        yield batch

def calculate_metadata(df, save_path=META_PATH):
    """счет базовых метаданных"""
    try:
        meta = {
            "rows": len(df),
            "columns": df.shape[1],
            "column_names": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().astype(int).to_dict(),
            "unique_values": df.nunique().astype(int).to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_stats": df.describe().to_dict()
        }

        with open(save_path, "w") as f:
            json.dump(meta, f, indent=4)

        logger.info("Метаданные успешно сохранены")
        return meta

    except Exception as e:
        logger.error(f"Ошибка при расчете метаданных: {e}")
        raise


def ingest_data(file_path):
    conn = None
    try:
        # Загрузка данных
        df = pd.read_csv(file_path)
        logger.info(f"Файл загружен: {file_path}, shape={df.shape}")

        if df.empty:
            raise ValueError("Файл пустой")

        calculate_metadata(df)
        conn = sqlite3.connect(DB_PATH)
        logger.info("Подключение к БД установлено")

        df.head(0).to_sql('data', conn, if_exists='replace', index=False)

        for i, batch in enumerate(create_batches_stream(df, BATCH_SIZE, interval=0)):
            batch.to_sql('data', conn, if_exists='append', index=False)
            logger.info(f'Батч #{i} загружен, размер={len(batch)}')

        conn.commit()
        logger.info("Данные успешно загружены в БД")

        return df

    except Exception as e:
        logger.error(f'Ошибка в ingest_data: {e}')
        raise

    finally:
        if conn:
            conn.close()
            logger.info("Соединение с БД закрыто")