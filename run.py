import argparse
import pandas as pd
import logging
import os
import json
import joblib

from data_collect import ingest_data
from data_analysis import clean_data, feature_engineering
from training import run_training_pipeline
from data_analysis import run_analysis
from training import handle_missing_values

logging.basicConfig(
    level=logging.INFO,
    filename="logs/app.log",
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MODEL_PATH = "models/final_model.pkl"

def run_inference(file_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Модель не найдена")

    df = pd.read_csv(file_path)

    df = clean_data(df)
    df = feature_engineering(df)
    df = handle_missing_values(df)

    if "CLAIM_PAID" in df.columns:
        df = df.drop(columns=["CLAIM_PAID"])

    model = joblib.load(MODEL_PATH)

    preds = model.predict(df)
    df["predict"] = preds

    output_path = "reports/inference_result.csv"
    os.makedirs("reports", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Inference завершен: {output_path}")
    return output_path

def run_update():
    try:
        df = ingest_data("data/motor_data11-14lats.csv")
        analysis = run_analysis(df)
        with open("reports/data_analysis.json", "w") as f:
            json.dump(analysis, f, indent=4)

        df = clean_data(df)
        df = feature_engineering(df)

        run_training_pipeline(df)

        print("Update успешно завершен")
        return True

    except Exception as e:
        print(f"Ошибка update: {e}")
        return False

def run_summary():
    report = {}

    try:
        best_model = None
        best_score = -1

        for model_type in ["RF", "MLP"]:
            log_path = f"models/log_{model_type}.json"

            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    lines = [json.loads(line) for line in f]

                report[model_type] = lines

                for entry in lines:
                    score = entry["metrics"]["roc_auc"]
                    if score > best_score:
                        best_score = score
                        best_model = entry

        report["best_model"] = best_model

        output_path = "reports/summary.json"
        os.makedirs("reports", exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"Summary сохранен: {output_path}")
        return output_path

    except Exception as e:
        print(f"Ошибка summary: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-mode", required=True, choices=["inference", "update", "summary"])
    parser.add_argument("-file", required=False)

    args = parser.parse_args()

    if args.mode == "inference":
        if not args.file:
            raise ValueError("Нужно указать -file для inference")
        run_inference(args.file)

    elif args.mode == "update":
        run_update()

    elif args.mode == "summary":
        run_summary()


if __name__ == "__main__":
    main()