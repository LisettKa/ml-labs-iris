import time
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

from ml_labs.config import Lab2Config

# 1. Инициализация конфига
config = Lab2Config()

# 2. Загрузка данных
df = pd.read_csv(config.data.path)
X = df.drop(columns=[config.target_col, "species"])
y = df[config.target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=config.data.random_state, stratify=y
)


# 3. Функции для оценки
def get_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


# MLflow настройка
mlflow.set_experiment("lab2_iris_ml")


# РЕЖИМ 1: FULL DATASET

print("--- Обучение на полном датасете ---")
start_time = time.time()

with mlflow.start_run(run_name="full_dataset"):
    mlflow.log_params(config.model.model_dump())

    # Предобработка на полном датасете
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение
    clf = SGDClassifier(
        loss="log_loss",
        max_iter=config.model.max_iter,
        random_state=config.data.random_state,
    )
    clf.fit(X_train_scaled, y_train)

    # Оценка
    y_pred = clf.predict(X_test_scaled)
    full_metrics = get_metrics(y_test, y_pred)

    full_time = time.time() - start_time
    full_metrics["time_sec"] = full_time

    print(f"Метрики: {full_metrics}")
    mlflow.log_metrics(full_metrics)
    mlflow.sklearn.log_model(clf, "model")

    # Сохраняем модель локально
    joblib.dump(clf, "reports/LAB2/ml_model_full.joblib")


# РЕЖИМ 2: CHUNKS (STREAMING)

print("\n--- Обучение на чанках ---")
start_time = time.time()

with mlflow.start_run(run_name="chunks_dataset"):
    mlflow.log_params(config.model.model_dump())

    scaler = StandardScaler()
    chunk_size = config.data.chunk_size
    clf = SGDClassifier(
        loss="log_loss", max_iter=1, random_state=config.data.random_state
    )

    is_first_chunk = True
    for i in range(0, len(X_train), chunk_size):
        X_chunk = X_train.iloc[i : i + chunk_size]
        y_chunk = y_train.iloc[i : i + chunk_size]

        if is_first_chunk:
            scaler.fit(X_chunk)
            X_chunk_scaled = scaler.transform(X_chunk)
            is_first_chunk = False
        else:
            X_chunk_scaled = scaler.transform(X_chunk)

        if i == 0:
            clf.partial_fit(X_chunk_scaled, y_chunk, classes=np.unique(y))
        else:
            clf.partial_fit(X_chunk_scaled, y_chunk)

    # Оценка
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)
    chunks_metrics = get_metrics(y_test, y_pred)

    chunks_time = time.time() - start_time
    chunks_metrics["time_sec"] = chunks_time

    print(f"Метрики: {chunks_metrics}")
    mlflow.log_metrics(chunks_metrics)

    joblib.dump(clf, "reports/LAB2/ml_model_chunks.joblib")

# Сохраняем итоговую таблицу метрик
results = pd.DataFrame(
    [
        {"mode": "full", "accuracy": full_metrics["accuracy"], "time_sec": full_time},
        {
            "mode": "chunks",
            "accuracy": chunks_metrics["accuracy"],
            "time_sec": chunks_time,
        },
    ]
)
print("\nИтоговое сравнение:")
print(results)

# Сохраняем таблицу в CSV для отчета
results.to_csv("reports/LAB2/ml_metrics.csv", index=False)
print("\nСохранено в reports/LAB2/ml_metrics.csv")
