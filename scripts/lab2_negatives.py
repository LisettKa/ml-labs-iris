from pydantic import ValidationError
from ml_labs.config import DataConfig, ModelConfig
import logging

logging.basicConfig(
    filename="reports/LAB2/config_negative_test.log", level=logging.INFO, filemode="w"
)
logger = logging.getLogger(__name__)


def check_bad_config(name, config_class, kwargs):
    try:
        config_class(**kwargs)
        logger.error(f"[FAIL] {name}: конфиг принят, хотя должен быть отклонен.")
        print(f"[FAIL] {name}: конфиг НЕ отклонен!")
    except ValidationError as e:
        logger.info(f"[OK] {name}: конфиг успешно отклонен.")
        logger.info(f"Ошибка: {e.errors()}\n")
        print(f"[OK] {name}: конфиг отклонен.")


print("Запуск негативных контролей...")
# 1. Отрицательный размер чанка
check_bad_config("Отрицательный chunk_size", DataConfig, {"chunk_size": -10})
# 2. Отрицательный C
check_bad_config("Отрицательный C", ModelConfig, {"C": -1.0})
# 3. Неверный режим
check_bad_config("Неизвестный режим", ModelConfig, {"mode": "invalid_mode"})

print("Лог сохранен в reports/LAB2/config_negative_test.log")
