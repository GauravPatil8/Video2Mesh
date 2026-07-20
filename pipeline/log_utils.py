import logging
import time
from functools import wraps

log_file = "pipeline.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Started: {func.__name__}")

        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            end = time.perf_counter()
            logger.info(
                f"Finished: {func.__name__} | "
                f"Elapsed: {end - start:.4f}s"
            )

    return wrapper