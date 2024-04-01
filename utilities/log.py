import logging

logging.getLogger('matplotlib').setLevel(logging.INFO)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

FORMAT = '%(asctime)-15s | %(levelname)-8s | %(lineno)04d | %(filename)-8s | %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")