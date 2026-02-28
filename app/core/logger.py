import logging
import sys
from loguru import logger
from app.config import LOG_LEVEL, LOG_FILE_PATH

# intercepting standard logging messages is crucial because many libraries (like uvicorn, fastapi, etc.)
# use the standard 'logging' module. We need 'import logging' to access the standard library's
# LogRecord and Handler classes to redirect those logs to loguru.

class InterceptHandler(logging.Handler):
    """
    Default handler from examples in loguru documentation.
    This handler intercepts all logging messages from the standard logging module
    and redirects them to loguru.
    """
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def configure_logging():
    """
    Configures loguru to replace standard logging and handle all logs.
    """
    # Remove default logger to avoid duplication
    logger.remove()

    # 1. Add a sink to stderr with a nice format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=LOG_LEVEL,
    )

    # 2. Add a sink to a rotating file
    if LOG_FILE_PATH:
        logger.add(
            LOG_FILE_PATH,
            rotation="10 MB",      # Rotate after file size exceeds 10 MB
            retention="10 days",   # Keep logs for 10 days
            compression="zip",     # Compress old logs
            level=LOG_LEVEL,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )

    # 3. Configure standard logging to use InterceptHandler
    # intercept everything at the root logger level
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # 4. Silence/Redirect specific libraries that might double-log or be too verbose
    # We want uvicorn to go through loguru
    for log_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
        logging_logger = logging.getLogger(log_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False
