import logging
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """
    Formats log records as a JSON string for production-ready monitoring.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_object = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_object['exc_info'] = self.formatException(record.exc_info)

        return json.dumps(log_object)