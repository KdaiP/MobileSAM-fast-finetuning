import logging
from logging.handlers import RotatingFileHandler

class Logger:
    def __init__(self, log_file='app.log', level=logging.INFO, max_log_size=5 * 1024 * 1024, backup_count=3):
        """
        Initialize the logger.

        :param log_file: Name of the log file.
        :param level: Logging level.
        :param max_log_size: Maximum size of log file before it gets rotated.
        :param backup_count: Number of old log files to keep.
        """
        # Logger basic configuration
        logging.basicConfig(level=level,
                            format="%(asctime)s [%(levelname)s] - %(message)s",
                            handlers=[logging.StreamHandler()])

        # Create a logger
        self.logger = logging.getLogger()
        # The logger will handle all levels from the specified and above
        self.logger.setLevel(level)

        # Create a file handler for logging
        file_handler = RotatingFileHandler(log_file, maxBytes=max_log_size, backupCount=backup_count)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        """
        Return the logger instance.

        :return: logger instance.
        """
        return self.logger
    
if __name__ == '__main__':
    # Usage:
    log = Logger(log_file="training.log").get_logger()
    log.info("This is an info message.")
    log.error("This is an error message.")
