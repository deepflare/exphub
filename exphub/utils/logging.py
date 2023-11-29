import sys

from loguru import logger as logging


class StdoutToLoguru:
    """Redirects stdout to loguru logger.

    Example:
        >>> with StdoutToLoguru():
        >>>     print("Hello debug loguru!")
        >>> with StdoutToLoguru(LoguruLevel.INFO):
        >>>     print("Hello info loguru!")
        >>> print("Hello stdout!")
        2023-11-29 15:55:55.365 | DEBUG    | __main__:write:32 - Hello debug loguru!
        2023-11-29 15:55:56.133 | INFO     | __main__:write:32 - Hello info loguru!
        Hello stdout!
    """
    def __init__(self, level: str = "DEBUG"):
        self.stdout = None
        self.level = level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            logging.log(self.level, line.rstrip())

    def __enter__(self):
        self.stdout = sys.stdout    # type: ignore[assignment]
        sys.stdout = self    # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout    # type: ignore[assignment]
