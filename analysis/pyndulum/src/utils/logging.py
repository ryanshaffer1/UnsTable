"""Contains variables defining the configuration of logging channels/handlers used in the project.

    Documentation on logging and logging config files:
        https://docs.python.org/3/library/logging.config.html
        https://docs.python.org/3/howto/logging.html
        https://stackoverflow.com/questions/7507825/where-is-a-complete-example-of-logging-config-dictconfig

    Variables:
        LOGGING_CONFIG (dict): json-like, configures logging formatters and handlers
"""  # fmt: skip

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "level": "DEBUG",
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
        "light_formatter": {
            "format": "%(asctime)s: %(levelname)s: %(message)s",
            "datefmt": "%H:%M:%S",
        },
        "heavy_formatter": {
            "format": "%(asctime)s: [%(module)s] %(levelname)s: %(message)s",
        },
    },
    "handlers": {
        "console_handler": {
            "level": "DEBUG",
            "formatter": "light_formatter",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "level": "DEBUG",
            "formatter": "heavy_formatter",
            "class": "logging.FileHandler",
            "filename": "logfile.log",
            "mode": "w",
        },
    },
    "loggers": {
        "log": {  # root logger
            "handlers": ["console_handler", "file_handler"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}
