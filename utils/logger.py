import datetime
import logging
import os

from config.env_config import EnvironmentConfig


class LOG:
    """
    A convenience class that handles logging to a file.
    """

    __instance = None

    def __init__(self):
        if self.__instance is not None:
            raise Exception('err')
        logging_level = logging.DEBUG if EnvironmentConfig.get_config('logging_debug') else logging.INFO
        self.setup_logger(level=logging_level)
        LOG.__instance = self

    def log_info(self, msg):
        self.LOG.info(msg)

    def log_debug(self, msg):
        self.LOG.debug(msg)

    def get_logger(self):
        return self.LOG

    def setup_logger(self, name=__file__, level=logging.INFO):
        print(f'Logging level :: {level}')
        logger = logging.getLogger(name)

        if getattr(logger, '_init_done__', None):
            logger.setLevel(level)
            self.LOG = logger
            return

        logger._init_done__ = True
        logger.propagate = False
        logger.setLevel(level)

        formatter = logging.Formatter("%(asctime)s:%(levelname)s::[%(module)s:%(lineno)d]::%(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(level)

        now = datetime.datetime.now()
        date_format = '%d_%m_%Y_%H_%M_%S'
        format_date = now.strftime(date_format)

        print(f'Creating logger {format_date}')

        if not os.path.exists('./output'):
            os.makedirs('./output')
        fh = logging.FileHandler(f'./output/log-{format_date}.log')
        fh.setFormatter(formatter)
        fh.setLevel(level)

        del logger.handlers[:]
        logger.addHandler(handler)
        logger.addHandler(fh)

        self.LOG = logger
        return

    @staticmethod
    def get_instance():
        if LOG.__instance is None:
            LOG()
            return LOG.__instance
        return LOG.__instance

    @staticmethod
    def info(msg):
        LOG.get_instance().log_info(msg)

    @staticmethod
    def debug(msg):
        LOG.get_instance().log_debug(msg)
