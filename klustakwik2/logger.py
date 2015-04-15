'''
Logging module

Written as a wrapper around the Python logging module. Making it a wrapper should make it easier
to refactor it into the logging solution of phy later on.
'''

import logging

__all__ = ['logger', 'log_to_file', 'file_log_level', 'console_log_level', 'log_message']

logger = logging.getLogger('klustakwik')
logger.propagate = False
logger.setLevel(logging.DEBUG)

#: Translation from string representation to number
LOG_LEVELS = {'CRITICAL': logging.CRITICAL,
              'ERROR': logging.ERROR,
              'WARNING': logging.WARNING,
              'INFO': logging.INFO,
              'DEBUG': logging.DEBUG}

FILE_HANDLER = None

def log_to_file(fname, level=logging.INFO):
    if isinstance(level, str):
        level = LOG_LEVELS[level.upper()]
    FILE_HANDLER = logging.FileHandler(fname, 'wt')
    FILE_HANDLER.setLevel(level)
    FILE_HANDLER.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(name)s: %(message)s'))
    logger.addHandler(FILE_HANDLER)
    
def file_log_level(level):
    if isinstance(level, str):
        level = LOG_LEVELS[level.upper()]
    FILE_HANDLER.setLevel(level)

# create console handler with a higher log level
CONSOLE_HANDLER = logging.StreamHandler()
CONSOLE_HANDLER.setLevel(logging.INFO)
CONSOLE_HANDLER.setFormatter(logging.Formatter('%(levelname)-8s %(name)s: %(message)s'))
logger.addHandler(CONSOLE_HANDLER)

def console_log_level(level):
    if isinstance(level, str):
        level = LOG_LEVELS[level.upper()]
    CONSOLE_HANDLER.setLevel(level)

def log_message(level, msg):
    '''
    Adds a log message at the specified log level.
    
    Log levels (strings) can be one of (In descending order of severity): critical, error, warning,
    info, debug.
    '''
    if isinstance(level, str):
        level = LOG_LEVELS[level.upper()]
    logger.log(level, msg)
