'''
Logging module

Written as a wrapper around the Python logging module. Making it a wrapper should make it easier
to refactor it into the logging solution of phy later on.
'''

import logging

__all__ = ['logger', 'log_to_file', 'file_log_level', 'console_log_level', 'log_message',
           'log_suppress_hierarchy', 'log_suppress_name', 'log_remove_filters',
           ]

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

def log_message(level, msg, name=None):
    '''
    Adds a log message at the specified log level.
    
    Log levels (strings) can be one of (In descending order of severity): critical, error, warning,
    info, debug.
    '''
    if isinstance(level, str):
        level = LOG_LEVELS[level.upper()]
    if name is None or name=='':
        name = 'klustakwik'
    else:
        name = 'klustakwik.'+name
    logger = logging.getLogger(name)
    logger.log(level, msg)


class HierarchyFilter(object):
    def __init__(self, name='', inclusive=True):
        self.name = name
        self.nlen = len(name)
        self.inclusive = inclusive

    def filter(self, record):
        if self.nlen == 0:
            return 0
        elif self.inclusive and self.name==record.name:
            return 0
        elif not self.inclusive and self.name==record.name:
            return 1
        elif record.name.find(self.name, 0, self.nlen) != 0:
            return 1
        return not (record.name[self.nlen] == ".")


# See brian2.utils.logger for details
class NameFilter(object):
    '''
    A class for suppressing log messages ending with a certain name.
    
    Parameters
    ----------
    name : str
        The name to suppress. See `BrianLogger.suppress_name` for details.
    '''
    
    def __init__(self, name):
        self.name = name
    
    def filter(self, record):
        '''
        Filter out all messages ending with a certain name.
        '''
        # The last part of the name
        record_name = record.name.split('.')[-1]
        return self.name != record_name


def log_suppress_hierarchy(name, inclusive=True, console=True, file=False):
    filter = HierarchyFilter(name, inclusive=inclusive)
    if console:
        CONSOLE_HANDLER.addFilter(filter)
    if file:
        FILE_HANDLER.addFilter(filter)

def log_suppress_name(name, console=True, file=False):
    filter = NameFilter(name)
    if console:
        CONSOLE_HANDLER.addFilter(filter)
    if file:
        FILE_HANDLER.addFilter(filter)

def log_remove_filters(console=True, file=True):
    if console:
        CONSOLE_HANDLER.filters = []
    if file:
        FILE_HANDLER.filters = []
