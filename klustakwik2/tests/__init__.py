import nose
import os
import klustakwik2

__all__ = ['run']

def run():
    klustakwik2.console_log_level('critical')
    dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return nose.run(argv=['', dirname,
                          '-c=',  # no config file loading
                          '-I', '^\.',
                          '-I', '^_',
                          '--logging-clear-handlers',
                          '--nologcapture',
                          ])
