import nose
import os

__all__ = ['run']

def run():
    dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    nose.run(argv=['', dirname,
                   '-c=',  # no config file loading
                   '-I', '^\.',
                   '-I', '^_',
                   '--nologcapture',
#                    '--exe',
                   ])
        