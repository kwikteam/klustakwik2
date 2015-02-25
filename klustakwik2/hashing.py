from numpy import *
from hashlib import sha1

__all__ = ['hash_array']

def hash_array(data):
    return sha1(data.view(uint8)).hexdigest()
