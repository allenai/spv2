from stringmatch._stringmatch import lib
from stringmatch._stringmatch import ffi
import numpy as np

def match(key: str, text: str):
    '''
    Find the location of the substring in text with the
    minimum edit distance (Levenshtein) to key.
    '''
    return lib.match(key, text)

def capitalization_features(token: str):
    return np.frombuffer(
        ffi.buffer(
            ffi.cast(
                "float[7]",
                lib.capitalization_features(token))),
        dtype=np.float32)
