from stringmatch._stringmatch import lib


def match(key: str, text: str):
    '''
    Find the location of the substring in text with the
    minimum edit distance (Levenshtein) to key.
    '''
    return lib.match(key, text)
