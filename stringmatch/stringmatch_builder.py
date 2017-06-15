import cffi
import os

ffibuilder = cffi.FFI()
cur_dir = os.path.dirname(os.path.abspath(__file__))
with open(cur_dir + '/stringmatch.cpp') as f:
    code = f.read()
ffibuilder.set_source(
    '_stringmatch', code,
    source_extension='.cpp',
)

ffibuilder.cdef('''
typedef struct {
    int start_pos;
    int end_pos;
    int cost;
} MatchResult;
MatchResult match(const wchar_t* a, const wchar_t* b);
''')

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
