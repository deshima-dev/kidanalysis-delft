import lmfit
from lmfit import Parameter
from .fit import fit

def cmp(a, b):
    return (a > b) - (a < b)

# lmfit version check
# fromhttp://stackoverflow.com/questions/1714027/version-number-comparison
import re
def mycmp(version1, version2):
    def normalize(v):
        return [int(x) for x in re.sub(r'(\.0+)*$','', v).split(".")]
    return cmp(normalize(version1), normalize(version2))
assert mycmp(lmfit.__version__, '0.9.0') >= 0










