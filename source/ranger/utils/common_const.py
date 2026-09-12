from _init import *

import numpy as np

###########################################################################################

ENCODING = "UTF-8"
DELIM_KEY = "\t"

FILE_EXT = "."

NULL_FLOAT = np.finfo(np.float32).min

###########################################################################################

class TXT_OPTION:
    OFF = 0
    UPPER = 1
    LOWER = 2
    RM_SPACE = 4
    
class LOG_OPTION:
    STDOUT = 1
    STDERR = 2
