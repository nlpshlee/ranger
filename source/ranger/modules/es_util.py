from _init import *

from ranger.modules.common_const import *
from ranger.modules import string_util

from datetime import datetime

###########################################################################################

ENCODING = "UTF-8"

def get_primary_key(inputs: list, delim='\t'):
    return string_util.to_hash(delim.join(inputs), ENCODING)

def get_datetime_now():
    date_now = datetime.now()
    return date_now.strftime("%Y-%m-%d %H:%M:%S")
