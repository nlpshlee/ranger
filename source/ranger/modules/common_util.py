from _init import *

from ranger.modules.common_const import *

from datetime import datetime
import time

###########################################################################################

def logging(msg: str, option=LOG_OPTION.STDOUT):
    if check_option(option, LOG_OPTION.STDOUT):
        print(msg)
    if check_option(option, LOG_OPTION.STDERR):
        print(msg, file=sys.stderr)

def logging_error(call_path: str, e: Exception):
    logging(f"### (ERROR) {call_path} error : {e}\n", LOG_OPTION.STDERR)

def check_option(option1: int, option2: int):
    if option1 == option2:
        return True
    elif (option1 & option2) != 0:
        return True
    else:
        return False

def rm_option(option1: int, option2: int):
    return (option1 | option2) ^ option2

def get_time_ms():
    return int(round(time.time_ns() / 1000000))

def get_datetime_now(expression: str="%Y-%m-%d %H:%M:%S"):
    date_now = datetime.now()
    return date_now.strftime(expression)

