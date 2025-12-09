import time
from datetime import datetime

from ranger.utils.common_const import *


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


def get_time_ms():
    return int(round(time.time_ns() / 1000000))


def get_datetime_now(expression: str="%Y-%m-%d %H:%M:%S"):
    date_now = datetime.now()
    return date_now.strftime(expression)


def get_elapsed_time_ms(start, end=None):
    if end is None:
        end = get_time_ms()

    elapsed_ms = end - start

    elapsed = elapsed_ms // 1000
    h = elapsed // 3600
    m = (elapsed % 3600) // 60
    s = elapsed % 60

    elapsed_str = f'{h:02d}:{m:02d}:{s:02d}'

    return elapsed_ms, elapsed_str

