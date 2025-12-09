from _init import *

from ranger.utils.common_const import *
from ranger.utils import common_utils

import hashlib, re


###########################################################################################

def is_empty(text: str, trim_flag=True):
    if text is None:
        return True
    
    if trim_flag:
        text = text.strip()
    
    if len(text) == 0:
        return True
    
    return False


###########################################################################################

def trim(text_list: list, rm_empty_flag: bool):
    if not rm_empty_flag:
        for i in range(len(text_list)):
            if text_list[i] is None:
                text_list[i] = ""
            else:
                text_list[i] = str(text_list[i]).strip()
    else:
        result = []

        for i in range(len(text_list)):
            if not is_empty(text_list[i], True):
                result.append(str(text_list[i]).strip())
        
        return result


def rm_space(text: str):
    return re.sub(r'[ \t\n]+', '', text)


def rm_multi_space(text: str):
    text = re.sub(r'[ ]+', ' ', text)
    text = re.sub(r'[\t]+', '\t', text)
    text = re.sub(r'[\n]+', '\n', text)

    return text


###########################################################################################

def refine_txt(text: str, txt_option=TXT_OPTION.OFF):
    if common_utils.check_option(txt_option, TXT_OPTION.OFF):
        return text

    if common_utils.check_option(txt_option, TXT_OPTION.UPPER):
        text = text.upper()
    elif common_utils.check_option(txt_option, TXT_OPTION.LOWER):
        text = text.lower()

    if common_utils.check_option(txt_option, TXT_OPTION.RM_SPACE):
        text = rm_space(text)

    return text


def to_hash(text: str, encoding=ENCODING):
    hash_obj = hashlib.sha1((text.encode(encoding)))
    #hash_obj = hashlib.sha256((text.encode(encoding)))
    
    hex_dig = hash_obj.hexdigest()
    return hex_dig

