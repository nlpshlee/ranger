from _init import *

from ranger.modules.common_const import *
from ranger.modules import common_util

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
    if common_util.check_option(txt_option, TXT_OPTION.OFF):
        return text

    if common_util.check_option(txt_option, TXT_OPTION.UPPER):
        text = text.upper()
    elif common_util.check_option(txt_option, TXT_OPTION.LOWER):
        text = text.lower()

    if common_util.check_option(txt_option, TXT_OPTION.RM_SPACE):
        text = rm_space(text)

    return text


def to_hash(text: str, encoding=ENCODING):
    hash_obj = hashlib.sha1((text.encode(encoding)))
    #hash_obj = hashlib.sha256((text.encode(encoding)))
    
    hex_dig = hash_obj.hexdigest()
    return hex_dig


###########################################################################################

""" def test_is_empty():
    print(is_empty(None, True)) # -> True
    print(is_empty("", True)) # -> True
    print(is_empty(" \t\n", True)) # -> True
    print(is_empty("ABC", True)) # -> False
    print(is_empty(" ABC\t", True)) # -> False
    print(is_empty("\tABC\n", True)) # -> False
    print(is_empty(" \t\nABC \t\n", True)) # -> False
    print()

    print(is_empty(None, False)) # -> True
    print(is_empty("", False)) # -> True
    print(is_empty(" \t\n", False)) # -> False
    print(is_empty("ABC", False)) # -> False
    print(is_empty(" ABC\t", False)) # -> False
    print(is_empty("\tABC\n", False)) # -> False
    print(is_empty(" \t\nABC \t\n", False)) # -> False


def test_trim():
    text_list = [None, "   ", " \t\nABC \t\n"]
    trim(text_list, False)

    # -> ['', '', 'ABC"]
    for i in range(len(text_list)):
        print(f"[{i}]\t[{text_list[i]}]")
    print()
    
    text_list = [None, " \t", " \tABC"]
    text_list_rm = trim(text_list, True)

    for i in range(len(text_list)):
        print(f"[{i}]\t[{text_list[i]}]")
    print()

    # -> ['ABC"]
    for i in range(len(text_list_rm)):
        print(f"[{i}]\t[{text_list_rm[i]}]")


def test_hash():
    text = ("[ABC]"*10)
    print(f"text : {text}, to_hash : {to_hash(text)}")


def test_rm_space():
    print(rm_space('   A    B\tC\nD'))
    print(rm_multi_space('   A     \tB\t\t\tC\n\n\tD'))



if __name__ == "__main__":
    # test_is_empty()
    # test_trim()
    # test_hash()
    test_rm_space() """

