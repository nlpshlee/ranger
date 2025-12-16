from _init import *

from ranger.utils.common_const import *
from ranger.utils import string_utils

###########################################################################################

def add_str_list_int(in_dict: dict, key_list: list, value: int, txt_option=TXT_OPTION.OFF):
    if in_dict != None:
        for i in range(len(key_list)):
            key = str(key_list[i])
            add_str_int(in_dict, key, value, txt_option)


def add_str_int(in_dict: dict, key: str, value: int, txt_option=TXT_OPTION.OFF):
    if in_dict != None:
        if txt_option != TXT_OPTION.OFF:
            key = string_utils.refine_txt(key, txt_option)

        if key in in_dict:
            value_prev = int(in_dict[key])
            in_dict[key] = value_prev + value
        else:
            in_dict[key] = value

###########################################################################################

def get_window(in_list: list, idx: int, window_size: int, delim='', pad='$'):
    result = []
    in_len = len(in_list)
    
    # 앞 부분 저장
    start = idx - window_size
    if start < 0:
        result += [pad] * (start*-1)
        start = 0
    
    if start < idx:
        result.extend(in_list[start:idx])
    
    # 현재 부분 저장
    result.append(in_list[idx])
    
    # 뒷 부분 저장
    end = idx + window_size
    
    if end < in_len:
        result.extend(in_list[idx+1:end+1])
    else:
        if idx+1 < in_len:
            result.extend(in_list[idx+1:in_len])
        
        end = end - in_len + 1
        if end > 0:
            result += [pad] * end
    
    return delim.join(result)


def chunks(arr, n):
    for i in range(0, len(arr), n):
        yield arr[i: i + n]

###########################################################################################

'''
    key를 기준으로 정렬
        - is_reverse = False : 오름 차순
        - is_reverse = True : 내림 차순
'''
def sorted_dict_key(in_dict: dict, is_reverse=False):
    return dict(sorted(in_dict.items(), key=lambda item:item[0], reverse=is_reverse))


'''
    value를 기준으로 정렬
        - is_reverse = False : 오름 차순
        - is_reverse = True : 내림 차순
'''
def sorted_dict_value(in_dict: dict, is_reverse=False):
    return dict(sorted(in_dict.items(), key=lambda item:item[1], reverse=is_reverse))


'''
    key를 기준으로 오름 차순 정렬, value를 기준으로 내림 차순 정렬
'''
def sorted_dict(in_dict: dict):
    return sorted_dict_value(sorted_dict_key(in_dict, False), True)

