from _init import *

import pickle

from ranger.utils.common_const import *
from ranger.utils import common_utils, string_utils

###########################################################################################

def get_file_name(file_path: str, rm_ext_flag=False):
    file_name = os.path.basename(file_path)

    if rm_ext_flag:
        idx = file_name.rfind(FILE_EXT)

        if idx != -1:
            file_name = file_name[:idx]
    
    return file_name

def get_file_paths(in_path: str, inner_flag=True):
    file_paths = []

    if inner_flag:
        for (parent_path, dirs, file_names) in os.walk(in_path):
                for file_name in file_names:
                    file_path = os.path.join(parent_path, file_name)

                    if os.path.isfile(file_path):
                        file_paths.append(file_path)
    else:
        file_names = os.listdir(in_path)
        for file_name in file_names:
            file_path = os.path.join(in_path, file_name)

            if os.path.isfile(file_path):
                file_paths.append(file_path)

    return file_paths

###########################################################################################

def make_parent(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

def open_file(file_path: str, encoding=ENCODING, mode='r'):
    if mode.find('w') != -1 or mode.find('a') != -1:
        make_parent(file_path)

    if string_utils.is_empty(encoding, True) or mode.find('b') != -1:
        return open(file_path, mode)
    else:
        return open(file_path, mode, encoding=encoding)

def open_file_read(file_path: str, encoding=ENCODING, is_bin=False):
    if not is_bin:
        return open_file(file_path, encoding, 'r')
    else:
        return open_file(file_path, None, 'rb')

###########################################################################################

def preprocess(input: str):
    return input.strip()

###########################################################################################

def load_set(in_set: set, in_file_path: str, encoding=ENCODING, delim=DELIM_KEY, txt_option=TXT_OPTION.OFF, is_bin=False):
    if exists(in_file_path):
        in_file = open_file_read(in_file_path, encoding, is_bin)

        while 1:
            line = readline(in_file, is_bin)
            if not line:
                break

            line = preprocess(line)
            if len(line) == 0:
                continue

            # 'delim'을 인자로 받았다면, 분리 후, 'key' 값만 사용
            if delim != None:
                terms = line.split(delim)
                line = terms[0]
                line = line.strip()
                if len(line) == 0:
                    continue

            if txt_option != TXT_OPTION.OFF:
                line = string_utils.refine_txt(line, txt_option)

            in_set.add(line)
        in_file.close()

###########################################################################################

def load_dict(in_dict: dict, is_value_int: bool, in_file_path: str, encoding=ENCODING, delim=DELIM_KEY, txt_option=TXT_OPTION.OFF, is_bin=False):
    if exists(in_file_path):
        in_file = open_file_read(in_file_path, encoding, is_bin)

        while 1:
            line = readline(in_file, is_bin)
            if not line:
                break

            line = preprocess(line)
            if len(line) == 0:
                continue
        
            terms = line.split(delim)
            terms = string_utils.trim(terms, True)
            if len(terms) != 2:
                continue

            if txt_option != TXT_OPTION.OFF:
                terms[0] = string_utils.refine_txt(terms[0], txt_option)

            #if terms[0] in in_dict:
                #print("중복 제거 :", terms[0], in_dict[terms[0]])
        
            if is_value_int:
                in_dict[terms[0]] = int(terms[1])
            else:
                in_dict[terms[0]] = terms[1]
        in_file.close()

###########################################################################################

def write_set(out_set: set, out_file_path: str, encoding=ENCODING, is_reverse=False):
    file = open_file(out_file_path, encoding, 'w')

    out_list = list(out_set)
    out_list.sort(reverse = is_reverse)

    for i in range(len(out_list)):
        file.write(f"{out_list[i]}\n")
    
    file.close()

def write_dict(out_dict: dict, out_file_path: str, encoding=ENCODING, delim=DELIM_KEY):
    file = open_file(out_file_path, encoding, 'w')

    items = out_dict.items()
    for item in items:
        file.write(f"{item[0]}{delim}{item[1]}\n")
    
    file.close()

###########################################################################################

def exists(file_path: str):
    if file_path == None or len(file_path) == 0:
        return False

    if os.path.exists(file_path) and os.path.isfile(file_path):
        return True

    return False

###########################################################################################

def convert_txt_to_bin(txt_file_path: str, bin_file_path: str, encoding=ENCODING):
    if exists(txt_file_path):
        try:
            txt_file = open_file(txt_file_path, encoding, 'r')
            bin_file = open_file(bin_file_path, None, 'wb')

            while 1:
                line = txt_file.readline()
                if not line:
                    break
                
                bin_list = [format(ord(c), 'b') for c in line]
                pickle.dump(bin_list, bin_file)
            
            txt_file.close()
            bin_file.close()
            return True

        except Exception as e:
            common_utils.logging_error("file_util.convert_txt_to_bin()", e)
    
    return False

def convert_bin_to_txt(bin_file_path: str, txt_file_path: str, encoding=ENCODING):
    if exists(bin_file_path):
        try:
            bin_file = open_file(bin_file_path, None, 'rb')
            txt_file = open_file(txt_file_path, encoding, 'w')
        
            while 1:
                bin_list = pickle.load(bin_file)
                line = ''.join(chr(int(c, 2)) for c in bin_list)
                txt_file.write(f"{line}")

            bin_file.close()
            txt_file.close()
            return True

        except Exception as e:
            common_utils.logging_error("file_util.convert_bin_to_txt()", e)
    
    return False

###########################################################################################

def readline(in_file, is_bin=False):
    if not is_bin:
        return in_file.readline()
    else:
        try:
            bin_list = pickle.load(in_file)
        except:
            return None
        
        return ''.join(chr(int(c, 2)) for c in bin_list)

def get_cnt_line(in_file_path: str, encoding=ENCODING, is_bin=False):
    try:
        if exists(in_file_path):
            in_file = open_file_read(in_file_path, encoding, is_bin)
            cnt = 0

            while 1:
                line = readline(in_file, is_bin)
                if not line:
                    break
            
                cnt += 1
        
            in_file.close()
            return cnt

    except Exception as e:
        common_utils.logging_error("file_util.get_cnt_line()", e)
    
    return -1
