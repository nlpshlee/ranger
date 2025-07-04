from _init import *

import json

from ranger.modules.common_const import *
from ranger.modules import common_util, file_util

###########################################################################################

def str_to_dict(json_str: str):
    try:
        # 문자열을 읽을 때는, loads() 호출
        return json.loads(json_str)

    except Exception as e:
        common_util.logging_error("json_util.str_to_dict()", e)
        return None

def to_str(input, indent=4):
    try:
        return json.dumps(input, ensure_ascii=False, indent=indent)

    except Exception as e:
        common_util.logging_error("json_util.to_str()", e)
        return ""

def load_file_to_dict(in_file_path: str, encoding=ENCODING):
    try:
        if file_util.exists(in_file_path):
            file = file_util.open_file(in_file_path, encoding, 'r')

            # 파일을 읽을 때는, load() 호출
            json_dict = json.load(file)

            return json_dict

    except Exception as e:
        common_util.logging_error("json_util.load_file_to_dict()", e)
        return None

    return None

def write_file(input, out_file_path: str, encoding=ENCODING, indent=4):
    try:
        file = file_util.open_file(out_file_path, encoding, 'w')
        file.write(to_str(input, indent))
        file.close()
        return True

    except Exception as e:
        common_util.logging_error("json_util.write_file()", e)
        return False

###########################################################################################

''' if __name__ == "__main__":
    work_dir = "C:\\nlpshlee\\machine_learning\\morpheme_analyzer\\emjeol_classification_based_cnn\\"

    in_file_path = work_dir + "/conf/ma_conf.json"
    encoding = "utf8"

    json_dict = load_file_to_dict(in_file_path, encoding)
    print(type(json_dict))
    print(f"{json_dict}\n")

    json_str = dict_to_str(json_dict, '\t')
    print(type(json_str))
    print(f"{json_str}\n")

    json_dict = str_to_dict(json_str)
    print(type(json_dict))
    print(f"{json_dict}\n") '''