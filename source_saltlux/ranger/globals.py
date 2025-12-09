# from pathlib import Path
import yaml


with open('./globals.yml', 'r') as in_file:
    data = yaml.safe_load(in_file)


class DEBUG:
    LOG = data['DEBUG']['LOG']
    ERROR = data['DEBUG']['ERROR']
    VLLM = data['DEBUG']['VLLM']
    CORAG = data['DEBUG']['CORAG']
    CORAG_SEARCH = data['DEBUG']['CORAG_SEARCH']
    CORAG_TEST = data['DEBUG']['CORAG_TEST']
    CHAIN = data['DEBUG']['CHAIN']

