from pathlib import Path
import yaml


globals_file_path = Path(__file__).resolve()
globals_dir = globals_file_path.parent

with open(f'{globals_dir}/globals.yml', 'r') as in_file:
    data = yaml.safe_load(in_file)


class DEBUG:
    LOG = data['DEBUG']['LOG']
    ERROR = data['DEBUG']['ERROR']
    VLLM = data['DEBUG']['VLLM']
    CORAG = data['DEBUG']['CORAG']
    CORAG_SEARCH = data['DEBUG']['CORAG_SEARCH']
    CORAG_TEST = data['DEBUG']['CORAG_TEST']
    CHAIN = data['DEBUG']['CHAIN']
    CHAIN_TIME = data['DEBUG']['CHAIN_TIME']
    CHAIN_SERVER = data['DEBUG']['CHAIN_SERVER']


ELASTICSEARCH_CONFIG = {
    'host': data['ELASTICSEARCH_CONFIG']['HOST'],
    'port': data['ELASTICSEARCH_CONFIG']['PORT']
}


RETRIEVER_SERVER_CONFIG = {
    'host': data['RETRIEVER_SERVER_CONFIG']['HOST'],
    'port': data['RETRIEVER_SERVER_CONFIG']['PORT'],
    'url_retrieve': data['RETRIEVER_SERVER_CONFIG']['URL_RETRIEVE']
}


VLLM_CONFIG = {
    'model_name': data['VLLM_CONFIG']['MODEL_NAME'],
    'device': f'cuda:{data["VLLM_CONFIG"]["DEVICE"]}',
    'dtype': data['VLLM_CONFIG']['DTYPE'],
    'max_seq_length': data['VLLM_CONFIG']['MAX_SEQ_LENGTH'],
    'max_new_tokens': data['VLLM_CONFIG']['MAX_NEW_TOKENS'],
    'temperature': data['VLLM_CONFIG']['TEMPERATURE'],
    'gpu_memory_utilization': data['VLLM_CONFIG']['GPU_MEMORY_UTILIZATION'],
    'n_log_prob': data['VLLM_CONFIG']['N_LOG_PROB']
}


CORAG_CONFIG = {
    'top_k_query': data['CORAG_CONFIG']['TOP_K_QUERY'],
    'top_k_sub_query': data['CORAG_CONFIG']['TOP_K_SUB_QUERY'],
    'task_desc': data['CORAG_CONFIG']['TASK_DESC']
}


CHAIN_SERVER_CONFIG = {
    'host': data['CHAIN_SERVER_CONFIG']['HOST'],
    'port': data['CHAIN_SERVER_CONFIG']['PORT'],
    'debug': data['CHAIN_SERVER_CONFIG']['DEBUG'],
    'url_generate': data['CHAIN_SERVER_CONFIG']['URL_GENERATE'],
    'url_reset': data['CHAIN_SERVER_CONFIG']['URL_RESET']
}
CHAIN_SERVER_CONFIG['url'] = f'http://{CHAIN_SERVER_CONFIG["host"]}:{CHAIN_SERVER_CONFIG["port"]}'
CHAIN_SERVER_CONFIG['url_full_generate'] = f'{CHAIN_SERVER_CONFIG["url"]}{CHAIN_SERVER_CONFIG["url_generate"]}'
CHAIN_SERVER_CONFIG['url_full_reset'] = f'{CHAIN_SERVER_CONFIG["url"]}{CHAIN_SERVER_CONFIG["url_reset"]}'

