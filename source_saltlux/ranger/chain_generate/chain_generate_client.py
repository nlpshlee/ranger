from _init import *

import time, requests, json

from ranger.utils import json_utils
from ranger.corag.corag_result import QueryResult


def request_chain_generate(datas, batch_size, n_chains, chain_depth, adapter_path=''):
    url = CHAIN_SERVER_CONFIG['url_full_generate']
    headers = {'Content-Type': 'application/json'}

    # 전송할 데이터 구성
    payload = {
        'datas': datas,
        'batch_size': batch_size,
        'n_chains': n_chains,
        'chain_depth': chain_depth,
        'adapter_path': adapter_path
    }

    try:
        if DEBUG.CHAIN_CLIENT:
            print(f'# [CHAIN_CLIENT] chain_generate_client.request_chain_generate() data_size : {len(datas)}')
        
        start_time = time.time()
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        elapsed = time.time() - start_time

        if response.status_code == 200:
            response_json = response.json()

            if DEBUG.CHAIN_CLIENT:
                print(f'# [CHAIN_CLIENT] chain_generate_client.request_chain_generate() response_size : {response_json["count"]}, elapsed_time : {elapsed:.2f}(s)\n')
            
            deserialized_responses = []

            for result in response_json['results']:
                deserialized_responses.append(QueryResult.from_dict(result))

            return deserialized_responses
        else:
            if DEBUG.CHAIN_CLIENT | DEBUG.ERROR:
                print(f'# [ERROR] chain_generate_client.request_chain_generate() code : {response.status_code}, {response.text}\n')
                return None
    except Exception as e:
        if DEBUG.CHAIN_CLIENT | DEBUG.ERROR:
            print(f'# [ERROR] chain_generate_client.request_chain_generate() CHAIN_SERVER_CONFIG :\n{json_utils.to_str(CHAIN_SERVER_CONFIG)}\n')
            print(f'# [ERROR] chain_generate_client.request_chain_generate() error : {e}\n')
            return None


def request_reset():
    url = CHAIN_SERVER_CONFIG['url_full_reset']
    headers = {'Content-Type': 'application/json'}

    try:
        if DEBUG.CHAIN_CLIENT:
            print(f'# [CHAIN_CLIENT] chain_generate_client.request_reset()')
        
        response = requests.post(url, headers=headers, data=[])

        if response.status_code == 200:
            return True
        else:
            if DEBUG.CHAIN_CLIENT | DEBUG.ERROR:
                print(f'# [ERROR] chain_generate_client.request_reset() code : {response.status_code}, {response.text}\n')
                return False
    except Exception as e:
        if DEBUG.CHAIN_CLIENT | DEBUG.ERROR:
            print(f'# [ERROR] chain_generate_client.request_reset() CHAIN_SERVER_CONFIG :\n{json_utils.to_str(CHAIN_SERVER_CONFIG)}\n')
            print(f'# [ERROR] chain_generate_client.request_reset() error : {e}\n')
            return False

