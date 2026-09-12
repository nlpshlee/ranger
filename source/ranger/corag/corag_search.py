from _init import *

import requests
from typing import List, Dict

from ranger.utils import common_utils


def search_by_http(query: str, topk=5,
                   host=RETRIEVER_SERVER_CONFIG['host'],
                   port=RETRIEVER_SERVER_CONFIG['port'],
                   url_retrieve=RETRIEVER_SERVER_CONFIG['url_retrieve'],
                   timeout=30) -> List[Dict]:

    url = f"http://{host}:{port}{url_retrieve}/"  # 엔드포인트에 슬래시 포함

    payload = {
        'query_text': query,
        'retrieval_method': 'retrieve_from_elasticsearch',
        'max_hits_count': topk,
        'document_type': 'paragraph_text'
    }
    
    search_start = common_utils.get_time_ms()
    if DEBUG.CORAG_SEARCH:
        print(f'# [CORAG_SEARCH] corag_search.search_by_http() start : [{query}], timeout : {timeout}s')
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)

        elapsed_ms, elapsed_str = common_utils.get_elapsed_time_ms(search_start)
        if DEBUG.CORAG_SEARCH:
            print(f'# [CORAG_SEARCH] corag_search.search_by_http() end : [{query}], elapsed_time : {elapsed_str} ({elapsed_ms}ms)\n')

        if response.status_code == 200:
            result = response.json()
            """
            result['retrieval'] -> 이런 게 max_hits_count만큼 있음
            [
                {
                    "id": "1",
                    "title": title,
                    "paragraph_text": contents,
                    "score": score,
                    "corpus_name": None,
                }
            ]
            """
            return result['retrieval']
        else:
            if DEBUG.CORAG_SEARCH | DEBUG.ERROR:
                print(f'# [ERROR] corag_search.search_by_http() 검색 실패 : [{query}], elapsed_time : {elapsed_str}, status_code : {response.status_code}, response : {response.text}\n')
            return []
    except requests.exceptions.Timeout:
        if DEBUG.CORAG_SEARCH | DEBUG.ERROR:
            _, elapsed_str = common_utils.get_elapsed_time_ms(search_start)
            print(f'# [ERROR] corag_search.search_by_http() 검색 타임 아웃 : [{query}], elapsed_time : {elapsed_str}, timeout : {timeout}s\n')
        return []
    except requests.exceptions.ConnectionError as e:
        if DEBUG.CORAG_SEARCH | DEBUG.ERROR:
            _, elapsed_str = common_utils.get_elapsed_time_ms(search_start)
            print(f'# [ERROR] corag_search.search_by_http() 검색 연결 오류 : [{query}], elapsed_time : {elapsed_str}, error_msg : {e}\n')
        return []
    except Exception as e:
        if DEBUG.CORAG_SEARCH | DEBUG.ERROR:
            _, elapsed_str = common_utils.get_elapsed_time_ms(search_start)
            print(f'# [ERROR] corag_search.search_by_http() 검색 예외 발생 : [{query}], elapsed_time : {elapsed_str}, error_msg : {e}\n')
        return []

