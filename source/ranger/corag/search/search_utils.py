import requests
import time
from typing import List, Dict


def search_by_http(query: str, topk=5, host: str = 'localhost', port: int = 8000, timeout: int = 30) -> List[Dict]:
    url = f"http://{host}:{port}/retrieve/"  # 엔드포인트에 슬래시 포함
    payload = {
        'query_text': query,
        'retrieval_method': 'retrieve_from_elasticsearch',
        'max_hits_count': topk,
        'document_type': 'paragraph_text'
    }
    
    start_time = time.time()
    # logger.info(f"검색 시작: {query[:50]}... (타임아웃: {timeout}초)")
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        elapsed_time = time.time() - start_time
        # logger.info(f"검색 완료: {query[:50]}... (소요시간: {elapsed_time:.2f}초)")
        
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
            print(f"# [error] 검색 실패: {query[:50]}... (상태코드: {response.status_code}, 응답: {response.text})")
            return []
    except requests.exceptions.Timeout:
        elapsed_time = time.time() - start_time
        print(f"# [error] 검색 타임아웃: {query[:50]}... (경과시간: {elapsed_time:.2f}초, 타임아웃: {timeout}초)")
        return []
    except requests.exceptions.ConnectionError as e:
        elapsed_time = time.time() - start_time
        print(f"# [error] 검색 연결 오류: {query[:50]}... (경과시간: {elapsed_time:.2f}초, 오류: {e})")
        return []
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"# [error] 검색 중 예외 발생: {query[:50]}... (경과시간: {elapsed_time:.2f}초, 오류: {e})")
        return []

