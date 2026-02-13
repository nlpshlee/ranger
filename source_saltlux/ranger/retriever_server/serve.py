import traceback
from time import perf_counter
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .unified_retriever import UnifiedRetriever

from globals import *


# 검색기 객체 생성
es_host = f'http://{ELASTICSEARCH_CONFIG["host"]}/'
es_port = ELASTICSEARCH_CONFIG['port']

retriever = UnifiedRetriever(es_host=es_host, es_port=es_port)

# FastAPI 서버 객체 생성
app = FastAPI()


# 'GET /' 요청에 '서버 정상 작동' 메시지 반환
@app.get("/")
async def index():
    return {"message": "Hello! This is a retriever server."}


# 'POST /retrieve/' 요청이 오면, 요청 본문 전체를 JSON으로 파싱
@app.post(f'{RETRIEVER_SERVER_CONFIG["url_retrieve"]}/')
async def retrieve(arguments: Request):  # see the corresponding method in unified_retriever.py
    try:
        arguments = await arguments.json()

        # retrieval_method 키로 검색 방식 선택 -> retrieve_from_elasticsearch만 허용
        retrieval_method = arguments.pop("retrieval_method")
        assert retrieval_method in ("retrieve_from_elasticsearch")

        # 검색 소요 시간 설정
        start_time = perf_counter()

        # 실제 검색 결과
        retrieval = getattr(retriever, retrieval_method)(**arguments)
        end_time = perf_counter()
        time_in_seconds = round(end_time - start_time, 1)
        return {"retrieval": retrieval, "time_in_seconds": time_in_seconds}
    except Exception as e:
        err_msg = str(e)
        tb = traceback.format_exc()

        if DEBUG.ERROR:
            print(f'# [ERROR] serve.retrieve() err_msg : {err_msg}\n\n{tb}\n', flush=True)
        
        return JSONResponse(
            status_code=500,
            content={
                'error_type': 'ServerExecutionError',
                'message': err_msg,
                'traceback': tb
            }
        )

