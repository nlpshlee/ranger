from _init import *

from typing import List
from flask import Flask, request, jsonify
from flask_restful import Api, Resource

from ranger.utils import common_utils, json_utils
from ranger.corag.corag_result import QueryResult
from ranger.chain_generate.chain_generator import ChainGenerator


seed = COMMON_CONFIG['seed']
common_utils.set_seed(seed)


if DEBUG.CHAIN_SERVER:
    print(f'\n{"="*100}\n# [CHAIN_SERVER] Starting ChainGenerator server...\n')
    print(f'# [CHAIN_SERVER] CHAIN_SERVER_CONFIG :\n{json_utils.to_str(CHAIN_SERVER_CONFIG)}\n')


# 1. ChainGenerator 초기화 (서버 시작 시 1회만 실행됨)
chain_generator = ChainGenerator(VLLM_CONFIG, CORAG_CONFIG)
# chain_generator._engine._seed = seed

# 2. Flask App 및 API 정의
app = Flask(__name__)
api = Api(app)


class ChainGenerateServer(Resource):
    def post(self):
        req_json = request.get_json()

        # req_json.get() 함수의 마지막 인자는 디폴트 값
        datas = req_json.get('datas', [])
        batch_size = req_json.get('batch_size', 1)
        n_chains = req_json.get('n_chains', 5)
        chain_depth = req_json.get('chain_depth', 5)
        adapter_path = req_json.get('adapter_path', '')

        results: List[List[QueryResult]] = chain_generator.generate(
            datas=datas,
            batch_size=batch_size,
            n_chains=n_chains,
            chain_depth=chain_depth,
            adapter_path=adapter_path
        )

        serialized_results = self._serialize(results)
        
        return jsonify({
            'status': 'success',
            'count': len(serialized_results),
            'results': serialized_results
        })


    def _serialize(self, results: List[List[QueryResult]]):
        serialized_results = []

        for result in results:
            for query_result in result:
                serialized_results.append(query_result.to_dict())

        return serialized_results


class ChainGenerateReset(Resource):
    def post(self):
        chain_generator.reset()

        return jsonify({'status': 'success'})


# 3. URL 맵핑
api.add_resource(ChainGenerateServer, CHAIN_SERVER_CONFIG['url_generate'])
api.add_resource(ChainGenerateReset, CHAIN_SERVER_CONFIG['url_reset'])

if DEBUG.CHAIN_SERVER:
        print(f'# [CHAIN_SERVER] ChainGenerator server started.\n{"="*100}\n')


'''
    CUDA_VISIBLE_DEVICES=0 python -u -m chain_generate.chain_generate_server > ./logs/chain_generate_server.log

    # 포트 확인
    sudo lsof -i :7200
'''
if __name__ == '__main__':
    host = CHAIN_SERVER_CONFIG['host']
    port = CHAIN_SERVER_CONFIG['port']
    debug = CHAIN_SERVER_CONFIG['debug']

    app.run(host=host, port=port, debug=debug)

