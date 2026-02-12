from _init import *

from typing import List

from ranger.utils import common_utils, container_utils, json_utils
from ranger.vllm.vllm_engine import VllmEngine
from ranger.corag.corag_agent import CoragAgent
from ranger.corag.corag_result import QueryResult
from ranger.chain_generate.chain_generate_time import ChainGenerateTime


class ChainGenerator:
    def __init__(self, vllm_config: dict, corag_config: dict):
        self._vllm_config = vllm_config
        self._corag_config = corag_config

        self._engine: VllmEngine = None
        self._corag_agent: CoragAgent = None

        self._chain_generate_time = ChainGenerateTime()

        self._init_modules()
        common_utils.check_gpu_memory(do_print=DEBUG.CHAIN, msg='[ChainGenerator init]')


    def _init_modules(self):
        if DEBUG.CHAIN:
            print(f'# [CHAIN] ChainGenerator._init_modules() vllm_config :\n{json_utils.to_str(self._vllm_config)}\n')

        self._engine = VllmEngine(
            model_name=self._vllm_config['model_name'],
            device=self._vllm_config['device'],
            dtype=self._vllm_config['dtype'],
            max_seq_length=self._vllm_config['max_seq_length'],
            max_new_tokens=self._vllm_config['max_new_tokens'],
            temperature=self._vllm_config['temperature'],
            top_p=self._vllm_config['top_p'],
            top_k=self._vllm_config['top_k'],
            gpu_memory_utilization=self._vllm_config['gpu_memory_utilization'],
            n_log_prob=self._vllm_config['n_log_prob']
        )

        if DEBUG.CHAIN:
            print(f'\n# [CHAIN] ChainGenerator._init_modules() corag_config :\n{json_utils.to_str(self._corag_config)}\n')
        
        self._corag_agent = CoragAgent(
            engine=self._engine,
            top_k_query=self._corag_config['top_k_query'],
            top_k_sub_query=self._corag_config['top_k_sub_query'],
            task_desc=self._corag_config['task_desc']
        )


    def reset(self):
        self._corag_agent.reset()
        self._chain_generate_time.reset()


    def generate(self, datas, batch_size, n_chains, chain_depth, adapter_path='', temperature=-9, top_p=-9, top_k=-9) -> List[List[QueryResult]]:
        results = []

        for i, datas_batch in enumerate(container_utils.chunks(datas, batch_size)):
            self._chain_generate_time.check_time()

            query_results: List[QueryResult] = self._corag_agent.generate_batch(
                datas=datas_batch,
                n_chains=n_chains,
                chain_depth=chain_depth,
                adapter_path=adapter_path,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )

            results.append(query_results)

            self._chain_generate_time.check_time(len(datas_batch) * n_chains, self._corag_agent._engine._called_cnt)

        if DEBUG.CHAIN_TIME:
            self._chain_generate_time.print_time()
        
        return results

