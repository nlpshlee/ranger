from _init import *

import threading, math
from typing import List, Dict
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from ranger.utils import common_utils

from ranger.vllm.vllm_engine import VllmEngine
from ranger.corag import corag_utils, corag_search, corag_prompts
from ranger.corag.corag_arguments import CoragArguments
from ranger.corag.corag_result import ChainResult, QueryResult


class CoragAgent:
    def __init__(self,
                 engine: VllmEngine, top_k_query: int, top_k_sub_query: int, task_desc: str,
                 corpus: Dataset=corag_utils.load_corpus('corag/kilt-corpus', 'train')):

        self._engine = engine
        self._top_k_query = top_k_query
        self._top_k_sub_query = top_k_sub_query
        self._task_desc = task_desc
        self._corpus = corpus # 검색 대상 문서

        self._corag_args = CoragArguments()

        self._tokenizer: PreTrainedTokenizerFast = self._engine._tokenizer
        self._lock = threading.Lock()

        self._batch_idx = 0
        self._adapter_path: str


    def reset(self):
        self._batch_idx = 0
        self._engine.reset()


    def _search_doc_for_query(self, query_results: List[QueryResult]):
        for query_result in query_results:
            searcheds: List[Dict] = corag_search.search_by_http(
                query=query_result._query,
                topk=self._top_k_query
            )

            query_result._doc_ids = [searched['id'] for searched in searcheds]
            query_result._docs = corag_utils.format_documents_for_final_answer(
                args=self._corag_args,
                context_doc_ids=query_result._doc_ids,
                tokenizer=self._tokenizer, corpus=self._corpus,
                lock=self._lock
            )


    def _generate_sub_querys(self, query_results: List[QueryResult]):
        inputs = []

        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    # 서브 쿼리 생성을 위한 프롬프트 생성
                    sub_query_prompt = corag_prompts.get_generate_sub_query_prompt(
                        query=query_result._query,
                        past_subqueries=chain_result._sub_querys,
                        past_subanswers=chain_result._sub_answers,
                        task_desc=self._task_desc
                    )

                    # get_generate_sub_query_prompt() 에서 [prompt] 형식으로 size '1'인 리스트로 반환함
                    inputs.append(sub_query_prompt[0])

        # 서브 쿼리 생성
        sub_querys = self._engine.generate_batch(
            datas=inputs,
            return_completion_output=False,
            adapter_path=self._adapter_path
        )

        idx = 0
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    normalized_sub_query = corag_utils.normalize_sub_query(sub_querys[idx])
                    chain_result._sub_querys.append(normalized_sub_query)
                    idx += 1


    def _search_doc_for_sub_query(self, query_results: List[QueryResult]):
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    searcheds: List[Dict] = corag_search.search_by_http(
                        query=chain_result._sub_querys[-1],
                        topk=self._top_k_sub_query
                    )
                    
                    doc_ids = [searched['id'] for searched in searcheds]
                    docs = [corag_utils.format_input_context(self._corpus[int(doc_id)]) for doc_id in doc_ids][::-1]
                    chain_result._doc_ids_list.append(doc_ids)
                    chain_result._docs_list.append(docs)


    def _generate_sub_answers(self, query_results: List[QueryResult]):
        inputs = []
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    sub_answer_prompt = corag_prompts.get_generate_intermediate_answer_prompt(
                        subquery=chain_result._sub_querys[-1],
                        documents=chain_result._docs_list[-1]
                    )

                    inputs.append(sub_answer_prompt[0])

        sub_answers = self._engine.generate_batch(
            datas=inputs,
            return_completion_output=False,
            adapter_path=self._adapter_path
        )
        
        idx = 0
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    normalized_sub_answer = corag_utils.normalize_answer(sub_answers[idx])
                    chain_result._sub_answers.append(normalized_sub_answer)
                    idx += 1


    def _check_step_final_answers(self, query_results: List[QueryResult]):
        inputs = []

        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    final_answer_prompt = corag_prompts.get_generate_final_answer_prompt(
                        query=query_result._query,
                        past_subqueries=chain_result._sub_querys or [],
                        past_subanswers=chain_result._sub_answers or [],
                        task_desc=self._task_desc,
                        documents=query_result._docs
                    )

                    inputs.append(final_answer_prompt[0])
        
        final_answer_completion_output_list = self._engine.generate_batch(
            datas=inputs,
            return_completion_output=True,
            adapter_path=self._adapter_path
        )

        idx = 0
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    final_answer, completion_output = final_answer_completion_output_list[idx]
                    normalized_final_answer = corag_utils.normalize_answer(final_answer)
                    chain_result._final_answers.append(normalized_final_answer)
                    chain_result._log_probs.append(self._engine.get_generated_log_prob(completion_output))

                    # answer_set 은 이미 소문자로 변환해서 저장된 상태이고, final_answer 은 소문자로 norm 처리되어 있음
                    if corag_utils.compare_answers(query_result._answer_set, normalized_final_answer):
                        chain_result._is_stop = True

                    idx += 1



                    if DEBUG.CORAG_TEST:
                        print(f'\n# [CORAG_TEST] CoragAgent._check_step_final_answers()')
                        print(f'\tquery : {query_result._query}')
                        print(f'\tanswers : {query_result._answers}\n')
                        
                        print(f'\tfinal_answer : {normalized_final_answer}')

                        log_prob, toks, tok_ids, tok_log_probs = self._engine.get_generated_log_prob(completion_output, return_all=True)
                        print(f'\tfinal_answer_prob : {math.exp(log_prob)}\n')

                        for i, (tok, tok_id, tok_log_prob) in enumerate(zip(toks, tok_ids, tok_log_probs)):
                            print(f'\t\ttok : [ {tok} ], id : {tok_id}, prob : {math.exp(tok_log_prob)}')
                            if i == 5:
                                break
                        print()


    def _check_all_stop(self, query_results: List[QueryResult]):
        all_stopped = True

        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    all_stopped = False
                    break
            if not all_stopped:
                break
        
        return all_stopped


    def _get_count_processing_chains(self, query_results: List[QueryResult], check_depth):
        count = 0

        for query_result in query_results:
            for chain_result in query_result._chain_results:
                depth = len(chain_result._final_answers)

                if check_depth == depth:
                    count += 1

        return count


    def generate_batch(self, datas: list, n_chains: int, chain_depth: int, adapter_path: str) -> List[QueryResult]:
        self._batch_idx += 1
        self._adapter_path = adapter_path

        # 각 쿼리에 대한 결과 객체 초기화
        query_results = []
        for data in datas:
            query_result = QueryResult()
            query_result._query_id = data['query_id']
            query_result._query = data['query']
            query_result._answers = [corag_utils.normalize_answer(answer, to_lower=False) for answer in data['answers']]
            query_result._answer_set = set([answer.lower() for answer in query_result._answers])
            query_result._hop = data['hop'] if 'hop' in data.keys() else -1
            query_result._chain_results = [ChainResult() for _ in range(n_chains)]
            query_results.append(query_result)
        
        # 쿼리 문서 검색
        self._search_doc_for_query(query_results)
        
        # 각 depth 마다 서브 쿼리 생성 및 처리
        for depth in range(chain_depth):
            depth_start = common_utils.get_time_ms()

            # 서브 쿼리 생성
            self._generate_sub_querys(query_results)

            # 서브 쿼리 문서 검색
            self._search_doc_for_sub_query(query_results)

            # 서브 답변 생성
            self._generate_sub_answers(query_results)

            # 서브 스텝 마다, 최종 답변 생성하고 실제 정답과 비교
            self._check_step_final_answers(query_results)

            if DEBUG.CORAG:
                count_processing_chains = self._get_count_processing_chains(query_results, depth+1)
                elapsed_ms, elapsed_str = common_utils.get_elapsed_time_ms(depth_start)
                print(f'# [CORAG] CoragAgent.generate_batch() [{self._batch_idx} batch] [{depth+1} depth] [{count_processing_chains} chains], elapsed_time : {elapsed_str} ({elapsed_ms})ms')

            # 모든 체인이 중단되었는지 확인
            if self._check_all_stop(query_results):
                break

        return query_results

