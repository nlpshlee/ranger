from _init import *

import threading, math
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from ranger.utils import common_utils

from ranger.vllm.vllm_engine import VllmEngine
from ranger.corag import corag_utils, corag_search, corag_prompts
from ranger.corag.corag_arguments import CoragArguments
from ranger.corag.corag_result import ChainResult, QueryResult


class CoragAgent:
    '''
        문서 검색은 리트리버 서버로의 HTTP 호출이라 GPU와 무관한데,
        체인 수만큼 순차 호출하면 depth 소요 시간의 30% 가량을 차지함 -> 병렬 처리
        (리트리버/ES 를 과도하게 때리지 않도록 워커 수는 제한)
    '''
    SEARCH_MAX_WORKERS = 8

    # 최종 답변 앞에 붙는 특수 토큰 (프롬프트가 '<STOP>' / '<CONTINUE>' 를 요구하지만, 변형도 함께 인식)
    CONTINUE_FORMS = ['<CONTINUE>', '<continue>', '[CONTINUE]', '[continue]', '(CONTINUE)', '(continue)']
    STOP_FORMS = ['<STOP>', '<stop>', '[STOP]', '[stop]', '(STOP)', '(stop)']


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
        self._temperature: float
        self._top_p: float
        self._top_k: int

        self._is_eval: bool

        '''
            [토큰 비용 공정 비교] 최종 답변 생성 시점
                False (기본) : 매 스텝마다 final_answer 생성
                               <STOP>/<CONTINUE> 로 조기 종료를 판단하는 모델(우리 SFT/RL)은 이게 필수
                True         : 마지막 스텝에서만 final_answer 생성
                               조기 종료 능력이 없는 모델(순정 LLM, CoRAG 등)은 중간 최종답변이
                               실제 추론 절차에 없는 계산이므로, 과금하면 베이스라인에 불리하게 왜곡됨
        '''
        self._final_answer_last_only: bool = False


    def reset(self):
        self._batch_idx = 0
        self._engine.reset()


    @staticmethod
    def _add_token_counts(chain_result: ChainResult, token_counts, idx: int):
        # 엔진이 반환한 (prompt_tokens, generated_tokens) 를 체인에 누적
        if idx < len(token_counts):
            prompt_tokens, gen_tokens = token_counts[idx]
            chain_result._prompt_tokens += prompt_tokens
            chain_result._gen_tokens += gen_tokens


    def _search_doc_for_query(self, query_results: List[QueryResult]):
        if not query_results:
            return

        with ThreadPoolExecutor(max_workers=self.SEARCH_MAX_WORKERS) as executor:
            searcheds_list = list(executor.map(
                lambda qr: corag_search.search_by_http(query=qr._query, topk=self._top_k_query),
                query_results
            ))

        # 코퍼스 접근(format_documents_for_final_answer)은 lock 을 쓰므로 순차 처리
        for query_result, searcheds in zip(query_results, searcheds_list):
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

                    inputs.append(sub_query_prompt)
                    chain_result._sub_query_prompts.append(sub_query_prompt)

        # 서브 쿼리 생성
        sub_querys = self._engine.generate_batch(
            datas=inputs,
            return_completion_output=False,
            adapter_path=self._adapter_path,
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k
        )

        # 직전 호출의 요청별 토큰 수 (다음 generate_batch 호출 전에 확보)
        token_counts = self._engine._last_token_counts

        idx = 0
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    self._add_token_counts(chain_result, token_counts, idx)
                    chain_result._sub_querys_raw.append(sub_querys[idx])

                    normalized_sub_query = corag_utils.normalize_sub_query(sub_querys[idx])
                    chain_result._sub_querys.append(normalized_sub_query)
                    idx += 1


    def _search_doc_for_sub_query(self, query_results: List[QueryResult]):
        targets = [cr for qr in query_results for cr in qr._chain_results if not cr._is_stop]

        if not targets:
            return

        with ThreadPoolExecutor(max_workers=self.SEARCH_MAX_WORKERS) as executor:
            searcheds_list = list(executor.map(
                lambda cr: corag_search.search_by_http(query=cr._sub_querys[-1], topk=self._top_k_sub_query),
                targets
            ))

        # 코퍼스 접근은 순차 처리 (datasets 객체 동시 접근 회피)
        for chain_result, searcheds in zip(targets, searcheds_list):
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

                    inputs.append(sub_answer_prompt)
                    chain_result._sub_answer_prompts.append(sub_answer_prompt)

        sub_answers = self._engine.generate_batch(
            datas=inputs,
            return_completion_output=False,
            adapter_path=self._adapter_path,
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k
        )
        
        token_counts = self._engine._last_token_counts

        idx = 0
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    self._add_token_counts(chain_result, token_counts, idx)
                    chain_result._sub_answers_raw.append(sub_answers[idx])

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
                        documents=query_result._docs,
                        # 조기 종료를 쓰지 않는 베이스라인에는 <STOP>/<CONTINUE> 지시문을 넣지 않음
                        # (해당 토큰을 학습한 적이 없어 답변 품질을 떨어뜨릴 수 있고, CoRAG 원 방식 프롬프트와도 달라짐)
                        add_decide_prompt=(not self._final_answer_last_only)
                    )

                    inputs.append(final_answer_prompt)
                    chain_result._final_answer_prompts.append(final_answer_prompt)
        
        '''
            [중요] 최종 답변 생성 온도
                - 평가 시 : greedy (재현성 / best-of-n 선택 기준의 일관성)
                - 학습 시 : 다른 추론 시점과 동일하게 정책 분포에서 샘플링

                  temperature=0.0 으로 만든 토큰은 정책이 '샘플링한 행동'이 아니라서
                  policy gradient 대상이 될 수 없음 (= 로스를 전파하면 안 되는 토큰이 됨)
                  최종 답변에도 학습이 되게 하려면 반드시 샘플링이어야 함
        '''
        if self._is_eval:
            final_answer_temperature, final_answer_top_p, final_answer_top_k = 0.0, 1.0, -1
        else:
            final_answer_temperature, final_answer_top_p, final_answer_top_k = self._temperature, self._top_p, self._top_k

        final_answer_completion_output_list = self._engine.generate_batch(
            datas=inputs,
            return_completion_output=True,
            adapter_path=self._adapter_path,
            temperature=final_answer_temperature,
            top_p=final_answer_top_p,
            top_k=final_answer_top_k
        )

        token_counts = self._engine._last_token_counts

        idx = 0
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    self._add_token_counts(chain_result, token_counts, idx)
                    final_answer, completion_output = final_answer_completion_output_list[idx]

                    # 학습은 원문(특수 토큰 포함)으로, 정답 비교/지표는 normalize 후 값으로
                    chain_result._final_answers_raw.append(final_answer)

                    '''
                        [중요] 특수 토큰은 normalize '이전'의 원문에서 판정해야 함
                            normalize_answer() 는 구두점을 전부 제거하므로 '<STOP>' 이 'stop' 이 되어
                            평문으로 시작하는 답변("Stop signs are red")과 구분할 수 없게 됨
                            -> 조기 종료를 학습하지 않은 모델에서 '가짜 종료'가 발생하고, 답변 앞부분까지 잘려나감

                        프롬프트가 요구하는 형태는 '<STOP>' / '<CONTINUE>' (대문자 + 꺾쇠) 이므로
                        꺾쇠/괄호 형태만 인정하고, 괄호 없는 평문 'stop' 은 종료 신호로 보지 않음
                    '''
                    '''
                        [중요] 특수 토큰 처리를 학습/평가 동일하게 수행

                        (A-1) 리워드/지표 계산용 텍스트에서는 접두어를 제거
                            - 제거하지 않으면 '<STOP> john wayne' -> 'stop john wayne' 이 되어
                              f1('john wayne', 'stop john wayne') = 0.8 로 정답인데도 점수가 깎이고,
                              정답 비교(exact match)도 실패함
                            - 즉 특수 토큰을 생성할수록 손해라서, RL 이 토큰을 지우는 방향으로 학습됨
                            - 학습 타깃은 _final_answers_raw (접두어 포함 원문) 이므로,
                              정책은 자기가 샘플링한 <STOP> 토큰에 정상적으로 credit 을 받음

                        (A-2) 종료 판정을 '모델이 생성한 특수 토큰' 으로 수행
                            - 기존에는 학습 시 정답 일치(오라클)로만 종료시켜서,
                              모델의 <STOP> 출력이 체인 길이에 아무 영향을 주지 못했음
                              -> 종료가 정책의 행동이 아니게 되어 학습이 불가능했고,
                                 LENGTH 리워드도 정책이 제어할 수 없는 값이었음
                            - 이제 종료는 정책의 행동이며, 짧고 정확하면 리워드가 높아짐
                    '''
                    answer_text = final_answer.strip()

                    is_continue, answer_text = corag_utils.truncate_starts(answer_text, self.CONTINUE_FORMS)

                    if not is_continue:
                        chain_result._is_stop, answer_text = corag_utils.truncate_starts(answer_text, self.STOP_FORMS)

                    normalized_final_answer = corag_utils.normalize_answer(answer_text, to_lower=True)

                    chain_result._final_answers.append(normalized_final_answer)
                    chain_result._log_probs.append(self._engine.get_generated_log_prob(completion_output))

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
        '''
            현재 depth 에서 실제로 처리된(= 아직 중단되지 않은) 체인 수

            [주의] _final_answers 로 세면 안 됨
                final_answer 는 final_answer_last_only 모드에서 마지막 스텝에만 1개 쌓이므로,
                depth 와 개수가 어긋나 항상 0 이 찍힘

                _sub_querys 는 모드와 무관하게 매 depth 마다 정확히 1개씩 쌓이므로 이걸 기준으로 셈
        '''
        count = 0

        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if check_depth == len(chain_result._sub_querys):
                    count += 1

        return count


    def generate_batch(self, datas: list, n_chains: int, chain_depth: int, adapter_path='', temperature=-9, top_p=-9, top_k=-9, is_eval=False, final_answer_last_only=False) -> List[QueryResult]:
        self._batch_idx += 1
        self._final_answer_last_only = final_answer_last_only
        self._adapter_path = adapter_path
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._is_eval = is_eval

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

            '''
                조기 종료 능력이 없는 모델은 마지막 스텝에서만 최종 답변을 생성
                (중간 스텝의 최종 답변은 어차피 종료 판단에 쓰이지 않으므로 순수 낭비)
            '''
            is_last_depth = (depth == chain_depth - 1)

            if self._final_answer_last_only and not is_last_depth:
                if DEBUG.CORAG:
                    elapsed_ms, elapsed_str = common_utils.get_elapsed_time_ms(depth_start)
                    print(f'# [CORAG] CoragAgent.generate_batch() [{self._batch_idx} batch] [{depth+1} depth] [skip final_answer], elapsed_time : {elapsed_str} ({elapsed_ms})ms')
                continue

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

