from _init import *

import re, string, threading, time
from typing import Optional, List, Dict
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ranger.modules.common_const import *
from ranger.modules import string_util

from ranger.corag.search.search_utils import search_by_http
from ranger.corag.data_utils import format_documents_for_final_answer, format_input_context
from ranger.corag.prompts import get_generate_subquery_prompt, get_generate_intermediate_answer_prompt, get_generate_final_answer_prompt
from ranger.corag.utils import batch_truncate
from ranger.corag.data_utils import load_corpus
from ranger.corag.config import Arguments

from ranger.vllm.vllm_agent import VllmAgent


IS_TEST = False





def _normalize_subquery(subquery: str):
    subquery = subquery.strip()
    if subquery.startswith('"') and subquery.endswith('"'):
        subquery = subquery[1:-1]
    if subquery.startswith('Intermediate query'):
        subquery = re.sub(r'^Intermediate query \d+: ', '', subquery)

    return subquery


def _normalize_answer(text, punc_chars=string.punctuation, punc_repl=""):
    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def replace_punctuation(s):
        to_replace = set(punc_chars)
        return "".join(punc_repl if ch in to_replace else ch for ch in s)

    def white_space_fix(s):
        return " ".join(s.split())

    text = replace_punctuation(text)
    text = remove_articles(text)
    text = white_space_fix(text)

    return text


def _compare_answer(final_answer: str, answer: str, txt_option=TXT_OPTION.LOWER|TXT_OPTION.RM_SPACE):
    final_answer = string_util.refine_txt(final_answer, txt_option)
    answer = string_util.refine_txt(answer, txt_option)

    if final_answer == answer:
        return True

    return False





class QueryResult:
    def __init__(self) -> None:
        self._query_id = ''
        self._query = ''
        self._answer = ''
        self._chain_results = []
        self._doc_ids = []
        self._documents = []


class ChainResult:
    def __init__(self) -> None:
        self._sub_querys = []
        self._sub_answers = []
        self._doc_ids_list = []             # 2차원 배열
        self._documents_list = []           # 2차원 배열
        self._final_answers = []
        self._log_probs_list = []           # 2차원 배열
        self._log_likes = []
        self._is_stop = False
        self._reward = -1                   # 0 ~ 1
    

    def print_chain(self):
        chain_depth = len(self._sub_querys)
        print(f'ChainResult.print_chain() is_stop(correct) : {self._is_stop}, chain_depth : {chain_depth}')
        print(f'\t[log_like]\t[sub_query]\t[doc_ids]\t[sub_answer]\t[final_answer]\n')

        for i in range(chain_depth):
            print(f'\t{self._log_likes[i]}\t{self._sub_querys[i]}\t{self._doc_ids_list[i]}\t{self._sub_answers[i]}\t{self._final_answers[i]}')
        print(f'\n\treward : {self._reward}\n')





class CoRagAgent:
    def __init__(self,
                 vllm_agent: VllmAgent, max_model_len: int, max_token_gen: int, temperature: float,
                 top_k_query: int, top_k_sub_query: int,
                 corpus: Dataset=load_corpus()):

        self._vllm_agent = vllm_agent
        self._corpus = corpus # 검색 대상 문서

        self._corag_args = Arguments()

        self._tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self._vllm_agent._model_name)
        self._lock = threading.Lock()

        self._task_desc: str
        self._max_model_len = max_model_len
        self._max_token_gen = max_token_gen
        self._temperature = temperature
        self._top_k_query = top_k_query
        self._top_k_sub_query = top_k_sub_query

        self._batch_idx = 0
    

    def reset(self):
        self._batch_idx = 0


    def _truncate_long_messages(self, messages: List[Dict]):
        for msg in messages:
            if len(msg['content']) < 2 * self._max_model_len:
                continue

            with self._lock:
                msg['content'] = batch_truncate(
                    [msg['content']], tokenizer=self._tokenizer, max_length=self._max_model_len, truncate_from_middle=True
                )[0]


    def generate_sub_querys(self, query_results: List[QueryResult]):
        inputs = []

        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    # 서브쿼리 생성 프롬프트 생성
                    sub_query_prompt = get_generate_subquery_prompt(
                        query=query_result._query,
                        past_subqueries=chain_result._sub_querys,
                        past_subanswers=chain_result._sub_answers,
                        task_desc=self._task_desc
                    )

                    self._truncate_long_messages(sub_query_prompt)
                    inputs.append(sub_query_prompt)
        
        # VLLM을 통해 서브쿼리 생성
        sub_querys = self._vllm_agent.generate_batch(
            messages=inputs,
            max_token_gen=self._max_token_gen,
            temperature=self._temperature,
            return_toks_log_probs=False
        )

        idx = 0
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    normalized_sub_query = _normalize_subquery(sub_querys[idx])
                    chain_result._sub_querys.append(normalized_sub_query)
                    idx += 1


    def search_doc_for_sub_query(self, query_results: List[QueryResult]):
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    searcheds: List[Dict] = search_by_http(
                        query=chain_result._sub_querys[-1],
                        topk=self._top_k_sub_query
                    )
                    
                    doc_ids = [searched['id'] for searched in searcheds]
                    documents = [format_input_context(self._corpus[int(doc_id)]) for doc_id in doc_ids][::-1]
                    chain_result._doc_ids_list.append(doc_ids)
                    chain_result._documents_list.append(documents)


    def generate_sub_answers(self, query_results: List[QueryResult]):
        inputs = []
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    sub_answer_prompt = get_generate_intermediate_answer_prompt(
                            subquery=chain_result._sub_querys[-1],
                            documents=chain_result._doc_ids_list[-1],
                    )
                    self._truncate_long_messages(sub_answer_prompt)
                    inputs.append(sub_answer_prompt)

        sub_answers = self._vllm_agent.generate_batch(
            messages=inputs,
            max_token_gen=self._max_token_gen,
            temperature=self._temperature,
            return_toks_log_probs=False
        )
        
        idx = 0
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    normalized_sub_answer = _normalize_answer(sub_answers[idx])
                    chain_result._sub_answers.append(normalized_sub_answer)
                    idx += 1


    def search_doc_for_query(self, query_results: List[QueryResult]):
        for query_result in query_results:
            searcheds: List[Dict] = search_by_http(
                query=query_result._query,
                topk=self._top_k_query
            )

            query_result._doc_ids = [searched['id'] for searched in searcheds]
            query_result._documents = format_documents_for_final_answer(
                args=self._corag_args,
                context_doc_ids=query_result._doc_ids,
                tokenizer=self._tokenizer, corpus=self._corpus,
                lock=self._lock
            )


    def check_step_final_answers(self, query_results: List[QueryResult]):
        inputs = []

        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    final_answer_prompt = get_generate_final_answer_prompt(
                        query=query_result._query,
                        past_subqueries=chain_result._sub_querys or [],
                        past_subanswers=chain_result._sub_answers or [],
                        task_desc=self._task_desc,
                        documents=query_result._documents,
                    )

                    self._truncate_long_messages(final_answer_prompt)
                    inputs.append(final_answer_prompt)
        
        final_answer_tok_log_probs_list = self._vllm_agent.generate_batch(
            messages=inputs,
            max_token_gen=self._max_token_gen,
            temperature=0.0,
            return_toks_log_probs=True
        )

        idx = 0
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    final_answer, (toks, log_probs) = final_answer_tok_log_probs_list[idx]
                    normalized_final_answer = _normalize_answer(final_answer)
                    chain_result._final_answers.append(normalized_final_answer)
                    chain_result._log_probs_list.append(log_probs)
                    chain_result._log_likes.append(sum(log_probs) / len(log_probs))

                    if _compare_answer(normalized_final_answer, query_result._answer):
                        chain_result._is_stop = True

                    idx += 1

                    if IS_TEST:
                        print(f'\nfinal_answer : {normalized_final_answer}')

                        import math
                        for tok, log_prob in zip(toks, log_probs):
                            print(f'\ttok : {tok}, prob : {math.exp(log_prob)}')
                        print()


    def check_all_stop(self, query_results: List[QueryResult]):
        all_stopped = True

        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    all_stopped = False
                    break
            if not all_stopped:
                break
        
        return all_stopped


    def generate_batch(self, task_desc: str, datas: list, n_chains: int, chain_depth: int) -> List[QueryResult]:
        self._batch_idx += 1
        self._task_desc = task_desc

        # 각 쿼리에 대한 결과 객체 초기화
        query_results = []
        for data in datas:
            query_result = QueryResult()
            query_result._query_id = data['query_id']
            query_result._query = data['query']
            query_result._answer = _normalize_answer(data['answers'][0])
            query_result._chain_results = [ChainResult() for _ in range(n_chains)]
            query_results.append(query_result)
        
        self.search_doc_for_query(query_results)
        
        # 각 depth마다 서브쿼리 생성 및 처리
        for depth in range(chain_depth):
            print(f"{'='*50}\n[{self._batch_idx} Batch] {depth+1} Depth Start\n{'='*50}\n")
            start_time_depth = time.time()

            # 서브 쿼리 생성
            self.generate_sub_querys(query_results)

            # 문서 검색
            self.search_doc_for_sub_query(query_results)

            # 서브 답변 생성
            self.generate_sub_answers(query_results)

            # 서브 스텝 마다, 최종 답변 확인
            self.check_step_final_answers(query_results)

            print(f"\n[{self._batch_idx} Batch] {depth+1} Depth End, Elapsed Time (s) : {(time.time()-start_time_depth):.2f}\n")

            # 모든 체인이 중단되었는지 확인
            if self.check_all_stop(query_results):
                break

        return query_results

