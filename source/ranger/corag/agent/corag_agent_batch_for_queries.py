import re, string, threading, time
from itertools import chain
from copy import deepcopy
from typing import Optional, List, Dict, Tuple
from datasets import Dataset
from openai.types.chat import ChatCompletion
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from logger_config import logger
from vllm_client import VllmClient, get_vllm_model_id
from search.search_utils import search_by_http
from data_utils import format_documents_for_final_answer, format_input_context, parse_answer_logprobs
from prompts import get_generate_subquery_prompt, get_generate_intermediate_answer_prompt, get_generate_final_answer_prompt
from agent.agent_utils import RagPath
from utils import batch_truncate
from ranger.corag.data_utils import load_corpus
from ranger.corag.config import Arguments


def _normalize_subquery(subquery: str):
    subquery = subquery.strip()
    if subquery.startswith('"') and subquery.endswith('"'):
        subquery = subquery[1:-1]
    if subquery.startswith('Intermediate query'):
        subquery = re.sub(r'^Intermediate query \d+: ', '', subquery)

    return subquery


def _normalize_answer(text, punc_chars, punc_repl):
    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def replace_punctuation(s):
        to_replace = set(punc_chars)
        return "".join(punc_repl if ch in to_replace else ch for ch in s)

    def white_space_fix(s):
        return " ".join(s.split())

    text = text.lower()
    text = replace_punctuation(text)
    text = remove_articles(text)
    text = white_space_fix(text)

    return text


class QueryResult:
    def __init__(self) -> None:
        self._query_id = ''
        self._query = ''
        self._answer = ''
        self._chain_results = []


class ChainResult:
    def __init__(self) -> None:
        self._sub_querys = []
        self._sub_answers = []
        self._doc_ids = []                  # 2차원 배열
        self._documents = []
        self._final_answers = []
        self._log_probs_list = []           # 2차원 배열
        self._scores = []
        self._is_stop = False
    

    def print_chain(self):
        chain_depth = len(self._sub_querys)
        print(f'ChainResult.print_chain() is_stop(correct) : {self._is_stop}, chain_depth : {chain_depth}')
        print(f'\t[score]\t[sub_query]\t[doc_ids]\t[sub_answer]\t[final_answer]\n')

        for i in range(chain_depth):
            print(f'\t{self._scores[i]}\t{self._sub_querys[i]}\t{self._doc_ids[i]}\t{self._sub_answers[i]}\t{self._final_answers[i]}')
        print()


class CoRagAgent:
    def __init__(self, vllm_client: VllmClient, corpus: Dataset=load_corpus()):
        self._vllm_client = vllm_client
        self._corpus = corpus # 검색 대상 문서
        self._tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self._vllm_client.model)
        self._lock = threading.Lock()

        self._task_desc: str
        self._n_chains: int
        self._chain_depth: int
        self._max_length: int
        self._temperature: int
        self._num_workers: int


    def _truncate_long_messages(self, messages: List[Dict], max_length=-1):
        if max_length == -1:
            max_length = self._max_length

        for msg in messages:
            if len(msg['content']) < 2 * max_length:
                continue

            with self._lock:
                msg['content'] = batch_truncate(
                    [msg['content']], tokenizer=self._tokenizer, max_length=max_length, truncate_from_middle=True
                )[0]


    def make_sub_querys(self, query_results: List[QueryResult]):
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
        sub_querys = self._vllm_client.batch_call_chat(
            messages=inputs,
            temperature=self._temperature
        )

        sub_batch_idx = 0
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    normalized_subquery = _normalize_subquery(sub_querys[sub_batch_idx])
                    chain_result._sub_querys.append(normalized_subquery)
                    sub_batch_idx += 1
    

    def make_doc_ids(self, query_results: List[QueryResult]):
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    retriever_results: List[Dict] = search_by_http(query=chain_result._sub_querys[-1])
                    
                    doc_ids = [res['id'] for res in retriever_results]
                    documents = [format_input_context(self._corpus[int(doc_id)]) for doc_id in doc_ids][::-1]
                    chain_result._doc_ids.append(doc_ids)
                    chain_result._documents.append(documents)

    
    def make_sub_answers(self, query_results: List[QueryResult], temperature=0.0, max_tokens=128):
        inputs = []
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    sub_answer_prompt = get_generate_intermediate_answer_prompt(
                            subquery=chain_result._sub_querys[-1],
                            documents=chain_result._doc_ids[-1],
                    )
                    self._truncate_long_messages(sub_answer_prompt)
                    inputs.append(sub_answer_prompt)

        sub_answers = self._vllm_client.batch_call_chat(
            messages=inputs,
            temperature=temperature, 
            max_tokens=max_tokens
        )
        
        sub_batch_idx = 0
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    sub_answer = sub_answers[sub_batch_idx]
                    chain_result._sub_answers.append(sub_answer)
                    sub_batch_idx += 1


    def generate_final_answer(self, corag_sample: RagPath, task_desc: str, documents: Optional[List[str]], max_length: int, **kwargs) -> str:
        messages: List[Dict] = get_generate_final_answer_prompt(
            query=corag_sample.query,
            past_subqueries=corag_sample.past_subqueries or [],
            past_subanswers=corag_sample.past_subanswers or [],
            task_desc=task_desc,
            documents=documents,
        )
        self._truncate_long_messages(messages, max_length=max_length)

        final_answer = self._vllm_client.call_chat(messages=messages, **kwargs)
        temp = self._vllm_client.call_chat(messages=messages, max_tokens=1, return_str=False, extra_body={"prompt_logprobs": 1})
        log_probs: List[float] = parse_answer_logprobs(temp)

        return final_answer, log_probs
    
    
    def predict_final_answer(self, path, temperature=0.0, max_tokens=128, topk=20):
        args = Arguments()
        retriever_results: List[Dict] = search_by_http(query=path.query, topk=topk)
        context_doc_ids: List[str] = [res['id'] for res in retriever_results]
        
        documents: List[str] = format_documents_for_final_answer(
            args=args,
            context_doc_ids=context_doc_ids,
            tokenizer=self._tokenizer, corpus=self._corpus,
            lock=self._lock
        )

        final_answer, log_probs = self.generate_final_answer(
            corag_sample=path,
            task_desc=self._task_desc,
            documents=documents,
            max_length=args.max_len,
            temperature=temperature, max_tokens=max_tokens
        )

        return final_answer, log_probs


    def check_final_answers(self, query_results: List[QueryResult]):
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                if not chain_result._is_stop:
                    rag_path = RagPath(
                        query_id=query_result._query_id,
                        query=query_result._query,
                        past_subqueries=chain_result._sub_querys,
                        past_subanswers=chain_result._sub_answers,
                        past_doc_ids=chain_result._doc_ids,
                        past_finalanswers=chain_result._final_answers,
                        answer=query_result._answer
                    )

                    final_answer, log_probs = self.predict_final_answer(rag_path)
                    chain_result._final_answers.append(final_answer)
                    chain_result._log_probs_list.append(log_probs)
                    chain_result._scores.append(sum(log_probs) / len(log_probs))

                    if _normalize_answer(final_answer, punc_chars=string.punctuation, punc_repl="") == _normalize_answer(query_result._answer, punc_chars=string.punctuation, punc_repl=""):
                        chain_result._is_stop = True
    

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


    def batch_chain_generate(
            self, task_desc: str,
            datas: list,
            n_chains: int,
            chain_depth: int,
            max_length=4096,
            temperature=0.7,
            num_workers=4,
            **kwargs
    ) -> List[QueryResult]:
        self._task_desc = task_desc
        self._n_chains = n_chains
        self._chain_depth = chain_depth
        self._max_length = max_length
        self._temperature = temperature
        self._num_workers = num_workers
        start_time_query = time.time()

        # 각 쿼리에 대한 결과 객체 초기화
        query_results = []
        for data in datas:
            query_result = QueryResult()
            query_result._query_id = data['query_id']
            query_result._query = data['query']
            query_result._answer = data['answers'][0]
            query_result._chain_results = [ChainResult() for _ in range(n_chains)]
            query_results.append(query_result)
        
        # 각 depth마다 서브쿼리 생성 및 처리
        for depth in range(chain_depth):
            print(f"{'='*50}\nDepth {depth} Start\n{'='*50}\n")
            start_time_depth = time.time()

            # 서브 쿼리 생성
            self.make_sub_querys(query_results)

            # # 문서 검색
            self.make_doc_ids(query_results)

            # 서브 답변 생성
            self.make_sub_answers(query_results)

            # 서브 스텝 마다, 최종 답변 확인
            self.check_final_answers(query_results)

            print(f"{depth} 번째 Depth 종료, 총 경과 시간: {time.time() - start_time_depth}")

            # 모든 체인이 중단되었는지 확인
            if self.check_all_stop(query_results):
                break

        return query_results

