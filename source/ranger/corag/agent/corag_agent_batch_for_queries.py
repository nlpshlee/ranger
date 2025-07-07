import re
import string
import threading
import time
from itertools import chain
from copy import deepcopy
from typing import Optional, List, Dict, Tuple
from datasets import Dataset
from openai.types.chat import ChatCompletion
from ranger.corag.inference.qa_utils import _normalize_answer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from logger_config import logger
from vllm_client import VllmClient, get_vllm_model_id
from search.search_utils import search_by_http
from data_utils import format_documents_for_final_answer, format_input_context, parse_answer_logprobs
from prompts import get_generate_subquery_prompt, get_generate_intermediate_answer_prompt, get_generate_final_answer_prompt
from agent.agent_utils import RagPath
from agent.agent_utils_2 import RagPath_2
from utils import batch_truncate
from ranger.corag.data_utils import load_corpus

class QueryResult:
    def __init__(self) -> None:
        self._query_idx = -1
        self._query = ''
        self._answer = ''
        self._chain_results = []

class ChainResult:
    def __init__(self) -> None:
        self._sub_querys = []
        self._sub_query_set = set()
        self._sub_answers = []
        self._doc_ids = []
        self._documents = []
        self._is_stop = False

tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
lock = threading.Lock()        
def _truncate_long_messages(messages: List[Dict], max_length: int):
    for msg in messages:
        if len(msg['content']) < 2 * max_length:
            continue

        with lock:
            msg['content'] = batch_truncate(
                [msg['content']], tokenizer=tokenizer, max_length=max_length, truncate_from_middle=True
            )[0]

def _make_sub_querys(query_results: List[QueryResult], vllm_client):
    inputs = []
    for query_result in query_results:
        for chain_result in query_result._chain_results:
            if not chain_result._is_stop:
                # 서브쿼리 생성 프롬프트 생성
                sub_query_prompt = get_generate_subquery_prompt(
                    query=query_result._query,
                    past_subqueries=chain_result._sub_querys,
                    past_subanswers=chain_result._sub_answers,
                    task_desc="answer multi-hop questions"
                )
                _truncate_long_messages(sub_query_prompt, max_length=4096)
                inputs.append(sub_query_prompt)
    
    # VLLM을 통해 서브쿼리 생성
    sub_querys = vllm_client.batch_call_chat(
        messages=inputs,
        temperature=0.7,
    )
    # sub_querys = vllm_client.new_call_chat(
    #     messages=inputs,
    #     temperature=0.7
    # )
    # print(sub_querys)
    # 생성된 서브쿼리 정규화 및 저장
    sub_batch_idx = 0
    for query_result in query_results:
        for chain_result in query_result._chain_results:
            if not chain_result._is_stop:
                normalized_subquery = _normalize_subquery(sub_querys[sub_batch_idx])
                # if normalized_subquery in chain_result._sub_querys:
                #     print(f'"{normalized_subquery}" is already exist.')
                #     chain_result._is_stop = True
                #     continue
                chain_result._sub_querys.append(normalized_subquery)
                sub_batch_idx += 1

def _make_doc_ids(query_results: List[QueryResult], corpus):
    for query_result in query_results:
        for chain_result in query_result._chain_results:
            if not chain_result._is_stop:
                retriever_results: List[Dict] = search_by_http(query=chain_result._sub_querys[-1])
                
                doc_ids = [res['id'] for res in retriever_results]
                documents = [format_input_context(corpus[int(doc_id)]) for doc_id in doc_ids][::-1]
                chain_result._doc_ids.append(doc_ids)
                chain_result._documents.append(documents)
                # print(chain_result._sub_querys[-1], chain_result._doc_ids[-1])                 

def _make_sub_answers(query_results: List[QueryResult], vllm_client):
    inputs = []
    for query_result in query_results:
        for chain_result in query_result._chain_results:
            if not chain_result._is_stop:
                # print(chain_result._sub_querys, chain_result._doc_ids)
                sub_answer_prompt = get_generate_intermediate_answer_prompt(
                        subquery=chain_result._sub_querys[-1],
                        documents=chain_result._doc_ids[-1],
                    )
                _truncate_long_messages(sub_answer_prompt, max_length=4096)
                inputs.append(sub_answer_prompt)

    sub_answers = vllm_client.batch_call_chat(
        messages=inputs,
        temperature=0., 
        max_tokens=128
    )
    # sub_answers = vllm_client.new_call_chat(
    #     messages=inputs,
    #     temperature=0., 
    #     max_tokens=128
    #     )
    # print(sub_answers)
    
    sub_batch_idx = 0
    for query_result in query_results:
        for chain_result in query_result._chain_results:
            if not chain_result._is_stop:
                sub_answer = sub_answers[sub_batch_idx]
                chain_result._sub_answers.append(sub_answer)
                
                sub_batch_idx += 1

def _check_answers(query_results: List[QueryResult]):
    for query_result in query_results:
        for chain_result in query_result._chain_results:
            if not chain_result._is_stop:
                if _normalize_answer(chain_result._sub_answers[-1], punc_chars=string.punctuation, punc_repl="") == _normalize_answer(query_result._answer, punc_chars=string.punctuation, punc_repl=""):
                    chain_result._is_stop = True
                    # print(f"{chain_result._sub_answers[-1]} == {query_result._answer}")

def _check_final_answers(query_results: List[QueryResult], predict_final_answer):
    # 매 스텝마다 파이널 엔서 생성하고 멈추는 코드
    for query_result in query_results:
        for chain_result in query_result._chain_results:
            if not chain_result._is_stop:
                pred = predict_final_answer(RagPath(
                    query=query_result._query,
                    past_subqueries=chain_result._sub_querys,
                    past_subanswers=chain_result._sub_answers,
                    past_doc_ids=chain_result._doc_ids
                ))
                if _normalize_answer(pred, punc_chars=string.punctuation, punc_repl="") == _normalize_answer(query_result._answer, punc_chars=string.punctuation, punc_repl=""):
                    chain_result._is_stop = True
                    # print(pred, query_result._answer)


def _normalize_subquery(subquery: str) -> str:
    subquery = subquery.strip()
    if subquery.startswith('"') and subquery.endswith('"'):
        subquery = subquery[1:-1]
    if subquery.startswith('Intermediate query'):
        subquery = re.sub(r'^Intermediate query \d+: ', '', subquery)

    return subquery

def unflatten_messages(flat_list: List[List[List[Dict]]], lengths: List[int]) -> List[str]:
    result = []
    index = 0
    for length in lengths:
        result.append(flat_list[index:index + length])
        index += length
    return result

class CoRagAgent:

    def __init__(
            self, vllm_client: VllmClient, corpus: Dataset=load_corpus()
    ):
        self.vllm_client = vllm_client
        self.corpus = corpus
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(vllm_client.model)
        self.lock = threading.Lock()
    
    def batch_sample_path_2(
            self, query_list: List[str], task_desc: str,
            max_path_length: int = 5,
            num_of_chains: int = 5,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            num_workers: int = 4,
            answers: List[str] = None,
            query_ids: List[str] = [],
            **kwargs
    ) -> List[RagPath]:
        
        query_start_time = time.time()
        # 각 쿼리에 대한 결과 객체 초기화
        query_results = []
        for query, answer, query_id in zip(query_list, answers, query_ids):
            query_result = QueryResult()
            query_result._query = query
            query_result._answer = answer[0]
            query_result._query_id= query_id
            query_result._chain_results = [ChainResult() for _ in range(num_of_chains)]
            query_results.append(query_result)
        
        # 각 depth마다 서브쿼리 생성 및 처리
        # for depth in range(max_path_length):
        for depth in range(max_path_length):
            print(f"\n{'='*50}")
            print(f"Depth {depth} 시작")
            print(f"{'='*50}")
            
            start_time = time.time()
            # 서브쿼리 생성
            _make_sub_querys(query_results, self.vllm_client)
            # 생성된 서브쿼리 출력
            # for query_idx, query_result in enumerate(query_results):
            #     print(f"\n쿼리 {query_idx + 1}: {query_result._query}")
            #     for chain_idx, chain_result in enumerate(query_result._chain_results):
            #         if chain_result._sub_querys:
            #             print(f"  체인 {chain_idx + 1}:")
            #             for subq_idx, subq in enumerate(chain_result._sub_querys):
            #                 print(f"    서브쿼리 {subq_idx + 1}: {subq}")
            
            _make_doc_ids(query_results, self.corpus)
            _make_sub_answers(query_results, self.vllm_client)
            
            if False:
                _check_answers(query_results)
            else:
                _check_final_answers(query_results, self.predict_final_answer)
            
            
            # print(f"Depth 종료 시간: {time.time() - start_time}")
            print(f"{depth} 번째 Depth 종료, 총 경과 시간: {time.time() - query_start_time}")
            # for query_idx, query_result in enumerate(query_results):
            #     print(f"\n쿼리 {query_idx + 1}: {query_result._query}, 정답 {query_result._answer}")
            #     for chain_idx, chain_result in enumerate(query_result._chain_results):
            #         if chain_result._sub_querys and chain_result._sub_answers:
            #             print(f"  체인 {chain_idx + 1}:")
            #             for subq_idx, (subq, suba) in enumerate(zip(chain_result._sub_querys, chain_result._sub_answers)):
            #                 print(f"    서브쿼리 {subq_idx + 1}: {subq}")
            #                 print(f"    서브답변 {subq_idx + 1}: {suba}")
            #                 if chain_result._doc_ids and len(chain_result._doc_ids) > subq_idx:
            #                     print(f"    검색된 문서 ID: {chain_result._doc_ids[subq_idx]}")
            #                 print()
            
            # break  # 테스트를 위해 첫 번째 depth만 실행
            
            # 모든 체인이 중단되었는지 확인
            all_stopped = True
            for query_result in query_results:
                for chain_result in query_result._chain_results:
                    if not chain_result._is_stop:
                        all_stopped = False
                        break
                if not all_stopped:
                    break
            
            if all_stopped:
                break
        
        # 최종 RagPath 객체 생성
        paths = []
        for query_result in query_results:
            for chain_result in query_result._chain_results:
                path = RagPath_2(
                    query=query_result._query,
                    query_id=query_result._query_id,
                    answer=query_result._answer,
                    past_subqueries=chain_result._sub_querys,
                    past_subanswers=chain_result._sub_answers,
                    past_doc_ids=chain_result._doc_ids if hasattr(chain_result, '_doc_ids') else []
                )
                paths.append(path)
        
        return paths
    
    def batch_sample_path(
            self, query_list: str, task_desc: str,
            max_path_length: int = 5,
            num_of_chains: int = 5,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            num_workers: int = 4,
            answers = None,
            **kwargs
    ) -> List[RagPath]:
        def retrieval_doc_ids_by_subqueries(subqueries_per_query: List[List[str]]) -> Tuple[List[List[List[str]]], List[List[str]]]:
            retrieved_doc_ids_by_query: List[List[List[str]]] = []
            retrieved_documents_by_query: List[List[str]] = []
            
            for query_subqueries in subqueries_per_query:
                if not query_subqueries:
                    retrieved_doc_ids_by_query.append([])
                    retrieved_documents_by_query.append([])
                    continue
                    
                search_results_by_subquery: List[List[Dict]] = [search_by_http(query=subquery) for subquery in query_subqueries]
                doc_ids_by_subquery: List[List[str]] = []
                documents_by_subquery: List[List[str]] = []
                
                for subquery_search_results in search_results_by_subquery:
                    retrieved_doc_ids: List[str] = [result['id'] for result in subquery_search_results]
                    formatted_documents: List[str] = [format_input_context(self.corpus[int(doc_id)]) for doc_id in retrieved_doc_ids][::-1]
                    doc_ids_by_subquery.append(retrieved_doc_ids)
                    documents_by_subquery.append(formatted_documents)
                    
                retrieved_doc_ids_by_query.append(doc_ids_by_subquery)
                retrieved_documents_by_query.append(documents_by_subquery)
                
            return retrieved_doc_ids_by_query, retrieved_documents_by_query
        
        queries_length = len(query_list)
        query_results = [QueryResult() for _ in range(queries_length)]
                
        past_subqueries_list: List[List[List[str]]] = [[[] for _ in range(num_of_chains)] for _ in range(queries_length)]
        past_subanswers_list: List[List[List[str]]] = [[[] for _ in range(num_of_chains)] for _ in range(queries_length)]
        past_doc_ids_list: List[List[List[List[str]]]] = [[[] for _ in range(num_of_chains)] for _ in range(queries_length)]

        active_chains: List[List[bool]] = [[True] * num_of_chains for _ in range(queries_length)]
        
        
        for depth in range(max_path_length*4):
            depth_start_time = time.time()
            
            # A. 활성화된 체인에 대해서 서브쿼리 프롬프트 생성
            subquery_prompts_per_query: List[List[List[Dict]]] = []
            for query_idx, query in enumerate(query_list):
                subquery_prompts_per_chain: List[List[Dict]] = []
                for chain_idx in range(num_of_chains):
                    if not active_chains[query_idx][chain_idx]:
                        subquery_prompts_per_chain.append([])
                        continue
                    messages = get_generate_subquery_prompt(
                        query=query,
                        past_subqueries=past_subqueries_list[query_idx][chain_idx],
                        past_subanswers=past_subanswers_list[query_idx][chain_idx],
                        task_desc=task_desc,
                    )
                    self._truncate_long_messages(messages, max_length=max_message_length)
                    subquery_prompts_per_chain.append(messages)
                subquery_prompts_per_query.append(subquery_prompts_per_chain)
            # b. 서브쿼리 프롬프트 flatten
            flat_subqueries = list(chain.from_iterable(subquery_prompts_per_query))
            # c. vllm에 서브쿼리 요청
            generated_subqueries: List[str] = self.vllm_client.batch_call_chat(
                messages=flat_subqueries,
                temperature=temperature,
                num_workers=num_workers,
                **kwargs
            )
            # normalize
            generated_subqueries = [_normalize_subquery(sq) for sq in generated_subqueries]
            
            # d. 받은 응답 unflatten
            num_chains_per_query = [len(chains) for chains in subquery_prompts_per_query]
            subqueries_per_query = unflatten_messages(generated_subqueries, num_chains_per_query)
            
            retrieved_doc_ids_by_query, retrieved_documents_by_query = retrieval_doc_ids_by_subqueries(subqueries_per_query)
            
            # 서브답변 프롬프트 생성
            subanswer_prompts_per_query: List[List[List[Dict]]] = []
            for query_idx, (subqueries, documents) in enumerate(zip(subqueries_per_query, retrieved_documents_by_query)):
                subanswer_prompts_per_chain: List[List[Dict]] = []
                for chain_idx, (subquery, docs) in enumerate(zip(subqueries, documents)):
                    if not active_chains[query_idx][chain_idx]:
                        subanswer_prompts_per_chain.append([])
                        continue
                    messages: List[Dict] = get_generate_intermediate_answer_prompt(
                        subquery=subquery,
                        documents=docs,
                    )
                    self._truncate_long_messages(messages, max_length=max_message_length)
                    subanswer_prompts_per_chain.append(messages)
                subanswer_prompts_per_query.append(subanswer_prompts_per_chain)

            # flatten
            flat_subanswers = list(chain.from_iterable(subanswer_prompts_per_query))

            # predict
            generated_subanswers: List[str] = self.vllm_client.batch_call_chat(
                messages=flat_subanswers,
                temperature=0.,
                max_tokens=128,
                num_workers=num_workers
            )
            # Unflatten
            num_chains_per_query = [len(chains) for chains in subanswer_prompts_per_query]
            subanswers_per_query = unflatten_messages(generated_subanswers, num_chains_per_query)

            # 각 쿼리와 체인에 대해 서브쿼리와 서브답변 저장
            for query_idx, (subqueries, subanswers, doc_ids_list) in enumerate(zip(subqueries_per_query, subanswers_per_query, retrieved_doc_ids_by_query)):
                # print(f"Query {query_idx}, Answer: {answers[query_idx][0]}")
                for chain_idx, (subquery, subanswer, doc_ids) in enumerate(zip(subqueries, subanswers, doc_ids_list)):
                    if not active_chains[query_idx][chain_idx]:
                        continue
                    # 중복 체크
                    if subquery in past_subqueries_list[query_idx][chain_idx]:
                        print(f"중복된 쿼리 생성. {subquery}")
                        # active는 그대로 True, append하지 않음 (다음 depth에서 다시 시도)
                        continue
                    # 중복이 아니면 append 및 active 처리
                    past_subqueries_list[query_idx][chain_idx].append(subquery)
                    past_subanswers_list[query_idx][chain_idx].append(subanswer)
                    past_doc_ids_list[query_idx][chain_idx].append(doc_ids)
                    
                    # answer와 일치하는지 확인
                    if _normalize_answer(subanswer, punc_chars=string.punctuation, punc_repl="") == _normalize_answer(answers[query_idx][0], punc_chars=string.punctuation, punc_repl=""):
                        active_chains[query_idx][chain_idx] = False
                    if len(past_subqueries_list[query_idx][chain_idx]) >= max_path_length:
                        active_chains[query_idx][chain_idx] = False
            #         print(f" Chain {chain_idx} - {past_subanswers_list[query_idx][chain_idx]}")
            #     print("-"*50)
            # print("\n"*3)
            
            depth_end_time = time.time()
            depth_time = depth_end_time - depth_start_time
            print(f"Depth {depth} 전체 실행 시간: {depth_time:.2f}초")

            # 모든 체인이 비활성화되었는지 확인
            if all(not any(chains) for chains in active_chains):
                break
        paths: List[RagPath] = []
        for query_idx, query in enumerate(query_list):
            for chain_idx in range(num_of_chains):
                paths.append(RagPath(
                    query=query,
                    past_subqueries=past_subqueries_list[query_idx][chain_idx],
                    past_subanswers=past_subanswers_list[query_idx][chain_idx],
                    past_doc_ids=past_doc_ids_list[query_idx][chain_idx],
                ))
        
        return paths
    
    def sample_path(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            **kwargs
    ) -> RagPath:
        past_subqueries: List[str] = kwargs.pop('past_subqueries', [])
        past_subanswers: List[str] = kwargs.pop('past_subanswers', [])
        past_doc_ids: List[List[str]] = kwargs.pop('past_doc_ids', [])
        assert len(past_subqueries) == len(past_subanswers) == len(past_doc_ids)

        subquery_temp: float = temperature
        num_llm_calls: int = 0
        max_num_llm_calls: int = 4 * (max_path_length - len(past_subqueries))
        while len(past_subqueries) < max_path_length and num_llm_calls < max_num_llm_calls:
            num_llm_calls += 1
            messages: List[Dict] = get_generate_subquery_prompt(
                query=query,
                past_subqueries=past_subqueries,
                past_subanswers=past_subanswers,
                task_desc=task_desc,
            )
            self._truncate_long_messages(messages, max_length=max_message_length)

            subquery: str = self.vllm_client.call_chat(messages=messages, temperature=subquery_temp, **kwargs)
            subquery = _normalize_subquery(subquery)

            if subquery in past_subqueries:
                subquery_temp = max(subquery_temp, 0.7)
                continue

            subquery_temp = temperature
            subanswer, doc_ids = self._get_subanswer_and_doc_ids(
                subquery=subquery, max_message_length=max_message_length
            )

            past_subqueries.append(subquery)
            past_subanswers.append(subanswer)
            past_doc_ids.append(doc_ids)

        return RagPath(
            query=query,
            past_subqueries=past_subqueries,
            past_subanswers=past_subanswers,
            past_doc_ids=past_doc_ids,
        )


    def generate_final_answer(
            self, corag_sample: RagPath, task_desc: str,
            max_message_length: int = 4096,
            documents: Optional[List[str]] = None, **kwargs
    ) -> str:
        messages: List[Dict] = get_generate_final_answer_prompt(
            query=corag_sample.query,
            past_subqueries=corag_sample.past_subqueries or [],
            past_subanswers=corag_sample.past_subanswers or [],
            task_desc=task_desc,
            documents=documents,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        return self.vllm_client.call_chat(messages=messages, **kwargs)

    def predict_final_answer(self, path):
        from src.config import Arguments
        import threading
        
        args = Arguments(output_dir="/home/softstone/saltlux/data/output")
        tokenizer_lock: threading.Lock = threading.Lock()
        
        retriever_results: List[Dict] = search_by_http(query=path.query, topk=20)
        context_doc_ids: List[str] = [res['id'] for res in retriever_results]
        
        documents: List[str] = format_documents_for_final_answer(
            args=args,
            context_doc_ids=context_doc_ids,
            tokenizer=self.tokenizer, corpus=self.corpus,
            lock=tokenizer_lock
        )

        prediction: str = self.generate_final_answer(
            corag_sample=path,
            task_desc="answer multi-hop questions",
            documents=documents,
            max_message_length=args.max_len,
            temperature=0., max_tokens=128
        )
        return prediction
    
    def _truncate_long_messages(self, messages: List[Dict], max_length: int):
        for msg in messages:
            if len(msg['content']) < 2 * max_length:
                continue

            with self.lock:
                msg['content'] = batch_truncate(
                    [msg['content']], tokenizer=self.tokenizer, max_length=max_length, truncate_from_middle=True
                )[0]

    def sample_subqueries(self, query: str, task_desc: str, n: int = 10, max_message_length: int = 4096, **kwargs) -> List[str]:
        messages: List[Dict] = get_generate_subquery_prompt(
            query=query,
            past_subqueries=kwargs.pop('past_subqueries', []),
            past_subanswers=kwargs.pop('past_subanswers', []),
            task_desc=task_desc,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        completion: ChatCompletion = self.vllm_client.call_chat(messages=messages, return_str=False, n=int(1.5 * n), **kwargs)
        subqueries: List[str] = [_normalize_subquery(c.message.content) for c in completion.choices]
        subqueries = list(set(subqueries))[:n]

        return subqueries

    def _get_subanswer_and_doc_ids(
            self, subquery: str, max_message_length: int = 4096
    ) -> Tuple[str, List]:
        retriever_results: List[Dict] = search_by_http(query=subquery)
        # doc_ids: List[str] = [res['doc_id'] for res in retriever_results]
        doc_ids: List[str] = [res['id'] for res in retriever_results]
        documents: List[str] = [format_input_context(self.corpus[int(doc_id)]) for doc_id in doc_ids][::-1]
        
        messages: List[Dict] = get_generate_intermediate_answer_prompt(
            subquery=subquery,
            documents=documents,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        subanswer: str = self.vllm_client.call_chat(messages=messages, temperature=0., max_tokens=128)
        return subanswer, doc_ids


    def tree_search(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            expand_size: int = 4, num_rollouts: int = 2, beam_size: int = 1,
            **kwargs
    ) -> RagPath:
        return self._search(
            query=query, task_desc=task_desc,
            max_path_length=max_path_length,
            max_message_length=max_message_length,
            temperature=temperature,
            expand_size=expand_size, num_rollouts=num_rollouts, beam_size=beam_size,
            **kwargs
        )

    def best_of_n(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            n: int = 4,
            **kwargs
    ) -> RagPath:
        sampled_paths: List[RagPath] = []
        for idx in range(n):
            path: RagPath = self.sample_path(
                query=query, task_desc=task_desc,
                max_path_length=max_path_length,
                max_message_length=max_message_length,
                temperature=0. if idx == 0 else temperature,
                **kwargs
            )
            sampled_paths.append(path)

        scores: List[float] = [self._eval_single_path(p) for p in sampled_paths]
        return sampled_paths[scores.index(min(scores))]

    def _search(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            expand_size: int = 4, num_rollouts: int = 2, beam_size: int = 1,
            **kwargs
    ) -> RagPath:
        candidates: List[RagPath] = [RagPath(query=query, past_subqueries=[], past_subanswers=[], past_doc_ids=[])]
        for step in range(max_path_length):
            new_candidates: List[RagPath] = []
            for candidate in candidates:
                new_subqueries: List[str] = self.sample_subqueries(
                    query=query, task_desc=task_desc,
                    past_subqueries=deepcopy(candidate.past_subqueries),
                    past_subanswers=deepcopy(candidate.past_subanswers),
                    n=expand_size, temperature=temperature, max_message_length=max_message_length
                )
                for subquery in new_subqueries:
                    new_candidate: RagPath = deepcopy(candidate)
                    new_candidate.past_subqueries.append(subquery)
                    subanswer, doc_ids = self._get_subanswer_and_doc_ids(
                        subquery=subquery, max_message_length=max_message_length
                    )
                    new_candidate.past_subanswers.append(subanswer)
                    new_candidate.past_doc_ids.append(doc_ids)
                    new_candidates.append(new_candidate)

            if len(new_candidates) > beam_size:
                scores: List[float] = []
                for path in new_candidates:
                    score = self._eval_state_without_answer(
                        path, num_rollouts=num_rollouts,
                        task_desc=task_desc,
                        max_path_length=max_path_length,
                        temperature=temperature,
                        max_message_length=max_message_length
                    )
                    scores.append(score)

                # lower scores are better
                new_candidates = [c for c, s in sorted(zip(new_candidates, scores), key=lambda x: x[1])][:beam_size]

            candidates = new_candidates

        return candidates[0]

    def _eval_single_path(self, current_path: RagPath, max_message_length: int = 4096) -> float:
        messages: List[Dict] = get_generate_intermediate_answer_prompt(
            subquery=current_path.query,
            documents=[f'Q: {q}\nA: {a}' for q, a in zip(current_path.past_subqueries, current_path.past_subanswers)],
        )
        messages.append({'role': 'assistant', 'content': 'No relevant information found'})
        self._truncate_long_messages(messages, max_length=max_message_length)

        response: ChatCompletion = self.vllm_client.call_chat(
            messages=messages,
            return_str=False,
            max_tokens=1,
            extra_body={
                "prompt_logprobs": 1
            }
        )
        answer_logprobs: List[float] = parse_answer_logprobs(response)

        return sum(answer_logprobs) / len(answer_logprobs)

    def _eval_state_without_answer(
            self, path: RagPath, num_rollouts: int, task_desc: str,
            max_path_length: int = 3,
            temperature: float = 0.7,
            max_message_length: int = 4096
    ) -> float:
        assert len(path.past_subqueries) > 0
        if num_rollouts <= 0:
            return self._eval_single_path(path)

        rollout_paths: List[RagPath] = []
        for _ in range(num_rollouts):
            rollout_path: RagPath = self.sample_path(
                query=path.query, task_desc=task_desc,
                max_path_length=min(max_path_length, len(path.past_subqueries) + 2), # rollout at most 2 steps
                temperature=temperature, max_message_length=max_message_length,
                past_subqueries=deepcopy(path.past_subqueries),
                past_subanswers=deepcopy(path.past_subanswers),
                past_doc_ids=deepcopy(path.past_doc_ids),
            )
            rollout_paths.append(rollout_path)

        scores: List[float] = [self._eval_single_path(p) for p in rollout_paths]
        # TODO: should we use the min score instead?
        return sum(scores) / len(scores)