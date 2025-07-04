import torch
import re
import threading
import string
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from copy import deepcopy
from typing import Optional, List, Dict, Tuple
from datasets import Dataset
from openai.types.chat import ChatCompletion
from transformers import AutoTokenizer, AutoModelForCausalLM

from search.logger_config import logger
from vllm_client import VllmClient, get_vllm_model_id
from search.search_utils import search_by_http
from data_utils import format_input_context, parse_answer_logprobs
from prompts import *
from agent.agent_utils import RagPath
from search.e5_searcher import E5Searcher
from utils import batch_truncate

from datasets import load_dataset

def _normalize_answer(text, punc_chars=string.punctuation, punc_repl=""):
    """Lower text and remove punctuation, articles and extra whitespace."""

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

class CoRagAgent:
    def __init__(
            self, model_name: str,
            task_desc: str,
            corpus: Dataset = load_dataset("corag/kilt-corpus", split="train"),
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.corpus = corpus
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        self.task_desc = task_desc
        self.device = device
        self.lock = threading.Lock()
    
    def extract_last_assistant_response(self, decoded: str) -> str:
        # find all assistant responses
        assistant_chunks = re.findall(
            r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>",
            decoded,
            re.DOTALL,
        )

        # extract last assistant response
        return assistant_chunks[-1].strip() if assistant_chunks else decoded.strip()
    
    def apply_chat_template(
            self, messages: List[Dict]
    )-> str:
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True, # 모델 응답 유도
            tokenize=False # 토큰화 X (문자열 출력)
        )

    def get_response_from_messages(
            self, messages: List[Dict], temp=0.7
    ) -> str:
        prompt = self.apply_chat_template(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            if temp > 0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=True,
                    temperature=temp,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=False
        )

        output = self.extract_last_assistant_response(response)
        return output
    
    def _truncate_long_messages(self, messages: List[Dict], max_length: int):
        for msg in messages:
            if len(msg['content']) < 2 * max_length:
                continue

            with self.lock:
                msg['content'] = batch_truncate(
                    [msg['content']], tokenizer=self.tokenizer, max_length=max_length, truncate_from_middle=True
                )[0]
    
    def generate_subquery(
            self, query: str,
            max_message_length: int = 4096, **kwargs
    ) -> List[str]:
        messages: List[Dict] = get_generate_subquery_prompt(
            query=query,
            past_subqueries=kwargs.pop('past_subqueries', []),
            past_subanswers=kwargs.pop('past_subanswers', []),
            task_desc=self.task_desc,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        subquery: str = self.get_response_from_messages(messages=messages, temp=0.7)
        return self._normalize_subquery(subquery=subquery)
    
    def _normalize_subquery(self, subquery: str) -> str:
        subquery = subquery.strip()
        if subquery.startswith('"') and subquery.endswith('"'):
            subquery = subquery[1:-1]
        if subquery.startswith('Intermediate query'):
            subquery = re.sub(r'^Intermediate query \d+: ', '', subquery)

        return subquery
    
    def generate_subanswer(
            self, subquery: str, documents: List[str], max_message_length: int = 4096
    ) -> str:
        # retriever_results: List[Dict] = search_by_http(query=subquery)
        # doc_ids: List[str] = [res['doc_id'] for res in retriever_results]
        # documents: List[str] = [format_input_context(self.corpus[int(doc_id)]) for doc_id in doc_ids][::-1]

        messages: List[Dict] = get_generate_intermediate_answer_prompt(
            subquery=subquery,
            documents=documents,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        subanswer: str = self.get_response_from_messages(messages=messages, temp=0.)
        return subanswer
    
    def generate_final_answer(
            self, corag_sample: RagPath,
            max_message_length: int = 4096,
            documents: Optional[List[str]] = None
    ) -> str:
        messages: List[Dict] = get_generate_final_answer_prompt(
            query=corag_sample.query,
            past_subqueries=corag_sample.past_subqueries or [],
            past_subanswers=corag_sample.past_subanswers or [],
            task_desc=self.task_desc,
            documents=documents,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        return self.get_response_from_messages(messages=messages, temp=0.)

    def decide_stop_or_not(
        self, query: str,
        subqueries: List[str], subanswers: List[str], max_message_length:int = 4096
    )-> str:
        messages: List[Dict] = decide_stop_or_not_prompt(
            query=query,
            past_subqueries=subqueries,
            past_subanswers=subanswers
        )
        
        self._truncate_long_messages(messages=messages, max_length=max_message_length)
        return self.get_response_from_messages(messages=messages, temp=0.)
    
    def compute_log_likelihood(
            self, chain: RagPath,
            answers: List[str]
    )-> float:
        """
        Compute log P(answer | query, subquery_{1:L}, subanswer_{1:L})
        """

        query:str = chain.query
        subqueries: List[str] = chain.past_subqueries
        subanswers: List[str] = chain.past_subanswers
        answers: str = " ".join(answers).strip()

        context = f"query: {query}\n"
        for i, (q, a) in enumerate(zip(subqueries, subanswers)):
            context += f"subquery: {q}\n"
            context += f"subanswer: {a}\n"

        # 전체 입력 구성 + 토큰화
        input_text = context + "\nfinal answer: " + answers
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # 정답 answer만 토큰화해서 시작 위치 찾기
        context_ids = self.tokenizer(context + "\nfinal answer: ", return_tensors="pt").input_ids
        context_len = context_ids.shape[1]

        # context 부분은 마스킹 처리된 label 생성
        labels = input_ids.clone().to(self.device)
        labels[0, :context_len] = -100

        with torch.no_grad():
            outputs = self.model(input_ids, labels=labels)
            log_likelihood = -outputs.loss

        return round(log_likelihood.item(), 4)

    def sample_path(
            self, query: str,
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
                task_desc=self.task_desc,
            )
            self._truncate_long_messages(messages, max_length=max_message_length)

            subquery: str = self.get_response_from_messages(messages=messages)
            subquery = self._normalize_subquery(subquery)

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

    def tree_search(
            self, query: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            expand_size: int = 4, num_rollouts: int = 2, beam_size: int = 1,
            **kwargs
    ) -> RagPath:
        return self._search(
            query=query, task_desc=self.task_desc,
            max_path_length=max_path_length,
            max_message_length=max_message_length,
            temperature=temperature,
            expand_size=expand_size, num_rollouts=num_rollouts, beam_size=beam_size,
            **kwargs
        )

    def best_of_n(
            self, query: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            n: int = 4,
            **kwargs
    ) -> RagPath:
        sampled_paths: List[RagPath] = []
        for idx in range(n):
            path: RagPath = self.sample_path(
                query=query, task_desc=self.task_desc,
                max_path_length=max_path_length,
                max_message_length=max_message_length,
                temperature=0. if idx == 0 else temperature,
                **kwargs
            )
            sampled_paths.append(path)

        scores: List[float] = [self._eval_single_path(p) for p in sampled_paths]
        return sampled_paths[scores.index(min(scores))]

    def _search(
            self, query: str,
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
                    query=query, task_desc=self.task_desc,
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
                        task_desc=self.task_desc,
                        max_path_length=max_path_length,
                        temperature=temperature,
                        max_message_length=max_message_length
                    )
                    scores.append(score)

                # lower scores are better
                new_candidates = [c for c, s in sorted(zip(new_candidates, scores), key=lambda x: x[1])][:beam_size]

            candidates = new_candidates

        return candidates[0]
    
def main():
    config = {
        "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "index_dir": "/workspace/yeomsee/corag/data/e5-large-index",
        "task_desc": "answer multi-hop questions",
        "max_chain_length": 6
    }

    retriever = E5Searcher(index_dir=config["index_dir"])
    # RTX 8000 (48G) 기준으로 27개 정도 올라감 -> generate할 때, 최소 20기가가 올라갈텐데,, 이게 커버가 가능할까?
    corag_agent = CoRagAgent(
        model_name=config["model_name"],
        task_desc=config["task_desc"]
    )

    # random_sampling
    dataset = load_dataset("corag/multihopqa", "hotpotqa", split="train")
    sampled_dataset = dataset.shuffle(seed=97).select(range(100))

    correct_num, chain_len_num, correct_num_corag, chain_len_num_corag = 0, 0, 0, 0
    for i, data in enumerate(sampled_dataset):
        query_id = data["query_id"]
        query = data["query"]
        answers = _normalize_answer(" ".join(data["answers"]))
        # context_doc_ids = data["context_doc_ids"]
        subqueries = data["subqueries"]
        subanswers = data["subanswers"]

        corag_path = RagPath(
            query=query,
            past_subqueries=subqueries,
            past_subanswers=subanswers,
            past_docs=None,
            chain_length=len(subqueries),
        )
        
        max_score = float("-inf")
        best_chain = None
        for _ in range(16):
            params = {
            "past_subqueries": [],
            "past_subanswers": [],
            "documents": []
            }

            for chain_len in range(config["max_chain_length"]):
                # generate subquery
                subquery = corag_agent.generate_subquery(
                    query=query,
                    **params
                )
                params["past_subqueries"].append(subquery)
                
                # retrieve top-5 docs
                doc_ids_and_scores = retriever.batch_search([subquery], k=5)[0]
                doc_ids = [int(doc_id_and_score["doc_id"]) for doc_id_and_score in doc_ids_and_scores]
                documents = [retriever.corpus[doc_id] for doc_id in doc_ids]
                params["documents"].append(documents)

                """
                retriever.batch_search([queries], top_k)
                [
                    [
                        {'doc_id': 32758384, 'score': 0.82861328125},
                        {'doc_id': 9461995, 'score': 0.82177734375},
                        {'doc_id': 19642473, 'score': 0.82177734375},
                        {'doc_id': 13911014, 'score': 0.81689453125},
                        {'doc_id': 7412711, 'score': 0.81396484375}
                    ]
                ]
                """

                """
                retriever.corpus[0]
                {
                    'id': '0',
                    'contents': "Academy Award for Best Production Design\nThe Academy Award for Best Pr....",
                    'title': 'Academy Award for Best Production Design',
                    'wikipedia_id': '316'
                }
                """

                # generate subanswer
                subanswer = corag_agent.generate_subanswer(
                    subquery=subquery,
                    documents=documents
                )
                params["past_subanswers"].append(subanswer)

                # check if subanswer == answers -> terminate!
                if _normalize_answer(subanswer) == answers:
                    dynamic_ragpath = RagPath(
                        query=query,
                        past_subqueries=params['past_subqueries'],
                        past_subanswers=params['past_subanswers'],
                        past_docs=params["documents"],
                        chain_length=chain_len+1
                    )
                    break

                # continue chain?
                stop_token = corag_agent.decide_stop_or_not(
                    query=query,
                    subqueries=params['past_subqueries'],
                    subanswers=params['past_subanswers']
                )
                
                if (stop_token.strip().lower() == "yes") or (chain_len+1 == config["max_chain_length"]):
                    dynamic_ragpath = RagPath(
                        query=query,
                        past_subqueries=params['past_subqueries'],
                        past_subanswers=params['past_subanswers'],
                        past_docs=params["documents"],
                        chain_length=chain_len+1
                    )
                    break          
                
            # compute average log likelihood at the end of the chain
            score = corag_agent.compute_log_likelihood(dynamic_ragpath, answers=answers)
            # logger.info(f"score: {score}")

            if score > max_score:
                max_score = score
                best_chain = dynamic_ragpath
            
            if max_score > -0.05:
                break

        logger.info(f"query_id: {query_id}\n")
        logger.info(f"best dynamic chain info")
        best_chain_pred = corag_agent.generate_final_answer(
            best_chain,
            max_message_length=4096,
            documents=best_chain.past_docs
        )
        best_chain_pred = _normalize_answer(best_chain_pred)
        # logger.info(f"score: {max_score}")
        best_chain.print_info()
        logger.info(f"chain length: {best_chain.chain_length}")
        logger.info(f"answer_pred: {best_chain_pred}")
        logger.info(f"answer_gold: {answers}\n")
        
        if best_chain_pred == answers:
            correct_num += 1
        chain_len_num += best_chain.chain_length
        
        logger.info(f"corag_chain.info")
        corag_pred = corag_agent.generate_final_answer(
            corag_path,
            max_message_length=4096,
            documents=corag_path.past_docs
        )
        corag_chain_pred = _normalize_answer(corag_pred)
        corag_path.print_info()
        logger.info(f"chain length: {corag_path.chain_length}")
        logger.info(f"answer_pred: {corag_chain_pred}")
        logger.info(f"answer_gold: {answers}\n")

        if corag_chain_pred == answers:
            correct_num_corag += 1
        chain_len_num_corag += corag_path.chain_length
    
    logger.info(f"dynamic_chain_pred_avg: {correct_num/100:.3f}")
    logger.info(f"dynamic_chain_len_pred_avg: {chain_len_num/100:.3f}\n")

    logger.info(f"corag_chain_pred_avg: {correct_num_corag/100:.3f}")
    logger.info(f"corag_chain_len_pred_avg: {chain_len_num_corag/100:.3f}\n")

if __name__ == "__main__":
    main()