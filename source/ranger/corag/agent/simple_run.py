import torch
import re
import threading
import string

from copy import deepcopy
from typing import Optional, List, Dict, Tuple
from datasets import Dataset
from openai.types.chat import ChatCompletion
from transformers import AutoTokenizer, AutoModelForCausalLM

from search.logger_config import logger
from vllm_client import VllmClient, get_vllm_model_id
from search.search_utils import search_by_http
from data_utils import format_input_context, parse_answer_logprobs
from prompts import get_generate_subquery_prompt, get_generate_intermediate_answer_prompt, get_generate_final_answer_prompt
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
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=True,
                temperature=temp,
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
    
    def generate_subanswer(
            self, subquery: str, documents: List[str], max_message_length: int = 4096
    ) -> Tuple[str, List]:
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
    
    def compute_log_likelihood(
            self, chain: RagPath,
            answers: List[str]
    )-> float:
        """
        Compute log P(answer | query, subquery_{1:L}, subanswer_{1:L})
        """

        query:str = chain.pop('query', '')
        subqueries: List[str] = chain.pop('past_subqueries', [])
        subanswers: List[str] = chain.pop('past_subanswers', [])
        answers: str = " ".join(answers).strip()

        inputs = [f"query: {query}\n"]
        for i, (q, a) in enumerate(zip(subqueries, subanswers)):
            inputs.append(f"subquery: {q}\n")
            inputs.append(f"subanswer: {a}\n")

        prompt = "".join(inputs)
        full_prompt = prompt + f"answer: {answers}"
        
        messages = get_generate_final_answer_prompt(
            query=chain.query,
            past_subqueries=chain.past_subqueries,
            past_subanswers=chain.past_subanswers,
            task_desc=self.task_desc,
            documents=None # 선택적으로 넣을 수 있기는 함
        )

        prompt = self.apply_chat_template(messages)

        full_input = prompt + answer

        
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
        "index_dir": "/home/stv10121/saltlux/data/e5-large-index",
        "task_desc": "answer multi-hop questions",
    }

    # retriever = E5Searcher(index_dir=config["index_dir"])
    # RTX 8000 (48G) 기준으로 27개 정도 올라감 -> generate할 때, 최소 20기가가 올라갈텐데,, 이게 커버가 가능할까?
    corag_agent = CoRagAgent(
        model_name=config["model_name"],
        task_desc=config["task_desc"]
    )

    ragpath_gpt = RagPath(
        query="Say Yes features a former member of Destiny's Child who appeared in what 2002 movie?",
        past_subqueries=[
            "Who is the former Destiny's Child member featured in Say Yes?",
        ],
        past_subanswers=[
            "Michelle Williams, a former member of Destiny's Child, is featured in 'Say Yes' along with Beyoncé and Kelly Rowland.",
        ],
        past_doc_ids=None
    )
    
    for data in load_dataset("corag/multihopqa", "hotpotqa", split="train"):
        if data["query_id"]:
            ragpath_gpt = RagPath(
            query="Say Yes features a former member of Destiny's Child who appeared in what 2002 movie?",
            past_subqueries=[
                "Who is the former Destiny's Child member featured in Say Yes?",
            ],
            past_subanswers=None,
            past_doc_ids=data["context_doc_ids"]
            )

            




        

if __name__ == "__main__":
    main()