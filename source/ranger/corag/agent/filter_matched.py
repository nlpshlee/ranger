import torch
import json
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
from agent.new_corag_agent import *
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

    out_file_path = f"/workspace/yeomsee/corag/samples/pred_gold_matched_{len(sampled_dataset)}.jsonl"
    for _, data in enumerate(tqdm(sampled_dataset, total=len(sampled_dataset))):
        # extract
        query_id = data["query_id"]
        query = data["query"]
        answers = " ".join(data["answers"])
        # context_doc_ids = data["context_doc_ids"]
        subqueries = data["subqueries"]
        subanswers = data["subanswers"]

        corag_path = RagPath(
            query=query,
            past_subqueries=subqueries,
            past_subanswers=subanswers,
            past_docs=[],
            chain_length=len(subqueries),
        )
        
        doc_ids_and_scores = retriever.batch_search(subqueries, k=5)
        for i in range(len(doc_ids_and_scores)):
            doc_ids = [int(doc_id_and_score["doc_id"]) for doc_id_and_score in doc_ids_and_scores[i]]
            documents = [retriever.corpus[doc_id] for doc_id in doc_ids]
            corag_path.past_docs.append(documents)

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

        answer_pred = corag_agent.generate_final_answer(
            corag_path,
            max_message_length=4096,
            documents=corag_path.past_docs
        )

        if _normalize_answer(answer_pred) == _normalize_answer(answers):
            result_dict = {
                "query_id": query_id,
                "query": query,
                "subqueries": subqueries,
                "subanswers": subanswers,
                "final_answer": answer_pred
            }

            with open(out_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()