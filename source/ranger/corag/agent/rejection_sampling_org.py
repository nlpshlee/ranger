import torch
import json
import random
import re
import threading
import string
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
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
    sampled_dataset = dataset.shuffle(seed=45).select(range(500))

    for i, data in enumerate(tqdm(sampled_dataset, total=len(sampled_dataset))):
        out_file_path = "/workspace/yeomsee/corag/samples/samples_500.jsonl"
        
        query_id = data["query_id"]
        query = data["query"]
        answers = _normalize_answer(" ".join(data["answers"]))

        result_dict = {
            "query_id": query_id,
            "query": query,
            "chains": [],
            "answer_gold": answers
        }

        for _ in range(16):
            params = {
            "past_subqueries": [],
            "past_subanswers": [],
            "doc_ids": [],
            "documents": []
            }

            for chain_len in range(random.randint(1, 5)):
                # generate subquery
                subquery = corag_agent.generate_subquery(
                    query=query,
                    **params
                )
                params['past_subqueries'].append(subquery)
                
                # retrieve top-5 docs
                doc_ids_and_scores = retriever.batch_search([subquery], k=5)[0]
                doc_ids = [int(doc_id_and_score['doc_id']) for doc_id_and_score in doc_ids_and_scores]
                documents = [retriever.corpus[doc_id] for doc_id in doc_ids]
                params['doc_ids'].append(doc_ids)
                params['documents'].append(documents)

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
                params['past_subanswers'].append(subanswer)

                # check if subanswer == answers -> terminate!
                if _normalize_answer(subanswer) == answers or (chain_len+1) == config['max_chain_length']:
                    break      
                
            # get path
            corag_path = RagPath(
                query=query,
                past_subqueries=params['past_subqueries'],
                past_subanswers=params['past_subanswers'],
                past_doc_ids=params['doc_ids'],
                past_docs=params['documents']
            )

            # compute average log likelihood at the end of the chain
            score = corag_agent.compute_log_likelihood(corag_path, answers=answers)
            # logger.info(f"score: {score}")

            answer_pred = corag_agent.generate_final_answer(
                corag_path,
                max_message_length=4096,
                documents=corag_path.past_docs
            )

            result_dict["chains"].append(
                {
                    "subqueries": corag_path.past_subqueries,
                    "subanswers": corag_path.past_subanswers,
                    "doc_ids": corag_path.past_doc_ids,
                    "answer_pred": answer_pred,
                    "log_likelihood": score
                }
            )

        # save file
        with open(out_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()