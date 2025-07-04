import os
import json
import re
import string
from tqdm import tqdm

import transformers
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer

import torch
import torch.functional as F
from torch import Tensor
from torch.cuda import empty_cache

from typing import List, Dict, Optional
from datasets import load_dataset

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
retriever_id = "intfloat/e5-large-v2"
task_desc = "answer multi-hop questions"
# index_dir = "/home/stv10121/saltlux/data/e5-large-index"
output_path = "/home/stv10121/saltlux/data/pred_gold_not_match_new.jsonl"
dataset_names = ["hotpotqa"]

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

def get_generate_final_answer_prompt(
        query: str, past_subqueries: List[str], past_subanswers: List[str], task_desc: str,
        documents: Optional[List[str]] = None
) -> List[Dict]:

    assert len(past_subqueries) == len(past_subanswers)
    past = ''
    for idx in range(len(past_subqueries)):
        past += f"""Intermediate query {idx+1}: {past_subqueries[idx]}
Intermediate answer {idx+1}: {past_subanswers[idx]}\n"""
    past = past.strip()

    context = ''
    if documents:
        for idx, doc in enumerate(documents):
            context += f"""Doc {idx}: {doc}\n\n"""

    messages: List[Dict] = [
        {
            'role': 'user',
            'content': prompt
        }
    ]
    return messages

if __name__ == "__main__":
    # retriever = E5Searcher(index_dir=index_dir)
    retriever = SentenceTransformer(retriever_id)

    model_tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        tokenizer=model_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    existing_ids = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                existing_ids.append(obj["query_id"])

    with open(output_path, 'a', encoding='utf-8') as out_file:
        for dataset_name in dataset_names:
            # load qa only dataset
            datas = load_dataset("corag/multihopqa", dataset_name, split="train")

            # load knowledge source
            corpus = load_dataset("corag/kilt-corpus", split="train")
            # print(f"data_sample.keys(): {datas[0].keys()}")
            # dict_keys(['query', 'answers', 'query_id', 'context_doc_ids', 'subqueries', 'subanswers', 'predictions'])
            # print(f"corpus_sample.keys(): {corpus[0].keys()}")
            # dict_keys(['id', 'contents', 'title', 'wikipedia_id'])

            num=0
            for i, data in tqdm(enumerate(datas), total=len(datas)):
                # parsing
                query_id = data["query_id"]
                if query_id in existing_ids:
                    continue
                try:
                    query = data["query"]
                    answers = [_normalize_answer(answer) for answer in data["answers"]]
                    
                    subqueries_gold = data["subqueries"]
                    subanswers_gold = data["subanswers"]
                    assert len(subqueries_gold) == len(subanswers_gold)

                    context_gold_ids = data["context_doc_ids"]
                    contexts_golds = [f"{corpus[int(context_doc_id)]['title']}\n{corpus[int(context_doc_id)]['contents']}" for context_doc_id in context_gold_ids]
                    
                    input_texts = []
                    for subquery in subqueries:
                        input_texts.append(f"query: {subquery}")
                    for passage in contexts_golds:
                        input_texts.append(f"passage: {passage}")
                    
                    embeddings = retriever.encode(input_texts, normalize_embeddings=True)
                    embeddings = torch.tensor(embeddings)
                    
                    query_emb = embeddings[:len(subqueries)] # (hidden_dim, )
                    passage_embs = embeddings[len(subqueries):] # (num_passages, hidden_dim)

                    scores = (query_emb @ passage_embs.T) # (num_passages, )

                    docs = []
                    top_k = torch.topk(scores, k=5, dim=1)
                    for idx in range(len(subqueries)):
                        row_indices = top_k.indices[idx]
                        row_scores = top_k.values[idx]
                        print(f"subquery: {subqueries[idx]}")

                        sub_docs = []
                        for passage_idx, score in zip(row_indices, row_scores):
                            print(f"{contexts_golds[passage_idx]} {score:.2f}")
                            sub_docs.append(contexts_golds[passage_idx])
                        docs.append(sub_docs)

                    # for i in top_k.indices:
                    #     print(f"{contexts_golds[i]} {scores[i]:.2f}")

                    chain = [f"subquery {i+1}: {subqueries[i]}, subanswer {i+1}: {subanswers[i]}" for i in range(len(subanswers))]

                    # for intermeidates
                    past = ''
                    for idx in range(len(subqueries)):
                        past += f"""Intermediate query {idx+1}: {subqueries[idx]}\n\nIntermediate answer {idx+1}: {subanswers[idx]}\n"""
                    past = past.strip()

                    # for docs
                    context = ''
                    if docs:
                        for sub_docs in docs:
                            for idx, sub_doc in enumerate(sub_docs):
                                context += f"""Doc {idx}: {sub_doc}\n\n"""

                    # prompt
                    prompt = f"""Given the following intermediate queries and answers, generate a final answer for the main query by combining relevant information. Note that intermediate answers are generated by an LLM and may not always be accurate.
                    ## Documents
                    {context.strip()}

                    ## Intermediate queries and answers
                    {past}

                    ## Task description
                    {task_desc}

                    ## Main query
                    {query}

                    Respond with an appropriate answer only, do not explain yourself or output anything else."""
                    messages = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]

                    outputs = pipeline(
                        messages,
                        max_new_tokens=4096,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=model_tokenizer.eos_token_id,
                        truncation=True
                    )

                    predictions = [_normalize_answer(outputs[0]["generated_text"][-1]["content"])]
                    print(f"answers: {answers}")
                    print(f"predictions: {predictions}")

                    if predictions != answers:
                        num += 1
                        json_obj = {
                            "query_id": query_id,
                            "query": query,
                            "answers": answers,
                            "predictions": predictions,
                            "chain": chain,
                            "documents": docs
                        }
                        out_file.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
                    
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"[OOM] Skipped: {query_id}")
                        empty_cache()
                        continue
                    else:
                        raise e
                
                except Exception as e:
                    print(f"[ERROR] {query_id}: {e}")
                    continue

                print(f"{i} queries processed")
                print(f"{num} queries mismatched")