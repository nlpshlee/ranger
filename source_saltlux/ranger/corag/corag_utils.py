from _init import *

import re, string, random
from typing import List, Set, Dict, Optional
from datasets import load_dataset, Dataset
from threading import Lock
from contextlib import nullcontext
from transformers import PreTrainedTokenizerFast

from ranger.corag.corag_arguments import CoragArguments


def load_corpus(path: str, split: str) -> Dataset:
    corpus: Dataset = load_dataset(path, split=split)

    if DEBUG.LOG:
        print(f'# corag_utils.load_corpus() Loaded {len(corpus)} passages from {path}, split={split}')

    return corpus


def normalize_sub_query(subquery: str):
    subquery = subquery.strip()
    if subquery.startswith('"') and subquery.endswith('"'):
        subquery = subquery[1:-1]
    if subquery.startswith('Intermediate query'):
        subquery = re.sub(r'^Intermediate query \d+: ', '', subquery)

    return subquery


def normalize_answer(text: str, punc_chars=string.punctuation, punc_repl="", to_lower=True):
    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def replace_punctuation(s):
        to_replace = set(punc_chars)
        return "".join(punc_repl if ch in to_replace else ch for ch in s)

    def white_space_fix(s: str):
        return " ".join(s.split())

    if to_lower:
        text = text.lower()
    
    text = replace_punctuation(text)
    text = remove_articles(text)
    text = white_space_fix(text)

    return text


def compare_answers(answer_set: Set[str], predict: str):
    return predict in answer_set


def batch_truncate(texts: List[str], tokenizer: PreTrainedTokenizerFast,
                   max_length=128,
                   truncate_from_middle=False,
                   skip_special_tokens=False) -> List[str]:

    if not texts:
        return texts

    if truncate_from_middle:
        input_ids: List[List[int]] = tokenizer(texts, truncation=True, max_length=int(10**7))['input_ids']

        for idx in range(len(input_ids)):
            if len(input_ids[idx]) > max_length:
                half_length = max_length // 2
                input_ids[idx] = input_ids[idx][:half_length] + input_ids[idx][-half_length:]

    else:
        input_ids: List[List[int]] = tokenizer(texts, truncation=True, max_length=max_length)['input_ids']

    # Skip bos but keep other special tokens in the input texts
    if not skip_special_tokens:
        input_ids = [ids[1:] if ids[0] == tokenizer.bos_token_id else ids for ids in input_ids]

    return tokenizer.batch_decode(input_ids, skip_special_tokens=skip_special_tokens)


def format_input_context(doc: Dict[str, str]) -> str:
    title: str = doc.get('title', '')
    contents: str = doc['contents']

    if contents.startswith(title + '\n'):
        contents = contents[len(title) + 1:]

    return f'{title}\n{contents}'.strip()


def apply_context_placement_strategy(context_placement: str, contexts: List[str]) -> List[str]:
    # Assume the contexts are sorted by retrieval scores descending
    if context_placement == 'forward':
        return contexts
    elif context_placement == 'backward':
        return list(reversed(contexts))
    elif context_placement == 'random':
        random.shuffle(contexts)
        return contexts
    else:
        raise ValueError(f'Invalid context placement strategy: {context_placement}')


def format_documents_for_final_answer(args: CoragArguments,
                                       context_doc_ids: List[str],
                                       tokenizer: PreTrainedTokenizerFast,
                                       corpus: Dataset,
                                       lock: Optional[Lock]=None) -> List[str]:

    selected_doc_ids: List[str] = context_doc_ids[:args.num_contexts]
    documents: List[str] = [format_input_context(corpus[int(doc_id)]) for doc_id in selected_doc_ids]

    max_per_ctx_length: int = int(args.max_len / max(args.num_contexts, 1) * 1.2)
    with nullcontext() if lock is None else lock:
        documents = batch_truncate(documents, tokenizer=tokenizer, max_length=max_per_ctx_length)
    documents = apply_context_placement_strategy(args.context_placement, documents)

    return documents

