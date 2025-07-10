import numpy as np
import statsmodels.api as sm
import random
import json
import logging

from typing import List, Dict, Optional
from datasets import load_dataset, Dataset, load_from_disk
from openai.types.chat import ChatCompletion
from contextlib import nullcontext
from transformers import PreTrainedTokenizerFast
from threading import Lock
from sklearn.model_selection import train_test_split

from ranger.corag.config import Arguments


# logger config
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def batch_truncate(
        texts: List[str], tokenizer: PreTrainedTokenizerFast,
        max_length: int = 128, truncate_from_middle: bool = False,
        skip_special_tokens: bool = False
) -> List[str]:
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


def load_used_corpus() -> Dataset:
    corpus = load_from_disk("/home/stv10121/saltlux/data/kilt-corpus-used-dataset")
    logger.info(f'Loaded {len(corpus)} passages from corag/kilt-corpus-used-dataset')
    return corpus


def load_unused_corpus() -> Dataset:
    corpus = load_from_disk("/home/stv10121/saltlux/data/kilt-corpus-unused-dataset")
    logger.info(f'Loaded {len(corpus)} passages from corag/kilt-corpus-unused-dataset')
    return corpus


def load_corpus() -> Dataset:
    corpus: Dataset = load_dataset('corag/kilt-corpus', split='train')
    logger.info(f'Loaded {len(corpus)} passages from corag/kilt-corpus')
    return corpus


def load_json_file(file_path: str) -> List[Dict]:
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            datas = [json.loads(line) for line in f]
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            datas = json.load(f)
    else:
        raise ValueError(f"Invalid file path: {file_path}")
    return datas


def load_params(file_path: str):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['coef'], data['intercept'], data['best_threshold']


def load_features_and_labels(file_path: str):
    dataset = load_json_file(file_path)
    
    features_list, labels_list = [], []
    for data in dataset:
        chains = data['chains']
        for chain in chains:
            label = chain['label_gold']
            labels_list.append(label)
            
            feature_vector = chain['feature_vector']
            feature_list = [feature_vector['LS'], feature_vector['QR'], feature_vector['SD'], feature_vector['EA']]
            features_list.append(feature_list)
                
    return np.array(features_list), np.array(labels_list)
            
    # features_array, labels_array = np.array(features_list), np.array(labels_list)
    
    # X_train, X_test, y_train, y_test = train_test_split(features_array, labels_array, test_size=test_size, random_state=random_state)
    # X_train, X_test = sm.add_constant(X_train), sm.add_constant(X_test)
    # return X_train, X_test, y_train, y_test


def save_params_to_json(params: Dict, file_path: str):
    with open(file_path, 'w') as f:
        json.dump(params, f, indent='\t', ensure_ascii=False)


def log_random_samples(dataset: Dataset, num_samples: int = 3):
    from utils import log_truncate
    # Log a few random samples
    num_samples = min(num_samples, len(dataset))
    for index in random.sample(range(len(dataset)), num_samples):
        logger.info(f"\nSample {index} of the dataset:")
        for key, value in dataset[index].items():
            logger.info(f"################ {key}")
            logger.info(log_truncate(value))


def format_input_context(doc: Dict[str, str]) -> str:
    title: str = doc.get('title', '')
    contents: str = doc['contents']
    # contents: str = doc.get('paragraph_text', '')
    if contents.startswith(title + '\n'):
        contents = contents[len(title) + 1:]

    return f'{title}\n{contents}'.strip()


def parse_answer_logprobs(response: ChatCompletion) -> List[float]:
    prompt_logprobs: List[Dict] = response.prompt_logprobs[::-1]

    # Hacky: this only works for llama-3 models
    assert '128006' in prompt_logprobs[3], f"Invalid prompt logprobs: {prompt_logprobs}"
    prompt_logprobs = prompt_logprobs[4:] # Skip the added generation prompt

    answer_logprobs: List[float] = []
    for logprobs in prompt_logprobs:
        logprobs: Dict[str, Dict]
        prob_infos: List[Dict] = sorted(list(logprobs.values()), key=lambda x: x['rank'])
        if prob_infos[-1]['decoded_token'] == '<|end_header_id|>':
            # also skip the "\n\n" token
            answer_logprobs = answer_logprobs[:-1]
            break

        prob = prob_infos[-1]['logprob']
        answer_logprobs.append(prob)

    return answer_logprobs


def _apply_context_placement_strategy(context_placement: str, contexts: List[str]) -> List[str]:
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


def format_documents_for_final_answer(
        args: Arguments,
        context_doc_ids: List[str],
        tokenizer: PreTrainedTokenizerFast, corpus: Dataset,
        lock: Optional[Lock] = None
) -> List[str]:
    selected_doc_ids: List[str] = context_doc_ids[:args.num_contexts]
    documents: List[str] = [format_input_context(corpus[int(doc_id)]) for doc_id in selected_doc_ids]

    max_per_ctx_length: int = int(args.max_len / max(args.num_contexts, 1) * 1.2)
    with nullcontext() if lock is None else lock:
        documents = batch_truncate(documents, tokenizer=tokenizer, max_length=max_per_ctx_length)
    documents = _apply_context_placement_strategy(args.context_placement, documents)

    return documents