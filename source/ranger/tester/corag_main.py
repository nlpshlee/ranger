from _init import *

import os, json, logging, string, time
from typing import Dict, List
from tqdm import tqdm

from ranger.modules import file_util
from ranger.corag.agent.agent_utils import RagPath
from ranger.corag.data_utils import format_documents_for_final_answer, parse_answer_logprobs
from ranger.corag.inference.qa_utils import _normalize_answer
from ranger.corag.prompts import get_generate_intermediate_answer_prompt
from ranger.corag.search.search_utils import search_by_http
from ranger.corag.agent.corag_agent_batch_for_queries import CoRagAgent
from ranger.corag.vllm_client import VllmClient


logging.getLogger("httpx").setLevel(logging.WARNING)

PATH_LENGTH = 15
NUM_OF_CHAINS = 10
BATCH_SIZE=100

CURRENT_FILE_PATH = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH))))
OUTPUT_PATH = f'{CURRENT_DIR}/data/corag/output/option_data.jsonl'
print(f'OUTPUT_PATH : {OUTPUT_PATH}')
file_util.make_parent(OUTPUT_PATH)


config = {
    "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
    "task_desc": "answer multi-hop questions",
}
vllm_client: VllmClient = VllmClient(model=config['model_name'], host="localhost", port=7030, api_key="token-123")
corag_agent = CoRagAgent(vllm_client=vllm_client)

from datasets import load_dataset
dataset = load_dataset("corag/multihopqa", "hotpotqa", split="train").shuffle(seed=42)


def _eval_single_path(current_path, max_message_length: int = 4096) -> float:
    messages: List[Dict] = get_generate_intermediate_answer_prompt(
        subquery=current_path.query,
        documents=[f'Q: {q}\nA: {a}' for q, a in zip(current_path.past_subqueries, current_path.past_subanswers)],
    )
    # messages.append({'role': 'assistant', 'content': 'No relevant information found'})
    messages.append({'role': 'assistant', 'content': current_path.answer})
    
    response = vllm_client.call_chat(
        messages=messages,
        return_str=False,
        max_tokens=1,
        extra_body={
            "prompt_logprobs": 1
        }
    )
    answer_logprobs: List[float] = parse_answer_logprobs(response)

    return sum(answer_logprobs) / len(answer_logprobs)

def build_data_dict(paths: List[RagPath], scores: List[float], predicts: List[str]):
    assert len(paths) == len(predicts) == len(scores)
    assert not len(paths) % NUM_OF_CHAINS
    
    overall_data = []
    
    chunks = [(paths[i:i+NUM_OF_CHAINS], scores[i:i+NUM_OF_CHAINS], predicts[i:i+NUM_OF_CHAINS]) for i in range(0, len(paths), NUM_OF_CHAINS)]
    for chunk_paths, chunk_scores, chunk_predicts in chunks:
        result_dict = {
            "query_id": chunk_paths[0].query_id,
            "query": chunk_paths[0].query,
            "chain_depths": [],
            "chains": [],
            "answer_gold": chunk_paths[0].answer
        }
        for path, score, predict in zip(chunk_paths, chunk_scores, chunk_predicts):
            result_dict['chains'].append({
                "subqueries": path.past_subqueries,
                "subanswers": path.past_subanswers,
                "doc_ids": path.past_doc_ids,
                # "context_doc_ids": context_doc_ids,
                "predicted_answer": predict,
                "log_likelihood": score,
            })
        chain_depths = []
        for path, predict in zip(chunk_paths, chunk_predicts):
            if _normalize_answer(predict, punc_chars=string.punctuation, punc_repl="") == _normalize_answer(path.answer, punc_chars=string.punctuation, punc_repl=""):
                chain_depths.append(len(path.past_subqueries))
            else:
                chain_depths.append(-len(path.past_subqueries))
        result_dict['chain_depths'] = chain_depths
        overall_data.append(result_dict)
    return overall_data     

def predict_final_answer(path):
    from ranger.corag.config import Arguments
    import threading
    
    args = Arguments(output_dir="/home/softstone/saltlux/data/output")
    tokenizer_lock: threading.Lock = threading.Lock()
    
    retriever_results: List[Dict] = search_by_http(query=path.query, topk=20)
    context_doc_ids: List[str] = [res['id'] for res in retriever_results]
    
    documents: List[str] = format_documents_for_final_answer(
        args=args,
        context_doc_ids=context_doc_ids,
        tokenizer=corag_agent.tokenizer, corpus=corag_agent.corpus,
        lock=tokenizer_lock
    )

    prediction: str = corag_agent.generate_final_answer(
        corag_sample=path,
        task_desc="answer multi-hop questions",
        documents=documents,
        max_message_length=args.max_len,
        temperature=0., max_tokens=128
    )
    return prediction


# start_time = time.time()
# paths = corag_agent.batch_sample_path_2(
#     query_list=dataset['query'][:1],
#     task_desc=config['task_desc'],
#     max_path_length=15,
#     num_of_chains=10,
#     answers=dataset['answers'][:1]
# )

# print(paths)

total_samples = len(dataset)
print(f"전체 데이터셋 크기: {total_samples}")

# 이미 저장된 파일이 있는지 확인하고 처리된 데이터 수 파악
processed_samples = 0
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, 'r') as f:
        processed_samples = sum(1 for line in f)
    print(f"이미 처리된 데이터: {processed_samples}개")
    print(f"남은 처리할 데이터: {total_samples - processed_samples}개")
else:
    print("새로운 파일을 생성합니다.")

# 이미 처리된 데이터가 있으면 그 다음부터 시작
start_batch = processed_samples // BATCH_SIZE
start_idx = start_batch * BATCH_SIZE

if start_idx >= total_samples:
    print("모든 데이터가 이미 처리되었습니다.")
else:
    for batch_idx in tqdm(range(start_batch, (total_samples + BATCH_SIZE - 1) // BATCH_SIZE), desc="배치 처리 중"):
        current_start_idx = batch_idx * BATCH_SIZE
        end_idx = min(current_start_idx + BATCH_SIZE, total_samples)
        current_batch_size = end_idx - current_start_idx
        
        print(f"\n배치 {batch_idx + 1} 처리 중 ({current_start_idx+1}-{end_idx})")
        
        try:
            start_time = time.time()
            paths = corag_agent.batch_sample_path_2(
                query_ids=dataset['query_id'][current_start_idx:end_idx],
                query_list=dataset['query'][current_start_idx:end_idx],
                task_desc=config['task_desc'],
                max_path_length=PATH_LENGTH,
                num_of_chains=NUM_OF_CHAINS,
                answers=dataset['answers'][current_start_idx:end_idx],
            )
            print(f"쿼리 {current_batch_size}개 생성 시간: {time.time()-start_time:.2f}초")
            
            scores = [_eval_single_path(path) for path in paths]
            predicts = [predict_final_answer(path) for path in paths] 
            bundle = build_data_dict(paths, scores, predicts)
            
            for data in bundle:
                with open(OUTPUT_PATH, 'a') as f:
                    f.write(json.dumps(data) + '\n')
            
            print(f"배치 {batch_idx + 1} 완료 - {len(bundle)}개 결과 저장됨")
            
        except KeyboardInterrupt:
            print(f"\n사용자에 의해 중단됨. 배치 {batch_idx + 1}에서 중단되었습니다.")
            break
        except Exception as e:
            print(f"\n배치 {batch_idx + 1} 처리 중 오류 발생: {e}")
            print("다음 배치로 계속 진행합니다...")
            continue

    print(f"\n전체 처리 완료! 결과가 {OUTPUT_PATH}에 저장되었습니다.")

