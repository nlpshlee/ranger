import os, time, json
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple
from datasets import load_dataset

from ranger.corag.agent.agent_utils import RagPath
from ranger.corag.agent.corag_agent_batch_for_queries import CoRagAgent
from ranger.corag.data_utils import format_documents_for_final_answer, parse_answer_logprobs
from ranger.corag.prompts import get_generate_intermediate_answer_prompt, get_generate_subquery_prompt
from ranger.corag.search.search_utils import search_by_http
from ranger.corag.vllm_client import VllmClient


# HTTP 요청 로그 비활성화
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

PATH_LENGTH = 15
NUM_OF_CHAINS = 10
# OUTPUT_PATH = f"/home/softstone/saltlux/data/output/chains_depth{PATH_LENGTH}_chain{NUM_OF_CHAINS}.jsonl"
# OUTPUT_PATH = "/home/softstone/saltlux/data/output/sample_inference.jsonl"
OUTPUT_PATH = "/home/softstone/saltlux/data/output/data.jsonl"

config = {
    "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
    "task_desc": "answer multi-hop questions",
}
vllm_client: VllmClient = VllmClient(
    model=config['model_name'],
    host="localhost",
    port=7030,
    api_key="token-123"
)
corag_agent = CoRagAgent(vllm_client=vllm_client)

dataset = load_dataset("corag/multihopqa", "hotpotqa", split="train").shuffle(seed=42).select(range(100))

def _eval_single_path(current_path, answer="No relevant information found", max_message_length: int = 4096) -> float:
    messages: List[Dict] = get_generate_intermediate_answer_prompt(
        subquery=current_path.query,
        documents=[f'Q: {q}\nA: {a}' for q, a in zip(current_path.past_subqueries, current_path.past_subanswers)],
    )
    # messages.append({'role': 'assistant', 'content': 'No relevant information found'})
    messages.append({'role': 'assistant', 'content': answer})
    
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

def build_data_dict(query_id: str, query: str, answer: str, paths: List[RagPath], scores: List[float], predicts):
    assert len(paths) == len(predicts) == len(scores)
    result_dict = {
        "query_id": query_id,
        "query": query,
        "chain_depths": [],
        "chains": [],
        "answer_gold": answer
    }
    
    chain_depths = []
    for path, predict in zip(paths, predicts):
        if len(path.past_subqueries) < PATH_LENGTH:
            chain_depths.append(len(path.past_subqueries))
        else:
            if predict == answer:
                chain_depths.append(len(path.past_subqueries))
            else:
                chain_depths.append(-1)
    result_dict['chain_depths'] = chain_depths
    
    for path, score, predict in zip(paths, scores, predicts):
        result_dict['chains'].append({
            "subqueries": path.past_subqueries,
            "subanswers": path.past_subanswers,
            "doc_ids": path.past_doc_ids,
            # "context_doc_ids": context_doc_ids,
            "predicted_answer": predict,
            "log_likelihood": score,
        })
    return result_dict

def predict_final_answer(query, path):
    from ranger.corag.config import Arguments
    import threading
    
    args = Arguments()
    tokenizer_lock: threading.Lock = threading.Lock()
    
    retriever_results: List[Dict] = search_by_http(query=query, topk=20)
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
# %%

# paths = corag_agent.batch_sample_path_4(
#     query_list=dataset['query'][:3],
#     task_desc=config['task_desc'],
#     max_path_length=10,
#     num_of_chains=5,
#     answers=dataset['answers'][:3]
# )

# %%
# 저장된 데이터 확인
start_idx = 0
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        saved_data = [json.loads(line) for line in f if line.strip()]
        if saved_data:
            start_idx = len(saved_data)
            print(f"이미 {start_idx}개의 데이터가 저장되어 있습니다. {start_idx}부터 저장을 시작합니다.")

generate_times = []
for idx in tqdm(range(start_idx, 10)):
    query_id = dataset[idx]['query_id']
    query = dataset[idx]['query']
    answer = dataset[idx]['answers'][0]
    
    start_time = time.time()
    paths = corag_agent.batch_sample_path_2(
        query=query,
        task_desc=config['task_desc'],
        max_path_length=PATH_LENGTH,
        num_of_chains=NUM_OF_CHAINS,
        max_message_length=4096,
        temperature=0.7,
        num_workers=4,
        answer=answer,
        max_attempts=3
        )
    generate_times.append(time.time()-start_time)
    print(f"[{idx}] Batched Paths (max path length = {PATH_LENGTH}, number of chains = {NUM_OF_CHAINS}) 실행 시간: {generate_times[-1]:.4f}초")
    print(f"평균 실행 시간: {sum(generate_times) / len(generate_times):.4f}초")
    print(f"총 실행 시간: {sum(generate_times):.4f}초")
    scores = [_eval_single_path(path, answer=answer) for path in paths]
    predicts = [predict_final_answer(query, path) for path in paths]
    bundle = build_data_dict(query_id, query, answer, paths, scores, predicts)
    print(f"query: {query}, predicted answer: {predicts}, answer: {answer}")
    # 각 인덱스마다 결과를 파일에 저장
    with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(bundle, ensure_ascii=False) + "\n")
# %%



# %%
# batch_size = 3
# queries, answers = dataset['query'][:batch_size], dataset['answers'][:batch_size]
# queries, answers
# # %%
# start_time = time.time()
# batched_paths = corag_agent.batch_sample_path(
#     queries= queries,
#     task_desc=config['task_desc'],
#     max_path_length=PATH_LENGTH,
#     num_of_chains=NUM_OF_CHAINS,
#     max_message_length=4096,
#     temperature=0.7,
#     num_workers=4
# )
# print(f"Batched Paths (max path length = {PATH_LENGTH}, number of chains = {1}) 실행 시간: {time.time()- start_time:.4f}초")
# %%
# start_time = time.time()
# unbatched_paths = []
# for query in dataset['query'][:10]:
#     path = corag_agent.sample_path(
#         query=query,
#         task_desc=config['task_desc'],
#         max_path_length=PATH_LENGTH,
#         max_message_length=4096,
#         temperature=0.7,
#     )
#     unbatched_paths.append(path)
# print(f"Unbatched Paths (max path length = {PATH_LENGTH}, number of chains = {1}) 실행 시간: {time.time()- start_time:.4f}초")
# %%
# final answer도 배치로 생성하게 변경해야함
# search_http 이것도 5개아니라 여러개 가능하게 변경해야함
