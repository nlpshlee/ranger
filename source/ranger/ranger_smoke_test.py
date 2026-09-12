from _init import *

'''
    RangerTrainer 코어(롤아웃 -> advantage -> loss -> backward -> 파라미터 갱신) 배선 점검용 스모크 테스트

        - 체인 서버 / 리트리버 / vLLM 없이, GPU와 학습 모델만 있으면 실행됨
        - 같은 프롬프트에 대해 advantage 가 '+'인 completion 과 '-'인 completion 을 만들어 두고,
          몇 번의 업데이트 후에 log_prob 가 의도한 방향으로 움직이는지 확인

        실행 : CUDA_VISIBLE_DEVICES=0 python -u ranger_smoke_test.py
'''

import torch

from ranger.corag.corag_result import QueryResult, ChainResult
from ranger.reward.reward_calculator import RewardCalculator
from ranger.train.ranger_trainer import RangerTrainer


N_STEPS = 20
OUT_DIR = '/tmp/ranger_smoke_test'

GOOD_TEXT = 'Who wrote the novel Moby Dick?'
BAD_TEXT = 'zzzz qqqq xxxx vvvv'

PROMPT_MESSAGES = [
    {'role': 'user', 'content': 'Generate a simple follow-up question that helps answer the main query.\n\n## Main query\nWho is the author of Moby Dick?'}
]


def make_query_results():
    # advantage 가 '+1' 인 체인과 '-1' 인 체인을 각각 하나씩 구성
    query_result = QueryResult()
    query_result._query_id = 'smoke-0'
    query_result._query = 'Who is the author of Moby Dick?'
    query_result._answers = ['herman melville']
    query_result._answer_set = {'herman melville'}
    query_result._hop = 2

    for text, advantage in ((GOOD_TEXT, 1.0), (BAD_TEXT, -1.0)):
        chain_result = ChainResult()
        chain_result._sub_query_prompts = [PROMPT_MESSAGES]
        chain_result._sub_querys = [text]
        chain_result._advantage = advantage
        chain_result._reward = 1.0 if 0 < advantage else 0.0
        query_result._chain_results.append(chain_result)

    return [query_result]


def get_log_prob(trainer: RangerTrainer, text: str):
    # 해당 completion 의 토큰 평균 log_prob (학습 방향 확인용)
    sample = trainer._build_sample(
        PROMPT_MESSAGES, text, 0.0,
        trainer._tokenizer(trainer._tokenizer.eos_token, add_special_tokens=False)['input_ids'],
        VLLM_CONFIG['max_new_tokens']
    )
    input_ids, attention_mask, completion_mask, _, _ = trainer._collate_samples([sample])

    with torch.no_grad():
        log_probs = trainer._get_per_token_log_probs(trainer._model, input_ids, attention_mask).to(torch.float32)

    return ((log_probs * completion_mask).sum() / completion_mask.sum()).item()


def main():
    trainer = RangerTrainer(MODEL_CONFIG, RewardCalculator(REWARD_CONFIG['reward_option']), OUT_DIR)

    # 스모크 테스트에서는 매 스텝 바로 반영되도록 accumulation 해제
    trainer._gradient_accumulation_steps = 1

    query_results = make_query_results()

    before_good, before_bad = get_log_prob(trainer, GOOD_TEXT), get_log_prob(trainer, BAD_TEXT)
    print(f'\n[before] log_prob(good) : {before_good:.5f}, log_prob(bad) : {before_bad:.5f}\n')

    grad_norms = []
    for step in range(1, N_STEPS+1):
        loss, stats = trainer._train_batch(query_results)

        grad_norm = float(trainer._accelerator.clip_grad_norm_(trainer._model.parameters(), trainer._max_grad_norm))
        trainer._optimizer.step()
        trainer._optimizer.zero_grad(set_to_none=True)
        grad_norms.append(grad_norm)

        print(f'[{step:02d}] loss : {loss:+.6f}, grad_norm : {grad_norm:.6f}, kl : {stats["kl"]:.6f}, n_samples : {stats["n_samples"]}')

    after_good, after_bad = get_log_prob(trainer, GOOD_TEXT), get_log_prob(trainer, BAD_TEXT)
    print(f'\n[after ] log_prob(good) : {after_good:.5f}, log_prob(bad) : {after_bad:.5f}')
    print(f'[delta ] good : {after_good-before_good:+.5f} (증가해야 정상), bad : {after_bad-before_bad:+.5f} (감소해야 정상)\n')

    checks = {
        'gradient가 흐름 (grad_norm > 0)': 0 < max(grad_norms),
        'advantage(+) completion 의 log_prob 증가': before_good < after_good,
        'advantage(-) completion 의 log_prob 감소': after_bad < before_bad
    }

    for msg, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {msg}')

    print(f'\n=> {"ALL PASS" if all(checks.values()) else "FAILED"}\n')


if __name__ == '__main__':
    main()
