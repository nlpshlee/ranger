from _init import *

import numpy as np

from ranger.utils import common_utils, json_utils, container_utils, evaluation_utils
from ranger.reward.reward_calculator import RewardCalculator


seed = COMMON_CONFIG['seed']
common_utils.set_seed(seed)

work_dir = f'/raid/ai/home/jsyang/dev_env/git/repos/ranger'
data_dir = f'{work_dir}/data'
out_dir = f'{work_dir}/output'

test_data_path = f'{data_dir}/custom_multihopqa_eval_1000.jsonl'
test_datas = json_utils.load_file(test_data_path)

reward_calculator = RewardCalculator(REWARD_CONFIG['reward_option'])

DATASET_SRCS = ['hotpotqa', 'musique', '2wikimultihopqa', 'bamboogle']





# 데이터 분포 확인 및 분리
all_dataset = {}

for data in test_datas:
    dataset_src = data['source']

    if not dataset_src in all_dataset.keys():
        all_dataset[dataset_src] = [data]
    else:
        all_dataset[dataset_src].append(data)

print(f'\n# evaluation_runner [{DATASET_SRCS[0]}] dataset size : {len(all_dataset[DATASET_SRCS[0]])}')
print(f'# evaluation_runner [{DATASET_SRCS[1]}] dataset size : {len(all_dataset[DATASET_SRCS[1]])}')
print(f'# evaluation_runner [{DATASET_SRCS[2]}] dataset size : {len(all_dataset[DATASET_SRCS[2]])}')
print(f'# evaluation_runner [{DATASET_SRCS[3]}] dataset size : {len(all_dataset[DATASET_SRCS[3]])}\n')


def _print_score(name, scores):
    s_max = np.max(scores)
    s_median = np.median(scores)
    s_min = np.min(scores)
    s_mean = np.mean(scores)

    print(f'# evaluation_runner._print_score() [{name}] max : {s_max:.2f}, median : {s_median:.2f}, min : {s_min:.2f}, mean : {s_mean:.2f}')


def _run(prefix, datas, batch_size, n_chains, chain_depth, adapter_path, temperature, top_p, top_k):
    ems, f1s, rewards, advantages = evaluation_utils.evaluate(
        prefix,
        datas,
        batch_size,
        n_chains,
        chain_depth,
        adapter_path,
        temperature,
        top_p,
        top_k,
        reward_calculator
    )
    
    print()
    _print_score('EM', ems)
    _print_score('F1', f1s)
    _print_score('reward', rewards)
    _print_score('advantage', advantages)
    print()

    return ems, f1s, rewards, advantages


def run_all(prefix, all_dataset, batch_size, n_chains, chain_depth, adapter_path, temperature, top_p, top_k):
    all_ems, all_f1s, all_rewards, all_advantages = [], [], [], []

    for dataset_src in DATASET_SRCS:
        ems, f1s, rewards, advantages = _run(
            f'[{prefix}] [{dataset_src}]',
            all_dataset[dataset_src],
            batch_size,
            n_chains,
            chain_depth,
            adapter_path,
            temperature,
            top_p,
            top_k
        )

        all_ems.extend(ems)
        all_f1s.extend(f1s)
        all_rewards.extend(rewards)
        all_advantages.extend(advantages)
    
    print()
    _print_score('ALL EM', all_ems)
    _print_score('ALL F1', all_f1s)
    _print_score('ALL reward', all_rewards)
    _print_score('ALL advantage', all_advantages)
    print()





'''
    CUDA_VISIBLE_DEVICES=1 python -u evaluation_runner.py > ./logs/evaluation_runner.log
'''
if __name__ == "__main__":
    batch_size = 100
    chain_depth = 5

    # greedy
    run_all(
        prefix='CoRAG-8B greedy',
        all_dataset=all_dataset,
        batch_size=batch_size,
        n_chains=1,
        chain_depth=chain_depth,
        adapter_path='',
        temperature=0.0,
        top_p=1.0,
        top_k=-1
    )

    # best-of-n
    for n_chains in [5, 4, 3, 2]:
        for temperature in [0.6, 0.7, 0.8]:
            run_all(
                prefix='CoRAG-8B best_n',
                all_dataset=all_dataset,
                batch_size=batch_size,
                n_chains=n_chains,
                chain_depth=chain_depth,
                adapter_path='',
                temperature=temperature,
                top_p=0.95,
                top_k=-1
            )

