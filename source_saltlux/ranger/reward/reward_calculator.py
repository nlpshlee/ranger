from _init import *

import math
from typing import List
import numpy as np

from ranger.utils import common_utils
from ranger.corag.corag_result import QueryResult, ChainResult


class RewardOption:
    OFF = 0
    CORRECT = 1
    LENGTH = 2
    CONFIDENCE = 4
    ALL = CORRECT | LENGTH | CONFIDENCE


# 하나의 쿼리에 대한 모든 체인들의 리워드를 정규화하여 advantage 계산
def _calculate_advantage(chain_results: List[ChainResult]):
    # 1. NumPy 배열로 변환
    rewards = np.array([cr._reward for cr in chain_results], dtype=np.float32)

    # 2. 평균 및 표준편차 계산
    mean_reward = rewards.mean()
    std_reward = rewards.std() if len(rewards) > 1 else 0.0

    # 3. 정규화
    advantages = (rewards - mean_reward) / (std_reward + 1e-8)

    # 4. 결과 저장
    for i, chain_result in enumerate(chain_results):
        chain_result._advantage = float(advantages[i])


class RewardCalculator:
    def __init__(self, reward_option):
        self._reward_option = reward_option


    def _calculate_chain_reward(self, chain_result: ChainResult, hop: int):
        temp_rewards = []

        if common_utils.check_option(self._reward_option, RewardOption.CORRECT):
            temp_rewards.append(max(0.0, chain_result._f1))

        if common_utils.check_option(self._reward_option, RewardOption.LENGTH):
            hop = max(1.0, hop) # test 데이터에는 hop 정보가 없어서 '-1'인 상황 발생
            generated_chain_depth = max(1.0, len(chain_result._final_answers))

            '''
                hop=2, generated_chain_depth=1 -> 1.0
                hop=2, generated_chain_depth=2 -> 1.0
                hop=2, generated_chain_depth=3 -> 2/3 = 0.66
                hop=2, generated_chain_depth=4 -> 2/4 = 0.5
            '''
            if hop >= generated_chain_depth:
                temp_rewards.append(1.0)
            else:
                temp_rewards.append(hop / generated_chain_depth)

        if common_utils.check_option(self._reward_option, RewardOption.CONFIDENCE):
            avg_log_prob = sum(chain_result._log_probs) / len(chain_result._log_probs)
            temp_rewards.append(max(0.0, min(1.0, math.exp(avg_log_prob))))
        
        # 최대 3개의 부분 리워드 존재
        '''
            1. correct : 정답을 맞췄는가?
            2. length : 체인의 길이는 짧은가?
            3. confidence : 모델은 강하게 확신했는가?

            3가지 모두 0 ~ 1 사이의 확률값
            전부 곱하면 너무 작은 값이 될까봐 더하는 것은 안됨
                - 'correct'가 keeper 역할
                    - 정답을 아예 틀렸는데, 길이가 짧고 확신했다고 리워드가 높으면 안되기 때문
                    - 길이가 길고 확신을 못했다고 해도, 정답을 맞췄으면 어느정도 보장된 리워드가 제공되어야 함
            
            근데, 정답은 못 맞췄는데 너무 확신하면 리워드가 제공되기 때문에, 확신도는 일단 제거

            전부 곱하면서 값이 너무 작아지는 것을 '기하 평균'을 이용하여 완화 -> 이것도 음수에 대한 제곱근 처리 때문에 제거
                A x B x C -> n(3)_루트(A x B x C)
        '''
        mul_reward = math.prod(temp_rewards)

        return mul_reward


    def _calculate_reward(self, query_results: List[QueryResult]):
        for query_result in query_results:
            '''
                하나의 쿼리에서 생성된 모든 체인 별로 각각 em/f1 계산
                그 다음 em/f1 각각 가장 높은 값을 쿼리의 em/f1 으로 설정
            '''
            query_result.compute_metrics()

            chain_results: List[ChainResult] = query_result._chain_results
            for chain_result in chain_results:
                reward = self._calculate_chain_reward(chain_result, query_result._hop)
                if isinstance(reward, complex):
                    reward = reward.real

                chain_result._reward = float(max(0.0, reward))


    def calculate_reward_and_advantage(self, query_results: List[QueryResult]):
        self._calculate_reward(query_results)

        for query_result in query_results:
            _calculate_advantage(query_result._chain_results)

