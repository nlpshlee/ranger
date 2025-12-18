from _init import *

import math
from typing import List

from ranger.utils import common_utils
from ranger.corag.corag_result import QueryResult, ChainResult


class RewardOption:
    OFF = 0
    CORRECT = 1
    LENGTH = 2
    CONFIDENCE = 4
    ALL = CORRECT | LENGTH | CONFIDENCE


class RewardCalculator:
    def __init__(self, reward_option):
        self._reward_option = reward_option


    def _calculate_chain_reward(self, chain_result: ChainResult, hop: int):
        temp_rewards = []

        if common_utils.check_option(self._reward_option, RewardOption.CORRECT):
            temp_rewards.append(chain_result._f1)

        if common_utils.check_option(self._reward_option, RewardOption.LENGTH):
            generated_chain_depth = len(chain_result._final_answers)

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

            전부 곱하면서 값이 너무 작아지는 것을 '기하 평균'을 이용하여 완화
                A x B x C -> n(3)_루트(A x B x C)
        '''
        mul_reward = math.prod(temp_rewards)

        # 전부 곱한 값이 '0'이라면, 'correct(f1)'가 '0'인 경우임 (나머지는 '0'이 될 수 없음) -> 강한 패널티
        if mul_reward == 0.0:
            return 0.0
        
        # 모든 부분 리워드의 곱을 기하 평균으로 반환
        return mul_reward ** (1.0 / len(temp_rewards))


    def calculate_reward(self, query_results: List[QueryResult]):
        for query_result in query_results:
            '''
                하나의 쿼리에서 생성된 모든 체인 별로 각각 em/f1 계산
                그 다음 em/f1 각각 가장 높은 값을 쿼리의 em/f1 으로 설정
            '''
            query_result.compute_metrics()

            chain_results: List[ChainResult] = query_result._chain_results
            for chain_result in chain_results:
                reward = self._calculate_chain_reward(chain_result, query_result._hop)
                chain_result._reward = reward

