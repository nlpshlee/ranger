from _init import *

import math
from typing import List

from ranger.modules import common_util
from ranger.corag_agent import ChainResult
from ranger.vllm.vllm_engine import VllmEngine


class RewardOption:
    OFF = 0
    LENGTH = 1
    CORRECT = 2
    CONFIDENCE = 4
    ALL = LENGTH | CORRECT | CONFIDENCE


class RewardModel():
    def __init__(self,
                 penalty_short: float,
                 penalty_long: float,
                 missing_prob: float):

        self._penalty_short = penalty_short                                 # 정답보다 한 스텝 짧을 때 마다 깎이는 점수
        self._penalty_long = penalty_long                                   # 정답보다 한 스텝 길 때 마다 깎이는 점수 (더 큰 페널티)
        self._missing_log_prob = math.log(missing_prob)                     # 정답 토큰이 상위 N개 토큰에 없는 경우의 패널티 확률 (내부적으론 log로 변환해서 사용됨)


    def _calculate_chain_len_reward_linear(self, hop: int, chain_len: int) -> float:
        if chain_len == hop:
            return 1.0
        elif chain_len < hop:
            diff = hop - chain_len
            reward = 1.0 - (diff * self._penalty_short)
        else:
            diff = chain_len - hop
            reward = 1.0 - (diff * self._penalty_long)

        return max(0.0, reward)


    def _calculate_chain_len_reward_gaussian(self, hop: int, chain_len: int) -> float:
        diff = chain_len - hop

        if diff == 0:
            return 1.0
        elif diff < 0:
            penalty = self._penalty_short
        else:
            penalty = self._penalty_long
            
        return math.exp(-(diff**2) / (2 * penalty**2))


    '''
        hop과의 길이를 비교하여 선형적으로 감소하는 보상을 계산합니다.
        길이가 길어지는 것에 더 큰 페널티를 부여합니다.
        길이가 일치할 때, 최대 리워드
    '''
    def _calculate_chain_len_reward(self, hop: int, chain_result: ChainResult) -> float:
        chain_len = len(chain_result._sub_querys)
        return self._calculate_chain_len_reward_linear(hop, chain_len)


    def _calculate_confidence_increases(self, log_likes: list):
        increases = 0

        num_compare = len(log_likes)-1
        for i in range(num_compare):
            if log_likes[i] < log_likes[i+1]:
                increases += 1
        
        return increases / num_compare


    def _calculate_confidence(self, answers: List[str], chain_result: ChainResult, vllm_engine: VllmEngine, alpha=0.5):
        chain_len = len(chain_result._sub_querys)

        if answers is None:
            log_likes_by_step = chain_result._log_likes
        else:
            log_likes_by_step = []
            for i in range(chain_len):
                completion_output = chain_result._completion_outputs[i]

                log_likes_by_step.append(
                    max(
                        [vllm_engine.get_generated_log_like(completion_output, answer, missing_log_prob=self._missing_log_prob) for answer in answers]
                    )
                )
        
        if chain_len == 1:
            return math.exp(log_likes_by_step[0])
        else:
            # 1. 과정 점수 : 스텝 간의 likelihood가 꾸준히 증가했는가?
            reward_increases = self._calculate_confidence_increases(log_likes_by_step)

            # 2. 결과 점수 : 마지막 스텝에서의 likelihood가 높은 값인가?
            reward_last_step = math.exp(log_likes_by_step[-1])

            # 3. [과정+결과] 가중 합산 최종 리워드
            return (alpha * reward_increases) + ((1-alpha) * reward_last_step)


    def calculate_chain_reward(self, reward_option: int, answers: List[str], hop: int, chain_result: ChainResult, vllm_engine: VllmEngine):
        chain_reward = 0

        if common_util.check_option(reward_option, RewardOption.LENGTH):
            chain_reward += self._calculate_chain_len_reward(hop, chain_result)
        if common_util.check_option(reward_option, RewardOption.CORRECT):
            chain_reward += chain_result._f1
        if common_util.check_option(reward_option, RewardOption.CONFIDENCE):
            chain_reward += self._calculate_confidence(None, chain_result, vllm_engine)
        
        return chain_reward

