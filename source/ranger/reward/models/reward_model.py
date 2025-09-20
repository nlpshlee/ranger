from _init import *

import math
from typing import List
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ranger.modules import common_util
from ranger.corag_agent import ChainResult


class RewardOption:
    OFF = 0
    LENGTH = 1
    CORRECT = 2
    CONFIDENCE = 4
    ALL = LENGTH | CORRECT | CONFIDENCE


class RewardModel():
    def __init__(self,
                 model_name: str,
                 penalty_short: float,
                 penalty_long: float,
                 penalty_missing_prob: float):

        self._model_name = model_name
        self._penalty_short = penalty_short                                 # 정답보다 한 스텝 짧을 때 마다 깎이는 점수
        self._penalty_long = penalty_long                                   # 정답보다 한 스텝 길 때 마다 깎이는 점수 (더 큰 페널티)
        self._penalty_missing_prob = math.log(penalty_missing_prob)         # 정답 토큰이 상위 N개 토큰에 없는 경우의 패널티 확률 (내부적으론 log로 변환해서 사용됨)

        self._tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self._model_name)


    def _calculate_chain_len_reward_linear(self, chain_len: int, hop: int) -> float:
        if chain_len == hop:
            return 1.0
        elif chain_len < hop:
            diff = hop - chain_len
            reward = 1.0 - (diff * self._penalty_short)
        else:
            diff = chain_len - hop
            reward = 1.0 - (diff * self._penalty_long)

        return max(0.0, reward)


    def _calculate_chain_len_reward_gaussian(self, chain_len: int, hop: int) -> float:
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
    def _calculate_chain_len_reward(self, chain_result: ChainResult, hop: int) -> float:
        chain_len = len(chain_result._sub_querys)
        return self._calculate_chain_len_reward_linear(chain_len, hop)


    def _calculate_confidence_unit(self, chain_result: ChainResult, answer: str):
        answer_toks = self._tokenizer(answer, add_special_tokens=False)


    def _calculate_confidence(self, chain_result: ChainResult, answers: List[str]):
        for answer in answers:
            self._calculate_confidence_unit(chain_result, answer)
        
        return 0


    def calculate_chain_reward(self, reward_option: int, answers: List[str], hop: int, chain_result: ChainResult):
        chain_reward = 0

        if common_util.check_option(reward_option, RewardOption.LENGTH):
            chain_reward += self._calculate_chain_len_reward(chain_result, hop)
        if common_util.check_option(reward_option, RewardOption.CORRECT):
            chain_reward += chain_result._f1
        if common_util.check_option(reward_option, RewardOption.CONFIDENCE):
            chain_reward += self._calculate_confidence(chain_result, answers)
        
        return chain_reward

