from _init import *

from typing import List

from ranger.modules import common_util
from ranger.reward.models.reward_model import RewardModel
from ranger.corag_agent import QueryResult, ChainResult
from ranger.vllm.vllm_engine import VllmEngine


class RewardCalculator:
    def __init__(self, reward_config: dict, use_gpu_ids=[0, 1]):
        self._reward_option: int
        self._reward_config = reward_config
        self._use_gpu_ids = use_gpu_ids

        self._set_config_and_init_model(self._reward_config)


    def _set_config_and_init_model(self, reward_config: dict):
        self._reward_option = reward_config['reward_option']

        self._reward_model = RewardModel(reward_config['penalty_short'],
                                         reward_config['penalty_long'],
                                         reward_config['missing_prob'])

        common_util.check_gpu_memory(self._use_gpu_ids, '[Reward Calculator Init]')


    def calculate_reward(self, query_results: List[QueryResult], vllm_engine: VllmEngine):
        for query_result in query_results:
            query_result.compute_metrics()
            chain_results: List[ChainResult] = query_result._chain_results

            for chain_result in chain_results:
                reward = self._reward_model.calculate_chain_reward(self._reward_option,
                                                                   query_result._answers,
                                                                   query_result._hop,
                                                                   chain_result,
                                                                   vllm_engine)

                chain_result._reward = reward

