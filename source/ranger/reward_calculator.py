from _init import *

from typing import List, Any

from ranger.modules import common_util
from ranger.reward.models.reward_model import RewardModel
from ranger.corag_agent import QueryResult, ChainResult
from ranger.corag.agent.agent_utils import RagPath


class RewardCalculator:
    def __init__(self, model_params_path: str, use_gpu_ids=[0, 1]):
        self._model_params_path = model_params_path
        self._use_gpu_ids = use_gpu_ids

        self._reward_model = RewardModel(self._model_params_path)
        common_util.check_gpu_memory(self._use_gpu_ids, '[Reward Calculator Init]')


    def calculate_reward(self, query_results: List[QueryResult]):
        # 체인 별로 리워드 계산인데, 리워드 모델의 배치 함수가 for문 돌면서 체인을 개별 호출하고 있어서, 여기도 걍 개별 호출함
        for query_result in query_results:
            chain_results: List[ChainResult] = query_result._chain_results

            for chain_result in chain_results:
                rag_path = RagPath(
                    query=query_result._query,
                    past_subqueries=chain_result._sub_querys,
                    past_doc_ids=chain_result._doc_ids_list,
                    past_subanswers=chain_result._sub_answers,
                    query_id=None,
                    answer=None,
                    past_finalanswers=None
                )
                
                reward = self._reward_model.calculate_reward(rag_path)
                chain_result._reward = reward

