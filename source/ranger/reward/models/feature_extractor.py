import numpy as np
import time

from tqdm import tqdm
from typing import Dict, List
from ranger.corag.agent.agent_utils import RagPath
from ranger.reward.utils import calculate_similarity
from ranger.reward.inference.qa_utils import _normalize_answer
from ranger.reward.data_utils import load_corpus
from ranger.reward.models.nli_model import NLIModel


class FeatureExtractor:
    def __init__(self, corpus=None):
        if corpus is None:
            self.corpus = load_corpus()
        else:
            self.corpus = corpus
        # self.nlp = spacy.load('en_core_web_md')
        self.nli = NLIModel()


    def _print_info(self, features_list: List[Dict]):
        length_scores = [features['LS'] for features in features_list]
        query_relevances = [features['QR'] for features in features_list]
        subquery_diversities = [features['SD'] for features in features_list]
        error_admittings = [features['EA'] for features in features_list]
        # step_relevances = [features['SR'] for features in features_list]
        answer_confidences = [features['AC'] for features in features_list]
        # document_relevances = [features['DR'] for features in features_list]

        print("### LS Reward ###")
        self._print_stats(length_scores)

        print("\n### QR Reward ###")
        self._print_stats(query_relevances)

        print("\n### SD Reward ###")
        self._print_stats(subquery_diversities)
        
        print("\n### EA Reward ###")
        self._print_stats(error_admittings)

        print(f"\n### AC Reward ###")
        self._print_stats(answer_confidences)
        # print("\n### SR Reward ###")
        # self._print_stats(step_relevances)


    def _print_stats(self, answer_list):
        print(f"max: {max(answer_list):.2f}")
        print(f"min: {min(answer_list):.2f}")
        print(f"average: {np.mean(answer_list):.2f}")


    def batch_extract_features(self, ragpath_list: List[RagPath]) -> List[Dict]:
        start_time = time.time()
        feature_vectors = [self.extract_feature_vector(ragpath) for ragpath in tqdm(ragpath_list, desc="Extracting features ...")]
        elapsed_time = time.time() - start_time
        print(f"batch feature extraction elapsed_time: {elapsed_time}")
        return feature_vectors

    
    def extract_feature_vector(self, ragpath: RagPath) -> Dict:
        LS = self.extract_length_score(ragpath) # O
        QR = self.extract_query_relevance(ragpath) # O
        SD = self.extract_subquery_diversity(ragpath) # O
        EA = self.extract_error_admitting(ragpath) # O
        # SR = self.extract_step_relevance(ragpath) # X
        # AC = self.extract_answer_confidence(ragpath) # O

        feature_vector = {
            'LS': LS,
            'QR': QR,
            'SD': SD,
            'EA': EA,
            # 'SR': SR,
            # 'DR': DR
        }
        return feature_vector


    def extract_length_score(self, ragpath: RagPath) -> int:
        subqueries = ragpath.past_subqueries
        subanswers = ragpath.past_subanswers
        assert len(subqueries) == len(subanswers)
        return 1 / len(subqueries)


    def extract_query_relevance(self, ragpath: RagPath) -> float:
        query = ragpath.query
        subqueries = ragpath.past_subqueries
        
        premises = [query] * len(subqueries)
        scores = self.nli.batch_predict(premises, subqueries)

        return np.mean(scores)
    

    def extract_subquery_diversity(self, ragpath: RagPath) -> float:
        subqueries = ragpath.past_subqueries

        # Compute Jaccard similarity between adjacent subqueries
        if len(subqueries) == 1:
            return 1
        else:
            similarities = [1 - calculate_similarity('jaccard', subqueries[i], subqueries[i+1]) for i in range(len(subqueries)-1)]
        return np.mean(similarities)


    def extract_error_admitting(self, ragpath: RagPath) -> float:
        subanswers = ragpath.past_subanswers

        # Count the number of subanswers that are "no relevant information found"
        error_admitting_count = sum([-1 if _normalize_answer(subanswer) == 'no relevant information found' else 0 for subanswer in subanswers])
        return error_admitting_count / len(subanswers)


    # 개선 필요
    def extract_step_relevance(self, ragpath: RagPath) -> float:
        subqueries = ragpath.past_subqueries
        subanswers = ragpath.past_subanswers
        assert len(subqueries) == len(subanswers)

        # Compute Jaccard similarity between subqueries and subanswers
        if len(subqueries) == 1:
            return 0
        else:
            # similarities = [calculate_similarity('jaccard', subanswers[i], subqueries[i+1]) for i in range(len(subqueries)-1)]
            similarities = [calculate_similarity('overlap_coef', subanswers[i], subqueries[i+1]) for i in range(len(subqueries)-1)]
        return np.mean(similarities)


    def extract_answer_confidence(self, ragpath: RagPath) -> float:
        log_likelihood = ragpath.log_likelihood
        return log_likelihood