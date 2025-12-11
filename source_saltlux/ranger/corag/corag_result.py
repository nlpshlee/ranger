from _init import *

from typing import List, Set, Union
import collections


def _metric_max_over_answers(metric_fn, answers: Union[List[str], Set[str]], predict: str):
    return max(
        metric_fn(answer, predict) for answer in answers
    )


def _compute_em(answer_set: Set[str], predict: str):
    if predict in answer_set:
        return 1
    
    return 0


def _compute_f1(answer: str, predict: str):
    answer_toks = answer.split()
    predict_toks = predict.split()

    common = (collections.Counter(answer_toks) & collections.Counter(predict_toks))
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(predict_toks)
    recall = 1.0 * num_same / len(answer_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


'''
    answer_set, predict 모두 _normalize_answer(to_lower=True) 수행되어 있음
    
        - answer_set은 원래는 공백까지 제거하여 초기화 하고, predict도 공백 제거해서 검사하는 로직이었는데
            - 일단은, answers를 그대로 set으로 변환한 상태 (속도 때문에 여기 저기 최적화 하는 중..)
'''
def _compute_em_and_f1(answer_set: Set[str], predict: str):
    em = _compute_em(answer_set, predict)
    f1 = _metric_max_over_answers(_compute_f1, answer_set, predict)

    return em, f1


class ChainResult:
    def __init__(self):
        self._sub_querys = []
        self._sub_answers = []
        self._doc_ids_list = []             # 2차원 배열
        self._docs_list = []                # 2차원 배열
        self._final_answers = []
        self._log_probs = []
        self._is_stop = False
        self._reward = -1                   # 0 ~ 1
        self._advantage = -1
        self._completion = ''
        self._em = -1
        self._f1 = -1


    def make_get_completion(self, delim='\n'):
        completion_parts = []
        chain_depth = len(self._sub_querys)

        for i in range(chain_depth):
            sub_query = self._sub_querys[i]
            sub_answer = self._sub_answers[i]
            final_answer = self._final_answers[i]

            completion_part = f'<sub_query>{sub_query}</sub_query><sub_answer>{sub_answer}</sub_answer><final_answer>{final_answer}</final_answer>'
            completion_parts.append(completion_part)
        
        self._completion = f'{delim}<chain>' + delim.join(completion_parts) + f'{delim}</chain>'
        return self._completion


    def compute_metrics(self, answer_set: Set[str]):
        self._em, self._f1 = _compute_em_and_f1(answer_set, self._final_answers[-1])


    def print_chain(self):
        chain_depth = len(self._sub_querys)
        print(f'# [CORAG_TEST] ChainResult.print_chain() is_stop(correct) : {self._is_stop}, chain_depth : {chain_depth}')
        print(f'\t[log_prob]\t[sub_query]\t[doc_ids]\t[sub_answer]\t[final_answer]\n')

        for i in range(chain_depth):
            print(f'\t{self._log_probs[i]}\t{self._sub_querys[i]}\t{self._doc_ids_list[i]}\t{self._sub_answers[i]}\t{self._final_answers[i]}')
        print(f'\n\treward : {self._reward}')
        print(f'\tadvantage : {self._advantage}\n')


class QueryResult:
    def __init__(self):
        self._query_id = ''
        self._query = ''
        self._answers = List[str]
        self._answer_set: Set[str]
        self._hop = -1
        self._chain_results: List[ChainResult] = []
        self._doc_ids = []
        self._docs = []
        self._em = -1
        self._f1 = -1
    

    def compute_metrics(self):
        all_em, all_f1 = [], []

        for chain_result in self._chain_results:
            chain_result.compute_metrics(self._answer_set)

            all_em.append(chain_result._em)
            all_f1.append(chain_result._f1)
        
        self._em = max(all_em)
        self._f1 = max(all_f1)

