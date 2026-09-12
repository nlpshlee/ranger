from _init import *

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from ranger.utils import tokenizer_utils


class SftDataset(Dataset):
    def __init__(self, datas: list, tokenizer: PreTrainedTokenizerFast, max_length: int, ignore_index=-100):
        super().__init__()
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._ignore_index = ignore_index

        self._datas = []
        self._preprocess(datas)


    def _preprocess(self, datas: list):
        sources = [[data['source']] for data in datas]
        targets = [data['target'] + self._tokenizer.eos_token for data in datas]

        source_ids_list = tokenizer_utils.tokenize_apply_chat_template_and_truncate(
            datas=sources,
            tokenizer=self._tokenizer,
            max_length=self._max_length
        )

        target_ids_list = self._tokenizer(targets, add_special_tokens=False)['input_ids']

        for source_ids, target_ids in zip(source_ids_list, target_ids_list):
            input_ids = source_ids + target_ids

            '''
                '입력+정답'의 길이가 최대 길이를 넘어가면 스킵
                    - 스킵하지 않으려면, 입력 부분에 대해서 정답 부분의 길이를 뺀 값으로 최대 길이를 설정해야 하는데
                      그렇게 하려면, 배치 단위로 못 하고 하나씩 토크나이징 해야함
                    - 그리고 입력 부분을 강제로 자르고 학습을 하는 것 자체가 문맥을 훼손한 상태로 로스를 계산하는거라 위험할 수 있음
            '''
            if len(input_ids) > self._max_length:
                continue

            attention_mask = [1] * len(input_ids)
            labels = [self._ignore_index] * len(source_ids) + target_ids

            self._datas.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })

        if DEBUG.TRAIN:
            print(f'# [TRAIN] SftDataset._preprocess() datas size : {len(datas)} -> preprocessed size : {len(self._datas)}')


    def __len__(self):
        return len(self._datas)


    def __getitem__(self, index):
        return self._datas[index]

