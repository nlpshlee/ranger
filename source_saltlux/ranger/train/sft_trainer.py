from _init import *

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

from ranger.utils import common_utils, tokenizer_utils
from ranger.train.sft_dataset import SftDataset


IGNORE_INDEX = -100


class SftTrainer:
    def __init__(self, model_name, dtype, max_seq_length, ignore_index=IGNORE_INDEX):
        self._model_name = model_name
        self._dtype = dtype
        self._torch_dtype = getattr(torch, self._dtype)
        self._max_seq_length = max_seq_length
        self._ignore_index = ignore_index

        self._model: AutoModelForCausalLM = None
        self._tokenizer: PreTrainedTokenizerFast = None
        self._data_collator: DataCollatorForSeq2Seq = None
        self._peft_config: LoraConfig = None
        self._training_args: TrainingArguments = None

        self._train_dataset: SftDataset = None
        self._eval_dataset: SftDataset = None


    def _logging(self, msg: str):
        if DEBUG.TRAIN:
            print(f'# [TRAIN] SftTrainer.{msg}')


    def init_model(self, peft_config: LoraConfig=None):
        self._logging('init_model() Initialization model')
        common_utils.check_gpu_memory(do_print=DEBUG.TRAIN, msg='[Before model init]')

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=self._torch_dtype,
            device_map='auto',
            trust_remote_code=False,
            attn_implementation='flash_attention_2'
        )

        self._tokenizer: PreTrainedTokenizerFast = tokenizer_utils.load_tokenizer(self._model_name, 'right')

        self._data_collator = DataCollatorForSeq2Seq(
            tokenizer=self._tokenizer,
            model=self._model,
            padding=True,
            label_pad_token_id=self._ignore_index
        )

        self._peft_config = peft_config
        if self._peft_config is not None:
            self._logging(f'init_model() peft_config :\n{self._peft_config}\n')

            self._model = get_peft_model(self._model, self._peft_config)
            self._model.print_trainable_parameters()


    def set_and_get_sft_dataset(self, train_datas, eval_datas):
        self._train_dataset = SftDataset(train_datas, self._tokenizer, self._max_seq_length, self._ignore_index)
        self._eval_dataset = SftDataset(eval_datas, self._tokenizer, self._max_seq_length, self._ignore_index)

        return self._train_dataset, self._eval_dataset


    def train(self, training_args: TrainingArguments, train_dataset: SftDataset=None, eval_dataset: SftDataset=None):
        self._training_args = training_args
        self._logging(f'train() training_args :\n{self._training_args}\n')
        self._logging(f'train() train start : {common_utils.get_datetime_now()}\n')
        train_start = common_utils.get_time_ms()

        if train_dataset is None:
            train_dataset = self._train_dataset
        if eval_dataset is None:
            eval_dataset = self._eval_dataset

        trainer = Trainer(
            model=self._model,
            args=self._training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self._data_collator
        )

        trainer.train()

        _, train_elapsed_str = common_utils.get_elapsed_time_ms(train_start)
        self._logging(f'train() train end : {common_utils.get_datetime_now()}, elapsed : {train_elapsed_str}\n')

