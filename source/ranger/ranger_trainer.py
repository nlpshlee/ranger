from _init import *

from typing import List

from unsloth import FastLanguageModel
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig
# from transformers.models.llama.modeling_llama import LlamaForCausalLM
# from peft.peft_model import PeftModelForCausalLM
# from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from ranger.modules import common_util, json_util, container_util
from ranger.chain_generator import ChainGenerator
from ranger.reward_calculator import RewardCalculator
from ranger.corag_agent import QueryResult, ChainResult


class RANGERTrainer(Trainer):
    def __init__(self,
                 chain_generator: ChainGenerator,
                 reward_calculator: RewardCalculator,
                 model_config: dict,
                 grpo_config: GRPOConfig,
                 use_gpu_ids=[0, 1]):
        
        self._chain_generator = chain_generator
        self._reward_calculator = reward_calculator
        self._model_config = model_config
        self._grpo_config = grpo_config
        self._use_gpu_ids = use_gpu_ids

        self._model_name: str
        self._device: str
        self._gpu_memory_utilization: float
        self._dtype: str
        self._max_model_len: int
        self._load_in_4bit: bool
        self._trust_remote_code: bool
        self._lora_r: int
        self._lora_target_modules: int
        self._lora_alpha: int
        self._use_gradient_checkpointing: int
        self._set_config(self._model_config)

        '''
            transformers.Trainer 상속 변수
        '''
        self.model: AutoModelForCausalLM
        self.processing_class: AutoTokenizer

        self._init_model()

        self._train_datas: list
        self._test_datas: list


    def _set_config(self, model_config):
        self._model_name                            = model_config['model_name']
        self._device                                = model_config['device']
        self._gpu_memory_utilization                = model_config['gpu_memory_utilization']
        self._dtype                                 = model_config['dtype']
        self._max_model_len                         = model_config['max_model_len']
        self._load_in_4bit                          = model_config['load_in_4bit']
        self._trust_remote_code                     = model_config['trust_remote_code']
        self._lora_r                                = model_config['lora_r']
        self._lora_target_modules                   = model_config['lora_target_modules']
        self._lora_alpha                            = model_config['lora_alpha']
        self._use_gradient_checkpointing            = model_config['use_gradient_checkpointing']


    def _init_model(self):
        print(f'\n# RANGERTrainer._init_model() model_config : {json_util.to_str(self._model_config)}\n')

        if type(self._model_name) is str:
            print('# RANGERTrainer._init_model() Instantiating model')

            self.model, self.processing_class = FastLanguageModel.from_pretrained(
                model_name=self._model_name,
                max_seq_length=self._max_model_len,
                gpu_memory_utilization=self._gpu_memory_utilization,
                dtype=self._dtype,
                load_in_4bit=self._load_in_4bit,
                trust_remote_code=self._trust_remote_code
            )
            self.model = self.model.to(self._device)
            common_util.check_gpu_memory(self._use_gpu_ids, '[Base Model Init]')

            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self._lora_r,
                target_modules=self._lora_target_modules,
                lora_alpha=self._lora_alpha,
                use_gradient_checkpointing=self._use_gradient_checkpointing
            )
            self.model = self.model.to(self._device)
            common_util.check_gpu_memory(self._use_gpu_ids, '[PEFT Model Init]')

            print(f'\tmodel : {type(self.model)}')
            print(f'\ttokenizer : {type(self.processing_class)}\n')
        else:
            print(f'# RANGERTrainer._init_model() Failed : {self._model_name}')


    def train(self, train_datas, batch_size, n_chains, chain_depth):
        print(f'\n# RANGERTrainer.train() train_data_size : {len(train_datas)}, batch_size : {batch_size}, n_chains : {n_chains}, chain_depth : {chain_depth}\n')
        self._train_datas = train_datas

        for i, datas_batch in enumerate(container_util.chunks(self._train_datas, batch_size)):
            print(f'# RANGERTrainer.train() batch {i+1} : datas size : {len(datas_batch)}({i*batch_size} ~ {(i+1)*batch_size-1})\n')

            # 학습 배치와 체인 배치를 동일하게 맞춰야 하므로, chain_generate() 호출하고, [0]만 가져오면 됨 (size == 1)
            query_results: List[QueryResult] = self._chain_generator.chain_generate(datas_batch, batch_size, n_chains, chain_depth)[0]

            # 체인 별로 리워드까지 계산
            self._reward_calculator.calculate_reward(query_results)



            for query_result in query_results:
                print(f'query_id : {query_result._query_id}, query : {query_result._query}, answer : {query_result._answer}\n')

                chain_results: List[ChainResult] = query_result._chain_results
                for chain_idx, chain_result in enumerate(chain_results):
                    print(f'chain_idx : {chain_idx}')
                    chain_result.print_chain()

