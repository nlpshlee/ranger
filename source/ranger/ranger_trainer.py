from _init import *

from typing import List
import numpy as np
import wandb

import torch
from torch.nn.functional import log_softmax
from torch.optim import AdamW
# torch.autograd.set_detect_anomaly(True)

from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from unsloth import FastLanguageModel

from ranger.modules import common_util, json_util, container_util
from ranger.chain_generator import ChainGenerator
from ranger.reward_calculator import RewardCalculator
from ranger.corag_agent import QueryResult, ChainResult


NULL_FLOAT = np.finfo(np.float32).min
DO_REVIEW_TRAIN, DO_REVIEW_TEST = True, True


class RANGERTrainer(Trainer):
    def __init__(self,
                 chain_generator: ChainGenerator,
                 reward_calculator: RewardCalculator,
                 model_config: dict,
                 grpo_config: dict,
                 out_dir: str,
                 use_gpu_ids=[0]):
        
        self._chain_generator = chain_generator
        self._reward_calculator = reward_calculator
        self._model_config = model_config
        self._grpo_config = grpo_config
        self._out_dir = out_dir
        self._use_gpu_ids = use_gpu_ids

        self._model_name                            : str
        self._device                                : str
        self._gpu_memory_utilization                : float
        self._dtype                                 : str
        self._max_model_len                         : int
        self._load_in_4bit                          : bool
        self._trust_remote_code                     : bool
        self._lora_r                                : int
        self._lora_target_modules                   : List[str]
        self._lora_alpha                            : int
        self._use_gradient_checkpointing            : bool
        self._gradient_accumulation_steps           : int
        self._lr                                    : float
        self._epsilon                               : float
        self._beta                                  : float
        self._set_config(self._model_config, self._grpo_config)

        '''
            '_'가 없는 변수는 transformers.Trainer 상속 변수
        '''
        self.model                                  : AutoModelForCausalLM
        self.processing_class                       : AutoTokenizer
        self.optimizer                              : torch.optim.Optimizer
        self.accelerator                            : Accelerator # Gradient Accumulation
        self._tokenizer                             : AutoTokenizer # 편의를 위한 래퍼 변수 (processing_class)
        self._init_model()

        self._train_datas: list
        self._test_datas: list

        super().__init__(
            model=self.model,
            processing_class=self.processing_class,
            optimizers=(self.optimizer, None)
        )

        # Output 경로 설정
        self._run_time = common_util.get_datetime_now("%Y-%m-%d-%H-%M-%S")
        self._adapter_path = f'{self._out_dir}/lora_adapter_{self._run_time}'

        self._global_step = 0
        self._train_epoch_results = []
        self._test_epoch_results = []


    def _set_config(self, model_config: dict, grpo_config: dict):
        print(f'\n# RANGERTrainer._set_config() model_config : {json_util.to_str(model_config)}\n')
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

        print(f'\n# RANGERTrainer._set_config() grpo_config : {json_util.to_str(grpo_config)}\n')
        self._gradient_accumulation_steps           = grpo_config['gradient_accumulation_steps']
        self._lr                                    = grpo_config['learning_rate']
        self._epsilon                               = grpo_config['epsilon']
        self._beta                                  = grpo_config['beta']


    def _init_model(self):
        if type(self._model_name) is str:
            print('# RANGERTrainer._init_model() Instantiating model')

            '''
                base 모델을 먼저 초기화하고, 이 모델을 이용하여 RoLA 튜닝을 위한 PEFT 모델을 초기화
                최종 사용은 PEFT 모델
            '''
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

            self.optimizer = AdamW(self.model.parameters(), lr=self._lr)

            # 배치를 '1'로 했을 때, 발생하는 학습 붕괴를 막기 위해, 그래디언트를 누적하여 반영하는 라이브러리
            os.environ['ACCELERATE_TORCH_DEVICE'] = self._device
            self.accelerator = Accelerator(gradient_accumulation_steps=self._gradient_accumulation_steps)
            self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            common_util.check_gpu_memory(self._use_gpu_ids, f'[Accelerator Init] with device:{self.accelerator.device}')

            # ranger 멤버 변수에 trainer 상속 변수 할당
            self._tokenizer = self.processing_class

            print(f'\tmodel : {type(self.model)}')
            print(f'\ttokenizer : {type(self._tokenizer)}')
            print(f'\toptimizer : {type(self.optimizer)}')
            print(f'\taccelerator : {type(self.accelerator)}')
            print(f'\tconfig_device : {self._device}, accelerator_device : {self.accelerator.device}\n')
        else:
            print(f'# RANGERTrainer._init_model() Failed : {self._model_name}')


    def _reset_epoch(self):
        self._chain_generator.reset()
    
    
    def _save_adapter_for_vllm(self, step_name: str, step: int):
        print(f'\n# RANGERTrainer._save_adapter_for_vllm() {step_name} : {step}')

        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(self._adapter_path)

        self.accelerator.wait_for_everyone()
        common_util.check_gpu_memory(self._use_gpu_ids, '[adapter saved]')


    def _calculate_advantages(self, query_result: QueryResult, epsilon=1e-8):
        # 하나의 Query에 대한 모든 체인의 리워드를 바탕으로 advantage 계산
        chain_results: List[ChainResult] = query_result._chain_results

        # 1. 모든 체인의 리워드를 텐서로 변환
        rewards = torch.tensor([chain_result._reward for chain_result in chain_results], device=self._device)

        # 2. 그룹(하나의 query) 내 리워드의 평균과 표준편차 계산
        mean_reward = rewards.mean()

        if rewards.numel() > 1:
            std_reward = rewards.std()
        else:
            std_reward = 0

        # 3. Advantage 계산: (개별 리워드 - 평균 리워드) / (표준편차 + epsilon)
        advantages = (rewards - mean_reward) / (std_reward + epsilon)

        # 4. 계산된 advantage를 각 ChainResult 객체에 저장
        for i, chain_result in enumerate(chain_results):
            chain_result._advantage = advantages[i].item() # 스칼라 값으로 저장


    '''
        - 입력된 시퀀스 전체에 대한 토큰별 log 확률 계산
            - n번째 토큰의 log 확률은 n-1번째 토큰까지의 입력(logit)을 통해 계산됨

        - Example).
            batch_size = 1
            vocab_size = 8
            sequence_length = 6
            mock_input_ids = torch.tensor([[2, 5, 1, 7, 3, 0]])
            mock_logits = torch.randn(batch_size, sequence_length, vocab_size)
            ------------------------------------------------------------------

            input_ids : torch.Size([1, 6])
            tensor([[2, 5, 1, 7, 3, 0]])

            logits : torch.Size([1, 6, 8])
            tensor([[[ 1.2338, -0.9403,  0.6188, -2.4464,  1.0180,  0.4548, -0.0093,
                    2.0530],                                                                -> [5]
                    [-1.4869, -0.1504,  0.9678,  0.7750, -2.2476,  0.4794,  2.6689,
                    2.1249],                                                                -> [1]
                    [-0.3414, -0.5537,  1.7023, -1.1219, -0.4809, -0.3780,  1.3905,
                    0.1745],                                                                -> [7]
                    [ 1.3006,  0.0372,  0.2913, -0.1514,  1.5692,  1.1073, -0.3339,
                    -1.1672],                                                               -> [3]
                    [-1.1066, -0.1595, -0.0377, -0.9362, -1.6921,  1.7293,  0.1520,
                    0.5527],                                                                -> [0]
                    [-0.9953,  0.4688,  0.6880, -0.3639,  0.2553, -1.8593,  2.0697,
                    -1.5089]]])                                                             -> X
            
            shifted_logits : torch.Size([1, 5, 8])
            tensor([[[ 1.2338, -0.9403,  0.6188, -2.4464,  1.0180,  0.4548, -0.0093,
                    2.0530],
                    [-1.4869, -0.1504,  0.9678,  0.7750, -2.2476,  0.4794,  2.6689,
                    2.1249],
                    [-0.3414, -0.5537,  1.7023, -1.1219, -0.4809, -0.3780,  1.3905,
                    0.1745],
                    [ 1.3006,  0.0372,  0.2913, -0.1514,  1.5692,  1.1073, -0.3339,
                    -1.1672],
                    [-1.1066, -0.1595, -0.0377, -0.9362, -1.6921,  1.7293,  0.1520,
                    0.5527]]])
            
            shifted_target_ids : torch.Size([1, 5])
            tensor([[5, 1, 7, 3, 0]])

            log_probs_all : torch.Size([1, 5, 8])
            tensor([[[-1.7051, -3.8792, -2.3200, -5.3852, -1.9208, -2.4840, -2.9481,
                    -0.8858],
                    [-4.9015, -3.5650, -2.4468, -2.6396, -5.6623, -2.9352, -0.7458,
                    -1.2897],
                    [-2.9522, -3.1644, -0.9084, -3.7327, -3.0917, -2.9887, -1.2202,
                    -2.4363],
                    [-1.4571, -2.7205, -2.4664, -2.9091, -1.1884, -1.6504, -3.0916,
                    -3.9249],
                    [-3.5280, -2.5808, -2.4590, -3.3576, -4.1135, -0.6921, -2.2693,
                    -1.8687]]])

            log_probs : torch.Size([1, 5])
            tensor([[-2.4840, -3.5650, -2.4363, -2.9091, -3.5280]])
    '''
    def _get_per_token_logps(self, model: AutoModelForCausalLM, input_ids, attention_mask):
        # 1. 전체 시퀀스에 대한 logits 획득
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits

        # 2. 마지막 logit 제거 (다음 토큰이 없으므로)
        shifted_logits = logits[:, :-1, :]

        # 3. 정답 토큰 ID도 위치를 맞춤 (맨 앞 토큰 제거)
        shifted_target_ids = input_ids[:, 1:]

        # 4. log_softmax를 적용하여 log 확률 계산
        log_probs_all = log_softmax(shifted_logits, dim=-1)

        # 5. 실제 정답 토큰 ID에 해당하는 log 확률만 추출
        log_probs = torch.gather(log_probs_all, 2, shifted_target_ids.unsqueeze(-1)).squeeze(-1)

        # print(f'\n\n input_ids : {input_ids.shape}\n{input_ids}')
        # print(f'\n\n logits : {logits.shape}\n{logits}')
        # print(f'\n\n shifted_logits : {shifted_logits.shape}\n{shifted_logits}')
        # print(f'\n\n shifted_target_ids : {shifted_target_ids.shape}\n{shifted_target_ids}')
        # print(f'\n\n log_probs_all : {log_probs_all.shape}\n{log_probs_all}')
        # print(f'\n\n log_probs : {log_probs.shape}\n{log_probs}')
        # sys.exit(-1)
        return log_probs.to(self._device)


    # _calculate_loss_per_batch() 를 테스트하기 위한 함수
    def __calculate_loss_per_batch_test(self):
        import torch.nn.functional as F
        batch_size = 2
        max_seq_len = 12
        vocab_size = 20

        query_len_1 = 4
        completion_len_1 = 7
        query_len_2 = 5
        completion_len_2 = 5

        ref_logits = torch.randn(batch_size, max_seq_len-1, vocab_size)
        policy_logits = ref_logits + torch.randn_like(ref_logits) * 0.1

        ref_log_probs_all = F.log_softmax(ref_logits, dim=-1)
        policy_log_probs_all = F.log_softmax(policy_logits, dim=-1)

        mock_target_ids = torch.randint(0, vocab_size, (batch_size, max_seq_len - 1))
        ref_logps = torch.gather(ref_log_probs_all, 2, mock_target_ids.unsqueeze(-1)).squeeze(-1).to(self._device)
        policy_logps = torch.gather(policy_log_probs_all, 2, mock_target_ids.unsqueeze(-1)).squeeze(-1).to(self._device)

        advantages_tensor = torch.tensor([[0.8], [-0.5]], dtype=torch.float32).to(self._device)
        all_query_lens = [query_len_1, query_len_2]

        attention_mask = torch.zeros(batch_size, max_seq_len).to(self._device)
        attention_mask[0, :query_len_1 + completion_len_1] = 1
        attention_mask[1, :query_len_2 + completion_len_2] = 1

        self._calculate_loss_per_batch(ref_logps, policy_logps, advantages_tensor, all_query_lens, attention_mask)
        '''
            ref_logps : torch.Size([2, 11])
            policy_logps : torch.Size([2, 11])
            advantages_tensor : torch.Size([2, 1])
            all_query_lens : 2
            attention_mask : torch.Size([2, 12])
            log_ratio : torch.Size([2, 11])
            ratio : torch.Size([2, 11])
            loss_unclipped : torch.Size([2, 11])
            loss_clipped : torch.Size([2, 11])
            kl_div : torch.Size([2, 11])


            loss_per_token : torch.Size([2, 11])
            tensor([[-0.7121, -0.8580, -0.8700, -0.6896, -0.8417, -0.8137, -0.9159, -0.9061,
                    -0.8092, -0.7385, -0.7031],
                    [ 0.5209,  0.5046,  0.5403,  0.4435,  0.5769,  0.5455,  0.5829,  0.4438,
                    0.4310,  0.4966,  0.4960]], device='cuda:0')


            query_lens_tensor : torch.Size([2])
            tensor([4, 5], device='cuda:0')


            full_mask : torch.Size([2, 11])
            tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.]], device='cuda:0')


            indices : torch.Size([2, 11])
            tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
                    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]], device='cuda:0')


            prompt_mask : torch.Size([2, 11])
            tensor([[1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.]], device='cuda:0')


            final_mask : torch.Size([2, 11])
            tensor([[0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0.],
                    [0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0.]], device='cuda:0')


            masked_loss : torch.Size([2, 11])
            tensor([[-0.0000, -0.0000, -0.0000, -0.0000, -0.8417, -0.8137, -0.9159, -0.9061,
                    -0.8092, -0.7385, -0.0000],
                    [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.5455,  0.5829,  0.4438,
                    0.4310,  0.0000,  0.0000]], device='cuda:0')


            sum_masked_loss : torch.Size([2])
            tensor([-5.0252,  2.0031], device='cuda:0')


            sum_mask : torch.Size([2])
            tensor([6., 4.], device='cuda:0')


            batch_loss : torch.Size([])
            -0.16838595271110535
        '''


    def _calculate_loss_per_batch(self, ref_logps, policy_logps, advantages_tensor, all_query_lens, attention_mask):
        # print(f'\n\nref_logps : {ref_logps.shape}')
        # print(f'policy_logps : {policy_logps.shape}')
        # print(f'advantages_tensor : {advantages_tensor.shape}')
        # print(f'all_query_lens : {len(all_query_lens)}')
        # print(f'attention_mask : {attention_mask.shape}')

        # r(i) 값 계산
        log_ratio = policy_logps - policy_logps.detach()
        ratio = torch.exp(log_ratio)

        # unclipped loss : r(i) * advantage
        loss_unclipped = ratio * advantages_tensor

        # clipped loss : clip(r(i), 1-ε, 1+ε) * advantage
        loss_clipped = torch.clip(ratio, 1-self._epsilon, 1+self._epsilon) * advantages_tensor

        # KL-Divergence 페널티 항 추가
        kl_div = torch.exp(ref_logps - policy_logps) - (ref_logps - policy_logps) - 1

        # GRPO loss
        '''
            - 우리의 목적 함수는 다음과 같음
                - PPO 목적 함수는 '최대화', KL 페널티는 '최소화'

                    - 여기서 PPO 목적 함수는 'torch.min(loss_unclipped, loss_clipped)' 이 부분임
                        - 엄청 쉽게 생각하면, 단순하게 이 부분임??? [ r(i) * advantage ]
                
                - PyTorch 옵티마이저는 Loss를 '최소화'하므로, 이 목표에 맞게 수식을 변환해야 함

                    - PPO 목적 함수를 'A', KL 페널티를 'B'라고 한다면,
                        - 최종 수식은 '-A + B'가 되어야 함
                        - 이걸 다시 변환해서 '-(A - B)'로 계산
        '''
        ppo_objective = torch.min(loss_unclipped, loss_clipped)
        loss_per_token = -(ppo_objective - (self._beta * kl_div))

        '''
            - prompt 부분은 제외하고, padding이 아닌 completion 부분만 마스킹
            - 그 다음 completion 부분에 대해서만 loss를 최종 계산

                - 각 시퀀스의 query 길이를 텐서로 변환
                  attention_mask를 이용해 패딩 부분을 제외하고,
                  각 시퀀스의 query 길이만큼 앞부분을 추가로 제외
                  shifted_logps에 맞춰 마스크도 맨 앞 제거
        '''
        query_lens_tensor = torch.tensor(all_query_lens, device=self._device)
        full_mask = attention_mask[:, 1:]

        indices = torch.arange(full_mask.shape[1], device=self._device).expand(full_mask.shape[0], -1)
        prompt_mask = (indices < query_lens_tensor.unsqueeze(1)).float()
        final_mask = full_mask * (1 - prompt_mask)

        '''
            마스크를 이용해서 completion에 대한 loss만 최종 계산
        '''
        masked_loss = loss_per_token * final_mask
        sum_masked_loss = masked_loss.sum(dim=1)
        sum_mask = final_mask.sum(dim=1).clamp(min=1e-8) # 이건 파라미터로 안 빼도 되겠지..?
        batch_loss = (sum_masked_loss / sum_mask).mean()

        # print(f'\n\n loss_per_token : {loss_per_token.shape}\n{loss_per_token}')
        # print(f'\n\n query_lens_tensor : {query_lens_tensor.shape}\n{query_lens_tensor}')
        # print(f'\n\n full_mask : {full_mask.shape}\n{full_mask}')
        # print(f'\n\n indices : {indices.shape}\n{indices}')
        # print(f'\n\n prompt_mask : {prompt_mask.shape}\n{prompt_mask}')
        # print(f'\n\n final_mask : {final_mask.shape}\n{final_mask}')
        # print(f'\n\n masked_loss : {masked_loss.shape}\n{masked_loss}')
        # print(f'\n\n sum_masked_loss : {sum_masked_loss.shape}\n{sum_masked_loss}')
        # print(f'\n\n sum_mask : {sum_mask.shape}\n{sum_mask}')
        # print(f'\n\n batch_loss : {batch_loss.shape}\n{batch_loss}')
        # sys.exit(-1)
        return batch_loss
    
    
    def _train_batch(self, query_results: List[QueryResult]):
        # 1. 데이터 수집 단계 : 모든 체인(chain)의 데이터를 계산에 필요한 파이썬 리스트에 저장
        all_full_texts = []
        all_query_lens = []
        all_advantages = []

        for query_result in query_results:
            query = query_result._query
            query_len = len(self._tokenizer(query, add_special_tokens=False)['input_ids'])

            chain_results: List[ChainResult] = query_result._chain_results
            for chain_result in chain_results:
                completion = chain_result.make_get_completion()

                all_full_texts.append(query + completion)
                all_query_lens.append(query_len)
                all_advantages.append(chain_result._advantage)
        
        # 2. 배치 토크나이징 단계 : 수집된 모든 텍스트를 한 번에 토크나이징하고, 가장 긴 시퀀스에 맞춰 패딩
        inputs = self._tokenizer(
            all_full_texts,
            padding=True,
            truncation=True,
            max_length=self._max_model_len,
            return_tensors="pt"
        ).to(self._device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # advantage 리스트도 텐서로 변환 (full_size, 1) 형태로 브로드캐스팅 준비
        advantages_tensor = torch.tensor(all_advantages, device=self._device).unsqueeze(1)

        # 3. 배치 단위 모델 추론 : 배치 전체에 대하여 단 한 번만 모델 호출
        '''
            - 여기서, RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation 이 에러가 발생함
                - 이 에러는 backward를 위한 변수의 값이 forward 상태와 달라지면서 발생하는 문제임
                    - 예를 들어, [forward1 : backward1] 인 상황에서, forward2가 실행되면, backward1이 실행될 때,
                      동일한 변수에 대해 forward1의 값이 forward2의 값으로 변경되었기 때문에 backward1을 계산할 수 없게되는 문제임
            
            - 아래 코드에서는 ref_model과 model이 기존 코드에서는 완전한 독립 객체가 아니라서, forward의 간섭이 발생함
                - 해결책으로 init_model() 부분에서 완전한 독립 개체로 초기화하도록 변경

                    - 젠장, 별짓 다해봤는데 해결 안 되길래, 다시 원본 코드 뒤져봤더니
                    - 참조 모델을 아예 다른 모델을 쓰는게 아니라면, PEFT 모델에서 어댑터만 비활성화해서 base 모델로 활용하는 방식
                    - 이렇게 해야, 'inplace operation' 에러의 근본 원인이 해결되는 것임
                    - 개열받..

            - _get_per_token_logps() 함수를 두번 호출하는건 문제가 되지 않음
                - 어댑터 부분을 제외한 모델의 기존 파라미터는 두 번의 forward를 수행해도 내부적으로는 전혀 변경되지 않기 때문임

            - 하아... 결국 accelerator 사용해서, 언랩핑된 모델 기준으로 해야 해결됨

            - 찐 해결 : _get_per_token_logps() 함수에서 아래처럼 변경
                - outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                - 'use_cache=False' 이 부분 추가하니까 해결은 됨
        '''
        # 레퍼런스 모델로부터 log 확률 계산 : self.model의 어댑터를 비활성화하여 계산
        with torch.inference_mode():
            with self.accelerator.unwrap_model(self.model).disable_adapter():
                ref_logps = self._get_per_token_logps(self.model, input_ids, attention_mask)
                ref_logps = ref_logps.detach()
        
        # 현재 모델(Policy)로부터 log 확률 계산
        policy_logps = self._get_per_token_logps(self.model, input_ids, attention_mask)

        # 4. batch 단위 loss 계산
        batch_loss = self._calculate_loss_per_batch(ref_logps, policy_logps, advantages_tensor, all_query_lens, attention_mask)
        # batch_loss = self.__calculate_loss_per_batch_test()

        # 5. 모델 업데이트
        '''
            - 모든 체인에 대한 그래디언트가 누적된 후, 모델 파라미터를 한 번만 업데이트

            - ref_model은 아래 설정으로 인하여 학습되지 않음
                1. 위에서 ref_logps를 계산할 때, torch.no_grad() 사용하고, detach() 해줌
                2. optimizer를 설정할 때, AdamW(self.model.parameters(), ...)와 같이 self.model의 파라미터만 전달했음
            
            # accelerator 사용
                - 기존 zero_grad(), step() 은 train() 함수에서 관리됨
        '''
        print(f'\n# RANGERTrainer._train_batch() Batch Loss: {batch_loss}\n')
        # self.optimizer.zero_grad()
        # batch_loss.backward()
        # self.optimizer.step()
        self.accelerator.backward(batch_loss)

        return batch_loss.item()


    def train(self, train_datas, test_datas, epochs, batch_size, n_chains, chain_depth):
        print(f'\n# RANGERTrainer.train() [train_data_size : {len(train_datas)}], [epochs : {epochs}, batch_size : {batch_size}], [n_chains : {n_chains}, chain_depth : {chain_depth}]')
        print(f'\tstart datetime : {common_util.get_datetime_now()}\n')
        train_start = common_util.get_time_ms()

        self._train_datas = train_datas
        self._test_datas = test_datas
        data_size = len(train_datas)

        # 0. [base] 성능 측정
        self.evaluate_all(0, batch_size, 1, chain_depth, True, True)

        for epoch in range(1, epochs+1):
            print(f'{"#"*25} RANGERTrainer.train() [ {epoch} epoch ] start : {common_util.get_datetime_now()} {"#"*25}\n')
            self._reset_epoch()
            epoch_start = common_util.get_time_ms()

            all_batch_loss = []
            batch_len = (data_size + batch_size - 1) // batch_size

            for batch_idx, datas_batch in enumerate(container_util.chunks(self._train_datas, batch_size)):
                self._global_step += 1
                start = batch_idx * batch_size
                end = min(data_size-1, (batch_idx+1) * batch_size-1)
                print(f'# RANGERTrainer.train() [ {batch_idx+1} batch ] start, datas size : {len(datas_batch)}({start} ~ {end})\n')
                batch_start = common_util.get_time_ms()

                # 1. 체인 생성 : 학습 배치와 체인 배치를 동일하게 맞춰야 하므로, chain_generate() 호출하고, [0]만 가져오면 됨 (size == 1)
                query_results: List[QueryResult] = self._chain_generator.chain_generate(datas_batch,
                                                                                        batch_size,
                                                                                        n_chains,
                                                                                        chain_depth,
                                                                                        self._adapter_path)[0]

                # 2. 체인 별 리워드 계산
                self._reward_calculator.calculate_reward(query_results)

                # 3. GRPO Advantage 계산 : 쿼리 별로 모든 체인에 대하여, Advantage 계산
                for query_result in query_results:
                    self._calculate_advantages(query_result)

                # 4. GRPO loss 계산 및 모델 학습
                batch_loss = self._train_batch(query_results)
                all_batch_loss.append(batch_loss)

                # 5. wandb 로깅
                self._wandb_logging_train_batch(epoch, batch_idx+1, query_results, batch_loss)

                # 6. Accelerator 를 이용한 gradient accumulation
                '''
                    - 싱글 GPU를 사용할 경우, self.accelerator.sync_gradients 는 무조건 True
                        - 애초에 멀티 GPU 환경에서 사용되는 라이브러리
                        - RANGER 처럼 배치를 1로 하는 경우, 직접 카운팅해서 제어해야 함
                '''
                is_update_step = (batch_idx+1) % self._gradient_accumulation_steps == 0
                is_last_step = (batch_idx+1) == batch_len

                if is_update_step or is_last_step:
                    self.optimizer.step()
                    self.optimizer.zero_grad() # 누적된 gradient 를 한 번에 반영해야 하므로, 먼저 step() 호출하고 다음에 zero_grad() 호출
                    print(f'\n# RANGERTrainer.train() [ {epoch} epoch - {batch_idx+1} batch ] : sync gradients (accelerator)')

                print(f'\n# RANGERTrainer.train() all_batch_loss[-10:] : {all_batch_loss[-10:]}')
                print(f'# RANGERTrainer.train() sum_batch_loss : {sum(all_batch_loss)}')
                print(f'# RANGERTrainer.train() avg_batch_loss : {sum(all_batch_loss) / len(all_batch_loss)}\n')

                _, batch_elapsed_str = common_util.get_elapsed_time_ms(batch_start)
                print(f'# RANGERTrainer.train() [ {batch_idx+1} batch ] end, elapsed_time : {batch_elapsed_str}\n')


            # 7. RANGERTrainer 내부 PEFT 모델의 어댑터를 ChainGenerator.VllmAgent(VllmEngine) 내부의 VLLM 모델로 전달하기 위해, 저장
            self._save_adapter_for_vllm('epoch', epoch)

            _, epoch_elapsed_str = common_util.get_elapsed_time_ms(epoch_start)
            print(f'{"#"*25} RANGERTrainer.train() [ {epoch} epoch ] end : {common_util.get_datetime_now()}, elapsed : {epoch_elapsed_str} {"#"*25}\n')

            # 8. 현재 epoch 학습 완료 후, 성능 측정
            self.evaluate_all(epoch, batch_size, 1, chain_depth, True, True)


        # 전체 epoch에 대해 각 epoch 별로 결과를 비교한 파일 생성 (검토 전용)
        if DO_REVIEW_TRAIN: self.write_results('train', self._train_epoch_results)
        if DO_REVIEW_TEST: self.write_results('test', self._test_epoch_results)
        
        _, train_elapsed_str = common_util.get_elapsed_time_ms(train_start)
        print(f'# RANGERTrainer.train() end datetime : {common_util.get_datetime_now()}, elapsed : {train_elapsed_str}\n')





    def evaluate_all(self, epoch, batch_size, n_chains, chain_depth, do_train=False, do_test=False):
        if do_train:
            epoch_result = self.evaluate('train', self._train_datas, epoch, batch_size, n_chains, chain_depth)
            if DO_REVIEW_TRAIN:
                self._train_epoch_results.append(epoch_result)
        if do_test:
            epoch_result = self.evaluate('test', self._test_datas, epoch, batch_size, n_chains, chain_depth)
            if DO_REVIEW_TEST:
                self._test_epoch_results.append(epoch_result)


    def evaluate(self, prefix, datas, epoch, batch_size, n_chains, chain_depth):
        prefix_org = prefix
        if epoch == 0:
            prefix = f'[base] {prefix}'
        else:
            prefix = f'[{epoch} epoch] {prefix}'

        print(f'\n{"#"*25} RANGERTrainer.evaluate() {prefix} start : {common_util.get_datetime_now()} {"#"*25}\n')
        self._reset_epoch()
        evaluate_start = common_util.get_time_ms()

        all_query_results = []

        with torch.no_grad():
            for batch_idx, datas_batch in enumerate(container_util.chunks(datas, batch_size)):
                print(f'# RANGERTrainer.evaluate() {prefix} - [ {batch_idx+1} batch ] start : {common_util.get_datetime_now()}')

                # 1. 체인 생성
                query_results: List[QueryResult] = self._chain_generator.chain_generate(datas_batch,
                                                                                        batch_size,
                                                                                        n_chains,
                                                                                        chain_depth,
                                                                                        self._adapter_path)[0]
                
                # 2. 체인 별 리워드 계산
                self._reward_calculator.calculate_reward(query_results)

                 # 3. GRPO Advantage 계산
                for query_result in query_results:
                    self._calculate_advantages(query_result)

                    # 4. 전체 서브 스텝이 연결된 하나의 체인 생성
                    chain_results: List[ChainResult] = query_result._chain_results
                    for chain_result in chain_results:
                        chain_result.make_get_completion()
                
                    # 5. 성능 측정
                    query_result.compute_metrics()

                all_query_results.extend(query_results)
        
        # 6. wandb 로깅
        self._wandb_logging_evaluate(prefix_org, epoch, query_results)
        
        _, evaluate_elapsed_str = common_util.get_elapsed_time_ms(evaluate_start)
        print(f'{"#"*25} RANGERTrainer.evaluate() {prefix} end : {common_util.get_datetime_now()}, elapsed : {evaluate_elapsed_str} {"#"*25}\n')

        return all_query_results
    

    def _wandb_logging_train_batch(self, epoch, batch, query_results: List[QueryResult], batch_loss):
        avg_reward, avg_advantage = _get_avg_reward_and_advantage(query_results)
        
        wandb.log({
            'epoch': epoch,
            'batch': batch,
            'train_batch/avg_reward': avg_reward,
            'train_batch/avg_advantage': avg_advantage,
            'train_batch/loss': batch_loss
        }, step=self._global_step)


    def _wandb_logging_evaluate(self, prefix, epoch, query_results: List[QueryResult]):
        avg_reward, avg_advantage = _get_avg_reward_and_advantage(query_results)
        avg_em, avg_f1 = _get_avg_em_and_f1(query_results)
        
        wandb.log({
            'epoch': epoch,
            f'evaluate_{prefix}/avg_reward': avg_reward,
            # f'evaluate_{prefix}/avg_advantage': avg_advantage,
            f'evaluate_{prefix}/avg_em': avg_em,
            f'evaluate_{prefix}/avg_f1': avg_f1
        }, step=self._global_step)


    def write_results(self, prefix, epoch_results:List[List[QueryResult]]):
        epoch_len = len(epoch_results)
        query_len = len(epoch_results[0])

        all_write_dict = []

        for query_idx in range(query_len):
            query_id = ''
            query = ''
            answer = ''
            all_chain_results = []

            is_same = True
            for epoch_idx in range(epoch_len):
                query_result: QueryResult = epoch_results[epoch_idx][query_idx]

                if len(query_id) != 0 and query_id != query_result._query_id:
                    is_same = False
                    break
                else:
                    query_id = query_result._query_id
                    query = query_result._query
                    answer = query_result._answer
                    all_chain_results.append(query_result._chain_results)
            
            if is_same:
                write_dict = {}
                write_dict['query_id'] = query_id
                write_dict['query'] = query
                write_dict['answer'] = answer

                epoch_outputs = {}
                prev_avg_reward, prev_avg_advantage = NULL_FLOAT, NULL_FLOAT

                for epoch_idx, chain_results in enumerate(all_chain_results):
                    epoch_output = {}

                    n_chains = len(chain_results)
                    all_reward, all_advantage = [], []

                    for chain_idx in range(n_chains):
                        chain_result: ChainResult = chain_results[chain_idx]
                        chain_output = {}

                        chain_output['reward'] = chain_result._reward
                        chain_output['advantage'] = chain_result._advantage
                        all_reward.append(chain_result._reward)
                        all_advantage.append(chain_result._advantage)

                        chain_depth = len(chain_result._sub_querys)
                        for depth_idx in range(chain_depth):
                            sub_query = chain_result._sub_querys[depth_idx]
                            sub_answer = chain_result._sub_answers[depth_idx]
                            final_answer = chain_result._final_answers[depth_idx]

                            chain_output[f'depth_{depth_idx+1}'] = {
                                'sub_query': sub_query,
                                'sub_answer': sub_answer,
                                'final_answer': final_answer
                            }
                        
                        epoch_output[f'chain_{chain_idx+1}'] = chain_output
                    
                    epoch_output['reward_all'] = all_reward
                    avg_reward, diff_reward_sign, diff_reward_value = _get_avg_and_diff_from_prev(all_reward, n_chains, prev_avg_reward)
                    epoch_output['reward_avg'] = avg_reward
                    epoch_output['reward_diff'] = [diff_reward_sign, diff_reward_value]

                    epoch_output['advantage_all'] = all_advantage
                    avg_advantage, diff_advantage_sign, diff_advantage_value = _get_avg_and_diff_from_prev(all_advantage, n_chains, prev_avg_advantage)
                    epoch_output['advantage_avg'] = avg_advantage
                    epoch_output['advantage_diff'] = [diff_advantage_sign, diff_advantage_value]

                    prev_avg_reward = avg_reward
                    prev_avg_advantage = avg_advantage
                    
                    if epoch_idx == 0:
                        epoch_outputs['base'] = epoch_output
                    else:
                        epoch_outputs[f'epoch_{epoch_idx}'] = epoch_output

                write_dict['epoch_outputs'] = epoch_outputs
                all_write_dict.append(write_dict)
        
        out_file_path = f'{self._out_dir}/output_dict_{self._run_time}_{prefix}.json'
        json_util.write_file(all_write_dict, out_file_path)


'''
    전체 값으로 평균을 계산하고, 이전 평균하고의 차이까지 계산
'''
def _get_avg_and_diff_from_prev(all_value: list, n_chains: int, prev_avg: float):
    avg_value = sum(all_value) / n_chains

    if (prev_avg == NULL_FLOAT and avg_value != NULL_FLOAT):
        diff_value = 0
        diff_sign = '-'
    else:
        diff_value = avg_value - prev_avg

        if prev_avg < avg_value:
            diff_sign = '+'
        elif prev_avg > avg_value:
            diff_sign = '-'
        else:
            diff_sign = '-'

    return avg_value, diff_sign, diff_value


def _get_avg_reward_and_advantage(query_results: List[QueryResult]):
    batch_rewards, batch_advantages = [], []

    for query_result in query_results:
        chain_results: List[ChainResult] = query_result._chain_results

        for chain_result in chain_results:
            batch_rewards.append(chain_result._reward)
            batch_advantages.append(chain_result._advantage)
    
    if len(batch_rewards) > 0:
        avg_reward = sum(batch_rewards) / len(batch_rewards)
        avg_advantage = sum(batch_advantages) / len(batch_advantages)
    else:
        avg_reward, avg_advantage = NULL_FLOAT, NULL_FLOAT
    
    return avg_reward, avg_advantage


def _get_avg_em_and_f1(query_results: List[QueryResult]):
    all_em, all_f1 = [], []

    for query_result in query_results:
        all_em.append(query_result._em)
        all_f1.append(query_result._f1)

    if len(all_em) > 0:
        avg_em = sum(all_em) / len(all_em)
        avg_f1 = sum(all_f1) / len(all_f1)
    else:
        avg_em, avg_f1 = NULL_FLOAT, NULL_FLOAT
    
    return avg_em, avg_f1

