from _init import *

import shutil, wandb, re
from typing import List
import numpy as np

import torch
from torch.optim import AdamW
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator

from ranger.utils import common_const, common_utils, json_utils, container_utils, tokenizer_utils, evaluation_utils
from ranger.corag.corag_result import QueryResult, ChainResult
from ranger.chain_generate.chain_generate_client import request_chain_generate, request_reset
from ranger.reward.reward_calculator import RewardCalculator


class RangerTrainer:
    def __init__(self,
                 model_config: dict,
                 reward_calculator: RewardCalculator,
                 out_dir: str):
        
        self._model_config = model_config
        self._reward_calculator = reward_calculator
        self._out_dir = out_dir

        self._set_config()

        # 모델 관련 변수
        self._model: AutoModelForCausalLM = None
        self._tokenizer: PreTrainedTokenizerFast = None
        self._optimizer: AdamW = None
        self._accelerator = Accelerator(gradient_accumulation_steps=self._gradient_accumulation_steps)
        self._device = self._accelerator.device
        self._local_gpu_id = self._accelerator.local_process_index

        self._logging(f'RangerTrainer._model_config :\n{json_utils.to_str(self._model_config)}\n', True)

        self._set_out_dir() # accelerator 초기화 이후에 호출
        self._init_model()

        self._global_step = 0


    def _logging(self, msg: str, main_only=False):
        if DEBUG.TRAIN:
            if main_only:
                if self._accelerator.is_main_process:
                    print(f'# [TRAIN][MAIN][{self._device}] {msg}')
            else:
                print(f'# [TRAIN][{self._device}] {msg}')


    def _set_config(self):
        self._model_name = self._model_config['model_name']
        self._dtype = self._model_config['dtype']
        self._max_seq_length = self._model_config['max_seq_length']
        self._lora_r = self._model_config['lora_r']
        self._lora_target_modules = self._model_config['lora_target_modules']
        self._lora_alpha = self._model_config['lora_alpha']
        self._gradient_accumulation_steps = self._model_config['gradient_accumulation_steps']
        self._learning_rate = self._model_config['learning_rate']
        self._epsilon = self._model_config['epsilon']
        self._kl_penalty = self._model_config['kl_penalty']
        self._max_grad_norm = self._model_config['max_grad_norm']
        self._use_gradient_checkpointing = self._model_config['use_gradient_checkpointing']
        self._resume_run_time = self._model_config['resume_run_time']

        self._torch_dtype = getattr(torch, self._dtype)


    def _set_out_dir(self):
        self._run_time = common_utils.get_datetime_now('%Y-%m-%d-%H-%M-%S')
        self._checkpoint_path = f'{self._out_dir}/checkpoint_{self._run_time}'
        self._checkpoint_history_path = f'{self._checkpoint_path}/checkpoint_history.txt'
        self._optimizer_path = f'{self._checkpoint_path}/optimizer.pt'
        self._adapter_path = f'{self._out_dir}/lora_adapter_{self._run_time}'
        self._adapter_history_path = f'{self._adapter_path}/adapter_history.txt'

        self._logging(f'RangerTrainer._set_out_dir() checkpoint_path : {self._checkpoint_path}')
        self._logging(f'RangerTrainer._set_out_dir() adapter_path : {self._adapter_path}\n')

        self._resume_checkpoint_path = ''
        self._resume_optimizer_path = ''
        self._resume_adapter_path = ''
        self._resume_epoch, self._resume_batch = -1, -1

        if self._resume_run_time:
            self._resume_checkpoint_path = f'{self._out_dir}/checkpoint_{self._resume_run_time}'
            self._resume_optimizer_path = f'{self._resume_checkpoint_path}/optimizer.pt'
            self._resume_adapter_path = f'{self._out_dir}/lora_adapter_{self._resume_run_time}'

            # 파일 복사는 오직 메인 프로세스만 수행 (안 그러면 서로 만들다가 충돌남)
            if self._accelerator.is_main_process:
                self._copy_from_resume()
            self._accelerator.wait_for_everyone()

            if os.path.exists(self._checkpoint_history_path):
                self._set_resume_epoch_batch(self._checkpoint_history_path)


    def _copy_from_resume(self):
        # 히스토리 파일만 복사하고, 나머지는 resume 경로에서 직접 로드 (무거운 모델 파일은 복사 X)
        if os.path.exists(self._resume_checkpoint_path):
            resume_checkpoint_history_path = f'{self._resume_checkpoint_path}/checkpoint_history.txt'

            if os.path.exists(resume_checkpoint_history_path):
                os.makedirs(self._checkpoint_path, exist_ok=True)
                shutil.copy2(resume_checkpoint_history_path, self._checkpoint_history_path)
                self._logging(f'RangerTrainer._copy_from_resume() [{resume_checkpoint_history_path}] -> [{self._checkpoint_history_path}]', True)

        # 어댑터는 전체 복사
        if os.path.exists(self._resume_adapter_path):
            shutil.copytree(self._resume_adapter_path, self._adapter_path)
            self._logging(f'RangerTrainer._copy_from_resume() [{self._resume_adapter_path}] -> [{self._adapter_path}]', True)


    def _set_resume_epoch_batch(self, history_path: str):
        with open(history_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            if lines:
                last_line = lines[-1].strip()
                pattern = re.compile(r'(\d+)\s*epoch,\s*(\d+)\s*batch')
                match = pattern.search(last_line)

                if match:
                    self._resume_epoch = int(match.group(1))
                    self._resume_batch = int(match.group(2))
            
            self._logging(f'RangerTrainer._get_resume_epoch_batch() resume point : [ {self._resume_epoch} epoch, {self._resume_batch} batch ]', True)


    def _init_model(self):
        self._logging(f'RangerTrainer._init_model() Initialization model')
        common_utils.check_gpu_memory(do_print=DEBUG.TRAIN, msg=f'[{self._device}][Before model init]')

        # 1. 기본 모델 초기화
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=self._torch_dtype,
            device_map={'': self._local_gpu_id},
            trust_remote_code=False,
            attn_implementation='flash_attention_2'
        )

        # 2. 토크나이저 초기화
        '''
            [중요] 학습 시에는 반드시 'right' 패딩
                - load_tokenizer()의 기본값이 'left'(생성용)인데,
                  loss 마스킹 로직이 '시퀀스가 0번 위치부터 시작한다'를 전제로 하기 때문에
                  left 패딩을 쓰면 prompt/completion 마스크가 통째로 어긋남
        '''
        self._tokenizer: PreTrainedTokenizerFast = tokenizer_utils.load_tokenizer(self._model_name, padding_side='right')

        common_utils.check_gpu_memory(do_print=DEBUG.TRAIN, msg=f'[{self._device}][Base model init]')

        # 3. PEFT (LoRA) 모델 초기화
        if os.path.exists(self._resume_checkpoint_path):
            self._logging(f'RangerTrainer._init_model() Resuming checkpoint from [{self._resume_checkpoint_path}]')

            self._model = PeftModel.from_pretrained(
                self._model,
                self._resume_checkpoint_path,
                is_trainable=True
            )
        else:
            self._logging(f'RangerTrainer._init_model() Setting up new LoRA Adapter...')

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self._lora_r,
                target_modules=self._lora_target_modules,
                lora_alpha=self._lora_alpha,
                # RL에서는 dropout이 있으면 reference/policy log_prob가 같은 입력에도 달라져서 ratio/KL이 흔들림
                lora_dropout=0.0
            )

            self._model = get_peft_model(self._model, peft_config)

        # 4. Gradient Checkpointing (반드시 LoRA 적용 '이후'에 설정)
        '''
            base 모델이 전부 freeze 되어 있으면, checkpoint 블록의 입력 텐서가 requires_grad=False 라서
            블록 출력에 grad_fn이 붙지 않음 -> LoRA 파라미터의 gradient가 전부 None 이 되어 학습이 아예 안 됨
                - enable_input_require_grads() : 임베딩 출력에 requires_grad를 강제로 부여
                - use_reentrant=False : 같은 이유로 기본값(True)에서는 grad가 끊김
        '''
        if self._use_gradient_checkpointing:
            self._model.enable_input_require_grads()
            self._model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            self._model.config.use_cache = False

        # 5. 학습 모드 (from_pretrained 직후에는 eval 모드 상태)
        self._model.train()

        if DEBUG.TRAIN and self._accelerator.is_main_process:
            print()
            self._model.print_trainable_parameters()

        common_utils.check_gpu_memory(do_print=DEBUG.TRAIN, msg=f'[{self._device}][PEFT model init]')

        # 6. Optimizer 초기화
        self._optimizer = AdamW([p for p in self._model.parameters() if p.requires_grad], lr=self._learning_rate)

        # 7. Accelerator Prepare (모델과 옵티마이저를 Accelerator가 관리하도록 래핑)
        self._model, self._optimizer = self._accelerator.prepare(self._model, self._optimizer)

        # 8. Resume Optimizer State
        if os.path.exists(self._resume_optimizer_path):
            self._logging(f'RangerTrainer._init_model() Resuming optimizer state from [{self._resume_optimizer_path}]')
            
            # Accelerator 내부 device로 매핑하여 로드
            opt_state = torch.load(self._resume_optimizer_path, map_location=self._device)
            self._optimizer.load_state_dict(opt_state)
        
        common_utils.check_gpu_memory(do_print=DEBUG.TRAIN, msg=f'[{self._device}][Model init complete]')


    def _save(self, epoch, batch):
        # [중요] 저장 시작 전 모든 프로세스 동기화
        self._accelerator.wait_for_everyone()

        if self._accelerator.is_main_process:
            saved_time = common_utils.get_datetime_now()
            self._logging(f'RangerTrainer._save() [{saved_time}] - [ {epoch} epoch, {batch} batch ]', main_only=True)

            # 1. 경로 생성 (혹시 없을 경우를 대비)
            os.makedirs(self._checkpoint_path, exist_ok=True)
            os.makedirs(self._adapter_path, exist_ok=True)

            # 2. 모델 래핑 해제 (한 번만 수행)
            # PEFT 모델이므로 save_pretrained 호출 시 'Adapter Config'와 'Weights'만 가볍게 저장됨
            unwrapped_model = self._accelerator.unwrap_model(self._model)

            # 3. [Checkpoint] 학습 재개용 저장 (Adapter + Optimizer)
            unwrapped_model.save_pretrained(self._checkpoint_path)
            self._accelerator.save(self._optimizer.state_dict(), self._optimizer_path)
            self._logging(f'RangerTrainer._save() Saved checkpoint : {self._checkpoint_path}', main_only=True)

            # 4. [Serving] vLLM 서빙용 어댑터 저장
            # (내용은 위와 같지만, 용도 분리를 위해 별도 경로에 복제 저장)
            unwrapped_model.save_pretrained(self._adapter_path)
            self._tokenizer.save_pretrained(self._adapter_path)
            self._logging(f'RangerTrainer._save() Saved adapter : {self._adapter_path}', main_only=True)

            # 5. 히스토리 파일 기록
            self._write_save_history(epoch, batch, saved_time)
        
        # 저장이 완료될 때까지 다른 GPU 대기
        self._accelerator.wait_for_everyone()

        if self._accelerator.is_main_process:
            common_utils.check_gpu_memory(do_print=DEBUG.TRAIN, msg=f'[{self._device}][Saved]')


    def _save_epoch_snapshot(self, epoch):
        '''
            에폭 단위 어댑터 스냅샷

            self._adapter_path 는 매 optimizer step 마다 '덮어쓰기' 되므로,
            학습이 끝난 뒤 "2 에폭보다 1 에폭이 더 좋았다" 는 걸 알아도 되돌릴 방법이 없음
            -> 에폭마다 별도 경로에 복사해두고, 평가 결과를 보고 최종 모델을 고를 수 있게 함
        '''
        self._accelerator.wait_for_everyone()

        if self._accelerator.is_main_process and os.path.exists(self._adapter_path):
            snapshot_path = f'{self._adapter_path}_epoch_{epoch}'

            if os.path.exists(snapshot_path):
                shutil.rmtree(snapshot_path)

            shutil.copytree(self._adapter_path, snapshot_path)
            self._logging(f'RangerTrainer._save_epoch_snapshot() Saved epoch snapshot : {snapshot_path}', main_only=True)

        self._accelerator.wait_for_everyone()


    def _write_save_history(self, epoch, batch, saved_time):
        msg = f'[{saved_time}] - [ {epoch} epoch, {batch} batch ], path : '

        with open(self._checkpoint_history_path, 'a', encoding='utf-8') as f:
            f.write(f'{msg}{self._checkpoint_path}\n')
        
        with open(self._adapter_history_path, 'a', encoding='utf-8') as f:
            f.write(f'{msg}{self._adapter_path}\n')


    def _reset_epoch(self):
        request_reset()


    def _get_per_token_log_probs(self, model, input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # 마지막 logit 제거 (다음 토큰이 없으므로) / 정답 토큰도 맨 앞 제거해서 위치를 맞춤
        shifted_logits = outputs.logits[:, :-1, :]
        shifted_target_ids = input_ids[:, 1:]

        '''
            [메모리] log_softmax()로 [B, T, V] 텐서를 따로 만들면
            (5 x 4096 x 128256) 만큼의 텐서가 backward 그래프에 그대로 남아서 수 GB 단위로 터짐
                - cross_entropy(reduction='none')는 backward에서 log_softmax를 다시 계산하므로
                  추가 [B, T, V] 텐서를 저장하지 않음 (= -log p(target))
        '''
        vocab_size = shifted_logits.size(-1)
        target_log_probs = -F.cross_entropy(
            shifted_logits.reshape(-1, vocab_size),
            shifted_target_ids.reshape(-1),
            reduction='none'
        ).view(shifted_target_ids.shape)

        return target_log_probs


    def _calculate_loss_per_batch(self, reference_log_probs, policy_log_probs, advantage_tensor, completion_mask):
        # FP16/BF16 오버플로우 및 정밀도 문제 방지를 위해 float32로 변환
        policy_log_probs = policy_log_probs.to(torch.float32)

        # r(i) 값 계산 (한 번의 rollout에 한 번만 업데이트하는 on-policy 이므로 ratio 자체는 1.0)
        log_ratio = policy_log_probs - policy_log_probs.detach()
        ratio = torch.exp(log_ratio)

        # PPO loss
        loss_unclipped = ratio * advantage_tensor
        loss_clipped = torch.clamp(ratio, 1.0-self._epsilon, 1.0+self._epsilon) * advantage_tensor
        ppo_objective = torch.min(loss_unclipped, loss_clipped)

        # KL-Divergence (k3 estimator, 항상 >= 0)
        if reference_log_probs is None:
            kl_div = torch.zeros_like(policy_log_probs)
        else:
            log_diff = reference_log_probs.to(torch.float32) - policy_log_probs
            kl_div = torch.exp(log_diff) - log_diff - 1

        # Ranger(GRPO) loss
        '''
            우리의 목적 함수는 다음과 같음
                - PPO 목적 함수는 '최대화', KL 페널티는 '최소화'
                - PyTorch 옵티마이저는 Loss를 '최소화'하므로, '-(A - B)' 형태로 변환
        '''
        loss_per_token = -(ppo_objective - (self._kl_penalty * kl_div))

        '''
            completion_mask 는 이미 '패딩 제외 + 프롬프트 제외 + shift 반영'이 끝난 마스크
            (예전처럼 attention_mask/query_len 으로 그 자리에서 만들면 패딩 방향에 따라 통째로 어긋남)
        '''
        masked_loss = loss_per_token * completion_mask
        sum_masked_loss = masked_loss.sum(dim=1)
        sum_mask = completion_mask.sum(dim=1).clamp(min=1.0)

        # 가중 평균은 호출부(_train_batch)에서 처리하므로, 여기서는 시퀀스 별 loss 를 그대로 반환
        per_seq_loss = sum_masked_loss / sum_mask

        # 학습이 실제로 돌고 있는지 확인하기 위한 진단 지표 (KL은 RL에서 가장 중요한 시그널)
        with torch.no_grad():
            n_token = completion_mask.sum().clamp(min=1.0)
            mean_kl = ((kl_div * completion_mask).sum() / n_token).item()

        return per_seq_loss, mean_kl


    '''
        [학습 대상 : 스텝 당 3가지 추론 시점 전부]
            하나의 스텝은 (서브 쿼리 생성 -> 문서 검색 -> 서브 답변 생성 -> 최종 답변 생성) 으로 이루어지고,
            이 중 '검색'을 뺀 3가지가 모델의 추론(= 정책의 행동)임

            따라서 policy gradient 는 3가지 추론 시점 전부에 전파되어야 하며,
            각각 '그 때 실제로 사용된 프롬프트 + 그 때 실제로 생성된 텍스트' 쌍으로 학습해야 함
                - 체인 하나가 D 스텝이면 학습 샘플은 D x 3 개가 되고, 전부 그 체인의 advantage 를 공유

            [주의] 3가지 모두 '샘플링'으로 생성되어야 학습 대상이 될 수 있음
                   (corag_agent._check_step_final_answers() 의 최종 답변은 평가 시에만 greedy)
    '''
    TRAIN_TARGETS = ('sub_query', 'sub_answer', 'final_answer')

    '''
        [마이크로 배치 : 개수가 아니라 '토큰 예산'으로 제한]
            프롬프트에 검색 문서가 들어가서 시퀀스 길이가 수십~4096 토큰으로 크게 요동침
            개수로 자르면 최악 길이(4096) 기준으로 잡아야 해서, 짧은 배치에서 GPU가 놀게 됨

            학습 피크 메모리는 사실상 '패딩 포함 총 토큰 수'에 비례함 (토큰당 약 0.9MB)
                - logits + log_softmax + backward grad : 3 x vocab(128256) x 2byte = 0.73MB/tok
                - gradient checkpointing 경계 hidden   : 28 x 3072 x 2byte        = 0.16MB/tok

            40960 토큰 -> 추정 피크 약 43GB (고정 6.7GB 포함, 140GB 대비 30%)
            추정이 2배 틀려도 85GB라 OOM 여유가 충분함
                - 실제 사용량은 로그의 [Cleaned GPU] 항목으로 확인 후 조정할 것
    '''
    MAX_MICRO_BATCH_TOKENS = 40960

    # 시퀀스가 아주 짧을 때 마이크로 배치가 과도하게 커지지 않도록 개수 상한도 둠
    MAX_MICRO_BATCH_SIZE = 32


    def _build_sample(self, messages, completion, advantage, eos_ids, max_new_tokens):
        prompt_ids = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        completion_ids = self._tokenizer(completion, add_special_tokens=False)['input_ids']

        if len(completion_ids) == 0:
            return None

        # 길이 제한으로 잘린 생성이 아니라면, 종료 토큰까지 학습 대상에 포함
        if len(completion_ids) < max_new_tokens:
            completion_ids = completion_ids + eos_ids

        # 프롬프트가 너무 길면 '앞쪽'을 자름 (생성 시 tokenize_apply_chat_template_and_truncate() 와 동일한 방식)
        max_prompt_len = self._max_seq_length - len(completion_ids)
        if max_prompt_len < 2:
            return None

        if max_prompt_len < len(prompt_ids):
            bos_token_id = self._tokenizer.bos_token_id

            if bos_token_id is not None and prompt_ids[0] == bos_token_id:
                prompt_ids = [bos_token_id] + prompt_ids[-(max_prompt_len-1):]
            else:
                prompt_ids = prompt_ids[-max_prompt_len:]

        return {'prompt_ids': prompt_ids, 'completion_ids': completion_ids, 'advantage': advantage, 'weight': 1.0}


    def _build_train_samples(self, query_results: List[QueryResult]):
        samples = []

        eos_ids = self._tokenizer(self._tokenizer.eos_token, add_special_tokens=False)['input_ids'] if self._tokenizer.eos_token else []
        max_new_tokens = VLLM_CONFIG['max_new_tokens']

        for query_result in query_results:
            for chain_result in query_result._chain_results:
                advantage = chain_result._advantage

                # 그룹 내 reward 가 전부 같으면 advantage 가 0 -> gradient 도 0 이므로 forward 자체를 스킵
                if abs(advantage) < 1e-6:
                    continue

                '''
                    [중요] 학습에는 반드시 '원문(_raw)' 을 사용해야 함
                        normalize_answer() 는 구두점을 전부 제거하므로 '<STOP>' 이 'stop' 이 되어버림
                        (<, > 가 string.punctuation 에 포함)

                        -> normalize 된 텍스트로 학습하면
                           (1) 모델이 샘플링하지 않은 토큰 시퀀스에 policy gradient 가 걸리고
                           (2) SFT 로 배운 <STOP>/<CONTINUE> 특수 토큰이 강화되기는커녕
                               평문 'stop' 을 뱉는 방향으로 밀림

                    _raw 가 비어 있으면(구버전 체인 서버 응답) 기존 필드로 폴백
                '''
                steps = []
                if 'sub_query' in self.TRAIN_TARGETS:
                    steps += list(zip(chain_result._sub_query_prompts, chain_result._sub_querys_raw or chain_result._sub_querys))
                if 'sub_answer' in self.TRAIN_TARGETS:
                    steps += list(zip(chain_result._sub_answer_prompts, chain_result._sub_answers_raw or chain_result._sub_answers))
                if 'final_answer' in self.TRAIN_TARGETS:
                    steps += list(zip(chain_result._final_answer_prompts, chain_result._final_answers_raw or chain_result._final_answers))

                chain_samples = []
                for messages, completion in steps:
                    if not messages or not completion:
                        continue

                    sample = self._build_sample(messages, completion, advantage, eos_ids, max_new_tokens)
                    if sample is not None:
                        chain_samples.append(sample)

                '''
                    [체인 단위 정규화]
                        advantage 를 받은 단위는 '샘플'이 아니라 '체인'이므로, 체인 하나가 loss 에 기여하는 양을 동일하게 맞춤

                        정답을 맞춰서 일찍 멈춘 체인은 스텝이 적어 샘플도 적고(advantage +),
                        끝까지 틀린 체인은 스텝이 많아 샘플도 많음(advantage -)
                        -> 그냥 샘플 평균을 내면 '틀린 체인의 음수 gradient'에 체계적으로 더 큰 가중치가 실림
                '''
                # 생성 텍스트가 전부 빈 문자열이면(normalize 결과 등) 샘플이 하나도 안 만들어질 수 있음
                if not chain_samples:
                    continue

                for sample in chain_samples:
                    sample['weight'] = 1.0 / len(chain_samples)

                samples += chain_samples

        return samples


    def _make_micro_batches(self, samples: List[dict]):
        '''
            패딩 후 총 토큰 수(개수 x 배치 내 최대 길이)가 예산을 넘지 않도록 묶음
            길이 순으로 정렬해서 담으면 같은 배치에 비슷한 길이끼리 모여 패딩 낭비가 최소화됨
        '''
        ordered = sorted(samples, key=lambda x: len(x['prompt_ids']) + len(x['completion_ids']), reverse=True)

        micro_batches = []
        cur_batch, cur_max_len = [], 0

        for sample in ordered:
            sample_len = len(sample['prompt_ids']) + len(sample['completion_ids'])
            next_max_len = max(cur_max_len, sample_len)

            is_over_tokens = self.MAX_MICRO_BATCH_TOKENS < ((len(cur_batch) + 1) * next_max_len)
            is_over_size = self.MAX_MICRO_BATCH_SIZE <= len(cur_batch)

            if cur_batch and (is_over_tokens or is_over_size):
                micro_batches.append(cur_batch)
                cur_batch, cur_max_len = [], 0
                next_max_len = sample_len

            cur_batch.append(sample)
            cur_max_len = next_max_len

        if cur_batch:
            micro_batches.append(cur_batch)

        return micro_batches


    def _collate_samples(self, samples: List[dict]):
        pad_token_id = self._tokenizer.pad_token_id
        max_len = max(len(s['prompt_ids']) + len(s['completion_ids']) for s in samples)

        all_input_ids, all_attention_mask, all_completion_mask, all_advantage, all_weight = [], [], [], [], []

        for sample in samples:
            prompt_len = len(sample['prompt_ids'])
            completion_len = len(sample['completion_ids'])

            ids = sample['prompt_ids'] + sample['completion_ids']
            pad_len = max_len - len(ids)

            # [중요] 오른쪽 패딩 (마스크 계산이 왼쪽 정렬을 전제로 함)
            all_input_ids.append(ids + ([pad_token_id] * pad_len))
            all_attention_mask.append(([1] * len(ids)) + ([0] * pad_len))

            '''
                shift 반영 : target 위치 i 는 input_ids[i+1] 을 예측
                    -> completion 토큰 [prompt_len, prompt_len+completion_len) 는
                       마스크 인덱스 [prompt_len-1, prompt_len+completion_len-1) 에 대응
            '''
            mask = [0.0] * (max_len - 1)
            for i in range(prompt_len-1, prompt_len+completion_len-1):
                mask[i] = 1.0

            all_completion_mask.append(mask)
            all_advantage.append(sample['advantage'])
            all_weight.append(sample['weight'])

        input_ids = torch.tensor(all_input_ids, dtype=torch.long, device=self._device)
        attention_mask = torch.tensor(all_attention_mask, dtype=torch.long, device=self._device)
        completion_mask = torch.tensor(all_completion_mask, dtype=torch.float32, device=self._device)
        advantage_tensor = torch.tensor(all_advantage, dtype=torch.float32, device=self._device).unsqueeze(1)
        weight_tensor = torch.tensor(all_weight, dtype=torch.float32, device=self._device)

        return input_ids, attention_mask, completion_mask, advantage_tensor, weight_tensor


    def _train_batch(self, query_results: List[QueryResult]):
        # 1. 데이터 준비 (실제 생성에 쓰인 프롬프트 + 실제 생성된 텍스트)
        samples = self._build_train_samples(query_results)
        n_samples = len(samples)

        if n_samples == 0:
            self._logging(f'RangerTrainer._train_batch() no trainable sample (all advantages are 0) -> skip')
            return 0.0, {'n_samples': 0, 'n_micro_batches': 0, 'n_chains': 0, 'kl': 0.0, 'completion_len': 0.0}

        # 체인 단위 정규화를 위한 전체 가중치 합 (= 학습 대상 체인 수)
        total_weight = sum(sample['weight'] for sample in samples)
        sum_loss, sum_kl = 0.0, 0.0

        micro_batches = self._make_micro_batches(samples)

        for micro_samples in micro_batches:
            input_ids, attention_mask, completion_mask, advantage_tensor, weight_tensor = self._collate_samples(micro_samples)

            # 2. 참조(reference) log_prob 계산 (adapter disable)
            '''
                inference_mode() 는 'inference tensor' 를 만들어서 autograd 그래프에 참여시킬 수 없음
                (RuntimeError: Inference tensors cannot be saved for backward) -> no_grad() 사용

                kl_penalty 가 0이면 KL 항 자체가 없으므로 reference forward 를 건너뜀 (학습 속도 2배)
            '''
            reference_log_probs = None

            if 0 < self._kl_penalty:
                with torch.no_grad():
                    with self._accelerator.unwrap_model(self._model).disable_adapter():
                        reference_log_probs = self._get_per_token_log_probs(self._model, input_ids, attention_mask).detach()

            # 3. 현재 정책(policy) log_prob 계산 (current model with adapter)
            policy_log_probs = self._get_per_token_log_probs(self._model, input_ids, attention_mask)

            # 4. loss 계산 및 역전파
            per_seq_loss, micro_kl = self._calculate_loss_per_batch(
                reference_log_probs, policy_log_probs, advantage_tensor, completion_mask
            )

            # 5. 체인 단위 가중 평균 + gradient accumulation 스케일링
            micro_loss = (per_seq_loss * weight_tensor).sum() / total_weight
            self._accelerator.backward(micro_loss / self._gradient_accumulation_steps)

            sum_loss += micro_loss.item()
            sum_kl += micro_kl * len(micro_samples)

            del reference_log_probs, policy_log_probs, per_seq_loss, micro_loss

        stats = {
            'n_samples': n_samples,
            'n_micro_batches': len(micro_batches),
            'n_chains': round(total_weight),      # advantage != 0 이라 실제로 학습에 쓰인 체인 수
            'kl': sum_kl / n_samples,
            'completion_len': sum(len(s['completion_ids']) for s in samples) / n_samples
        }
        self._logging(f'RangerTrainer._train_batch() stats : {stats}')

        # log 용으로는 accumulation 스케일링 전의 배치 loss 반환
        return sum_loss, stats


    def train(self, train_datas, test_datas, epochs, batch_size, n_chains, chain_depth):
        with self._accelerator.split_between_processes(train_datas) as train_datas_sharded:
            train_datas = train_datas_sharded
        with self._accelerator.split_between_processes(test_datas) as test_datas_sharded:
            test_datas = test_datas_sharded

        train_size = len(train_datas)

        # 기본 성능 측정
        self.evaluate(0, test_datas, batch_size, n_chains, chain_depth)

        self._logging(f'RangerTrainer.train() train_size : {train_size}, epochs : {epochs}, batch_size : {batch_size}, n_chains : {n_chains}, chain_depth : {chain_depth}')
        self._logging(f'RangerTrainer.train() train start : {common_utils.get_datetime_now()}')
        train_start = common_utils.get_time_ms()

        # accumulation 시작 전 gradient 초기화
        self._optimizer.zero_grad(set_to_none=True)

        for epoch in range(1, epochs+1):
            if epoch < self._resume_epoch:
                continue
            else:
                self._resume_epoch = -1

            epoch_start = common_utils.get_time_ms()

            if self._accelerator.is_main_process:
                self._reset_epoch()
            self._accelerator.wait_for_everyone()

            # 에폭마다 데이터 순서를 섞음 (프로세스 별로 다른 순서가 되지 않도록 epoch 기반 시드 사용)
            common_utils.shuffle_datas(train_datas, COMMON_CONFIG['seed'] + epoch)

            all_batch_loss = []
            batch_len = (train_size + batch_size - 1) // batch_size

            # 현재 accumulation 구간에서 실제로 gradient 가 쌓인 샘플 수
            accum_n_samples = 0

            for batch_idx, datas_batch in enumerate(container_utils.chunks(train_datas, batch_size)):
                if (batch_idx+1) <= self._resume_batch:
                    if (batch_idx+1) == self._resume_batch:
                        self._resume_batch = -1
                        self._global_step = ((epoch-1) * ((train_size + batch_size - 1) // batch_size)) + (batch_idx+1)
                    continue

                self._logging(f'RangerTrainer.train() {epoch} epoch, {batch_idx+1} batch start\t: {common_utils.get_datetime_now()}')
                batch_start = common_utils.get_time_ms()

                self._global_step += 1

                # 1. 체인 생성 (학습 배치와 체인 배치는 동일하게 맞춘 상태)
                # len(datas_batch) == batch_size == len(query_results)
                query_results: List[QueryResult] = request_chain_generate(
                    datas_batch,
                    batch_size,
                    n_chains,
                    chain_depth,
                    self._adapter_path
                )

                # 2. 체인 별로 각각 reward 계산
                # 3. grpo advantage 계산 (쿼리 별로 해당 쿼리의 체인 집합에 대하여 계산)
                self._reward_calculator.calculate_reward_and_advantage(query_results)
                
                # 4. 실제 train step (forward/backward)
                # 'batch_loss', 'all_batch_loss'는 둘 다 로깅 목적
                batch_loss, batch_stats = self._train_batch(query_results)
                all_batch_loss.append(batch_loss)
                accum_n_samples += batch_stats['n_samples']

                # 5. gradient accumulation
                '''
                    - 싱글 GPU를 사용할 경우, self.accelerator.sync_gradients 는 무조건 True
                        - 애초에 멀티 GPU 환경에서 사용되는 라이브러리
                        - RANGER 처럼 배치를 1로 하는 경우, 직접 카운팅해서 제어해야 함
                '''
                is_update_step = ((batch_idx+1) % self._gradient_accumulation_steps) == 0
                is_last_step = (batch_idx+1) == batch_len

                if self._accelerator.is_main_process:
                    common_utils.clear_gpu_memory()
                    common_utils.check_gpu_memory(do_print=DEBUG.TRAIN, msg=f'[{self._device}][Cleaned GPU]')

                # 여기는 메인 프로세스로 실행 X (내부적으로 알아서 동기화해서 파라미터 업데이트한다고 함)
                grad_norm = -1.0

                '''
                    구간 내 모든 배치가 스킵된 경우(그룹 내 리워드가 전부 같아 advantage 가 0)에는
                    쌓인 gradient 자체가 없으므로 optimizer step / 저장을 건너뜀
                        - 의미 없는 어댑터 재저장(1GB 디스크 쓰기)과 vLLM 재로드를 피함
                '''
                if (is_update_step or is_last_step) and (accum_n_samples == 0):
                    self._logging(f'RangerTrainer.train() {epoch} epoch, {batch_idx+1} batch no accumulated gradient -> skip optimizer step')

                elif is_update_step or is_last_step:
                    '''
                        [중요] RL은 advantage 스케일 때문에 gradient가 크게 튈 수 있으므로 clipping 필수
                        clip_grad_norm_() 반환값(clipping 전 norm)은 'gradient가 실제로 흐르는지' 확인하는 핵심 지표
                            - 계속 0.0 이면 LoRA에 gradient가 안 흐르는 것(= 학습이 전혀 안 되는 상태)
                    '''
                    if 0 < self._max_grad_norm:
                        grad_norm = float(self._accelerator.clip_grad_norm_(self._model.parameters(), self._max_grad_norm))

                    # 누적된 gradient 를 한 번에 반영해야 하므로, 먼저 step() 호출하고 다음에 zero_grad() 호출
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    self._logging(f'RangerTrainer.train() {epoch} epoch, {batch_idx+1} batch optimizer step (grad_norm : {grad_norm})')

                    # 6. 모델 저장 (여기도 그냥 호출, 내부에서 메인만 저장하도록 되어 있음)
                    # 파라미터가 갱신된 직후에만 저장해야, vLLM이 최신 정책으로 다음 롤아웃을 생성함
                    self._save(epoch, batch_idx+1)

                if is_update_step or is_last_step:
                    accum_n_samples = 0

                # wandb 에 별도 로깅
                self._wandb_logging_train_batch(epoch, batch_idx+1, batch_loss, batch_stats, grad_norm, query_results)

                self._logging(f'RangerTrainer.train() {epoch} epoch, {batch_idx+1} batch all_batch_loss\t: {all_batch_loss[-10:]}')
                self._logging(f'RangerTrainer.train() {epoch} epoch, {batch_idx+1} batch sum_batch_loss\t: {sum(all_batch_loss)}')
                self._logging(f'RangerTrainer.train() {epoch} epoch, {batch_idx+1} batch avg_batch_loss\t: {sum(all_batch_loss) / len(all_batch_loss)}')

                _, batch_elapsed_str = common_utils.get_elapsed_time_ms(batch_start)
                self._logging(f'RangerTrainer.train() {epoch} epoch, {batch_idx+1} batch end\t: {common_utils.get_datetime_now()}, elapsed : {batch_elapsed_str}')
            
            _, epoch_elapsed_str = common_utils.get_elapsed_time_ms(epoch_start)
            self._logging(f'RangerTrainer.train() {epoch} epoch end : {common_utils.get_datetime_now()}, elapsed : {epoch_elapsed_str}')

            # 7. 에폭 단위 어댑터 스냅샷 (평가 결과를 보고 최종 모델을 고르기 위함)
            self._save_epoch_snapshot(epoch)

            # 8. 에폭 단위로 성능 측정
            self.evaluate(epoch, test_datas, batch_size, n_chains, chain_depth)
        
        _, train_elapsed_str = common_utils.get_elapsed_time_ms(train_start)
        self._logging(f'RangerTrainer.train() train end : {common_utils.get_datetime_now()}, elapsed : {train_elapsed_str}')


    def evaluate(self, epoch, datas, batch_size, n_chains, chain_depth, temperature=-9, top_p=-9, top_k=-9):
        if self._accelerator.is_main_process:
            self._reset_epoch()
        self._accelerator.wait_for_everyone()

        prefix = '[base]' if epoch == 0 else f'[{epoch} epoch]'

        local_ems, local_f1s, local_rewards, local_advantages, local_sources, local_token_stats = evaluation_utils.evaluate(
            prefix,
            datas,
            batch_size,
            n_chains,
            chain_depth,
            self._adapter_path,
            temperature,
            top_p,
            top_k,
            self._reward_calculator
        )

        # source(벤치마크) 별 집계 - 혼합 평가셋이라 단일 숫자는 공개 논문 수치와 직접 비교 불가
        self._wandb_logging_evaluate_by_source(epoch, local_sources, local_ems, local_f1s, local_token_stats)

        # wandb 에 별도 로깅
        self._wandb_logging_evaluate(epoch, local_ems, local_f1s, local_rewards, local_advantages)


    def _wandb_logging_train_batch(self, epoch, batch_step, batch_loss, batch_stats: dict, grad_norm, query_results: List[QueryResult]):
        local_rewards = [cr._reward for qr in query_results for cr in qr._chain_results]
        local_advantages = [cr._advantage for qr in query_results for cr in qr._chain_results]

        t_rewards = torch.tensor(local_rewards, device=self._device, dtype=torch.float32)
        t_advantages = torch.tensor(local_advantages, device=self._device, dtype=torch.float32)
        t_batch_loss = torch.tensor([batch_loss], device=self._device, dtype=torch.float32)
        t_kl = torch.tensor([batch_stats['kl']], device=self._device, dtype=torch.float32)

        all_reward = self._accelerator.gather(t_rewards)
        all_advantage = self._accelerator.gather(t_advantages)
        all_batch_loss = self._accelerator.gather(t_batch_loss)
        all_kl = self._accelerator.gather(t_kl)

        if self._accelerator.is_main_process:
            avg_reward = all_reward.mean().item()
            avg_advantage = all_advantage.mean().item()
            avg_batch_loss = all_batch_loss.mean().item()

            '''
                [진단 지표]
                    - reward_std : 그룹 내 reward가 전부 같으면 advantage가 0이 되어 gradient가 사라짐
                                   0에 계속 붙어 있으면 학습 신호 자체가 없는 상태
                    - grad_norm  : 0.0 이 계속되면 gradient가 안 흐르는 것
                    - kl         : 0에서 서서히 증가해야 정상 (급증하면 lr/kl_penalty 조정 필요)
            '''
            wandb.log({
                'epoch': epoch,
                'batch_step': batch_step,
                'train_batch/reward': avg_reward,
                'train_batch/reward_std': all_reward.std().item() if 1 < len(all_reward) else 0.0,
                'train_batch/advantage': avg_advantage,
                'train_batch/batch_loss': avg_batch_loss,
                'train_batch/kl': all_kl.mean().item(),
                'train_batch/grad_norm': grad_norm,
                'train_batch/n_samples': batch_stats['n_samples'],
                'train_batch/n_chains': batch_stats['n_chains'],
                'train_batch/n_micro_batches': batch_stats['n_micro_batches'],
                'train_batch/completion_len': batch_stats['completion_len']
            }, step=self._global_step)

            self._logging(f'RangerTrainer._wandb_logging_train_batch() avg_reward : {avg_reward}, (gathered len : {len(all_reward)})')
            self._logging(f'RangerTrainer._wandb_logging_train_batch() avg_advantage : {avg_advantage}, (gathered len : {len(all_advantage)})')
            self._logging(f'RangerTrainer._wandb_logging_train_batch() avg_batch_loss : {avg_batch_loss}, (gathered len : {len(all_batch_loss)})')

        self._accelerator.wait_for_everyone()


    def _wandb_logging_evaluate_by_source(self, epoch, local_sources, local_ems, local_f1s, local_token_stats):
        '''
            [주의] source 별 집계는 '현재 프로세스가 담당한 데이터'에 대해서만 계산됨
                   멀티 GPU 로 데이터를 나눠 처리하는 경우 프로세스마다 부분 집계가 되므로,
                   왜곡된 값을 남기지 않도록 단일 프로세스일 때만 기록함
                   (전체 평균 EM/F1 은 _wandb_logging_evaluate() 에서 gather 하므로 항상 정확)
        '''
        if 1 < self._accelerator.num_processes:
            self._logging('RangerTrainer._wandb_logging_evaluate_by_source() multi-process -> skip (부분 집계 방지)', main_only=True)
            return

        if not self._accelerator.is_main_process:
            return

        source_scores = evaluation_utils.aggregate_by_source(local_sources, local_ems, local_f1s, **local_token_stats)
        prefix = '[base]' if epoch == 0 else f'[{epoch} epoch]'
        evaluation_utils.print_source_scores(f'# [TRAIN] {prefix}', source_scores)

        log_data = {'epoch': epoch}
        for source, row in source_scores.items():
            if source == 'ALL':
                continue
            log_data[f'evaluate_source/{source}/em'] = row['em']
            log_data[f'evaluate_source/{source}/f1'] = row['f1']
            log_data[f'evaluate_source/{source}/tokens_per_query'] = row['total_tokens']

        wandb.log(log_data, step=self._global_step)


    def _wandb_logging_evaluate(self, epoch, local_ems, local_f1s, local_rewards, local_advantages):
        # em 은 0/1 int 라서 dtype 을 지정하지 않으면 Long 텐서가 되어 mean() 이 실패함
        t_ems = torch.tensor(local_ems, device=self._device, dtype=torch.float32)
        t_f1s = torch.tensor(local_f1s, device=self._device, dtype=torch.float32)
        t_rewards = torch.tensor(local_rewards, device=self._device, dtype=torch.float32)
        t_advantages = torch.tensor(local_advantages, device=self._device, dtype=torch.float32)

        all_em = self._accelerator.gather(t_ems)
        all_f1 = self._accelerator.gather(t_f1s)
        all_reward = self._accelerator.gather(t_rewards)
        all_advantage = self._accelerator.gather(t_advantages)

        if self._accelerator.is_main_process:
            avg_em = all_em.mean().item()
            avg_f1 = all_f1.mean().item()
            avg_reward = all_reward.mean().item()
            avg_advantage = all_advantage.mean().item()

            wandb.log({
                'epoch': epoch,
                'evaluate/em': avg_em,
                'evaluate/f1': avg_f1,
                'evaluate/reward': avg_reward,
                'evaluate/advantage': avg_advantage
            }, step=self._global_step)

            self._logging(f'RangerTrainer._wandb_logging_evaluate() avg_em : {avg_em}, (gathered len : {len(all_em)})')
            self._logging(f'RangerTrainer._wandb_logging_evaluate() avg_f1 : {avg_f1}, (gathered len : {len(all_f1)})')
            self._logging(f'RangerTrainer._wandb_logging_evaluate() avg_reward : {avg_reward}, (gathered len : {len(all_reward)})')
            self._logging(f'RangerTrainer._wandb_logging_evaluate() avg_advantage : {avg_advantage}, (gathered len : {len(all_advantage)})')

        self._accelerator.wait_for_everyone()

