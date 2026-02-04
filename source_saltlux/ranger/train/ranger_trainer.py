from _init import *

import shutil, wandb, re
from typing import List
import numpy as np

import torch
from torch.optim import AdamW
from torch.nn.functional import log_softmax

from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator

from ranger.utils import common_const, common_utils, json_utils, container_utils, tokenizer_utils
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
        self._tokenizer: PreTrainedTokenizerFast = tokenizer_utils.load_tokenizer(self._model_name)

        # 3. Gradient Checkpointing
        if self._use_gradient_checkpointing:
            self._model.gradient_checkpointing_enable()
        
        common_utils.check_gpu_memory(do_print=DEBUG.TRAIN, msg=f'[{self._device}][Base model init]')

        # 4. PEFT (LoRA) 모델 초기화
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
                lora_dropout=0.05
            )

            self._model = get_peft_model(self._model, peft_config)
        
        if DEBUG.TRAIN and self._accelerator.is_main_process:
            print()
            self._model.print_trainable_parameters()

        common_utils.check_gpu_memory(do_print=DEBUG.TRAIN, msg=f'[{self._device}][PEFT model init]')

        # 5. Optimizer 초기화
        self._optimizer = AdamW(self._model.parameters(), lr=self._learning_rate)

        # 6. Accelerator Prepare (모델과 옵티마이저를 Accelerator가 관리하도록 래핑)
        self._model, self._optimizer = self._accelerator.prepare(self._model, self._optimizer)

        # 7. Resume Optimizer State
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


    def _write_save_history(self, epoch, batch, saved_time):
        msg = f'[{saved_time}] - [ {epoch} epoch, {batch} batch ], path : '

        with open(self._checkpoint_history_path, 'a', encoding='utf-8') as f:
            f.write(f'{msg}{self._checkpoint_path}\n')
        
        with open(self._adapter_history_path, 'a', encoding='utf-8') as f:
            f.write(f'{msg}{self._adapter_path}\n')


    def _reset_epoch(self):
        request_reset()


    # 하나의 쿼리에 대한 모든 체인들의 리워드를 정규화하여 advantage 계산
    def _calculate_advantage(self, chain_results: List[ChainResult]):
        # 1. NumPy 배열로 변환
        rewards = np.array([cr._reward for cr in chain_results], dtype=np.float32)

        # 2. 평균 및 표준편차 계산
        mean_reward = rewards.mean()
        std_reward = rewards.std() if len(rewards) > 1 else 0.0

        # 3. 정규화
        advantages = (rewards - mean_reward) / (std_reward + 1e-8)

        # 4. 결과 저장
        for i, chain_result in enumerate(chain_results):
            chain_result._advantage = float(advantages[i])


    def _get_per_token_log_probs(self, model, input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits

        # 마지막 logit 제거 (다음 토큰이 없으므로)
        shifted_logits = logits[:, :-1, :]
        del logits, outputs
        common_utils.clear_gpu_memory()

        # log softmax 변환
        log_probs = log_softmax(shifted_logits, dim=-1)
        del shifted_logits
        common_utils.clear_gpu_memory()

        # 정답 토큰 ID도 위치를 맞춤 (맨 앞 토큰 제거)
        shifted_target_ids = input_ids[:, 1:]

        # 실제 정답 토큰 ID에 해당하는 log 확률만 추출
        target_log_probs = torch.gather(log_probs, 2, shifted_target_ids.unsqueeze(-1)).squeeze(-1)
        del log_probs
        common_utils.clear_gpu_memory()

        return target_log_probs


    def _calculate_loss_per_batch(self, reference_log_probs, policy_log_probs, all_advantage_tensor, all_query_len, attention_mask):
        # FP16 오버플로우 방지를 위해 강제로 float32로 변환
        reference_log_probs = reference_log_probs.to(torch.float32)
        policy_log_probs = policy_log_probs.to(torch.float32)

        # r(i) 값 계산
        log_ratio = policy_log_probs - policy_log_probs.detach()
        ratio = torch.exp(log_ratio)

        # PPO loss
        loss_unclipped = ratio * all_advantage_tensor
        loss_clipped = torch.clamp(ratio, 1.0-MODEL_CONFIG['epsilon'], 1.0+MODEL_CONFIG['epsilon']) * all_advantage_tensor
        ppo_objective = torch.min(loss_unclipped, loss_clipped)

        # KL-Divergence
        kl_div = torch.exp(reference_log_probs - policy_log_probs) - (reference_log_probs - policy_log_probs) - 1

        # Ranger(GRPO) loss
        '''
            우리의 목적 함수는 다음과 같음
                - PPO 목적 함수는 '최대화', KL 페널티는 '최소화'

                - PyTorch 옵티마이저는 Loss를 '최소화'하므로, 이 목표에 맞게 수식을 변환해야 함
                    - PPO 목적 함수를 'A', KL 페널티를 'B'라고 한다면,
                        - 최종 수식은 '-A + B'가 되어야 함
                        - 이걸 다시 변환해서 '-(A - B)'로 계산
        '''
        loss_per_token = -(ppo_objective - (MODEL_CONFIG['kl_penalty'] * kl_div))

        '''
            - prompt 부분은 제외하고, padding이 아닌 completion 부분만 마스킹
            - 그 다음 completion 부분에 대해서만 loss를 최종 계산

                - 각 시퀀스의 query 길이를 텐서로 변환
                  attention_mask를 이용해 패딩 부분을 제외하고,
                  각 시퀀스의 query 길이만큼 앞부분을 추가로 제외
                  shifted_logps에 맞춰 마스크도 맨 앞 제거
        '''
        full_mask = attention_mask[:, 1:]
        all_query_len_tensor = torch.tensor(all_query_len, device=self._device).unsqueeze(1)
        indices = torch.arange(full_mask.shape[1], device=self._device).expand(full_mask.shape[0], -1)
        prompt_mask = (indices < all_query_len_tensor).float()
        final_mask = full_mask * (1 - prompt_mask)

        '''
            마스크를 이용해서 completion에 대한 loss만 최종 계산
        '''
        masked_loss = loss_per_token * final_mask
        sum_masked_loss = masked_loss.sum(dim=1)
        sum_mask = final_mask.sum(dim=1).clamp(min=1e-8)
        batch_loss = (sum_masked_loss / sum_mask).mean()

        return batch_loss


    def _train_batch(self, query_results: List[QueryResult]):
        # 1. 데이터 준비 (flattening)
        all_full_text = []
        all_query_len = []
        all_advantage = []

        for query_result in query_results:
            query = query_result._query
            query_len = len(self._tokenizer(query, add_special_tokens=False)['input_ids'])

            for chain_result in query_result._chain_results:
                completion = chain_result.make_get_completion()
                all_full_text.append(query + completion)
                all_query_len.append(query_len)
                all_advantage.append(chain_result._advantage)
        
        # 2. 배치 토크나이징
        inputs = self._tokenizer(
            all_full_text,
            padding=True,
            truncation=True,
            max_length=self._max_seq_length,
            return_tensors="pt"
        ).to(self._device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        all_advantage_tensor = torch.tensor(all_advantage, device=self._device, dtype=torch.float32).unsqueeze(1)

        # 3. 참조(reference) logits 계산 (adapter disable)
        with torch.inference_mode():
            with self._accelerator.unwrap_model(self._model).disable_adapter():
                reference_log_probs = self._get_per_token_log_probs(self._model, input_ids, attention_mask)
                reference_log_probs = reference_log_probs.detach() # 그래프 분리
        
        # 4. 현재 정책(policy) logits 계산 (current model with adapter)
        policy_log_probs = self._get_per_token_log_probs(self._model, input_ids, attention_mask)

        # 5. loss 계산 및 역전파
        batch_loss = self._calculate_loss_per_batch(
            reference_log_probs, policy_log_probs, all_advantage_tensor, all_query_len, attention_mask
        )

        # 6. gradient accumulation을 위해 loss를 n분의 1로 줄여줌
        loss_to_backward = batch_loss / self._gradient_accumulation_steps

        self._accelerator.backward(loss_to_backward)

        # reference_log_probs는 detach된 텐서라 명시적으로 지워주는 것이 좋음
        del reference_log_probs

        # log 용으로는 원래 loss 반환
        return batch_loss.item()


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

        for epoch in range(1, epochs+1):
            if epoch < self._resume_epoch:
                continue
            else:
                self._resume_epoch = -1

            epoch_start = common_utils.get_time_ms()

            if self._accelerator.is_main_process:
                self._reset_epoch()
            self._accelerator.wait_for_everyone()

            all_batch_loss = []
            batch_len = (train_size + batch_size - 1) // batch_size

            for batch_idx, datas_batch in enumerate(container_utils.chunks(train_datas, batch_size)):
                if (batch_idx+1) <= self._resume_batch:
                    if (batch_idx+1) == self._resume_batch:
                        self._resume_batch = -1
                        self._global_step = ((epoch-1) * train_size) + (batch_idx+1)
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

                # 2. 체인 별로 각각 리워드 계산
                self._reward_calculator.calculate_reward(query_results)

                # 3. grpo advantage 계산 (쿼리 별로 해당 쿼리의 체인 집합에 대하여 계산)
                for query_result in query_results:
                    self._calculate_advantage(query_result._chain_results)
                
                # 4. 실제 train step (forward/backward)
                # 'batch_loss', 'all_batch_loss'는 둘 다 로깅 목적
                batch_loss = self._train_batch(query_results)
                all_batch_loss.append(batch_loss)

                # 5. accelerator 를 이용한 gradient accumulation
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
                if (self._gradient_accumulation_steps == 1) or is_update_step or is_last_step:
                    # 누적된 gradient 를 한 번에 반영해야 하므로, 먼저 step() 호출하고 다음에 zero_grad() 호출
                    self._optimizer.step()
                    self._optimizer.zero_grad()
                    self._logging(f'RangerTrainer.train() {epoch} epoch, {batch_idx+1} batch sync gradients (accelerator)')

                    # 6. 모델 저장 (여기도 그냥 호출, 내부에서 메인만 저장하도록 되어 있음)
                    self._save(epoch, batch_idx+1)

                # wandb 에 별도 로깅
                self._wandb_logging_train_batch(epoch, batch_idx+1, batch_loss, query_results)

                self._logging(f'RangerTrainer.train() {epoch} epoch, {batch_idx+1} batch all_batch_loss\t: {all_batch_loss[-10:]}')
                self._logging(f'RangerTrainer.train() {epoch} epoch, {batch_idx+1} batch sum_batch_loss\t: {sum(all_batch_loss)}')
                self._logging(f'RangerTrainer.train() {epoch} epoch, {batch_idx+1} batch avg_batch_loss\t: {sum(all_batch_loss) / len(all_batch_loss)}')

                _, batch_elapsed_str = common_utils.get_elapsed_time_ms(batch_start)
                self._logging(f'RangerTrainer.train() {epoch} epoch, {batch_idx+1} batch end\t: {common_utils.get_datetime_now()}, elapsed : {batch_elapsed_str}')
            
            _, epoch_elapsed_str = common_utils.get_elapsed_time_ms(epoch_start)
            self._logging(f'RangerTrainer.train() {epoch} epoch end : {common_utils.get_datetime_now()}, elapsed : {epoch_elapsed_str}')

            # 7. 에폭 단위로 성능 측정
            self.evaluate(epoch, test_datas, batch_size, n_chains, chain_depth)
        
        _, train_elapsed_str = common_utils.get_elapsed_time_ms(train_start)
        self._logging(f'RangerTrainer.train() train end : {common_utils.get_datetime_now()}, elapsed : {train_elapsed_str}')


    def evaluate(self, epoch, datas, batch_size, n_chains, chain_depth):
        if self._accelerator.is_main_process:
            self._reset_epoch()
        self._accelerator.wait_for_everyone()

        prefix = '[base]' if epoch == 0 else f'[{epoch} epoch]'
        data_size = len(datas)

        self._logging(f'RangerTrainer.evaluate() {prefix} data_size : {data_size}, batch_size : {batch_size}, n_chains : {n_chains}, chain_depth : {chain_depth}')
        self._logging(f'RangerTrainer.evaluate() {prefix} eval start : {common_utils.get_datetime_now()}')
        eval_start = common_utils.get_time_ms()

        local_rewards = []
        local_advantages = []
        local_f1s = []

        for batch_idx, datas_batch in enumerate(container_utils.chunks(datas, batch_size)):
            self._logging(f'RangerTrainer.evaluate() {prefix} {batch_idx+1} batch start\t: {common_utils.get_datetime_now()}')
            batch_start = common_utils.get_time_ms()

            query_results: List[QueryResult] = request_chain_generate(
                datas_batch,
                batch_size,
                n_chains,
                chain_depth,
                self._adapter_path
            )

            self._reward_calculator.calculate_reward(query_results)

            for query_result in query_results:
                self._calculate_advantage(query_result._chain_results)

                best_reward = common_const.NULL_FLOAT
                best_advantage = common_const.NULL_FLOAT
                best_f1 = common_const.NULL_FLOAT

                for chain_result in query_result._chain_results:
                    if best_f1 < chain_result._f1:
                        best_reward = chain_result._reward
                        best_advantage = chain_result._advantage
                        best_f1 = chain_result._f1
                
                local_rewards.append(best_reward)
                local_advantages.append(best_advantage)
                local_f1s.append(best_f1)
            
            _, batch_elapsed_str = common_utils.get_elapsed_time_ms(batch_start)
            self._logging(f'RangerTrainer.evaluate() {prefix} {batch_idx+1} batch end\t: {common_utils.get_datetime_now()}, elapsed : {batch_elapsed_str}')

        # wandb 에 별도 로깅
        self._wandb_logging_evaluate(epoch, local_rewards, local_advantages, local_f1s)

        _, eval_elapsed_str = common_utils.get_elapsed_time_ms(eval_start)
        self._logging(f'RangerTrainer.evaluate() {prefix} eval end : {common_utils.get_datetime_now()}, elapsed : {eval_elapsed_str}')


    def _wandb_logging_train_batch(self, epoch, batch_step, batch_loss, query_results: List[QueryResult]):
        local_rewards = [cr._reward for qr in query_results for cr in qr._chain_results]
        local_advantages = [cr._advantage for qr in query_results for cr in qr._chain_results]

        t_rewards = torch.tensor(local_rewards, device=self._device)
        t_advantages = torch.tensor(local_advantages, device=self._device)
        t_batch_loss = torch.tensor([batch_loss], device=self._device)

        all_reward = self._accelerator.gather(t_rewards)
        all_advantage = self._accelerator.gather(t_advantages)
        all_batch_loss = self._accelerator.gather(t_batch_loss)

        if self._accelerator.is_main_process:
            avg_reward = all_reward.mean().item()
            avg_advantage = all_advantage.mean().item()
            avg_batch_loss = all_batch_loss.mean().item()

            wandb.log({
                'epoch': epoch,
                'batch_step': batch_step,
                'train_batch/reward': avg_reward,
                'train_batch/advantage': avg_advantage,
                'train_batch/batch_loss': avg_batch_loss
            }, step=self._global_step)

            self._logging(f'RangerTrainer._wandb_logging_train_batch() avg_reward : {avg_reward}, (gathered len : {len(all_reward)})')
            self._logging(f'RangerTrainer._wandb_logging_train_batch() avg_advantage : {avg_advantage}, (gathered len : {len(all_advantage)})')
            self._logging(f'RangerTrainer._wandb_logging_train_batch() avg_batch_loss : {avg_batch_loss}, (gathered len : {len(all_batch_loss)})')
        
        self._accelerator.wait_for_everyone()


    def _wandb_logging_evaluate(self, epoch, local_rewards, local_advantages, local_f1s):
        t_rewards = torch.tensor(local_rewards, device=self._device)
        t_advantages = torch.tensor(local_advantages, device=self._device)
        t_f1s = torch.tensor(local_f1s, device=self._device)

        all_reward = self._accelerator.gather(t_rewards)
        all_advantage = self._accelerator.gather(t_advantages)
        all_f1 = self._accelerator.gather(t_f1s)

        if self._accelerator.is_main_process:
            avg_reward = all_reward.mean().item()
            avg_advantage = all_advantage.mean().item()
            avg_f1 = all_f1.mean().item()

            wandb.log({
                'epoch': epoch,
                'evaluate/reward': avg_reward,
                'evaluate/advantage': avg_advantage,
                'evaluate/f1': avg_f1
            }, step=self._global_step)

            self._logging(f'RangerTrainer._wandb_logging_evaluate() avg_reward : {avg_reward}, (gathered len : {len(all_reward)})')
            self._logging(f'RangerTrainer._wandb_logging_evaluate() avg_advantage : {avg_advantage}, (gathered len : {len(all_advantage)})')
            self._logging(f'RangerTrainer._wandb_logging_evaluate() avg_f1 : {avg_f1}, (gathered len : {len(all_f1)})')
        
        self._accelerator.wait_for_everyone()

