from _init import *

import shutil

import torch
from torch.optim import AdamW

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator

from ranger.utils import common_utils, json_utils
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
        self._tokenizer: AutoTokenizer = None
        self._optimizer: AdamW = None
        self._accelerator = Accelerator(gradient_accumulation_steps=self._gradient_accumulation_steps)
        self._device = self._accelerator.device
        self._local_gpu_id = self._accelerator.local_process_index

        self._logging(f'RangerTrainer._model_config :\n{json_utils.to_str(self._model_config)}\n', True)

        self._set_out_dir() # accelerator 초기화 이후에 호출
        self._init_model()


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
        if self._resume_run_time:
            self._resume_checkpoint_path = f'{self._out_dir}/checkpoint_{self._resume_run_time}'
            self._resume_optimizer_path = f'{self._resume_checkpoint_path}/optimizer.pt'
            self._resume_adapter_path = f'{self._out_dir}/lora_adapter_{self._resume_run_time}'

            # 파일 복사는 오직 메인 프로세스만 수행 (안 그러면 서로 만들다가 충돌남)
            if self._accelerator.is_main_process:
                self._copy_from_resume()


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
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

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

