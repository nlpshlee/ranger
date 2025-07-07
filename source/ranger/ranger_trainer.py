from _init import *

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.import_utils import is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax

from ranger.grpo.repeat_sampler import RepeatRandomSampler # 클래스 떼서 새로 붙였답니다.
from ranger.grpo import init_utils
from ranger.grpo.prompt_processor import PromptProcessor # 1. 프롬프트 전처리 단계
from ranger.grpo.evaluation_utils import prediction_step as evaluation_prediction_step # 평가 함수
from ranger.grpo.logging_utils import log_metrics # 로깅 함수
from ranger.grpo.loss_utils import compute_loss as compute_loss_func # 손실 계산 함수


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RANGERTrainer(Trainer):
    """
    Group Relative Policy Optimization (GRPO) 방법을 위한 트레이너입니다. 이 알고리즘은 처음에
    [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300) 논문에서 제안되었습니다.

    예시:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # 더 많은 고유 문자를 가진 완성에 보상을 주는 더미 보상 함수입니다.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            훈련할 모델입니다. 다음 중 하나일 수 있습니다:

            - 문자열: huggingface.co의 모델 저장소 내에서 호스팅되는 사전 훈련된 모델의 *모델 ID*이거나,
              [`~transformers.PreTrainedModel.save_pretrained`]를 사용하여 저장된 모델 가중치를 포함하는 *디렉토리*의 경로입니다.
              예: `'./my_model_directory/'`. 모델은 `args.model_init_kwargs`의 키워드 인수를 사용하여
              [`~transformers.AutoModelForCausalLM.from_pretrained`]로 로드됩니다.
            - [`~transformers.PreTrainedModel`] 객체입니다. 인과적 언어 모델만 지원됩니다.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            보상을 계산하는 데 사용할 보상 함수들입니다. 보상을 계산하기 위해 모든 보상 함수를
            프롬프트와 완성과 함께 호출하고 보상을 합산합니다. 다음 중 하나일 수 있습니다:

            - 단일 보상 함수, 예를 들어:
                - 문자열: huggingface.co의 모델 저장소 내에서 호스팅되는 사전 훈련된 모델의 *모델 ID*이거나,
                [`~transformers.PreTrainedModel.save_pretrained`]를 사용하여 저장된 모델 가중치를 포함하는 *디렉토리*의 경로입니다.
                예: `'./my_model_directory/'`. 모델은 `num_labels=1`과 `args.model_init_kwargs`의 키워드 인수를 사용하여
                [`~transformers.AutoModelForSequenceClassification.from_pretrained`]로 로드됩니다.
                - [`~transformers.PreTrainedModel`] 객체: 시퀀스 분류 모델만 지원됩니다.
                - 사용자 정의 보상 함수: 함수는 프롬프트와 생성된 완성, 그리고 데이터셋의 추가 열들과 함께 제공됩니다.
                  보상 목록을 반환해야 합니다. 자세한 내용은 [사용자 정의 보상 함수 사용](#using-a-custom-reward-function)을 참조하세요.
            - 보상 함수들의 목록으로, 각 항목은 위의 유형 중 하나일 수 있습니다. 목록 내에서 서로 다른 유형을 혼합하는 것
            (예: 문자열 모델 ID와 사용자 정의 보상 함수)이 허용됩니다.
        args ([`GRPOConfig`], *optional*, 기본값 `None`):
            이 트레이너의 구성입니다. `None`인 경우 기본 구성이 사용됩니다.
        train_dataset ([`~datasets.Dataset`] 또는 [`~datasets.IterableDataset`]):
            훈련에 사용할 데이터셋입니다. `"prompt"` 열을 포함해야 합니다. 데이터셋의 추가 열들은 무시됩니다.
            샘플의 형식은 다음 중 하나일 수 있습니다:

            - [표준](dataset_formats#standard): 각 샘플은 일반 텍스트를 포함합니다.
            - [대화형](dataset_formats#conversational): 각 샘플은 구조화된 메시지를 포함합니다 (예: 역할과 내용).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] 또는 `dict[str, Union[Dataset, IterableDataset]]`):
            평가에 사용할 데이터셋입니다. `train_dataset`와 동일한 요구사항을 충족해야 합니다.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, 기본값 `None`):
            데이터를 처리하는 데 사용되는 처리 클래스입니다. 패딩 측면은 "left"로 설정되어야 합니다. `None`인 경우,
            처리 클래스는 [`~transformers.AutoTokenizer.from_pretrained`]를 사용하여 모델의 이름에서 로드됩니다.
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, 기본값 `None`):
            `reward_funcs`에 지정된 보상 함수에 해당하는 처리 클래스들입니다. 다음 중 하나일 수 있습니다:

            - 단일 처리 클래스: `reward_funcs`에 보상 함수가 하나만 포함된 경우 사용됩니다.
            - 처리 클래스들의 목록: `reward_funcs`의 보상 함수들의 순서와 길이와 일치해야 합니다.
            `None`으로 설정되거나, [`~transformers.PreTrainedModel`]에 해당하는 목록의 요소가 `None`인 경우,
            모델의 토크나이저는 [`~transformers.AutoTokenizer.from_pretrained`]를 사용하여 자동으로 로드됩니다.
            `reward_funcs`에서 사용자 정의 보상 함수(not [`~transformers.PreTrainedModel`])인 요소의 경우,
            `reward_processing_classes`의 해당 항목들은 무시됩니다.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, 기본값 `None`):
            훈련 루프를 사용자 정의하기 위한 콜백들의 목록입니다. [여기](https://huggingface.co/docs/transformers/main_classes/callback)에
            자세히 설명된 기본 콜백들의 목록에 추가됩니다.

            사용된 기본 콜백 중 하나를 제거하려면 [`~transformers.Trainer.remove_callback`] 메서드를 사용하세요.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, 기본값 `(None, None)`):
            사용할 옵티마이저와 스케줄러를 포함하는 튜플입니다. 모델의 [`AdamW`] 인스턴스와 `args`에 의해 제어되는
            [`get_linear_schedule_with_warmup`]에 의해 제공되는 스케줄러로 기본 설정됩니다.
        peft_config ([`~peft.PeftConfig`], *optional*, 기본값 `None`):
            모델을 래핑하는 데 사용되는 PEFT 구성입니다. `None`인 경우 모델은 래핑되지 않습니다.
    """

    _tag_names = ["trl", "grpo"]
    # 분리를 여러번 시도했지만...왜 안될까요.
    # vllm 사용 여부에 따라 생성 환경을 준비하는 함수
    def _prepare_generation_environment(self, args, processing_class):
        if self.use_vllm:
            # vllm이 설치되어 있는지 확인하고, 없으면 오류 발생
            if not is_vllm_available():
                raise ImportError(
                    "vLLM을 사용할 수 없고 `use_vllm`이 True로 설정되어 있습니다. vLLM을 사용하려면 "
                    "`pip install vllm`으로 설치하세요."
                )
            # 메인 프로세스에서만 vllm초기화: 분산 학습에서 메인 프로세스에서만 vllm 모델을 로드하여 메모리 효율성 확보
            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    if torch.cuda.device_count() == 1:
                        vllm_device = "cuda:0"  # 단일 GPU로 학습할 때의 특별한 경우: GPU를 공유
                    else:
                        vllm_device = f"cuda:{self.accelerator.num_processes}"  # 다음 GPU 인덱스 사용
                # 요청된 디바이스가 사용 가능한지 확인
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"vLLM용으로 요청된 디바이스({vllm_device})를 사용할 수 없습니다. vLLM을 사용할 때 "
                        "학습용 GPU 수를 제한하지 않고 있을 가능성이 높습니다. `--num_processes` 인자를 "
                        "머신에서 사용 가능한 GPU 수보다 낮은 값으로 설정하세요—일반적으로 하나를 줄이는 것이 "
                        f"충분합니다. 귀하의 경우: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # 요청된 디바이스가 학습에도 사용되고 있지 않은지 확인
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"요청된 디바이스 {vllm_device}가 학습에도 사용되고 있습니다. 더 높은 처리량과 "
                        "메모리 부족 오류를 방지하기 위해 vLLM용 전용 디바이스를 사용하는 것이 권장됩니다. "
                        "이것이 의도적인 경우라면 이 경고를 무시할 수 있지만 "
                        "`vllm_gpu_memory_utilization`을 적절히 조정해야 합니다."
                    )
                # vLLM은 accelerate와 호환되지 않습니다. 따라서 패치를 통해 (1) vLLM 모델을 원하는 디바이스에 배치하고
                # (2) 우리 설정에 맞지 않는 테스트를 피할 수 있도록 해야 합니다 (world_size_patch 및 profiling_patch).
                # vllm 모델 초기화
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )
                with world_size_patch, profiling_patch:
                    self.llm = LLM(
                        model=self.model.name_or_path, # 사용할 모델 경로
                        device=vllm_device, # vllm이 사용할 gpu
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization, # gpu메모리 사용률
                        dtype=self.args.vllm_dtype, # 모델 데이터 타입
                        enable_prefix_caching=True, # 프리픽스 캐싱 활성화(성능 향상)
                        max_model_len=self.args.vllm_max_model_len, # 최대 모델 길이
                    )
                # 샘플링 파라미터 설정
                self.sampling_params = SamplingParams(
                    temperature=args.temperature, # 샘플링 온도
                    max_tokens=self.max_completion_length, # 최대 생성 토큰 수
                )
            # 초기화 태그 설정
            self._last_loaded_step = 0  # 그래디언트 누적 중 불필요한 로딩을 방지하기 위한 태그

            # vLLM을 사용할 때, 메인 프로세스가 모델 가중치를 로드하는 책임을 집니다. 이는 프로세스 동기화 해제를
            # 일으킬 수 있고 DeepSpeed 초기화 중 멈춤 현상을 야기하는 것 같습니다. 이를 방지하기 위해 vLLM이
            # 완전히 초기화된 후 모든 프로세스를 동기화합니다. -> vllm 초기화 완료 후 모든 프로세스 동기화
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=processing_class.pad_token_id,
            )
    # 참조 모델과 보상 모델을 준비하는 함수
    def _prepare_reference_and_reward_models(self, args):
        # 참조 모델 준비, 분산 학습 환경에서 모델을 준비하는 함수. 참조 모델이 있을 때만 실행되는 블록인데 우리는 참조모델이 없고, 기존 모델을 참조 모델로 사용하기에 실행되지 않는 블록.
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))
        # 보상 모델을 쓰면 모델 준비 함수 호출(신경망 모델인 경우만), 근데 우린 커스텀 보상 함수를 쓰니까 실제로 동작하지 않음 - 사실상 동작하지 않는 코드
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def __init__(
        self,
        model: Union[str, PreTrainedModel], # 모델 경로 또는 모델 객체
        reward_funcs: Union[RewardFunc, list[RewardFunc]], # 보상 함수 리스트
        args: GRPOConfig = None, # GRPO 설정 객체, None이면 기본 설정 사용
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None, # 학습 데이터
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None, # 평가 데이터
        processing_class: Optional[PreTrainedTokenizerBase] = None, # 토크나이저 클래스
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None, # 보상 함수 토크나이저 클래스
        callbacks: Optional[list[TrainerCallback]] = None, # 학습 중에 호출될 콜백 리스트
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None), # 옵티마이저와 스케줄러
        peft_config: Optional["PeftConfig"] = None, # PEFT 설정
    ):
        # 1. 설정 검증 및 초기화: 모델과 args를 확인하고, args가 None이면 기본 설정 사용
        args = init_utils.check_args(model, args)
        # 2. 모델 초기화: 모델 경로 또는 모델 객체를 초기화하고, PEFT 설정을 적용
        model_init_kwargs = args.model_init_kwargs or {}
        self.model, model_id = init_utils.init_model(model, model_init_kwargs, args, peft_config)
        # 3. 참조 모델 초기화
        self.ref_model = None

        # Processing class
        # 4. 토크나이저 초기화: 토크나이저 클래스를 초기화하고, 모델에 맞는 토크나이저 자동 로드
        processing_class = init_utils.init_processing_class(processing_class, self.model)

        # Reward functions
        # 5. 보상 함수 초기화: 보상 함수를 초기화하고, 모델 초기화 인수를 적용 -> 우린 커스텀 함수여서 그대로 유지
        self.reward_funcs = init_utils.init_reward_funcs(reward_funcs, model_init_kwargs)

        # Reward weights
        # 6. 보상 가중치 초기화: 보상 가중치를 초기화하고, 모델 초기화 인수를 적용 -> 우린 아직 안씀 -> 그대로 1.0 유지 혹시 필요하면 이거 안하고 커스텀 리워드 함수 내에서 가중치 주면서 return해도 됨
        self.reward_weights = init_utils.init_reward_weights(args, self.reward_funcs)

        # Reward processing class
        # 7.보상함수용 토크나이저 초기화: 보상 함수에 대응하는 토크나이저 클래스를 초기화하고, 모델 초기화 인수를 적용 -> 우린 커스텀 함수여서 사실상 의미없이 실행되는 코드
        reward_processing_classes = init_utils.init_reward_processing_classes(
            self.reward_funcs, reward_processing_classes, model_init_kwargs
        )
        self.reward_processing_classes = reward_processing_classes

        # Data collator 배치 단위로 데이터를 묶어서 모델에 전달할 때,각 샘플을 어떻게 하나의 배치로 합칠지(패딩, 텐서 변환 등) 정의하는 함수 
        # 8. 데이터 콜레이터 설정 -> GRPO에서는 데이터 콜레이터가 필요없음, 그래서 기본값 사용
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        # 9. 학습 파라미터 설정
        self.max_prompt_length = args.max_prompt_length # 최대 프롬프트 길이
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.use_vllm = args.use_vllm # vllm 사용 여부

        self.beta = args.beta

        # 프롬프트 프로세서 초기화 아래 1단계 프롬프트 전처리 단계
        # 10. 프롬프트 프로세서 초기화, 프롬프트 전처리를 위한 프로세서 생성
        self.prompt_processor = PromptProcessor(processing_class, self.max_prompt_length)

        # 트레이너는 "input_ids" 키와 관련된 입력 텐서의 요소 수를 사용하여 FLOP(부동소수점 연산) 수를 추정합니다.
        # 그러나 GRPO에서는 샘플링된 데이터에 "input_ids" 키가 포함되지 않습니다. 대신 사용 가능한 키는 "prompt"입니다.
        # 결과적으로 트레이너는 다음 경고를 발생시킵니다:
        # "입력의 토큰 수를 추정할 수 없어서 부동소수점 연산이 계산되지 않습니다."
        # 이 경고를 억제하기 위해 모델의 "warnings_issued" 딕셔너리에서 "estimate_tokens" 키를 True로 설정합니다.
        # 이는 경고가 이미 발생했음을 나타내는 플래그 역할을 합니다.
        
        # 11. 경고 억제를 위한 플래그 설정
        self.model.warnings_issued["estimate_tokens"] = True

        # 12. 메트릭 초기화 -> 학습 중 메트릭 저장을 위한 딕셔너리.
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions # 완성 로깅 여부

        # 13. 부모 클래스 초기화, 부모 클래스는 트레이너 클래스의 기본 클래스인 Trainer 클래스 -> 이 클래스는 학습 데이터셋, 평가 데이터셋, 토크나이저, 콜백, 옵티마이저 등을 초기화하는 데 사용
        super().__init__(
            model=self.model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        # 14. 생성 배치 호환성 검증, 배치 크기와 생성 개수의 호환성 확인
        init_utils.check_generation_batch_compatibility(args, self.accelerator, self.num_generations)
        
        # num_generations가 per_device_train_batch_size를 초과할 때 transformers로 생성할 때 중복 완성본을 방지하기 위해
        # 각 프로세스가 고유한 시드를 받도록 보장합니다. vLLM을 사용하는 경우 건너뛸 수 있지만,
        # 모든 경우에 설정하는 것이 더 안전합니다.
        # 15. 시드 설정, 각 프로세스가 고유한 시드를 받도록 보장하여 중복 완성본 방지 -> 같은 질문 받았는데 다른 gpu에서 다른 답변이 나오도록하여 데이터 중복 방지
        set_seed(args.seed, device_specific=True)

        # 16. VLLM 환경 준비, vllm 사용 여부에 따라 생성 환경을 준비하는 함수
        self._prepare_generation_environment(args, processing_class)

        # 그래디언트 누적에는 스케일된 손실이 필요합니다. 일반적으로 부모 클래스의 손실 스케일링은
        # 모델이 손실 관련 kwargs를 받아들이는지 여부에 따라 달라집니다. 우리는 자체 손실을 계산하므로
        # 이 검사는 관련이 없습니다. 스케일링을 활성화하기 위해 self.model_accepts_loss_kwargs를 False로 설정합니다. -> grpo는 이거 사용안해서 false로 설정
        self.model_accepts_loss_kwargs = False

        # 17. 모델에 태그 추가, 모델에 태그를 추가하여 모델의 특성을 표시 -> 의미 x
        self.model.add_model_tags(self._tag_names)

        # 18. 참조 모델과 보상 모델 준비, 참조 모델과 보상 모델을 준비
        self._prepare_reference_and_reward_models(args)

    # Trainer가 실제로 사용할 데이터 칼럼을 명시. prompt 컬럼만 씀.  -> 프롬프트를 받아 완성문을 생성하기 때문
    def _set_signature_columns_if_needed(self):
        # `self.args.remove_unused_columns`가 True인 경우, 사용되지 않는 컬럼들이 제거됩니다.
        # 기본적으로 이 메서드는 `self._signature_columns`를 모델의 예상 입력으로 설정합니다.
        # GRPOTrainer에서는 데이터를 전처리하므로, 모델의 시그니처 컬럼을 사용하는 것은 작동하지 않습니다.
        # 대신 `training_step` 메서드에서 예상하는 컬럼으로 설정하므로, 이를 재정의합니다.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # 각 프롬프트를 여러 번 반복해서, 여러 GPU에 일관되게 분배. -> 각 프롬프트를 num_generations만큼 반복해서 여러 GPU에 일관되게 분배
    def _get_train_sampler(self) -> Sampler:
        # 각 프롬프트가 여러 프로세스에 걸쳐 반복되도록 보장하는 샘플러를 반환합니다.
        # 이를 통해 동일한 프롬프트가 서로 다른 GPU에 분배되어, 각 프롬프트 그룹 내에서 보상 계산 및 정규화가 올바르게 수행됩니다.
        # 모든 프로세스에서 동일한 시드를 사용하므로, 프롬프트 할당이 일관되게 유지되어 그룹 형성에 오류가 발생하지 않습니다.
        return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)

    # 평가용 데이터셋(eval_dataset)에서 데이터를 어떻게 뽑을지 결정하는 샘플러를 반환
    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # 각 프롬프트가 여러 프로세스에 걸쳐 반복되도록 보장하는 샘플러를 반환합니다.
        # 이를 통해 동일한 프롬프트가 서로 다른 GPU에 분배되어, 각 프롬프트 그룹 내에서 보상 계산 및 정규화가 올바르게 수행됩니다.
        # 모든 프로세스에서 동일한 시드를 사용하므로, 프롬프트 할당이 일관되게 유지되어 그룹 형성에 오류가 발생하지 않습니다.
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

    # 모델과 레퍼런스 모델에 대해 각 토큰별 로그 확률을 계산하는 함수, 모델(또는 레퍼런스 모델)이 생성한 시퀀스에서 각 토큰별 로그 확률(log probability)을 계산하는 함수
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # model: 모델 객체, input_ids: 입력 토큰 ID, attention_mask: 어텐션 마스크, logits_to_keep: 유지할 로짓 수
        # 1. 모델 추론
        # 시퀀스의 마지막 로짓은 나중에 제외되므로, `logits_to_keep`에 1을 더해줌
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        # 2. 마지막 로짓 제거 -> 마지막 토큰은 다음 토큰이 없으므로 예측대상이 아님
        logits = logits[:, :-1, :]  # (B, L-1, V), 마지막 로짓은 다음 토큰 예측에 해당하므로 제외
        # 3. 입력 토큰 조정, logits_to_keep 개수 만큼의 토큰만 유지
        input_ids = input_ids[:, -logits_to_keep:]

        # 4. 로짓 자르기
        # transformers<=4.48에서는 logits_to_keep 인자를 지원하지 않으므로, 여기서 직접 logits을 잘라냅니다.
        # 참고: https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  # 5. 입력 토큰에 대한 로그 확률 계산
        # 즉, GRPO에서 필요한 토큰별 확률 정보를 효율적으로 제공
    
    # 이 함수는 학습된 모델의 가중치를 vLLM 엔진으로 이동시키는 역할, 특히 PEFT 모델의 어댑터를 병합하고 가중치를 vLLM에 로드
    # vLLM은 특정 형태의 가중치 구조 요구, PEFT 모델의 어댑터를 병합해야 함
    # 즉, 학습된 모델을 vLLM에서 고속 추론할 수 있도록 준비
    def _move_model_to_vllm(self):
        # 분산 학습 환경에서 모델을 단일 프로세스로 언래핑, 생성에 적합한 형태로 모델 변환
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            # 1. 컴파일된 모듈 처리
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod
            # 2. PEFT 모델 처리
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                # 3. 모델 가중치 정리
                # base_model 및 base_layer 접두사 제거
                # 변환 전: "base_model.model.layers.0.self_attn.q_proj.base_layer.weight"
                # # 변환 후: "layers.0.self_attn.q_proj.weight"
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
                }
                # adapter 접두사(예: "_lora")가 포함된 값 제거
                state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                # 저장할 모듈일 때, 해당 접두사 제거 및 original_module은 버림
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else: # PEFT 모델이 아닌 경우
                state_dict = unwrapped_model.state_dict()
            if self.accelerator.is_main_process: # vllm에 가중치 로드, 메인 프로세스에서만 실행
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(state_dict.items())
            # 어댑터를 언머지하여 모델을 원래 상태로 복원
            # 가중치 로드 후에 수행해야 병합된 상태와 일치함
            if is_peft_model(unwrapped_model):
                unwrapped_model.unmerge_adapter()

    # 여기가 문젠데..
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        # 1. 프롬프트 전처리
        """
        inputs = [
                    {"prompt": "안녕하세요"},           # 첫 번째 쿼리
                    {"prompt": "날씨가 어때요?"},       # 두 번째 쿼리
                    {"prompt": "파이썬이 뭐예요?"},     # 세 번째 쿼리
                    {"prompt": "머신러닝이란?"}         # 네 번째 쿼리
                ]
        """
        device = self.accelerator.device # 현재 디바이스 확인
        processed_prompts = self.prompt_processor.process_prompts(inputs, device) # 프롬프트 전처리
        prompt_ids = processed_prompts["prompt_ids"] # 프롬프트 토큰 ID
        prompt_mask = processed_prompts["prompt_mask"] # 어텐션 마스크
        prompts = processed_prompts["prompts"] # 원본 프롬프트
        prompts_text = processed_prompts["prompts_text"] # 템플릿이 적용된 프롬프트

        # ------------------------------------------------------------
        # 2. 텍스트 생성
        # vLLM 또는 일반 생성 방식을 사용하여 완성문 생성
        if self.args.use_vllm:
            # 1. 모델 가중치 로드
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # 2. vLLM을 사용하여 완성문 생성: 모든 프롬프트를 수집하고 메인 프로세스에서 단일 호출로 사용
            all_prompts_text = gather_object(prompts_text) # 모든 프롬프트를 수집하고 메인 프로세스에서 단일 호출로 사용
            if self.accelerator.is_main_process: # 메인 프로세스에서만 실행
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False) # 완성문 생성
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs] # 완성문 토큰 ID
            else:
                completion_ids = [None] * len(all_prompts_text) # 완성문 토큰 ID
            # 메인 프로세스에서 모든 프로세스로 완성문을 브로드캐스트하여, 각 프로세스가 해당하는 슬라이스를 받도록 보장
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            # 3. 프로세스별 슬라이스 추출
            # 각 프로세스가 자신에게 해당하는 부분만 추출
            # 전체 결과에서 자신의 배치 부분만 가져옴
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # 4. 텐서 변환 및 패딩 -> 리스트를 파이토치 텐서로 변환, 배치 내에서 길이 맞춤을 위한 패딩  
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id) # 패딩
            # 5. 프롬프트와 완성문 연결 -> 프롬프트 토큰 + 생성된 완성문 토큰을 하나로 연결 -> 모델이 전체 시퀀스를 처리할 수 있도록 준비
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1) 
        else:
            # 일반 생성 경로
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # 프롬프트 길이를 계산하고 완성문 ID 추출 -> 보상 계산을 위해 완성문만 추출함.
            prompt_length = prompt_ids.size(1) # 프롬프트 길이 계산
            prompt_ids = prompt_completion_ids[:, :prompt_length] # 프롬프트 토큰만 추출
            completion_ids = prompt_completion_ids[:, prompt_length:] # 완성문 토큰만 추출
        """
            실제:
            completion_ids = tensor([
                # 첫 번째 쿼리 "안녕하세요"의 2개 응답
                [20, 21, 22, 23, 50256, 0, 30, 31, 32, 33, 34, 50256],
                
                # 두 번째 쿼리 "날씨가 어때요?"의 2개 응답
                [24, 25, 26, 27, 28, 50256, 35, 36, 37, 38, 50256, 0],
                
                # 세 번째 쿼리 "파이썬이 뭐예요?"의 2개 응답
                [29, 30, 31, 32, 50256, 0, 39, 40, 41, 42, 43, 50256],
                
                # 네 번째 쿼리 "머신러닝이란?"의 2개 응답
                [33, 34, 35, 36, 37, 50256, 44, 45, 46, 47, 48, 50256]
            ])
            예시: -> 원래는 어텐션 마스크 형태여야함.
            completions = [
                # 첫 번째 쿼리의 2개 응답(즉, 한 쿼리에 체인이 2개 있는 예제와 비슷)
                ["안녕하세요! 반갑습니다", "안녕하세요! 오늘도 좋은 하루 되세요"],
                
                # 두 번째 쿼리의 2개 응답
                ["오늘 날씨가 맑고 좋네요", "맑고 화창한 날씨입니다"],
                
                # 세 번째 쿼리의 2개 응답
                ["파이썬은 프로그래밍 언어입니다", "파이썬은 간단하고 강력한 언어입니다"],
                
                # 네 번째 쿼리의 2개 응답
                ["머신러닝은 AI의 한 분야입니다", "머신러닝은 데이터로부터 학습하는 기술입니다"]
            ]
        """

        # ------------------------------------------------------------
        # 3. 완성문 후처리 -> 완성문의 어텐션 마스크를 생성하는 부분, 즉 모델이 올바른 부분만 처리하도록 어텐션 마스크를 생성하는 단계
        # 1) EOS 토큰 위치 찾기, 완성문에서 EOS토큰의 위치를 찾음
        is_eos = completion_ids == self.processing_class.eos_token_id
        # 2) EOS 토큰 인덱스 계산, 완성문에서 EOS토큰의 위치를 찾음
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        # 3) 시퀀스 인덱스 생성 -> 각 위치의 인덱스를 생성
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        # 4) 완성문 마스크 생성
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # 5) 전체 어텐션 마스크 연결 -> 프롬프트 마스크와 완성문 마스크를 연결 -> 전체 시퀀스에 대한 어텐션 마스크 생성
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        # 6) 로짓 계산 범위 설정
        logits_to_keep = completion_ids.size(1)  # 완성 토큰에 대한 로짓만 계산하면 됨 # 완성문 -> 로짓 변환을 해야합니다!!!

        # ------------------------------------------------------------
        # 4. 참조 모델 로짓 계산 -> 정책 정규화를 위한 기준점을 제공(KL발산이나 정책 정규화 기준점)
        with torch.inference_mode():
            if self.ref_model is not None: # 참조 모델이 있는 경우 -> 별도의 참조 모델을 사용하여 로그 확률 계산, 참조 모델: 학습되지 않은 고정 모델
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:# 참조 모델이 없는 경우 -> 현재 학습중인 모델을 사용하되, 어댑터를 비활성화, 기존 모델의 가중치만 사용하여 로그 확률 계산 -> 저희는 이 방식대로 합니다.
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # ------------------------------------------------------------
        # 5. 완성문 후처리 -> 텍스트 레벨 후처리
        # 1) 토큰을 텍스트로 디코딩 -> 토큰 ID들을 실제 텍스트로 변환, 스페셜 토큰 제거
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True) 
        if is_conversational(inputs[0]): # 2) 입력이 대화 형식 데이터인지 확인
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else "" # 마지막 메시지가 assistant인 경우, 그 전 메시지의 content를 사용
                completions.append([{"role": "assistant", "content": bootstrap + completion}]) # 완성문을 대화 형식으로 변환
        else: # 대화 형식이 아닌 경우 -> 완성문을 그대로 사용
            completions = completions_text
            # completions = 각 프롬프트(배치 단위)에 대한 여러 완성문 생성
            """
            completions = [
                            ["안녕하세요! 반갑습니다", "안녕하세요! 오늘도 좋은 하루 되세요"],  # 프롬프트1의 2개 완성문
                            ["오늘 날씨가 좋네요", "맑고 화창한 날씨입니다"],                    # 프롬프트2의 2개 완성문
                            ["파이썬은 프로그래밍 언어입니다", "파이썬은 간단하고 강력한 언어입니다"],  # 프롬프트3의 2개 완성문
                            ["머신러닝은 AI의 한 분야입니다", "머신러닝은 데이터로부터 학습하는 기술입니다"]  # 프롬프트4의 2개 완성문
                        ]
            """
        # ------------------------------------------------------------
        # 6. 보상 계산
        # 1) 보상 저장 배열 초기화 -> 각 프롬프트와 보상 함수 조합에 대한 보상 저장 -> 프롬프트가 복제되어 있으므로 응답 개수에 따른 보상 함수 조합임
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # 컴파일된 모델과의 호환성을 위해 PretrainedModel 대신 Module 사용 # 보상이 신경망 모델일때인데 저희는 이거 아닙니다.
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # 생성 수에 맞춰 모든 입력 열을 반복 -> 저희는 커스텀 보상 함수이기 때문에 여기입니다. ## 시형이형 ##
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]] # 프롬프트와 완성문 제외 모든 키 추출
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys} 
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs) # 보상 함수 호출 -> 프롬프트와 완성문 및 추가 정보를 보상 함수에 전달
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device) # 보상 저장 배열에 저장

        # ------------------------------------------------------------
        # 7. 보상 advantage 계산
        # 1) 함수별 보상을 수집: 이 부분이 중요한 이유는 보상이 그룹별로 정규화되고 완성문이 프로세스 간에 분산될 수 있기 때문
        rewards_per_func = gather(rewards_per_func)

        # 2) 각 보상 함수의 출력에 가중치를 적용하고 합산
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # 3) 그룹별 보상 계산
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # 4) 보상을 정규화하여 advantage 계산
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # ------------------------------------------------------------
        # 데이터의 로컬 부분만 유지하도록 슬라이스 # 분산처리 후 각 GPU가 자신의 데이터만 정리해서 가져감 -> 메모리 효율성
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # 메트릭 기록
        reward_per_func = rewards_per_func.mean(0) # 각 보상 함수의 평균 보상 계산
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # 컴파일된 모델과의 호환성을 위해 PretrainedModel 대신 Module 사용
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())# 함수의 name속성을 이용해 메트릭 기록

        self._metrics["reward"].append(rewards.mean().item()) # 전체 평균 보상 기록
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item()) # 보상 표준편차 기록

        # 완성문 로깅 -> 학습 과정에서 생성된 완성문들을 로깅하는 부분
        if (
            self.log_completions # 로깅 활성화 여부
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to # wandb 사용 여부
        ):
            import pandas as pd

            # 로깅용 테이블 생성
            table = {
                "step": [str(self.state.global_step)] * len(rewards), # 현재 스텝
                "prompt": gather_object(prompts_text), # 프롬프트들
                "completion": gather_object(completions_text), # 완성문들
                "reward": rewards.tolist(), # 각 완성문의 보상
            }
            df = pd.DataFrame(table) # 테이블을 데이터프레임으로 변환

            if wandb.run is not None and self.accelerator.is_main_process: # wandb 사용 여부
                wandb.log({"completions": wandb.Table(dataframe=df)}) # 테이블을 wandb에 로깅

        return {
            "prompt_ids": prompt_ids, # 프롬프트 토큰 ID(배치 단위)
            "prompt_mask": prompt_mask, # 프롬프트 마스킹(배치 단위)
            "completion_ids": completion_ids, # 완성문 토큰 ID(배치 단위)
            "completion_mask": completion_mask, # 완성문 마스킹(배치 단위)
            "ref_per_token_logps": ref_per_token_logps, # 참조 모델 로짓 계산(배치 단위)
            "advantages": advantages, # 보상 advantage 계산(배치 단위)
        }
    # GRPO 손실 계산 함수
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return compute_loss_func(
            model=model,
            inputs=inputs,
            beta=self.beta,
            accelerator=self.accelerator,
            metrics=self._metrics,
            get_per_token_logps_func=self._get_per_token_logps,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch
        )

    # 학습 과정 중 평가(evaluation) 단계에서 모델의 성능을 측정하는 함수
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        return evaluation_prediction_step(
            model, 
            inputs, 
            prediction_loss_only, 
            ignore_keys,
            prepare_inputs_func=self._prepare_inputs,
            compute_loss_func=self.compute_loss,
            compute_loss_context_manager=self.compute_loss_context_manager
        )
    # 메트릭 로깅 함수
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        return log_metrics(
            logs=logs,
            metrics=self._metrics,
            super_log_func=super().log,
            start_time=start_time
        )
    # 모델 카드 생성 함수->사실상 의미 x
    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        `Trainer`에서 사용 가능한 정보를 사용하여 모델 카드의 초안을 생성합니다.

        Args:
            model_name (`str` 또는 `None`, *optional*, 기본값: `None`):
                모델의 이름.
            dataset_name (`str` 또는 `None`, *optional*, 기본값: `None`):
                훈련에 사용된 데이터셋의 이름.
            tags (`str`, `list[str]` 또는 `None`, *optional*, 기본값: `None`):
                모델 카드와 연결할 태그들.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
