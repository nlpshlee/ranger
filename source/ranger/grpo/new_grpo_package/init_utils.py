import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, AutoModelForSequenceClassification
from peft import get_peft_model

from ..grpo_config import GRPOConfig

# 모델과 args를 확인하고, args가 None인 경우 기본 GRPO 설정을 생성하는 함수
def check_args(model, args):
    # Args(인수)
    if args is None:
        model_name = model if isinstance(model, str) else model.config._name_or_path
        model_name = model_name.split("/")[-1]
        args = GRPOConfig(f"{model_name}-GRPO")
    return args

# 모델을 초기화하고 설정하는 함수 (문자열인 경우 모델 로드, PEFT 설정 적용)
def init_model(model, model_init_kwargs, args, peft_config):
    if isinstance(model, str):
        model_id = model # 모델 ID저장
        torch_dtype = model_init_kwargs.get("torch_dtype") # torch_dtype 처리, 모델의 데이터 타입 설정 가져옴
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass # 이미 올바른 형식이면 그대로 사용
        elif isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype) # 문자열로 된 데이터 타입을 실제 파이토치 데이터 타입으로 변환
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError( # 데이터 타입 오류 발생 시 예외 발생
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # 그래디언트 체크포인트 사용 여부에 따라 use_cache 설정 조정, 모델의 중간 계산 결과를 메모리에 저장
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        # 모델 로드, 모델 초기화 인수를 사용하여 모델 생성
        model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
    else:
        model_id = model.config._name_or_path
        if args.model_init_kwargs is not None:
            raise ValueError(
                "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                "This argument can only be used when the `model` argument is a string."
            )
    # PEFT 설정 적용
    if peft_config is not None:
        model = get_peft_model(model, peft_config)
    return model, model_id

# 토크나이저를 초기화하는 함수 (None인 경우 모델에 맞는 토크나이저 자동 로드)
def init_processing_class(processing_class, model):
    if processing_class is None:
        return AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
    return processing_class

# 보상 함수들의 토크나이저를 초기화하고 설정하는 함수, 우리는 일반 함수형 리워드를 사용하기 때문에 토크나이저를 초기화할 필요가 없음
# reward_processing_classes: 각 보상 함수에 대응하는 토크나이저 리스트
def init_reward_processing_classes(reward_funcs, reward_processing_classes, model_init_kwargs):
    if reward_processing_classes is None:
        reward_processing_classes = [None] * len(reward_funcs)
    elif not isinstance(reward_processing_classes, list):
        reward_processing_classes = [reward_processing_classes]
    else:
        if len(reward_processing_classes) != len(reward_funcs):
            raise ValueError("The number of reward processing classes must match the number of reward functions.")

    for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
        if isinstance(reward_func, PreTrainedModel):
            if reward_processing_class is None:
                reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
            if reward_processing_class.pad_token_id is None:
                reward_processing_class.pad_token = reward_processing_class.eos_token
            reward_func.config.pad_token_id = reward_processing_class.pad_token_id
            reward_processing_classes[i] = reward_processing_class
    return reward_processing_classes

# 보상 함수들을 초기화하는 함수 (문자열인 경우 시퀀스 분류 모델로 로드)
def init_reward_funcs(reward_funcs, model_init_kwargs):
    if not isinstance(reward_funcs, list):
        reward_funcs = [reward_funcs]
        # 우리는 함수형 커스텀 리워드이기 때문에 여러 보상 함수들이 리스트 형태로 반환되고 끝
    for i, reward_func in enumerate(reward_funcs):
        if isinstance(reward_func, str):
            reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                reward_func, num_labels=1, **model_init_kwargs
            )
    return reward_funcs

# 보상 가중치를 초기화하는 함수 (None인 경우 모든 가중치를 1로 설정), 보상 가중치: 각 보상 함수에 대한 가중치를 설정하는 파라미터.
def init_reward_weights(args, reward_funcs):
    if args.reward_weights is not None:
        if len(args.reward_weights) != len(reward_funcs):
            raise ValueError(
                f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                f"functions ({len(reward_funcs)})"
            )
        return torch.tensor(args.reward_weights, dtype=torch.float32)
    else:
        return torch.ones(len(reward_funcs), dtype=torch.float32) # 보상 가중치가 없으면 모든 가중치를 1로 설정하고 리턴

# 생성 배치 크기와 생성 횟수의 호환성을 확인하는 함수
# accelerator: 분산 학습을 위한 accelerator 객체(허깅페이스 라이브러리), num_generations: 각 프롬프트당 생성 응답 개수
def check_generation_batch_compatibility(args, accelerator, num_generations):
    num_processes = accelerator.num_processes # 프로세스 수
    global_batch_size = args.per_device_train_batch_size * num_processes # 전체 배치 크기 = 디바이스당 배치 크기 * 프로세스 개수
    possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0] # 전체 배치 크기로 나누어 떨어지는 생성 개수 (2부터 시작)
    if num_generations not in possible_values: # num_generations가 가능한 값에 없으면 오류 발생
        raise ValueError(
            f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
            f"divisible by the number of generations per prompt ({num_generations}). Given the current train "
            f"batch size, the valid values for the number of generations are: {possible_values}."
        )
    # 평가 전략이 "no"가 아닌 경우에만 실행, 평가 배치 크기와 생성횟수의 호환성을 검증하는 코드, 학습 중에 모델 성능을 평가할 때 사용되는 배치 크기와 생성 횟수가 호환되는지 확인
    if args.eval_strategy != "no":
        global_batch_size = args.per_device_eval_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if num_generations not in possible_values:
            raise ValueError(
                f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({num_generations}). Given the current "
                f"eval batch size, the valid values for the number of generations are: {possible_values}."
            )
