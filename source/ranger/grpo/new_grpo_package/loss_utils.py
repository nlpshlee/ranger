import torch
from typing import Dict, Any, Union


def compute_loss(
    model, 
    inputs: Dict[str, Any], 
    beta: float,
    accelerator,
    metrics: Dict[str, list],
    get_per_token_logps_func,
    return_outputs: bool = False, 
    num_items_in_batch: int = None
):
    """
    GRPO 손실을 계산하는 함수
    
    Args:
        model: 학습할 모델
        inputs: 모델 입력 데이터
        beta: KL 발산 가중치
        accelerator: Accelerate 객체
        metrics: 메트릭 저장 딕셔너리
        get_per_token_logps_func: 토큰별 로그 확률 계산 함수
        return_outputs: 출력 반환 여부
        num_items_in_batch: 배치 내 아이템 수
    
    Returns:
        계산된 손실값
    """
    if return_outputs:
        raise ValueError("The GRPOTrainer does not support returning outputs")
    
    # 모델의 토큰별 로그 확률을 계산
    # 1. 입력 데이터 추출 및 결합 -> 프롬프트와 완성문을 하나의 시퀀스로 결합, 완성문 부분의 길이를 게산(로짓 계산 범위)
    prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
    completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)  # 완성 토큰에 대한 로짓만 계산하면 됨
    # 2. 현재 모델의 로그 확률 계산 -> 현재 학습 중인 모델로 각 토큰의 로그 확률 계산, 완성 부분만 계산(완성문은 이미 주어졌기 때문)
    per_token_logps = get_per_token_logps_func(model, input_ids, attention_mask, logits_to_keep)

    # 3. 모델과 참조 모델 간의 KL 발산을 계산
    ref_per_token_logps = inputs["ref_per_token_logps"]
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

    # x - x.detach()는 x에서 그래디언트를 보존할 수 있게 함
    # 4. GRPO손실 계산
    # 공식: Loss = -[exp(log_π - log_π_detached) × advantage - β × KL]
    advantages = inputs["advantages"]
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean() # 최종 손실 게산

    # 메트릭 기록
    completion_length = accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item() # 완성문 길이
    metrics["completion_length"].append(completion_length) # 완성문 길이 기록

    mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean() # KL 발산 계산
    metrics["kl"].append(accelerator.gather_for_metrics(mean_kl).mean().item()) # KL 발산 기록

    return loss # 최종 손실 반환