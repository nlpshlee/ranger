from _init import *

import random, math, glob
from transformers import TrainingArguments
from peft import LoraConfig, TaskType

from ranger.utils import common_utils, json_utils, evaluation_utils, model_utils
from ranger.train.sft_trainer import SftTrainer


seed = COMMON_CONFIG['seed']
common_utils.set_seed(seed)


def load_datas(train_file_path, train_size):
    all_datas = json_utils.load_jsonl(train_file_path)

    random.shuffle(all_datas)

    train_datas = all_datas[:train_size]
    eval_datas = all_datas[train_size:]

    print(f'\n# sft_runner.load_datas() train_datas size : {len(train_datas)}')
    print(f'# sft_runner.load_datas() eval_datas size : {len(eval_datas)}\n')

    json_utils.write_jsonl(train_datas, f'{train_dir}/train_{train_size}.jsonl')
    json_utils.write_jsonl(eval_datas, f'{train_dir}/eval_{len(all_datas)-train_size}.jsonl')

    return train_datas, eval_datas


def get_peft_config(lora_r,
                    lora_alpha,
                    lora_dropout,
                    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                    bias='none',
                    task_type=TaskType.CAUSAL_LM):
    
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type
    )

    return peft_config


def get_training_args(out_dir,
                      num_epochs,
                      batch_size,
                      accumulation_steps,
                      learning_rate,
                      weight_decay,
                      warmup_ratio,
                      max_grad_norm,
                      save_strategy='steps',
                      save_steps=10,
                      eval_strategy='steps',
                      eval_steps=10,
                      logging_steps=10,
                      lr_scheduler_type='cosine',
                      load_best_model_at_end=True,
                      metric_for_best_model='eval_loss',
                      save_total_limit=5):
    
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,

        save_strategy=save_strategy,
        save_steps=save_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        logging_steps=logging_steps,

        lr_scheduler_type=lr_scheduler_type,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        save_total_limit=save_total_limit,

        bf16=True,
        do_train=True,
        label_names=['labels'],
        report_to='none'
    )

    return training_args



def train(model_name, dtype, max_seq_length,
        lora_r, lora_alpha, lora_dropout,
        num_epochs, batch_size, accumulation_steps, learning_rate, weight_decay, warmup_ratio, max_grad_norm,
        train_datas, eval_datas,
        save_and_eval_per_epoch, logging_per_epoch, out_dir):

    sft_trainer = SftTrainer(model_name, dtype, max_seq_length)
    peft_config = get_peft_config(lora_r, lora_alpha, lora_dropout)
    sft_trainer.init_model(peft_config)

    real_batch_size = batch_size * accumulation_steps
    steps_per_epoch = math.ceil(len(train_datas) / real_batch_size)
    save_and_eval_steps = max(1, steps_per_epoch // save_and_eval_per_epoch)
    logging_steps = max(1, steps_per_epoch // logging_per_epoch)

    training_args = get_training_args(
        out_dir, num_epochs, batch_size, accumulation_steps, learning_rate, weight_decay, warmup_ratio, max_grad_norm,
        save_steps=save_and_eval_steps, eval_steps=save_and_eval_steps, logging_steps=logging_steps
    )

    sft_trainer.set_and_get_sft_dataset(train_datas, eval_datas)
    sft_trainer.train(training_args)


def evaluate_all(checkpoint_dir, eval_datas):
    checkpoint_paths = glob.glob(f'{checkpoint_dir}/checkpoint-*')
    checkpoint_paths.sort(key=lambda x: int(x.split('-')[-1]))

    for checkpoint_path in checkpoint_paths:
        evaluation_utils.evaluate_sft(
            checkpoint_path,
            VLLM_CONFIG['dtype'],
            eval_datas,
            VLLM_CONFIG['max_seq_length'],
            VLLM_CONFIG['max_new_tokens']
        )





'''
    CUDA_VISIBLE_DEVICES=0 python -u sft_runner.py > ./logs/sft_runner.log
    CUDA_VISIBLE_DEVICES=0 python -u sft_runner.py > ./logs/sft_runner_v1_llama-3B.log
    CUDA_VISIBLE_DEVICES=0 python -u sft_runner.py > ./logs/sft_runner_v2_llama-3B.log
    CUDA_VISIBLE_DEVICES=0 python -u sft_runner.py > ./logs/sft_runner_v2_llama-8B.log
'''
if __name__ == "__main__":
    work_dir = f'/raid/ai/home/jsyang/dev_env/git/repos/ranger'
    data_dir = f'{work_dir}/data'
    sft_dir = f'{data_dir}/sft'



    data_version = 'v2'
    # base_model = 'meta-llama/Llama-3.2-3B-Instruct'
    base_model = 'meta-llama/Llama-3.1-8B-Instruct'

    # out_dir = f'{work_dir}/outputs/sft/v1_llama-3B'
    # out_dir = f'{work_dir}/outputs/sft/v2_llama-3B'
    out_dir = f'{work_dir}/outputs/sft/v2_llama-8B'

    if data_version == 'v1':
        train_dir = f'{sft_dir}/selected_train_v1_all'
        train_size = 29000
    elif data_version == 'v2':
        train_dir = f'{sft_dir}/selected_train_v2_only_first_sub_query'
        train_size = 27000

    train_file_path = f'{train_dir}/train_merged.jsonl'
    train_datas, eval_datas = load_datas(train_file_path, train_size)

    # train(
    #     base_model, VLLM_CONFIG['dtype'], VLLM_CONFIG['max_seq_length'],
    #     64, 128, 0.05,
    #     5, 2, 16, 5e-5, 0.01, 0.05, 1.0,
    #     train_datas, eval_datas,
    #     5, 5, out_dir
    # )

    # evaluate_all(
    #     out_dir,
    #     eval_datas
    # )



    model_dir = f'{work_dir}/outputs/sft/models'
    checkpoint_num = -1

    model_utils.merge_and_save(
        base_model,
        VLLM_CONFIG['dtype'],
        f'{out_dir}/checkpoint-{checkpoint_num}',
        f'{model_dir}/{data_version}_8B_checkpoint-{checkpoint_num}'
    )

