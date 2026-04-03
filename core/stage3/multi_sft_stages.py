#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Progressive Cognitive Internalization 训练脚本
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

OPTIONAL_IMPORT_ERROR = None

try:
    import torch
    from datasets import Dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
except ImportError as exc:
    OPTIONAL_IMPORT_ERROR = exc
    torch = None
    Dataset = None
    AutoConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    LoraConfig = None
    PeftModel = None
    prepare_model_for_kbit_training = None
    SFTTrainer = None
    SFTConfig = None
    DataCollatorForCompletionOnlyLM = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 数据加载与 ChatML 格式化
# ==============================================================================

def load_data(data_path: str) -> List[Dict[str, Any]]:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def require_training_dependencies():
    if OPTIONAL_IMPORT_ERROR is not None:
        raise ImportError(
            "Stage 3 training dependencies are missing. Install the packages in requirements.txt before running this script."
        ) from OPTIONAL_IMPORT_ERROR


def read_adapter_config(model_path: str) -> Dict[str, Any]:
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        return {}
    with open(adapter_config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_adapter_checkpoint(model_path: str) -> bool:
    return (Path(model_path) / "adapter_config.json").exists()


def resolve_model_config_path(model_path: str) -> str:
    adapter_config = read_adapter_config(model_path)
    return adapter_config.get("base_model_name_or_path", model_path)


def resolve_tokenizer_path(model_path: str) -> str:
    path = Path(model_path)
    if path.exists() and (
        (path / "tokenizer_config.json").exists()
        or (path / "tokenizer.json").exists()
        or (path / "special_tokens_map.json").exists()
    ):
        return model_path
    return resolve_model_config_path(model_path)


def should_create_new_lora_adapter(model_path: str) -> bool:
    return not is_adapter_checkpoint(model_path)


def infer_model_family(model_config, tokenizer) -> str:
    model_type = getattr(model_config, "model_type", "") or ""
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    model_name = getattr(tokenizer, "name_or_path", "") or ""

    probe = f"{model_type}\n{chat_template}\n{model_name}".lower()
    if "qwen" in probe or "<|im_start|>" in chat_template:
        return "qwen"
    if "llama" in probe or "<|start_header_id|>" in chat_template:
        return "llama"
    return "generic"


def _pick_existing_token_id(tokenizer, token: Optional[str]) -> Optional[int]:
    if not token:
        return None

    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None:
        return None
    if unk_token_id is not None and token_id == unk_token_id and token != tokenizer.unk_token:
        return None
    return token_id


def ensure_padding_token(tokenizer, model_family: str) -> bool:
    if tokenizer.pad_token_id is not None:
        return False

    candidate_tokens: List[str] = []
    if model_family == "llama":
        candidate_tokens.extend(["<|finetune_right_pad_id|>", "<|eot_id|>"])
    elif model_family == "qwen":
        candidate_tokens.extend(["<|endoftext|>", "<|im_end|>"])

    if tokenizer.eos_token:
        if isinstance(tokenizer.eos_token, list):
            candidate_tokens.extend([token for token in tokenizer.eos_token if isinstance(token, str)])
        elif isinstance(tokenizer.eos_token, str):
            candidate_tokens.append(tokenizer.eos_token)

    candidate_tokens.extend(["<|finetune_right_pad_id|>", "<|endoftext|>", "<|eot_id|>", "<|im_end|>"])

    seen = set()
    for token in candidate_tokens:
        if token in seen:
            continue
        seen.add(token)
        token_id = _pick_existing_token_id(tokenizer, token)
        if token_id is None:
            continue
        tokenizer.pad_token = token
        logger.info("Using existing pad token %s (id=%s)", token, token_id)
        return False

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    logger.warning("Tokenizer has no reusable pad token; added a new [PAD] token.")
    return True


def get_response_template(tokenizer) -> str:
    seed_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    prompt_without_generation = tokenizer.apply_chat_template(
        seed_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_with_generation = tokenizer.apply_chat_template(
        seed_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if not prompt_with_generation.startswith(prompt_without_generation):
        raise ValueError("Tokenizer chat template is incompatible with completion-only masking.")

    response_template = prompt_with_generation[len(prompt_without_generation):]
    if not response_template:
        raise ValueError("Failed to infer assistant response template from tokenizer chat template.")
    return response_template


def get_lora_target_modules(model) -> List[str]:
    candidate_suffixes = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    discovered = []
    module_names = [name for name, _ in model.named_modules()]
    for suffix in candidate_suffixes:
        if any(name.endswith(suffix) for name in module_names):
            discovered.append(suffix)

    if not discovered:
        raise ValueError("Unable to infer LoRA target modules from model structure.")
    return discovered

def format_sample_with_chat_template(sample: Dict[str, Any], stage: int, tokenizer) -> str:
    """
    使用 tokenizer 原生的 chat template 将特定阶段的 prompt 和 label 格式化。
    """
    prompt_key = f'prompt_stage{stage}'
    label_key = f'label_stage{stage}'

    prompt = sample.get(prompt_key, '')
    label = sample.get(label_key, '')

    messages = [
        {"role": "system", "content": "你是一个高度拟真的社交网络数字人类。"},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": label}
    ]

    # 使用 tokenizer 原生方法生成模型对应的对话格式
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False, 
        add_generation_prompt=False
    )
    return text

# ==============================================================================
# DEBUG 打印系统：彻底看清 Token 级别的细节
# ==============================================================================

def print_debug_info(tokenizer, sample_text: str, stage: int):
    """
    打印详细的调试信息，验证 ChatML 格式、Token IDs 以及 DataCollator 掩码位置。
    """
    print("\n" + "🔥" * 40)
    print(f"🐛 深度调试信息 [STAGE {stage}] (取训练集第 0 条样本)")
    print("🔥" * 40)
    
    # 1. 打印格式化后的纯文本
    print(f"\n1️⃣ [格式化后的完整文本]:\n{sample_text}\n")

    # 2. Tokenize 文本
    tokenized_output = tokenizer(sample_text)
    input_ids = tokenized_output['input_ids']
    
    print(f"2️⃣ [Token IDs 总长度]: {len(input_ids)}")
    print(f"3️⃣ [头部 Tokens (前 25 个)]: {tokenizer.convert_ids_to_tokens(input_ids[:25])}")
    print(f"4️⃣ [尾部 Tokens (最后 20 个)] (请检查是否有结束符): {tokenizer.convert_ids_to_tokens(input_ids[-20:])}\n")

    # 3. 构造 Response Template 并寻找掩码切分点
    response_template_str = get_response_template(tokenizer)
    response_template_ids = tokenizer.encode(response_template_str, add_special_tokens=False)
    
    print(f"5️⃣ [Response Template 字符串]: {repr(response_template_str)}")
    print(f"6️⃣ [Response Template Token IDs]: {response_template_ids}\n")

    # 模拟 DataCollatorForCompletionOnlyLM 的寻找逻辑
    match_idx = -1
    for i in range(len(input_ids) - len(response_template_ids) + 1):
        if input_ids[i:i+len(response_template_ids)] == response_template_ids:
            match_idx = i + len(response_template_ids)
            break

    if match_idx != -1:
        print(f"✅ [掩码匹配成功]: 找到了 Template！")
        print(f"   --> 模型将只对第 {match_idx} 个 Token 之后的内容计算 Loss。")
        loss_calc_tokens = tokenizer.convert_ids_to_tokens(input_ids[match_idx:])
        print(f"🎯 [实际计算 Loss 的内容 (应仅包含 Assistant 的回复和结束符)]:\n{loss_calc_tokens}\n")
    else:
        print(f"❌ [严重错误: 掩码匹配失败]: 无法在 input_ids 中找到 Response Template IDs！")
        print(f"   --> 模型将对整段对话（包括 User 的 Prompt）计算 Loss，这会导致模型彻底崩溃和复读！\n")
        
    print("🔥" * 40 + "\n")

# ==============================================================================
# 单阶段训练逻辑
# ==============================================================================

def train_single_stage(args, stage: int, current_model_path: str):
    require_training_dependencies()

    output_dir = os.path.join(args.output_dir, f"stage_{stage}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"🚀 开始训练 Stage {stage}")
    print(f"📦 当前基础模型: {current_model_path}")
    print(f"💾 输出目录: {output_dir}")
    print("=" * 70)

    # 1. 加载 Tokenizer
    model_config_path = resolve_model_config_path(current_model_path)
    tokenizer_path = resolve_tokenizer_path(current_model_path)

    model_config = AutoConfig.from_pretrained(model_config_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, padding_side="right")
    model_family = infer_model_family(model_config, tokenizer)
    added_new_pad_token = ensure_padding_token(tokenizer, model_family)
    logger.info(
        "Detected model family=%s, pad_token=%s, pad_token_id=%s",
        model_family,
        repr(tokenizer.pad_token),
        tokenizer.pad_token_id,
    )

    # 2. 准备数据集
    data = load_data(args.data_path)
    if args.max_samples is not None:
        data = data[:args.max_samples]

    formatted_texts = [format_sample_with_chat_template(sample, stage, tokenizer) for sample in data]
    train_dataset = Dataset.from_dict({"text": formatted_texts})

    # === 执行调试打印 ===
    # if len(formatted_texts) > 0:
    #     print_debug_info(tokenizer, formatted_texts[0], stage)

    # 3. 配置 Data Collator (极其关键)
    response_template_str = get_response_template(tokenizer)
    response_template_ids = tokenizer.encode(response_template_str, add_special_tokens=False)
    data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer)
    logger.info("Using response template: %r", response_template_str)

    # 4. 加载模型与量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True
    ) if args.use_qlora else None

    model = AutoModelForCausalLM.from_pretrained(
        model_config_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"
    )
    if added_new_pad_token:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # 5. 配置 LoRA
    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    peft_config = None
    if args.use_lora and is_adapter_checkpoint(current_model_path):
        model = PeftModel.from_pretrained(model, current_model_path, is_trainable=True)
        logger.info("Resuming shared LoRA adapter from %s", current_model_path)
    elif args.use_lora:
        target_modules = get_lora_target_modules(model)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        logger.info("Using new LoRA target modules: %s", target_modules)

    # 6. 配置训练参数 (加入防过拟合策略)
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs, # 推荐设为 1
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        bf16=args.bf16,
        max_seq_length=args.max_seq_length,
        packing=False,
        report_to="none",
        seed=42
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n🎉 Stage {stage} 训练完成！权重保存至: {output_dir}")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Multi-stage CoRE SFT Training")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--stage", type=int, default=0, help="0表示一次性跑完1,2,3; 否则只跑指定stage")
    parser.add_argument("--max_samples", type=int, default=None)
    
    # 训练参数
    parser.add_argument("--learning_rate", type=float, default=1e-5) # 调低默认学习率
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=8196)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["epoch", "steps"])
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)


    # LoRA 参数
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--use_qlora", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true", default=True)

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent if 'Path' in globals() else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(project_root, args.data_path)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)

    if args.stage == 0:
        print(">>> 启动完整多阶段课程学习 (Stage 1 -> 2 -> 3)")
        m1 = train_single_stage(args, 1, args.model_name_or_path)
        m2 = train_single_stage(args, 2, m1)
        m3 = train_single_stage(args, 3, m2)
        print("\n🏆 全流程多阶段微调完毕！")
    else:
        print(f">>> 仅执行 Stage {args.stage}")
        train_single_stage(args, args.stage, args.model_name_or_path)

if __name__ == "__main__":
    main()
