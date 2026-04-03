#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2.1: Backward reasoning path construction.
"""

import asyncio
import json
import sys
import argparse
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

# 路径设置
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.llm_client import LLMClient


SYSTEM_PROMPT = """你是一位顶尖的社会认知心理学专家与计算社会科学研究者。你擅长通过因果认知链分析社交媒体用户的行为。你需要根据用户的真实发言，反推为什么用户会这样发言，重点从其认知状态（情感、思维风格、立场和意图）等角度来分析。推理内容应该能促使模型在模拟该用户时，看到同样的上下文时给出同样的回复。"""

FOUR_DIMENSIONS = ["Emotion", "Thinking", "Stance", "Intent"]
STANCE_LABELS = {
    "CUT": "不明确、支持中方、支持美方",
    "FRIR": "支持降息、反对降息、不明确",
    "UE": "支持共和党、支持民主党、不明确",
    "DEI": "支持废除DEI、反对废除DEI、不明确",
}


def normalize_salience_scores(raw_scores, dominant_dimension=None):
    scores = {}
    total = 0.0

    for dim in FOUR_DIMENSIONS:
        value = raw_scores.get(dim, 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        value = max(value, 0.0)
        scores[dim] = value
        total += value

    if total <= 0:
        if dominant_dimension in FOUR_DIMENSIONS:
            return {
                dim: (0.7 if dim == dominant_dimension else 0.1)
                for dim in FOUR_DIMENSIONS
            }
        return {dim: 0.25 for dim in FOUR_DIMENSIONS}

    return {dim: round(scores[dim] / total, 4) for dim in FOUR_DIMENSIONS}


def build_history_context(data, history_turns=3):
    grouped = defaultdict(list)
    for item in data:
        grouped[item["user_id"]].append(item)

    for user_id in grouped:
        grouped[user_id].sort(
            key=lambda x: (
                x.get("timestep", ""),
                x.get("conversation_id", -1),
                x.get("sub_id", -1),
                x.get("chain_id", ""),
            )
        )

        for idx, sample in enumerate(grouped[user_id]):
            history_items = grouped[user_id][max(0, idx - history_turns):idx]
            if not history_items:
                sample["history_context"] = "No prior interaction history."
                continue

            lines = []
            for turn_id, item in enumerate(history_items, start=1):
                labels = item.get("curr_cognitive_labels", {})
                lines.append(
                    "\n".join(
                        [
                            f"Round {turn_id}",
                            f"- User reply: {item.get('curr_behavior', '')}",
                            (
                                "- Cognitive state: "
                                f"emotion={labels.get('emotion', 'unknown')}, "
                                f"thinking={labels.get('thinking_value', 'unknown')}, "
                                f"stance={labels.get('stance', 'unknown')}, "
                                f"intent={labels.get('intent', 'unknown')}"
                            ),
                        ]
                    )
                )
            sample["history_context"] = "\n\n".join(lines)

    return data


def build_reverse_inference_prompt(sample):
    """Construct the backward reasoning prompt."""
    prev_emotion = sample["prev_cognitive_labels"].get("emotion", "未知")
    prev_stance = sample["prev_cognitive_labels"].get("stance", "未知")
    prev_thinking_value = sample["prev_cognitive_labels"].get("thinking_value", "未知")
    prev_intent = sample["prev_cognitive_labels"].get("intent", "未知")

    curr_emotion = sample["curr_cognitive_labels"].get("emotion", "未知")
    curr_stance = sample["curr_cognitive_labels"].get("stance", "未知")
    curr_thinking_value = sample["curr_cognitive_labels"].get("thinking_value", "未知")
    curr_intent = sample["curr_cognitive_labels"].get("intent", "未知")

    dominant_dim = sample.get("dominant_dimension", "未知")
    salience_scores = normalize_salience_scores(
        sample.get("salience_scores", sample.get("ate_scores", {})),
        dominant_dimension=dominant_dim,
    )
    salience_text = "\n".join(
        f"- {dim}: {salience_scores[dim]:.4f}" for dim in FOUR_DIMENSIONS
    )

    behavior = sample.get("curr_behavior", "")
    curr_event = sample.get("curr_event", "未识别的触发事件")
    topic = sample.get("topic", "未知话题")
    topic_description = sample.get("topic_description", "")
    history_context = sample.get("history_context", "No prior interaction history.")
    current_stance_label = STANCE_LABELS.get(topic, "不明确")

    prompt = f"""{SYSTEM_PROMPT}

### [Known Information]

1. **User history**:
{history_context}

2. **Observed user reply**:
"{behavior}"

3. **Current triggering event**:
"{curr_event}"

4. **Topic background**:
- Topic: {topic}
- Stance options: {current_stance_label}
- Background: {topic_description}

5. **Cognitive state transition**:
- Previous state: emotion({prev_emotion}), stance({prev_stance}), thinking({prev_thinking_value}), intent({prev_intent})
- Current state: emotion({curr_emotion}), stance({curr_stance}), thinking({curr_thinking_value}), intent({curr_intent})

6. **Salience Guidance**:
{salience_text}

7. **Dominant dimension**:
{dominant_dim}

### [Task]

Please reconstruct the internal reasoning path that explains why this user produced the observed reply. The reasoning should be organized around the four core cognitive dimensions and remain consistent with the dominant dimension and the salience guidance.

### [Output Format]

Return JSON only:

{{
  "emotion_thinking": "How the event changes the user's emotion and how that emotion shapes the reply.",
  "thinking_type_thinking": "How the user processes the information and what kind of thinking style is used.",
  "stance_thinking": "How the user's stance is activated or maintained in this context.",
  "intent_thinking": "What communicative intent drives the final reply strategy."
}}
"""
    return prompt


def parse_json_response(response_text):
    """Extract JSON from raw LLM output."""
    try:
        if "```json" in response_text:
            response_text = response_text.split("```json", 1)[1].split("```", 1)[0]

        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start == -1 or end <= start:
            return None

        return json.loads(response_text[start:end])
    except Exception as e:
        print(f"解析失败详情: {e}")
        return None


async def generate_full_chain_sample(sem, llm_client, sample, pbar, max_retries=3):
    """Generate a four-dimension reasoning path for a single sample."""
    async with sem:
        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            try:
                prompt = build_reverse_inference_prompt(sample)
                temp = max(0.1, 0.7 - (retry_count * 0.2))
                response = await llm_client.generate(prompt, temperature=temp)
                parsed = parse_json_response(response)

                if parsed:
                    sample["emotion_thinking"] = parsed.get(
                        "emotion_thinking",
                        "No strong emotional shift is observed.",
                    )
                    sample["thinking_type_thinking"] = parsed.get(
                        "thinking_type_thinking",
                        "No clear thinking-style shift is observed.",
                    )
                    sample["stance_thinking"] = parsed.get(
                        "stance_thinking",
                        "No strong stance-driven reasoning is observed.",
                    )
                    sample["intent_thinking"] = parsed.get(
                        "intent_thinking",
                        "No clear communicative intent is observed.",
                    )
                    sample["generation_status"] = "success"
                    sample["retry_count"] = retry_count
                    success = True
                else:
                    retry_count += 1
            except Exception:
                retry_count += 1

        if not success:
            sample["generation_status"] = "failed"
            sample["retry_count"] = retry_count

        pbar.update(1)
        return sample


def build_xml_from_sample(item):
    return f"""<emotion>
{item.get('emotion_thinking', '')}
</emotion>
<thinking_type>
{item.get('thinking_type_thinking', '')}
</thinking_type>
<stance>
{item.get('stance_thinking', '')}
</stance>
<intent>
{item.get('intent_thinking', '')}
</intent>
<answer>
{item.get('curr_behavior', '')}
</answer>"""


async def main():
    parser = argparse.ArgumentParser(description="Step 2.1: Backward reasoning path construction")
    parser.add_argument("--dataset", type=str, default="four_topics", help="数据集名称")
    parser.add_argument("--model", type=str, default="gpt-4o", help="使用的模型")
    parser.add_argument("--concurrency", type=int, default=16, help="并发数")
    parser.add_argument("--limit", type=int, default=None, help="测试用限制处理数量")
    parser.add_argument("--max_retries", type=int, default=3, help="JSON解析最大重试次数")
    args = parser.parse_args()

    input_path = project_root / "core/step1_ate_analysis/output/step_5/final_personalized_ate_data.json"
    output_dir = project_root / "core/step2_sample_generate/output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "sft_training_data_with_full_chain_5d.json"

    if not input_path.exists():
        print(f"❌ 错误: 未找到输入文件 {input_path}")
        return

    print(f"加载个性化显著性数据: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = build_history_context(data, history_turns=3)

    if args.limit:
        data = data[:args.limit]
        print(f"测试模式: 限制处理 {args.limit} 条样本")

    print(f"准备为 {len(data)} 条样本生成四维推理路径...\n")

    llm_config = {
        "model_name": args.model,
        "temperature": 0.5,
        "api_key": "",
        "base_url": "",
    }
    llm_client = LLMClient(llm_config)

    print("测试 LLM 连接...")
    if not await llm_client.test_connection():
        print("❌ LLM 连接失败")
        await llm_client.close()
        return

    print("✅ LLM 连接成功\n")

    sem = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(data), desc="生成四维推理")
    tasks = [
        generate_full_chain_sample(sem, llm_client, sample, pbar, max_retries=args.max_retries)
        for sample in data
    ]
    final_data = await asyncio.gather(*tasks)

    await llm_client.close()

    print("\n构建 XML 训练标签...")
    xml_data = []
    for item in final_data:
        if item.get("generation_status") != "success":
            continue
        xml_data.append(
            {
                "chain_id": item.get("chain_id"),
                "user_id": item.get("user_id"),
                "topic": item.get("topic"),
                "dominant_dimension": item.get("dominant_dimension"),
                "xml": build_xml_from_sample(item),
                "generation_status": "success",
            }
        )

    success_count = len(xml_data)
    failed_count = len(final_data) - success_count

    print(f"\n{'=' * 70}")
    print("生成完成统计")
    print(f"{'=' * 70}")
    print(f"总样本数: {len(final_data)}")
    print(f"成功生成: {success_count} ({success_count / len(final_data) * 100:.1f}%)")
    print(f"生成失败: {failed_count} ({failed_count / len(final_data) * 100:.1f}%)")

    print(f"\n保存 SFT 训练数据至: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(xml_data, f, ensure_ascii=False, indent=2)

    print("\n✨ Step 2.1 处理完成！")


if __name__ == "__main__":
    asyncio.run(main())
