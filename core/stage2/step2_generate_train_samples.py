#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# 路径设置
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

FOUR_DIMENSIONS = ["Emotion", "Thinking", "Stance", "Intent"]
DOMINANT_TAG_MAP = {
    "Emotion": "emotion",
    "Thinking": "thinking_type",
    "Stance": "stance",
    "Intent": "intent",
}


def sort_user_chains(user_chains):
    return sorted(
        user_chains,
        key=lambda x: (
            x.get("timestep", ""),
            x.get("conversation_id", -1),
            x.get("sub_id", -1),
            x.get("chain_id", ""),
        ),
    )


def build_history_string(user_chains, current_chain_idx, n=3):
    """Build the previous n rounds of interaction history."""
    history_lines = []
    start_idx = max(0, current_chain_idx - n)

    for i in range(start_idx, current_chain_idx):
        chain = user_chains[i]
        round_num = i + 1
        post = chain.get("curr_behavior", "")
        labels = chain.get("curr_cognitive_labels", {})

        if not post:
            continue

        history_lines.append(
            "\n".join(
                [
                    f"Round {round_num}",
                    f"[User Reply]: {post}",
                    (
                        "[Cognitive State]: "
                        f"emotion={labels.get('emotion', 'unknown')}, "
                        f"thinking={labels.get('thinking_value', 'unknown')}, "
                        f"stance={labels.get('stance', 'unknown')}, "
                        f"intent={labels.get('intent', 'unknown')}"
                    ),
                ]
            )
        )

    return "\n\n".join(history_lines) if history_lines else "No prior interaction history."


def get_stance_range(topic):
    ranges = {
        "CUT": "unclear, pro-China, pro-US",
        "FRIR": "unclear, support rate cuts, oppose rate cuts",
        "UE": "unclear, support Republicans, support Democrats",
        "DEI": "unclear, support abolishing DEI, oppose abolishing DEI",
    }
    if topic is None:
        return "unclear"
    return ranges.get(topic.upper(), "unclear")


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


def build_salience_guidance(sample):
    scores = normalize_salience_scores(
        sample.get("salience_scores", sample.get("ate_scores", {})),
        dominant_dimension=sample.get("dominant_dimension"),
    )
    return "\n".join(f"- {dim}: {scores[dim]:.4f}" for dim in FOUR_DIMENSIONS)


def build_stage_prompt(sample, history_str, stage):
    topic = sample.get("topic", "Unknown")
    topic_desc = sample.get("topic_description", "No topic description.")
    dominant_dim = sample.get("dominant_dimension", "Unknown")
    current_stance_label = get_stance_range(topic)
    target_labels = sample.get("curr_cognitive_labels", {})

    common_sections = f"""### [System Persona]
You are a highly realistic digital human on social media. You should respond as this target user under the given context.

### [Topic Background]
- Topic: {topic}
- Stance options: {current_stance_label}
- Background: {topic_desc}

### [History]
{history_str}

### [Current Stimulus]
- Original post: {sample.get('original_post', '')}
- Direct reply target: {sample.get('context_post', '')}
- Triggering event: {sample.get('curr_event', '')}

### [Target Cognitive State]
- Emotion: {target_labels.get('emotion', 'unknown')}
- Thinking style: {target_labels.get('thinking_value', 'unknown')}
- Stance: {target_labels.get('stance', 'unknown')}
- Intent: {target_labels.get('intent', 'unknown')}
"""

    salience_section = f"""
### [Salience Guidance]
{build_salience_guidance(sample)}

### [Dominant Dimension]
{dominant_dim}
"""

    if stage == 1:
        stage_instruction = """### [Stage Instruction]
Generate the full explicit reasoning path in a fixed order:
1. emotion
2. thinking_type
3. stance
4. intent
5. answer

Output must use XML tags exactly:
<emotion>...</emotion>
<thinking_type>...</thinking_type>
<stance>...</stance>
<intent>...</intent>
<answer>...</answer>"""
        return common_sections + salience_section + "\n" + stage_instruction

    if stage == 2:
        dominant_tag = DOMINANT_TAG_MAP.get(dominant_dim, "emotion")
        stage_instruction = f"""### [Stage Instruction]
Focus only on the dominant cognitive dimension and then provide the final answer.
Keep only the dominant reasoning tag `<{dominant_tag}>...</{dominant_tag}>` and `<answer>...</answer>`.
Do not output the other reasoning dimensions."""
        return common_sections + salience_section + "\n" + stage_instruction

    stage_instruction = """### [Stage Instruction]
Do not output any reasoning process.
Generate only the final social media reply that matches the target cognitive state."""
    return common_sections + "\n" + stage_instruction


def parse_xml_sections(xml_text):
    sections = {}
    for tag in ["emotion", "thinking_type", "stance", "intent", "answer"]:
        match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", xml_text, flags=re.S)
        if match:
            sections[tag] = match.group(1).strip()
    return sections


def build_full_label(xml_sections):
    ordered_tags = ["emotion", "thinking_type", "stance", "intent", "answer"]
    return "\n".join(
        f"<{tag}>\n{xml_sections[tag]}\n</{tag}>"
        for tag in ordered_tags
        if tag in xml_sections
    )


def build_dominant_label(xml_sections, dominant_dimension):
    dominant_tag = DOMINANT_TAG_MAP.get(dominant_dimension, "emotion")
    parts = []
    if dominant_tag in xml_sections:
        parts.append(f"<{dominant_tag}>\n{xml_sections[dominant_tag]}\n</{dominant_tag}>")
    if "answer" in xml_sections:
        parts.append(f"<answer>\n{xml_sections['answer']}\n</answer>")
    return "\n".join(parts)


def build_multistage_training_sample(sample, history_str, xml_sections):
    training_sample = {
        "chain_id": sample["chain_id"],
        "user_id": sample["user_id"],
        "topic": sample.get("topic"),
        "dominant_dimension": sample.get("dominant_dimension"),
        "salience_scores": normalize_salience_scores(
            sample.get("salience_scores", sample.get("ate_scores", {})),
            dominant_dimension=sample.get("dominant_dimension"),
        ),
        "prompt_stage1": build_stage_prompt(sample, history_str, stage=1),
        "label_stage1": build_full_label(xml_sections),
        "prompt_stage2": build_stage_prompt(sample, history_str, stage=2),
        "label_stage2": build_dominant_label(xml_sections, sample.get("dominant_dimension")),
        "prompt_stage3": build_stage_prompt(sample, history_str, stage=3),
        "label_stage3": xml_sections.get("answer", ""),
    }
    return training_sample


def main():
    parser = argparse.ArgumentParser(description="Step 2.2: Compile multi-stage SFT training JSON")
    parser.parse_args()

    ate_data_path = project_root / "core/step1_ate_analysis/output/step_5/final_personalized_ate_data.json"
    chains_data_path = project_root / "core/step1_ate_analysis/output/step_2/causal_chains.json"
    xml_data_path = project_root / "core/step2_sample_generate/output/sft_training_data_with_full_chain_5d.json"

    output_dir = project_root / "core/step2_sample_generate/output"
    sft_out_path = output_dir / "train_sft_samples.json"

    for path in [ate_data_path, chains_data_path, xml_data_path]:
        if not path.exists():
            print(f"❌ 错误: 未找到依赖文件 {path}")
            sys.exit(1)

    print("正在加载数据...")
    with open(ate_data_path, "r", encoding="utf-8") as f:
        ate_data = json.load(f)

    with open(xml_data_path, "r", encoding="utf-8") as f:
        xml_list = json.load(f)
        xml_data_dict = {item["chain_id"]: item["xml"] for item in xml_list}

    with open(chains_data_path, "r", encoding="utf-8") as f:
        chains_raw = json.load(f)
        topic_info = {
            c["chain_id"]: (c.get("topic"), c.get("topic_description"))
            for c in chains_raw
        }

    print(f"加载了 {len(ate_data)} 条个体化显著性数据")
    print(f"加载了 {len(xml_data_dict)} 条有效 XML 标签数据")
    print(f"加载了 {len(topic_info)} 条话题上下文信息\n")

    user_to_chains = defaultdict(list)
    for item in ate_data:
        user_to_chains[item["user_id"]].append(item)

    for uid in user_to_chains:
        user_to_chains[uid] = sort_user_chains(user_to_chains[uid])

    sft_samples = []
    print("正在构建三阶段训练样本...")
    for uid, user_chains in user_to_chains.items():
        for idx, sample in enumerate(user_chains):
            cid = sample["chain_id"]
            if cid not in xml_data_dict:
                continue

            topic, topic_desc = topic_info.get(
                cid,
                (sample.get("topic", "Unknown"), sample.get("topic_description", "")),
            )
            sample["topic"] = topic
            sample["topic_description"] = topic_desc

            history_str = build_history_string(user_chains, idx, n=3)
            xml_sections = parse_xml_sections(xml_data_dict[cid])
            if "answer" not in xml_sections:
                continue

            sft_sample = build_multistage_training_sample(sample, history_str, xml_sections)
            sft_samples.append(sft_sample)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(sft_out_path, "w", encoding="utf-8") as f:
        json.dump(sft_samples, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print("三阶段训练数据构建完成！")
    print(f"{'=' * 70}")
    print(f"- SFT 样本数: {len(sft_samples)} -> {sft_out_path}")

    topic_counts = defaultdict(int)
    for sample in sft_samples:
        topic_counts[sample["topic"]] += 1
    print("\n各话题样本分布：")
    for topic, count in topic_counts.items():
        print(f"  - [{topic}]: {count} 条")

    print("\n✨ Step 2.2 处理完成，可以直接用于三阶段训练。")


if __name__ == "__main__":
    main()
