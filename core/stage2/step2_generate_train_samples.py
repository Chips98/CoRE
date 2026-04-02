#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# 路径设置
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def build_history_string(user_chains, current_chain_idx, n=3):
    """构建前 n 轮的历史交互信息"""
    history_str = ""
    start_idx = max(0, current_chain_idx - n)

    for i in range(start_idx, current_chain_idx):
        chain = user_chains[i]
        round_num = i + 1
        post = chain.get('prev_post', '')
        labels = chain.get('prev_cognitive_labels', {})

        if not post:
            continue

        history_str += f"第{round_num}轮\n"
        history_str += f"[用户发言]：{post}\n"
        history_str += f"[用户认知状态]：情感:{labels.get('emotion')}, 立场:{labels.get('stance')}, 思维风格:{labels.get('thinking_value')}, 意图:{labels.get('intent')}\n"

    return history_str if history_str else "无历史交互记录数据。"

def get_stance_range(topic):
    """根据话题获取立场标签范围"""
    ranges = {
        "CUT": "不明确、支持中方、支持美方",
        "FRIR": "不明确、支持降息、反对降息",
        "UE": "不明确、支持共和党、支持民主党",
        "DEI": "不明确、支持废除DEI、反对废除DEI"
    }
    if topic is None:
        return "不明确"
    return ranges.get(topic.upper(), "不明确")

def generate_prompt(sample, history_str):
    """生成全链路推理提示词 - 围绕四个评估维度指导，生成五维度推理"""
    topic = sample.get('topic', '未知话题')
    topic_desc = sample.get('topic_description', '无话题描述信息。')

    dominant_dim = sample.get('dominant_dimension', '未知')
    current_stance_label = get_stance_range(topic)

    prompt = f"""### [系统人格]
你是一个处于社交网络中的数字人类。你需要根据你的历史发言、当前面对的社交刺激，按照心理学路径逐步思考，并给出最终的回复。

### [话题背景]
- 话题名称：{topic}
- 该话题的立场标签：{current_stance_label}
- 背景描述：{topic_desc}

### [历史交互记忆]
{history_str}

### [当前刺激]
- 原始帖子：{sample.get('original_post', '')}
- 直接回复对象：{sample.get('context_post', '')}
- 触发事件：{sample.get('curr_event', '')}

### [任务指令]

在做出最终回复前，你必须进行深度的自我内省。请从以下五个维度的顺序进行思考，并将思考过程包裹在对应的 XML 标签中。

在分析时，请特别关注这四个评估角度：
- **认知状态一致性**: 你的发言是否准确体现了你的情感、思维风格、立场和意图？
- **上下文交互合理性**: 你的发言对触发事件的回应是否符合逻辑？
- **人设与语气拟真度**: 你的发言是否像真实的社交媒体用户，避免"AI味"？
- **主导因果跟随性**: 你的发言是否被正确的主导维度【{dominant_dim}】所驱动？

思考路径必须严格遵循以下格式（标准XML闭合标签）：

<emotion>基于触发事件，我的情绪发生了怎样的变化？</emotion>
<thinking_type>我应该采用何种思维风格来处理该信息？</thinking_type>
<stance>结合历史记忆与话题背景，我在这件事上的立场是什么？</stance>
<intent>我发表回复的最终社交意图是什么？</intent>
<other>是否有其他外界因素影响了我的判断？</other>
<answer>我最终生成的外显发言内容</answer>"""

    return prompt

def main():
    parser = argparse.ArgumentParser(description='Step 2.2: Compile Unified SFT Training JSON')
    args = parser.parse_args()

    # 1. 动态对齐统一路径
    ate_data_path = project_root / "core/step1_ate_analysis/output/step_5/final_personalized_ate_data.json"
    chains_data_path = project_root / "core/step1_ate_analysis/output/step_2/causal_chains.json"
    xml_data_path = project_root / "core/step2_sample_generate/output/sft_training_data_with_full_chain_5d.json"
    
    # 最终输出路径
    output_dir = project_root / "core/step2_sample_generate/output"
    sft_out_path = output_dir / "train_sft_samples.json"

    # 校验文件是否存在
    for p in [ate_data_path, chains_data_path, xml_data_path]:
        if not p.exists():
            print(f"❌ 错误: 未找到依赖文件 {p}")
            sys.exit(1)

    print("正在加载数据...")
    with open(ate_data_path, 'r', encoding='utf-8') as f:
        ate_data = json.load(f)

    with open(xml_data_path, 'r', encoding='utf-8') as f:
        xml_list = json.load(f)
        xml_data_dict = {item['chain_id']: item['xml'] for item in xml_list}

    with open(chains_data_path, 'r', encoding='utf-8') as f:
        chains_raw = json.load(f)
        topic_info = {c['chain_id']: (c.get('topic'), c.get('topic_description')) for c in chains_raw}

    print(f"加载了 {len(ate_data)} 条多话题 ATE 数据")
    print(f"加载了 {len(xml_data_dict)} 条有效 XML 标签数据")
    print(f"加载了 {len(topic_info)} 条话题上下文信息\n")

    # 2. 构建 ATE 分数字典 (chain_id -> ate_scores)
    ate_scores_dict = {item['chain_id']: item.get('ate_scores', {}) for item in ate_data}

    # 3. 按用户组织链条
    user_to_chains = defaultdict(list)
    for item in ate_data:
        user_to_chains[item['user_id']].append(item)

    for uid in user_to_chains:
        user_to_chains[uid].sort(key=lambda x: x['chain_id'])

    sft_samples = []

    # 4. 构造混合话题大一统样本
    print("正在构建全链路大一统训练样本...")
    for uid, u_chains in user_to_chains.items():
        for idx, sample in enumerate(u_chains):
            cid = sample['chain_id']

            # 补全话题信息
            topic, topic_desc = topic_info.get(cid, (sample.get('topic', '未知话题'), '无描述'))
            sample['topic'] = topic
            sample['topic_description'] = topic_desc

            # 构建历史记录
            history_str = build_history_string(u_chains, idx, n=3)

            # 生成 Prompt
            prompt_text = generate_prompt(sample, history_str)

            # SFT 样本：仅保留有成功生成 XML 标签的记录
            if cid in xml_data_dict:
                sft_sample = {
                    "chain_id": cid,
                    "user_id": uid,
                    "topic": topic,
                    "dominant_dimension": sample.get('dominant_dimension'),
                    "prompt": prompt_text,
                    "label": xml_data_dict[cid]
                }
                # 补充 ATE 分数
                if cid in ate_scores_dict:
                    sft_sample["ate_scores"] = ate_scores_dict[cid]
                sft_samples.append(sft_sample)

    # 5. 保存最终的 JSON 文件
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(sft_out_path, 'w', encoding='utf-8') as f:
        json.dump(sft_samples, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print("大一统训练数据构建完成！")
    print(f"{'='*70}")
    print(f"- SFT 统一样本数: {len(sft_samples)} -> {sft_out_path}")

    # 6. 打印不同话题的分布
    topic_counts = defaultdict(int)
    for s in sft_samples:
        topic_counts[s['topic']] += 1
    print("\n各话题样本分布：")
    for t, count in topic_counts.items():
        print(f"  - [{t}]: {count} 条")

    print("\n✨ Step 2 统一聚合完成，可以准备将此 JSON 喂给模型进行训练了！")

if __name__ == "__main__":
    main()