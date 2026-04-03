#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import json
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# 路径设置
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.llm_client import LLMClient
from utils.utils import group_and_sort_by_user

# ----------------------------------------------------------------------
# Prompt 定义
# ----------------------------------------------------------------------
def build_attribution_prompt(c_prev, event, c_curr, behavior, topic='', topic_description=''):
    """构造归因 + 主导维度推理 Prompt"""

    stance_labels = {
        "CUT": "不明确、支持中方、支持美方",
        "FRIR": "支持降息、反对降息、不明确",
        "UE": "支持共和党、支持民主党、不明确",
        "DEI": "支持废除DEI、反对废除DEI、不明确"
    }
    current_stance_label = stance_labels.get(topic, "不明确")

    prompt = f"""你是一个认知心理学归因专家。我们要分析该用户为什么会发生心理状态的变化（或者保持不变），并反向推理出主导该用户发言的核心认知维度。

请分析【触发事件】如何作用于【上一轮状态】，导致了【本轮状态】。
同时，基于用户的【真实发言】和【认知状态变化】，判断是哪个核心维度主导了用户的此次发言。

请生成一个【因果规则 (Rationale)】（200字以内），解释这种心理演变的内在逻辑，并说明推理出的主导维度。

【话题背景】
话题：{topic}
该话题的特定立场标签为：{current_stance_label}
背景描述：{topic_description}

---

[上一轮认知状态]:
{json.dumps(c_prev, ensure_ascii=False, indent=2)}

[本轮触发事件 (刺激源)]（这是一个直白的口语化动作描述）:
"{event}"

[本轮认知状态 (结果)]:
{json.dumps(c_curr, ensure_ascii=False, indent=2)}

[本轮实际发言 (行为)]:
"{behavior}"

**任务**：
1. 分析，是具体的什么逻辑或心理机制，导致了从"上一轮状态"到"本轮状态"的演变？
   如果是状态突变（如情绪激动），解释刺激源的作用；如果是状态惯性，解释信念的坚持。
   归因中应该体现该话题的特有特征（例如在DEI话题下提到"优绩主义"，在FRIR话题下提到"通胀风险"）。

2. 基于用户的【真实发言】和【认知状态变化】，判断是哪个核心维度主导了用户的此次发言？
   可选的维度（必须从中选择其一）：
   - Emotion (情感)：发言明显由情绪发泄、感性共鸣驱动。例如：辱骂回击、强烈共鸣、宣泄愤怒/悲伤。
   - Thinking (思维风格)：发言明显由事实核查、逻辑推理、个人经验或批判思维驱动。例如：辟谣、数据辩论、长文分析、逻辑纠错。
   - Stance (立场)：发言明显为了捍卫某一阵营立场驱动。当前话题立场为：{current_stance_label}。例如：政治站队、维护群体利益、口号式表达。
   - Intent (意图)：发言具有强烈的目标导向，由社交策略或预演预判驱动。例如：拉票、操纵舆论、反串、阴阳怪气、号召行动。
   - Other (其他)：发言为无意义的极短附和、纯标点，或不受上述四个维度强驱动。

请输出 JSON 格式：{{"rationale": "你的归因解释", "dominant_dimension": "Emotion/Thinking/Stance/Intent/Other"}}
只输出JSON，不要其他内容no think"""

    return prompt

# ----------------------------------------------------------------------
# 核心逻辑
# ----------------------------------------------------------------------

async def process_chain_node(sem, llm_client, chain_item, pbar):
    """处理单条因果链节点，生成 Rationale"""
    async with sem:
        try:
            prompt = build_attribution_prompt(
                chain_item['t_minus_1']['state'],
                chain_item['t']['event'],
                chain_item['t']['state'],
                chain_item['t']['behavior'],
                topic=chain_item.get('topic', ''),
                topic_description=chain_item.get('topic_description', '')
            )

            response = await llm_client.generate(prompt, temperature=0.3)

            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                result = json.loads(response[start:end])
                chain_item['rationale'] = result.get('rationale', '无归因')
                chain_item['t']['dominant_dimension'] = result.get('dominant_dimension', 'Other')
            else:
                chain_item['rationale'] = 'JSON解析失败'
                chain_item['t']['dominant_dimension'] = 'Other'

        except Exception as e:
            chain_item['rationale'] = f"处理出错: {str(e)[:50]}"
            chain_item['t']['dominant_dimension'] = 'Other'
        finally:
            pbar.update(1)
            return chain_item

async def main():
    parser = argparse.ArgumentParser(description='Stage 2: Chain Extraction & Attribution')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称 (e.g., four_topics, dei)')
    parser.add_argument('--model', type=str, default='gpt4o', help='使用的模型')
    parser.add_argument('--concurrency', type=int, default=16, help='并发数')
    parser.add_argument('--min_interactions', type=int, default=3, help='最小交互轮数')
    parser.add_argument('--limit', type=int, default=None, help='小样本测试：限制处理的链条数量')
    args = parser.parse_args()

    input_dir = project_root / "core/step1_ate_analysis/output/step_1"
    input_file = input_dir / "step1_data_with_events.json"

    output_dir = project_root / "core/step1_ate_analysis/output/step_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "causal_chains.json"

    if not input_file.exists():
        print(f"错误: Stage 1 输出文件不存在: {input_file}")
        return

    print(f"加载 Stage 1 数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("构建滑动窗口 (t-1 -> t)...")
    user_data = group_and_sort_by_user(data)

    chains_to_process = []

    for user_id, items in user_data.items():
        if len(items) < args.min_interactions:
            continue

        for i in range(1, len(items)):
            prev_item = items[i-1]
            curr_item = items[i]

            if 'cognitive_labels' not in prev_item or 'event_label' not in curr_item:
                continue

            chain_obj = {
                "user_id": user_id,
                "chain_id": f"{user_id}_{i-1}_to_{i}",
                "topic": prev_item.get('topic', '未知话题'),
                "topic_description": prev_item.get('topic_description', ''),
                "t_minus_1": {
                    "timestep": prev_item.get('timestep'),
                    "event": prev_item.get('event_label', ''), # 新增保留上一轮事件
                    "state": prev_item['cognitive_labels'],
                    "post": prev_item['target_post']
                },
                "t": {
                    "timestep": curr_item.get('timestep'),
                    "event": curr_item.get('event_label', '未识别事件'),
                    "state": curr_item['cognitive_labels'],
                    "behavior": curr_item['target_post']
                },
                "rationale": None 
            }
            chains_to_process.append(chain_obj)

    print(f"共需处理 {len(chains_to_process)} 条因果链")

    if args.limit and args.limit > 0:
        chains_to_process = chains_to_process[:args.limit]
        print(f"限制模式：仅处理前 {len(chains_to_process)} 条链条")

    llm_config = {
        "model_name": args.model,
        "temperature": 0.3,
        "api_key": "",
        "base_url": ""
    }
    llm_client = LLMClient(llm_config)

    print("测试 LLM 连接...")
    if not await llm_client.test_connection():
        print("错误: LLM 连接失败")
        await llm_client.close()
        return

    print("LLM 连接成功\n")

    sem = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(chains_to_process), desc="生成归因")

    tasks = [process_chain_node(sem, llm_client, item, pbar) for item in chains_to_process]
    final_chains = await asyncio.gather(*tasks)

    await llm_client.close()

    print(f"\n保存因果链到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_chains, f, ensure_ascii=False, indent=2)

    print(f"完成！共生成 {len(final_chains)} 条因果链")
    print("\n✅ Stage 2 处理完成！")

if __name__ == "__main__":
    asyncio.run(main())
