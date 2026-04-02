#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import asyncio
import json
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.llm_client import LLMClient

DIMENSION_ADJUSTMENT_GUIDELINES = """
1. **Emotion (情感)**:
   - 增加逻辑: 当归因提到"情绪爆发"、"受到冒犯"、"喜悦共鸣"或行为中包含明显的辱骂、宣泄、感性共鸣时。
   - 减少逻辑: 当归因强调理性分析、事实核查或冷静态度时。

2. **Thinking (思维风格)**:
   - 增加逻辑: 当归因提到"事实核查"、"逻辑推演"、"基于经验评价"或"批判性质疑"时。
   - 减少逻辑: 当归因强调情绪驱动或简单附和时。

3. **Stance (立场)**:
   - 增加逻辑: 当归因提到"维护阵营"、"政治站队"、"强化标签"或"身份认同"时。
   - 减少逻辑: 当归因强调中立或事实讨论时。

4. **Intent (意图)**:
   - 增加逻辑: 当归因提到"号召行动"、"拉票"、"操纵舆论"、"蓄意挑起冲突"或"阴阳怪气"时。
   - 减少逻辑: 当归因强调无意识或被动反应时。

5. **Other (其他)**:
   - 增加逻辑: 当发言极短、无明显认知特征、纯附和或标点符号时。
   - 减少逻辑: 当发言具有明确的认知特征时。
"""

def build_personalization_prompt(sample):
    base_ate = sample['ate_scores']
    rationale = sample['causal_rationale']
    behavior = sample['curr_behavior']
    curr_event = sample.get('curr_event', '未识别的触发事件')
    prev_state = sample['prev_cognitive_labels']
    curr_state = sample['curr_cognitive_labels']
    topic = sample.get('topic', '未知话题')
    
    # 兼容没有 topic_description 的情况
    topic_description = "" 

    stance_labels = {
        "CUT": "不明确、支持中方、支持美方",
        "FRIR": "支持降息、反对降息、不明确",
        "UE": "支持共和党、支持民主党、不明确",
        "DEI": "支持废除DEI、反对废除DEI、不明确"
    }
    current_stance_label = stance_labels.get(topic, "不明确")

    prompt = f"""你是一个社会认知科学专家。你的任务是根据用户的【心理归因】对【基准因果效应 (Base ATE)】进行微调，输出该样本个性化的 ATE 分数。

---

### [维度微调指南]
{DIMENSION_ADJUSTMENT_GUIDELINES}

---

### [话题背景]
话题：{topic}
该话题的特定立场标签为：{current_stance_label}

---

### [输入数据]
1. 【场景基准 ATE Scores】:
{json.dumps(base_ate, ensure_ascii=False, indent=2)}

2. 【本轮触发事件 (口语化描述)】:
"{curr_event}"

3. 【本轮个体心理归因 (Rationale)】:
"{rationale}"

4. 【本轮实际回复 (Behavior)】:
"{behavior}"

5. 【认知状态转移轨迹】:
从上一轮状态 {json.dumps(prev_state, ensure_ascii=False)}
演变为本轮状态 {json.dumps(curr_state, ensure_ascii=False)}

---

### [任务流程]
1. **对比分析**：结合【触发事件】和【归因】，判断该用户的心理机制是否比同类场景下的平均水平更倾向于某种特定维度。
2. **话题感知微调**：在【{topic}】话题背景下，考虑该维度在该话题中的特有强度。
3. **分数微调**：在基准 ATE 基础上进行小幅微调（通常在 ±0.4000 之间）。
4. **约束条件**：
   - ATE 代表事件对维度发生的"提升贡献值"，通常在 [-1, 1] 之间。
   - 如果基准分为 0 且 Rationale 中无明显证据支持该维度，请保持为 0 或极小值。
   - 保持微调后的分数体现出该样本与"平均水平"的细微差别。

### [输出格式]
直接输出 JSON，不要任何思考过程或解释。
{{
  "personalized_ate": {{
    "Emotion": 0.xxxx,
    "Thinking": 0.xxxx,
    "Stance": 0.xxxx,
    "Intent": 0.xxxx,
    "Other": 0.xxxx
  }},
  "adjustment_reason": "简述微调逻辑（100字以内）"
}}
"""
    return prompt

async def personalize_sample(sem, llm_client, sample, pbar):
    async with sem:
        if sample.get('is_single_interaction') or not sample.get('has_ate_calculation'):
            pass

        try:
            prompt = build_personalization_prompt(sample)
            response = await llm_client.generate(prompt, temperature=0.3)
            
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                result = json.loads(response[start:end])
                sample['ate_scores'] = result.get('personalized_ate', sample['ate_scores'])
                sample['personalization_log'] = result.get('adjustment_reason', 'LLM个性化')
            else:
                sample['personalization_log'] = "JSON解析失败，保持基准"
        except Exception as e:
            sample['personalization_log'] = f"处理异常: {str(e)[:50]}"
        finally:
            pbar.update(1)
            return sample

async def main():
    parser = argparse.ArgumentParser(description='Stage 5: ATE Personalization')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称')
    parser.add_argument('--model', type=str, default='gpt4o', help='模型名称')
    parser.add_argument('--concurrency', type=int, default=16, help='并发数')
    args = parser.parse_args()

    input_path = project_root / f"cem/step1_ate_analysis/output/step_4/train_data_with_ate.json"
    output_dir = project_root / f"cem/step1_ate_analysis/output/step_5"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "final_personalized_ate_data.json"

    if not input_path.exists():
        print(f"❌ 错误: 未找到输入文件 {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    llm_config = {
        "model_name": args.model,
        "temperature": 0.3,
        "api_key": "",
        "base_url": ""
    }
    llm_client = LLMClient(llm_config)

    if not await llm_client.test_connection():
        print("❌ LLM 连接失败")
        return

    sem = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(data), desc="个性化微调中")

    tasks = [personalize_sample(sem, llm_client, sample, pbar) for sample in data]
    final_data = await asyncio.gather(*tasks)

    await llm_client.close()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"✨ 全部处理完成！已保存至 {output_path}")

if __name__ == "__main__":
    asyncio.run(main())