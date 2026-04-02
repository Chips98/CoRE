#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: 全链路数据合成 - 5维度版本 (Full Chain Ground Truth Generation - 5D Version)
目标：基于个性化 ATE 分数和口语化事件，反向推理5个维度的结构化思考过程
"""

import asyncio
import json
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# 路径设置
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.llm_client import LLMClient

# 系统人格提示词
SYSTEM_PROMPT = """你是一位顶尖的社会认知心理学专家与计算社会科学研究者。你擅长通过"因果认知链"分析社交媒体用户的行为。你需要根据用户的真实发言，反推为什么用户会这样发言，重点从其认知状态（情感、思维风格、立场和意图）等角度来分析。推理内容应该能促使llm在模拟该用户时，看到同样的上下文时给出同样的回复。我需要这些推理内容来训练llm，你的任务就是推理出这些中间思考内容。"""

def build_reverse_inference_prompt(sample):
    """构建反向推理的Prompt - 围绕四个评估维度生成推理路径"""

    # 提取认知状态和行为
    prev_emotion = sample['prev_cognitive_labels'].get('emotion', '未知')
    prev_stance = sample['prev_cognitive_labels'].get('stance', '未知')
    prev_thinking_value = sample['prev_cognitive_labels'].get('thinking_value', '未知')
    prev_intent = sample['prev_cognitive_labels'].get('intent', '未知')

    curr_emotion = sample['curr_cognitive_labels'].get('emotion', '未知')
    curr_stance = sample['curr_cognitive_labels'].get('stance', '未知')
    curr_thinking_value = sample['curr_cognitive_labels'].get('thinking_value', '未知')
    curr_intent = sample['curr_cognitive_labels'].get('intent', '未知')

    # 提取ATE分数和主导维度（仅用于指导推理深度）
    dominant_dim = sample.get('dominant_dimension', '未知')
    ate_scores = sample.get('ate_scores', {})

    behavior = sample.get('curr_behavior', '')
    curr_event = sample.get('curr_event', '未识别的触发事件')
    topic = sample.get('topic', '未知话题')
    topic_description = sample.get('topic_description', '')

    # 根据话题获取立场标签
    stance_labels = {
        "CUT": "不明确、支持中方、支持美方",
        "FRIR": "支持降息、反对降息、不明确",
        "UE": "支持共和党、支持民主党、不明确",
        "DEI": "支持废除DEI、反对废除DEI、不明确"
    }
    current_stance_label = stance_labels.get(topic, "不明确")

    # 根据ATE分数判断各维度的重要性（用于指导推理深度）
    ate_importance = {}
    for dim in ['Emotion', 'Thinking', 'Stance', 'Intent', 'Other']:
        score = ate_scores.get(dim, 0)
        if score > 0.5:
            ate_importance[dim] = '高'
        elif score > 0.2:
            ate_importance[dim] = '中'
        else:
            ate_importance[dim] = '低'

    prompt = f"""{SYSTEM_PROMPT}

### [已知信息 - 已标注]

1. **用户真实发言** - 核心参考:
   "{behavior}"

2. **当前触发事件 (社交刺激源)**:
   "{curr_event}"

3. **话题背景**:
   - 话题: {topic}
   - 立场选项: {current_stance_label}
   - 背景: {topic_description}

4. **用户认知状态**:
   - 历史状态: 情感({prev_emotion}), 立场({prev_stance}), 思维({prev_thinking_value}), 意图({prev_intent})
   - 当前状态: 情感({curr_emotion}), 立场({curr_stance}), 思维({curr_thinking_value}), 意图({curr_intent})

5. **主导驱动维度**: 【{dominant_dim}】

### [任务指令]

你的任务是反向推导用户为什么会说出这句话。请从以下五个维度展开分析，确保推理能够解释用户发言的真实动机。

在分析时，请特别关注这四个评估角度：
- **认知状态一致性**: 发言是否准确体现了用户的情感、思维风格、立场和意图？
- **上下文交互合理性**: 发言对触发事件的回应是否符合逻辑？
- **人设与语气拟真度**: 发言是否像真实的社交媒体用户，避免"AI味"？
- **主导因果跟随性**: 发言是否被正确的主导维度【{dominant_dim}】所驱动？

### [五个维度的推演]

1. **Emotion (情感)** - 分析用户的情感状态如何受【{curr_event}】影响，从 {prev_emotion} 演变为 {curr_emotion}，以及情感与最终发言的关系

2. **Thinking (思维风格)** - 分析用户采用的思维加工方式（理性、经验、直觉等），如何从历史思维风格 {prev_thinking_value} 演变为 {curr_thinking_value}

3. **Stance (立场)** - 分析用户在该话题下的立场如何从 {prev_stance} 演变为 {curr_stance}，以及立场与群体认同的绑定关系

4. **Intent (意图)** - 分析用户的社交意图如何从 {prev_intent} 演变为 {curr_intent}，以及意图如何驱动最终的发言策略

5. **Other (其他)** - 分析是否有其他边缘因素干扰用户的判断

### [输出格式要求]

直接输出 JSON 格式，包含针对五个维度的思考过程（不包含ATE分数）：

{{
  "emotion_thinking": "基于情感维度的演化思考...",
  "thinking_type_thinking": "基于思维风格的加工过程...",
  "stance_thinking": "基于立场的判断与权衡...",
  "intent_thinking": "社交意图的形成过程...",
  "other_thinking": "其他边缘因素或环境噪音的干扰"
}}

只输出JSON，不要任何其他内容。"""

    return prompt

def parse_json_response(response_text):
    """更强大的 JSON 提取函数，支持 Markdown 代码块过滤"""
    try:
        # 1. 尝试寻找 Markdown JSON 块
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        
        # 2. 寻找第一个 { 和最后一个 }
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start == -1: return None
        
        json_str = response_text[start:end]
        # 3. 只清理危险的控制字符，不破坏换行
        return json.loads(json_str)
    except Exception as e:
        print(f"解析失败详情: {e}")
        return None

async def generate_full_chain_sample(sem, llm_client, sample, pbar, max_retries=3):
    """为单个样本生成五维度的推理内容，包含重试机制"""
    async with sem:
        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            try:
                prompt = build_reverse_inference_prompt(sample)
                # 随重试次数降低 temperature 以增加确定性
                temp = max(0.1, 0.7 - (retry_count * 0.2))
                response = await llm_client.generate(prompt, temperature=temp)

                parsed = parse_json_response(response)

                if parsed:
                    # 存储五个维度的推理内容（不包含ATE分数）
                    sample['emotion_thinking'] = parsed.get('emotion_thinking', '无明显情感波动。')
                    sample['thinking_type_thinking'] = parsed.get('thinking_type_thinking', '无深层思维加工。')
                    sample['stance_thinking'] = parsed.get('stance_thinking', '立场无明显作用。')
                    sample['intent_thinking'] = parsed.get('intent_thinking', '无明确社交意图。')
                    sample['other_thinking'] = parsed.get('other_thinking', '无其他干扰因素。')

                    sample['generation_status'] = 'success'
                    sample['retry_count'] = retry_count
                    success = True
                else:
                    retry_count += 1
            except Exception as e:
                retry_count += 1

        if not success:
            sample['generation_status'] = 'failed'
            sample['retry_count'] = retry_count

        pbar.update(1)
        return sample

async def main():
    parser = argparse.ArgumentParser(description='Step 2.1: Full Chain Ground Truth Generation - 5D Version')
    parser.add_argument('--dataset', type=str, default='four_topics', help='数据集名称')
    parser.add_argument('--model', type=str, default='gpt4o', help='使用的模型')
    parser.add_argument('--concurrency', type=int, default=16, help='并发数')
    parser.add_argument('--limit', type=int, default=None, help='测试用限制处理数量')
    parser.add_argument('--max_retries', type=int, default=3, help='JSON解析最大重试次数')
    args = parser.parse_args()

    # 1. 路径设置 (读取 Step 5 个体化 ATE 微调的结果)
    input_path = project_root / f"cem/step1_ate_analysis/output/step_5/final_personalized_ate_data.json"
    
    # 统一将 SFT 数据输出到一个专属文件夹中，混合保存
    output_dir = project_root / f"cem/step2_sample_generate/output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "sft_training_data_with_full_chain_5d.json"

    if not input_path.exists():
        print(f"❌ 错误: 未找到输入文件 {input_path}")
        return

    print(f"加载个性化 ATE 数据: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if args.limit:
        data = data[:args.limit]
        print(f"测试模式: 限制处理 {args.limit} 条样本")

    print(f"准备为 {len(data)} 条混合话题的样本生成 5 维度推理...\n")

    # 2. 初始化 LLM
    llm_config = {
        "model_name": args.model,
        "temperature": 0.5,
        "api_key": "",
        "base_url": ""
    }
    llm_client = LLMClient(llm_config)

    print("测试 LLM 连接...")
    if not await llm_client.test_connection():
        print("❌ LLM 连接失败")
        await llm_client.close()
        return

    print("✅ LLM 连接成功\n")

    # 3. 并发处理
    sem = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(data), desc="生成5维度推理")

    tasks = [generate_full_chain_sample(sem, llm_client, sample, pbar, max_retries=args.max_retries) for sample in data]
    final_data = await asyncio.gather(*tasks)

    await llm_client.close()

    # 4. 构建符合评估维度的 XML 格式数据
    print("\n构建符合四维度评估的 XML 结构的数据...")
    xml_data = []

    for item in final_data:
        if item.get('generation_status') == 'success':

            xml_str = f"""<emotion>
{item.get('emotion_thinking')} | {item.get('emotion_thinking')}
</emotion>
<thinking_type>
{item.get('thinking_type_thinking')}
</thinking_type>
<stance>
{item.get('stance_thinking')}
</stance>
<intent>
{item.get('intent_thinking')}
</intent>
<other>
{item.get('other_thinking')}
</other>
<answer>
{item.get('curr_behavior', '')}
</answer>"""
            xml_data.append({
                'chain_id': item.get('chain_id'),
                'user_id': item.get('user_id'),
                'topic': item.get('topic'),
                'dominant_dimension': item.get('dominant_dimension'),
                'xml': xml_str,
                'generation_status': 'success'
            })

    # 5. 统计生成结果
    success_count = len(xml_data)
    failed_count = len(final_data) - success_count

    print(f"\n{'='*70}")
    print(f"生成完成统计")
    print(f"{'='*70}")
    print(f"总样本数: {len(final_data)}")
    print(f"成功生成: {success_count} ({success_count/len(final_data)*100:.1f}%)")
    print(f"生成失败: {failed_count} ({failed_count/len(final_data)*100:.1f}%)")

    # 6. 保存结果
    print(f"\n保存 SFT 训练数据至: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(xml_data, f, ensure_ascii=False, indent=2)

    print("\n✨ Step 2.1 处理完成！")

if __name__ == "__main__":
    asyncio.run(main())