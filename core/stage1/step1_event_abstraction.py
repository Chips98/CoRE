#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import asyncio
import json
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# 路径设置：确保能导入项目根目录的模块
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.llm_client import LLMClient

def load_data(data_path):
    """加载数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# ----------------------------------------------------------------------
# Prompt 定义
# ----------------------------------------------------------------------
def build_event_prompt(item):
    """构造完整的 Prompt（包含系统指令和用户输入）"""
    topic = item.get('topic', '未知话题')
    topic_description = item.get('topic_description', '')

    prompt = f"""你是一个社会认知学专家。你的任务是分析社交媒体对话中的【刺激源】。
请基于用户的回复内容（Target Post），专注于分析用户是针对什么内容（Original/Context Post）进行回应的。
请将这个刺激源概括为一个简短的、具体的【事件描述】（15~30字）。

【话题背景】
话题：{topic}
背景描述：{topic_description}

【输出格式与严苛要求】
【严禁使用】：“针对...的言论进行反驳”、“回应...争议”等学术句式。
【必须使用】：第一人称视角的直白描述、带明确动作、带具体立场、口语化、接地气。

示例：
输入：上下文是特朗普要求大幅降息，用户在下面说“老头又在干预美联储了，这样通胀要爆炸”。
输出：{{"event_summary": "看到特朗普施压美联储降息"}}

输入：上下文是黑人小美人鱼剧照发布，用户在骂选角。
输出：{{"event_summary": "看到反对DEI选角的帖子"}}

输入：别人回复该用户：“你的数据全是编的，去看看劳工局官网吧”。
输出：{{"event_summary": "遭到对方的数据打脸"}}

输入：一个民主党支持者发帖说哈里斯必胜。
输出：{{"event_summary": "看到支持哈里斯的言论"}}

---

[背景帖子]: {item.get('original_post', '无')}
[直接回复对象]: {item.get('context_post', '无')}
[用户回复(仅供参考)]: {item.get('target_post', '无')}

请分析：在【{topic}】话题背景下，用户是在什么具体情境（事件）下做出回复的？
请输出 JSON 格式：{{"event_summary": "你的概括"}}
只输出JSON，不要其他内容no think"""

    return prompt

# ----------------------------------------------------------------------
# 核心逻辑
# ----------------------------------------------------------------------

async def process_single_item(sem, llm_client, item, pbar):
    """处理单条数据"""
    async with sem:
        try:
            # 如果已经跑过且有结果，跳过（断点续传）
            if 'event_label' in item and item['event_label']:
                pbar.update(1)
                return item

            # 构造完整的 prompt
            prompt = build_event_prompt(item)

            # 调用 LLM
            response = await llm_client.generate(prompt, temperature=0.1)

            # 解析 JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)
                item['event_label'] = result.get('event_summary', '未知事件')
            else:
                item['event_label'] = '解析失败'

        except Exception as e:
            item['event_label'] = f"处理出错: {str(e)[:50]}"
        finally:
            # 确保透传话题字段
            if 'topic' not in item:
                item['topic'] = '未知话题'
            if 'topic_description' not in item:
                item['topic_description'] = ''

            pbar.update(1)
            return item

async def main():
    parser = argparse.ArgumentParser(description='Stage 1: Event Abstraction')
    parser.add_argument('--dataset', type=str, default='four_topics', help='数据集名称 (e.g., four_topics, dei, ue)')
    parser.add_argument('--model', type=str, default='gpt4o', help='使用的模型')
    parser.add_argument('--concurrency', type=int, default=16, help='并发数')
    parser.add_argument('--limit', type=int, default=None, help='测试用，限制处理数量')
    args = parser.parse_args()

    # 1. 路径准备
    data_dir = project_root / f"data/{args.dataset}"
    input_file = data_dir / "train_data.json"

    # 输出路径设置
    output_dir = project_root / f"core/step1_ate_analysis/output/step_1"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "step1_data_with_events.json"

    print(f"加载数据: {input_file}")
    data = load_data(str(input_file))

    if args.limit:
        data = data[:args.limit]
        print(f"测试模式: 限制处理 {args.limit} 条样本")

    # 2. 初始化 LLM
    llm_config = {
        "model_name": args.model,
        "temperature": 0.1,
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

    # 3. 并发处理
    sem = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(data), desc="抽取事件")

    tasks = [process_single_item(sem, llm_client, item, pbar) for item in data]
    processed_data = await asyncio.gather(*tasks)

    await llm_client.close()

    # 4. 保存结果
    print(f"\n保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"完成！共处理 {len(processed_data)} 条数据")

if __name__ == "__main__":
    asyncio.run(main())