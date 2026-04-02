#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import json
import os
import sys
import argparse
import pickle
import numpy as np
import networkx as nx
import aiohttp
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.cluster import KMeans

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.llm_client import LLMClient

# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
async def get_embedding_from_text(text: str, session: aiohttp.ClientSession, embedding_config: dict) -> np.ndarray:
    data = {
        "model": embedding_config["model_name"],
        "input": text,
        "encoding_format": "float"
    }
    try:
        headers = {"Authorization": f"Bearer {embedding_config['api_key']}"}
        async with session.post(
            f"{embedding_config['api_base']}/embeddings",
            json=data,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=embedding_config['timeout'])
        ) as response:
            if response.status == 200:
                result = await response.json()
                if 'data' in result and len(result['data']) > 0 and 'embedding' in result['data'][0]:
                    return np.array(result['data'][0]['embedding'], dtype=np.float32)
            return None
    except Exception:
        return None

async def get_embeddings_batch(texts: list, embedding_config: dict, batch_size: int = 10) -> np.ndarray:
    async with aiohttp.ClientSession() as session:
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="获取嵌入向量"):
            batch = texts[i:i+batch_size]
            tasks = [get_embedding_from_text(text, session, embedding_config) for text in batch]
            batch_embeddings = await asyncio.gather(*tasks)
            embeddings.extend(batch_embeddings)
        dim = next((e.shape[0] for e in embeddings if e is not None), 1536)
        embeddings = [e if e is not None else np.zeros(dim, dtype=np.float32) for e in embeddings]
        return np.array(embeddings)

def serialize_state(state_dict):
    return (
        state_dict.get('emotion', '未知'),
        state_dict.get('stance', '未知'),
        state_dict.get('thinking_value', '未知'),
        state_dict.get('intent', '未知')
    )

async def generate_cluster_labels(llm_client, clusters, topic_name):
    """利用 LLM 为事件簇生成语义标签 (按话题隔离，强制口语化)"""
    cluster_labels = {}
    existing_labels = [] 
    
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\n正在生成 [{topic_name}] 话题下的事件簇标签...")
    for cid, texts in tqdm(sorted_clusters):
        samples = texts[:15]
        sample_str = "\n".join([f"- {t}" for t in samples])
        existing_str = ", ".join(existing_labels) if existing_labels else "无"

        prompt = f"""你是一个社会事件分析专家。请根据以下社交媒体讨论中的【事件样本】，总结出一个具体的【事件标签】。
当前讨论所属的话题背景是：【{topic_name}】

【输入数据】
事件样本：
{sample_str}

【当前已存在的其他标签】（请确保新标签与这些不同）：
{existing_str}
如果有一些簇很相似，请修改后出现的事件标签，确保其能看出与之前的事件不同。而不是后面加后缀"_1"来做简单区分。
注意，总结出来的事件是针对这个刺激源的。如“攻击特朗普的政策”这一新帖子被用户看到。我们要总结的是这一刺激源。

1. **具体化**：标签必须包含具体的“主体”和“核心行为/客体”。
   - ❌ 错误示范："代表性争议" (太抽象)、"政策讨论" (太宽泛)、"种族问题"。
   - ✅ 正确示范："讨论黑人女性超级英雄电影"、"质疑医学院录取标准"、"抗议迪士尼DEI政策"。
2. **唯一性**：如果样本内容与【已存在标签】非常相似，请尝试侧重其独特的一面，或使用更细粒度的描述。
3. **简洁性**：控制在 15 字左右。
4. **格式**：只输出标签文本，不要包含引号或任何解释no think

直接输出："""

        try:
            label = await llm_client.generate(prompt, temperature=0.2)
            label = label.strip().replace('"', '').replace("'", "").replace("label:", "")
            
            original_label = label
            counter = 1
            while label in existing_labels:
                label = f"{original_label}_{counter}"
                counter += 1
            
            cluster_labels[cid] = label
            existing_labels.append(label)
        except Exception as e:
            cluster_labels[cid] = f"{topic_name}_Cluster_{cid}"
            
    return cluster_labels

# ----------------------------------------------------------------------
# 核心逻辑
# ----------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description='Stage 3: Graph Construction (Topic Isolated)')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称')
    parser.add_argument('--n_clusters_per_topic', type=int, default=25, help='每个话题下的事件聚类数量')
    parser.add_argument('--model', type=str, default='gpt4o', help='生成标签用的LLM')
    parser.add_argument('--embedding_port', type=int, default=6862, help='Embedding API端口')
    parser.add_argument('--batch_size', type=int, default=16, help='Embedding批次大小')
    args = parser.parse_args()

    embedding_config = {
        "model_name": "Qwen3-Embedding-8B",
        "api_base": f"http://localhost:{args.embedding_port}/v1",
        "api_key": "dummy-key",
        "timeout": 30
    }

    input_dir = project_root / f"core/step1_ate_analysis/output/step_2"
    input_file = input_dir / "causal_chains.json"

    output_dir = project_root / f"core/step1_ate_analysis/output/step_3"
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_file = output_dir / "cognitive_graph.pkl"
    cluster_info_file = output_dir / "event_clusters.json"

    if not input_file.exists():
        print(f"错误: step2 输出文件不存在: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        chains = json.load(f)
    
    # 1. 按话题分组链条
    chains_by_topic = defaultdict(list)
    for chain in chains:
        topic = chain.get('topic', 'Unknown')
        chains_by_topic[topic].append(chain)

    llm_config = {
        "model_name": args.model,
        "temperature": 0.1,
        "api_key": "",
        "base_url": ""
    }
    llm_client = LLMClient(llm_config)
    is_llm_ready = await llm_client.test_connection()

    global_cluster_info = {}
    G = nx.MultiDiGraph() # 全局图谱

    # 2. 对每个话题独立进行聚类和建图
    for topic, topic_chains in chains_by_topic.items():
        print(f"\n{'='*50}")
        print(f"开始处理话题: [{topic}] (链条数量: {len(topic_chains)})")
        print(f"{'='*50}")

        events = [item['t']['event'] for item in topic_chains]
        rationales = [item.get('rationale', '') for item in topic_chains]

        clustering_texts = [f"事件: {e}\n归因: {r}" for e, r in zip(events, rationales)]

        embeddings = await get_embeddings_batch(clustering_texts, embedding_config, batch_size=args.batch_size)
        
        k = min(args.n_clusters_per_topic, len(events))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        clusters_text = defaultdict(list)
        for i, label in enumerate(labels):
            clusters_text[int(label)].append(events[i])
            
        if is_llm_ready:
            cluster_labels = await generate_cluster_labels(llm_client, clusters_text, topic)
        else:
            cluster_labels = {cid: f"{topic}_Cluster_{cid}" for cid in clusters_text}
            
        # 保存该话题的聚类信息
        global_cluster_info[topic] = {
            "k": k,
            "clusters": {
                cid: {
                    "label": lbl,
                    "count": len(clusters_text[cid]),
                    "examples": clusters_text[cid][:5]
                } for cid, lbl in cluster_labels.items()
            }
        }

        # 往全局图中添加节点和边
        for i, chain in enumerate(topic_chains):
            cid = int(labels[i])
            event_label = cluster_labels[cid]
            dominant_dim = chain['t'].get('dominant_dimension', 'Other')

            state_prev = serialize_state(chain['t_minus_1']['state'])
            state_curr = serialize_state(chain['t']['state'])

            if not G.has_node(state_prev):
                G.add_node(state_prev, count=0, type='state')
            G.nodes[state_prev]['count'] += 1

            if not G.has_node(state_curr):
                G.add_node(state_curr, count=0, type='state')
            G.nodes[state_curr]['count'] += 1

            # 添加边，增加 topic 属性以供后续隔离计算使用
            if G.has_edge(state_prev, state_curr, key=event_label):
                edge_data = G[state_prev][state_curr][event_label]
                edge_data['count'] += 1
                edge_data['raw_chains'].append(chain['chain_id'])
                if 'dominant_dims' not in edge_data:
                    edge_data['dominant_dims'] = defaultdict(int)
                edge_data['dominant_dims'][dominant_dim] += 1
            else:
                G.add_edge(state_prev, state_curr,
                           key=event_label,
                           topic=topic, # 强制写入话题
                           event_cluster=cid,
                           event_name=event_label,
                           count=1,
                           raw_chains=[chain['chain_id']],
                           dominant_dims=defaultdict(int, {dominant_dim: 1}))

    await llm_client.close()
    
    with open(cluster_info_file, 'w', encoding='utf-8') as f:
        json.dump(global_cluster_info, f, ensure_ascii=False, indent=2)
    print(f"\n话题聚类信息已保存至: {cluster_info_file}")

    print(f"图谱构建完成: 节点数 {G.number_of_nodes()}, 边数 {G.number_of_edges()}")
    with open(graph_file, 'wb') as f:
        pickle.dump(G, f)
        
    print("完成！")

if __name__ == "__main__":
    asyncio.run(main())