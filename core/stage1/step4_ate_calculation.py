#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import pickle
import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def calculate_state_ate(G, min_samples=3):
    """遍历图谱计算状态级 ATE，按 Topic 隔离计算基准概率"""
    print(f"开始计算状态级 ATE (最小样本阈值: {min_samples})...")
    updates_count = 0
    causal_rules = []

    for u in tqdm(G.nodes(), desc="计算各状态的ATE"):
        out_edges = list(G.out_edges(u, data=True, keys=True))
        if not out_edges:
            continue

        # 先按 topic 分组出边
        edges_by_topic = defaultdict(list)
        for _, v, key, d in out_edges:
            topic = d.get('topic', 'Unknown')
            edges_by_topic[topic].append((v, key, d))

        # 针对每个 topic 独立计算
        for topic, t_edges in edges_by_topic.items():
            # 计算该话题下的基准转移概率 (Baseline)
            total_transitions = sum(d['count'] for _, _, d in t_edges)
            target_counts = defaultdict(int)
            for v, _, d in t_edges:
                target_counts[v] += d['count']
            
            p_natural = {v: count / total_transitions for v, count in target_counts.items()}

            # 按事件分组统计
            event_groups = defaultdict(lambda: {'total': 0, 'targets': defaultdict(int)})
            for v, key, d in t_edges:
                event = key
                count = d['count']
                event_groups[event]['total'] += count
                event_groups[event]['targets'][v] += count

            # 计算 ATE 并更新图谱
            for v, key, d in t_edges:
                event = key
                group_stats = event_groups[event]
                n_event = group_stats['total']

                if n_event < min_samples:
                    p_intervention = p_natural[v]
                    is_reliable = False
                else:
                    p_intervention = group_stats['targets'][v] / n_event
                    is_reliable = True

                ate = p_intervention - p_natural[v]

                G[u][v][key]['p_nat'] = round(p_natural[v], 4)
                G[u][v][key]['p_int'] = round(p_intervention, 4)
                G[u][v][key]['ate'] = round(ate, 4)
                G[u][v][key]['weight'] = round(ate, 4)
                G[u][v][key]['reliable'] = is_reliable
                G[u][v][key]['n_samples'] = n_event

                updates_count += 1

                if is_reliable and abs(ate) > 0.1:
                    causal_rules.append({
                        "topic": topic,
                        "from_state": u,
                        "event": event,
                        "to_state": v,
                        "ate": round(ate, 4),
                        "p_int": round(p_intervention, 4),
                        "p_nat": round(p_natural[v], 4),
                        "support": n_event
                    })

    print(f"完成! 更新了 {updates_count} 条边的 ATE 权重。")
    return G, causal_rules

def calculate_dimension_ate(G, min_samples=3):
    """计算维度级 ATE，按 Topic 隔离"""
    print(f"\n开始计算维度级 ATE (最小样本阈值: {min_samples})...")
    lookup_table = {}

    for u in tqdm(G.nodes(), desc="计算维度ATE"):
        out_edges = list(G.out_edges(u, data=True, keys=True))
        if not out_edges:
            continue

        edges_by_topic = defaultdict(list)
        for _, v, key, d in out_edges:
            topic = d.get('topic', 'Unknown')
            edges_by_topic[topic].append((v, key, d))

        state_key = str(u)

        for topic, t_edges in edges_by_topic.items():
            total_dim_counts = defaultdict(int)
            total_samples = 0

            for _, _, d in t_edges:
                if 'dominant_dims' in d:
                    for dim, c in d['dominant_dims'].items():
                        if dim != 'Other':
                            total_dim_counts[dim] += c
                            total_samples += c

            if total_samples < min_samples:
                continue 

            p_natural = {dim: c / total_samples for dim, c in total_dim_counts.items()}

            event_dims = defaultdict(lambda: defaultdict(int))
            event_totals = defaultdict(int)

            for _, key, d in t_edges:
                event_name = key
                if 'dominant_dims' in d:
                    for dim, c in d['dominant_dims'].items():
                        if dim != 'Other':
                            event_dims[event_name][dim] += c
                            event_totals[event_name] += c

            for event_name, count in event_totals.items():
                if count < min_samples:
                    continue

                current_counts = event_dims[event_name]
                p_intervention = {dim: c / count for dim, c in current_counts.items()}

                ate_scores = {}
                all_dims = set(p_natural.keys()) | set(p_intervention.keys()) | {'Emotion', 'Thinking', 'Stance', 'Intent', 'Other'}

                for dim in all_dims:
                    p_int = p_intervention.get(dim, 0.0)
                    p_nat = p_natural.get(dim, 0.0)
                    ate_scores[dim] = round(p_int - p_nat, 4)

                # 将 topic 加入 Lookup Key
                lookup_key = f"[{topic}]_{state_key}::{event_name}"
                lookup_table[lookup_key] = ate_scores

    return lookup_table

def flatten_and_save_data(project_root, dataset, mode_lookup_full, chains, stage1_data):
    """扁平化数据"""
    print(f"\n开始数据扁平化...")

    original_data_map = {}
    for item in stage1_data:
        key = (item['user_id'], item['target_post'].strip())
        original_data_map[key] = item

    default_ate_scores = {'Emotion': 0.2, 'Thinking': 0.2, 'Stance': 0.2, 'Intent': 0.2, 'Other': 0.2}

    final_data = []
    processed_chain_ids = set()
    processed_samples = set()

    for scenario_key, scenario_data in tqdm(mode_lookup_full.items(), desc="处理场景"):
        ate_scores = scenario_data['ate_scores']
        samples = scenario_data['samples']

        for sample in samples:
            chain_id = sample['chain_id']
            processed_chain_ids.add(chain_id)

            key = (sample['user_id'], sample['t']['behavior'].strip())
            processed_samples.add(key)
            orig = original_data_map.get(key, {})

            item = {
                'chain_id': chain_id,
                'conversation_id': orig.get('conversation_id'),
                'user_id': sample['user_id'],
                'sub_id': orig.get('sub_id'),
                'timestep': orig.get('timestep'),
                'action_type': orig.get('action_type'),
                'topic': sample.get('topic', 'Unknown'), # 新增导出 topic
                'original_post': orig.get('original_post'),
                'context_post': orig.get('context_post'),
                'target_post': orig.get('target_post'),
                'dominant_dimension': sample['dominant_dimension'],
                'prev_cognitive_labels': sample['t_minus_1']['state'],
                'prev_post': sample['t_minus_1'].get('post', ''),
                'curr_cognitive_labels': sample['t']['state'],
                'curr_behavior': sample['t']['behavior'],
                'curr_event': sample['t'].get('event', ''),
                'causal_rationale': sample.get('rationale', ''),
                'ate_scores': ate_scores,
                'scenario_key': scenario_key,
                'is_single_interaction': False,
                'has_ate_calculation': True
            }
            final_data.append(item)

    for chain in tqdm(chains, desc="处理缺失链条"):
        chain_id = chain['chain_id']
        if chain_id not in processed_chain_ids:
            key = (chain['user_id'], chain['t']['behavior'].strip())
            processed_samples.add(key)
            orig = original_data_map.get(key, {})

            item = {
                'chain_id': chain_id,
                'conversation_id': orig.get('conversation_id'),
                'user_id': chain['user_id'],
                'sub_id': orig.get('sub_id'),
                'timestep': orig.get('timestep'),
                'action_type': orig.get('action_type'),
                'topic': chain.get('topic', 'Unknown'),
                'original_post': orig.get('original_post'),
                'context_post': orig.get('context_post'),
                'target_post': orig.get('target_post'),
                'dominant_dimension': chain['t'].get('dominant_dimension', 'Other'),
                'prev_cognitive_labels': chain['t_minus_1']['state'],
                'prev_post': chain['t_minus_1'].get('post', ''),
                'curr_cognitive_labels': chain['t']['state'],
                'curr_behavior': chain['t']['behavior'],
                'curr_event': chain['t'].get('event', ''),
                'causal_rationale': chain.get('rationale', ''),
                'ate_scores': default_ate_scores, 
                'scenario_key': 'insufficient_samples',
                'is_single_interaction': False,
                'has_ate_calculation': False 
            }
            final_data.append(item)

    single_interaction_count = 0
    for orig_item in stage1_data:
        key = (orig_item['user_id'], orig_item['target_post'].strip())
        if key not in processed_samples:
            single_interaction_count += 1
            item = {
                'chain_id': f"single_{orig_item['user_id']}_{single_interaction_count}",
                'conversation_id': orig_item.get('conversation_id'),
                'user_id': orig_item['user_id'],
                'sub_id': orig_item.get('sub_id'),
                'timestep': orig_item.get('timestep'),
                'action_type': orig_item.get('action_type'),
                'topic': orig_item.get('topic', 'Unknown'),
                'original_post': orig_item.get('original_post'),
                'context_post': orig_item.get('context_post'),
                'target_post': orig_item.get('target_post'),
                'dominant_dimension': orig_item.get('cognitive_labels', {}).get('dominant_dimension', 'Other'),
                'prev_cognitive_labels': {},
                'prev_post': '',
                'curr_cognitive_labels': orig_item.get('cognitive_labels', {}),
                'curr_behavior': orig_item['target_post'],
                'curr_event': orig_item.get('event_label', ''),
                'causal_rationale': '独立样本，无因果链',
                'ate_scores': default_ate_scores,
                'scenario_key': 'single_interaction',
                'is_single_interaction': True,
                'has_ate_calculation': False
            }
            final_data.append(item)

    for item in final_data:
        if item['dominant_dimension'] == 'Other':
            item['ate_scores'] = default_ate_scores
            item['has_ate_calculation'] = False

    output_dir = project_root / f"core/step1_ate_analysis/output/step_4"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train_data_with_ate.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    return final_data

def main():
    parser = argparse.ArgumentParser(description='Stage 4: ATE Calculation (Topic Isolated)')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称')
    parser.add_argument('--min_samples', type=int, default=3, help='计算ATE所需的最小事件样本数')
    args = parser.parse_args()

    input_dir = project_root / f"core/step1_ate_analysis/output/step_3"
    graph_file = input_dir / "cognitive_graph.pkl"
    chains_path = project_root / f"core/step1_ate_analysis/output/step_2/causal_chains.json"
    output_dir = project_root / f"core/step1_ate_analysis/output/step_4"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not graph_file.exists():
        print(f"错误: 输入图谱不存在: {graph_file}")
        return

    with open(graph_file, 'rb') as f:
        G = pickle.load(f)

    with open(chains_path, 'r', encoding='utf-8') as f:
        chains = json.load(f)

    stage1_file = project_root / f"core/step1_ate_analysis/output/step_1/step1_data_with_events.json"
    with open(stage1_file, 'r', encoding='utf-8') as f:
        stage1_data = json.load(f)

    G_weighted, rules = calculate_state_ate(G, min_samples=args.min_samples)
    dimension_lookup = calculate_dimension_ate(G, min_samples=args.min_samples)

    dimension_lookup_full = {}
    for chain in chains:
        state_prev = (
            chain['t_minus_1']['state'].get('emotion', '未知'),
            chain['t_minus_1']['state'].get('stance', '未知'),
            chain['t_minus_1']['state'].get('thinking_value', '未知'),
            chain['t_minus_1']['state'].get('intent', '未知')
        )
        state_curr = (
            chain['t']['state'].get('emotion', '未知'),
            chain['t']['state'].get('stance', '未知'),
            chain['t']['state'].get('thinking_value', '未知'),
            chain['t']['state'].get('intent', '未知')
        )
        topic = chain.get('topic', 'Unknown')

        event_name = "Unknown_Event"
        for u, v, key, data in G.edges(keys=True, data=True):
            if u == state_prev and v == state_curr and data.get('topic') == topic:
                event_name = key
                break

        # 匹配 Lookup Key
        lookup_key = f"[{topic}]_{str(state_prev)}::{event_name}"

        if lookup_key in dimension_lookup:
            if lookup_key not in dimension_lookup_full:
                dimension_lookup_full[lookup_key] = {
                    'ate_scores': dimension_lookup[lookup_key],
                    'samples': []
                }
            dimension_lookup_full[lookup_key]['samples'].append({
                'chain_id': chain['chain_id'],
                'user_id': chain['user_id'],
                'topic': topic,
                'dominant_dimension': chain['t'].get('dominant_dimension', 'Other'),
                't_minus_1': chain['t_minus_1'],
                't': chain['t'],
                'rationale': chain.get('rationale', '')
            })

    with open(output_dir / "weighted_cognitive_graph.pkl", 'wb') as f:
        pickle.dump(G_weighted, f)

    rules.sort(key=lambda x: abs(x['ate']), reverse=True)
    for r in rules:
        r['from_state'] = str(r['from_state'])
        r['to_state'] = str(r['to_state'])
    with open(output_dir / "top_causal_rules.json", 'w', encoding='utf-8') as f:
        json.dump(rules[:100], f, ensure_ascii=False, indent=2)

    with open(output_dir / "ate_mode_lookup.json", 'w', encoding='utf-8') as f:
        json.dump(dimension_lookup, f, ensure_ascii=False, indent=2)

    with open(output_dir / "ate_mode_lookup_with_samples.json", 'w', encoding='utf-8') as f:
        json.dump(dimension_lookup_full, f, ensure_ascii=False, indent=2)

    flatten_and_save_data(project_root, args.dataset, dimension_lookup_full, chains, stage1_data)
    print("完成 Stage 4 所有步骤。")

if __name__ == "__main__":
    main()