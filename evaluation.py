import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
from itertools import chain
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes-path', type=str, required=True, help='path to the nodes csv')
    parser.add_argument('--embeddings-path', type=str, required=True, help='path to the embeddings csv')
    parser.add_argument('--output-path', type=str, default="results.csv", help='path to the output csv')
    parser.add_argument('--top-k', type=int, default=1000, help='sorting all content nodes is expensive, so take the top k content nodes for each topic node')
    parser.add_argument('--k', type=int, default=10, help='used in recall, precision, and f1')
    return parser.parse_args()

def get_recall_precision_f1(ranks, k):
    ranks_unpacked = np.array(list(chain(*ranks.values())))
    tp = sum(ranks_unpacked < k)
    recall = tp / len(ranks_unpacked)
    precision = tp / (len(ranks) * k)
    f1 = 2 * (recall * precision) / (recall + precision) if recall + precision > 0 else 0
    return recall, precision, f1

def get_mean_reciprocal_rank(ranks, top_k):
    inverse_ranks = []
    for node_id in ranks:
        if len(ranks[node_id]) == 0:
            continue
        rank = np.min(ranks[node_id]) + 1
        if rank < top_k + 1:
            inverse_ranks.append(1 / rank)
        else:
            inverse_ranks.append(0)
    return np.mean(inverse_ranks)

def get_median_rank(ranks):
    median_ranks = []
    for node_id in ranks:
        if len(ranks[node_id]) == 0:
            continue
        median_rank = np.median(ranks[node_id]) + 1
        median_ranks.append(median_rank)
    return np.median(median_ranks)

def get_mean_rank(ranks):
    mean_ranks = []
    for node_id in ranks:
        if len(ranks[node_id]) == 0:
            continue
        mean_rank = np.mean(ranks[node_id]) + 1
        mean_ranks.append(mean_rank)
    return np.mean(mean_ranks)

def get_metrics(x, ranks):
    cur_ranks = {node_id: ranks[node_id] for node_id in x["node_id"]}
    recall, precision, f1 = get_recall_precision_f1(cur_ranks, args.k)
    mrr = get_mean_reciprocal_rank(cur_ranks, args.top_k)
    median_rank = get_median_rank(cur_ranks)
    mean_rank = get_mean_rank(cur_ranks)
    return pd.DataFrame({
        f"recall@{args.k}": [recall],
        f"precision@{args.k}": [precision],
        f"f1@{args.k}": [f1],
        "mean reciprocal rank": [mrr],
        "median rank": [median_rank],
        "mean rank": [mean_rank]
    })


if __name__ == '__main__':
    args = parse_arguments()
    print("Reading input data...", end=" ")
    nodes = pd.read_csv(args.nodes_path)
    embeddings = pd.read_csv(args.embeddings_path, index_col=0)
    
    topic_nodes = nodes[nodes["kind"] == "topic"].reset_index(drop=True)
    content_nodes = nodes[nodes["kind"] != "topic"].reset_index(drop=True)
    
    topic_nodes_embeddings = embeddings.loc[topic_nodes["node_id"], :].reset_index(drop=True)
    content_nodes_embeddings = embeddings.loc[content_nodes["node_id"], :].reset_index(drop=True)
    print("done")

    print("Calculating cosine similarity...", end=" ")
    cos_sim = cosine_similarity(topic_nodes_embeddings.to_numpy(), content_nodes_embeddings.to_numpy())
    print("done")
    
    print("Get predictions...", end=" ")
    # take top-k content nodes for each topic node
    cos_sim_partition = np.argpartition(cos_sim, kth=-args.top_k, axis=1)
    top_cos_sim = np.take_along_axis(cos_sim, cos_sim_partition[:,-args.top_k:], axis=1)
    top_cos_sim_argsort = np.argsort(-top_cos_sim, axis=1)
    cos_sim_argsort = np.take_along_axis(cos_sim_partition[:,-args.top_k:], top_cos_sim_argsort, axis=1)
    
    cos_sim_pred = content_nodes.loc[cos_sim_argsort.reshape(-1),
                                "content_id"].to_numpy().reshape((len(topic_nodes), args.top_k))
    predictions = {}
    for i in range(len(topic_nodes)):
        predictions[topic_nodes.loc[i, "node_id"]] = list(OrderedDict.fromkeys(cos_sim_pred[i]))
    print("done")
        
    # get ground truth
    print("Get ground truth...", end=" ")
    gt = content_nodes.groupby("parent_id").apply(lambda x: set(x["content_id"])).to_dict()
    for i in range(topic_nodes["level"].max(), -1, -1):
        for _, row in topic_nodes[topic_nodes["level"] == i].iterrows():
            gt.setdefault(row["parent_id"], set())
            gt.setdefault(row["node_id"], set())
            gt[row["parent_id"]].update(gt[row["node_id"]])
    for key in set(gt.keys()) - set(topic_nodes.node_id):
        del gt[key]
    print("done")
    
    # get ranks for every ground truth content
    print("Get ranks for each node...", end=" ")
    ranks = {}
    for node_id in gt:
        ranks[node_id] = []
        for content_id in gt[node_id]:
            if content_id in predictions[node_id]:
                ranks[node_id].append(predictions[node_id].index(content_id))
            else:
                ranks[node_id].append(args.top_k)
    print("done")

    print("Generating resultst...", end=" ")
    # calculate metrics using ranks
    results = get_metrics(topic_nodes, ranks=ranks)
    cur_results = topic_nodes.groupby("language").apply(
        get_metrics, ranks=ranks).reset_index().drop(columns=["level_1"])
    results = results.append(cur_results)
    cur_results = topic_nodes.groupby(["language", "condition"]).apply(
        get_metrics, ranks=ranks).reset_index().drop(columns=["level_2"])
    results = results.append(cur_results)
    cur_results = topic_nodes.groupby(["language", "condition", "channel_id"]).apply(
        get_metrics, ranks=ranks).reset_index().drop(columns=["level_3"])
    results = results.append(cur_results)
    print("done")
    
    # save
    results.to_csv(args.output_path, index=False)
    print(f"results saved to {args.output_path}")
