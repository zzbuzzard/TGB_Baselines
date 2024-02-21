import logging
import numpy as np
import warnings
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu
from utils.utils import get_neighbor_sampler, NeighborSampler
from utils.DataLoader import get_idx_data_loader, get_link_pred_data_TRANS_TGB, Data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

from tgb.linkproppred.evaluate import Evaluator

from plot_utils import get_temporal_edge_times


def query_pred_edge_batch(model_name: str, model: nn.Module,
                          src_node_ids: int, dst_node_ids: int, node_interact_times: float, edge_ids: int,
                          edges_are_positive: bool, num_neighbors: int, time_gap: int):
    """
    query the prediction probabilities for a batch of edges
    """
    if model_name in ['TGAT', 'CAWN', 'TCL']:
        # get temporal embedding of source and destination nodes
        # two Tensors, with shape (batch_size, node_feat_dim)
        batch_src_node_embeddings, batch_dst_node_embeddings = \
            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                              dst_node_ids=dst_node_ids,
                                                              node_interact_times=node_interact_times,
                                                              num_neighbors=num_neighbors)

    elif model_name in ['JODIE', 'DyRep', 'TGN']:
        # get temporal embedding of source and destination nodes
        # two Tensors, with shape (batch_size, node_feat_dim)
        batch_src_node_embeddings, batch_dst_node_embeddings = \
            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                              dst_node_ids=dst_node_ids,
                                                              node_interact_times=node_interact_times,
                                                              edge_ids=edge_ids,
                                                              edges_are_positive=edges_are_positive,
                                                              num_neighbors=num_neighbors)

    elif model_name in ['GraphMixer']:
        # get temporal embedding of source and destination nodes
        # two Tensors, with shape (batch_size, node_feat_dim)
        batch_src_node_embeddings, batch_dst_node_embeddings = \
            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                              dst_node_ids=dst_node_ids,
                                                              node_interact_times=node_interact_times,
                                                              num_neighbors=num_neighbors,
                                                              time_gap=time_gap)

    elif model_name in ['DyGFormer']:
        # get temporal embedding of source and destination nodes
        # two Tensors, with shape (batch_size, node_feat_dim)
        batch_src_node_embeddings, batch_dst_node_embeddings = \
            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                              dst_node_ids=dst_node_ids,
                                                              node_interact_times=node_interact_times)

    else:
        raise ValueError(f"Wrong value for model_name {model_name}!")
        batch_src_node_embeddings, batch_dst_node_embeddings = None, None

    return batch_src_node_embeddings, batch_dst_node_embeddings


def get_probabilities_by_time(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_data: Data,
                              src: int, dst: int, spacing: int = 1000, num_neighbors: int = 20, time_gap: int = 2000,
                              batch: int = 512):
    """Returns (times, predicted probabilities, event times)."""
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)
    model.eval()

    t_min = evaluate_data.node_interact_times.min().item()
    t_max = evaluate_data.node_interact_times.max().item()
    # r = t_max - t_min
    # t_min -= r * 0.2
    # t_max += r * 0.2
    # times = np.linspace(t_min, t_max, num=count)
    times = np.arange(t_min, t_max, step=spacing)
    count = len(times)
    src_ids = np.array([src] * count)
    dst_ids = np.array([dst] * count)

    with torch.no_grad():
        all_pred_probs = []

        # Batch across negative samples
        for start in tqdm(range(0, count, batch)):
            end = min(start + batch, count)

            # 1) Obtain probabilities for linspace times
            src_embeds, dst_embeds = \
                query_pred_edge_batch(model_name=model_name, model=model, src_node_ids=src_ids[start:end], dst_node_ids=dst_ids[start:end],
                                      node_interact_times=times[start:end], edge_ids=None, edges_are_positive=False,
                                      num_neighbors=num_neighbors, time_gap=time_gap)

            predicted_probs = model[1](input_1=src_embeds, input_2=dst_embeds).squeeze(dim=-1).sigmoid()
            all_pred_probs.append(predicted_probs.detach().cpu().numpy())

        predicted_probs = np.concatenate(all_pred_probs, axis=0)

        # 2) Obtain probabilities for actual times
        inds = np.logical_and(evaluate_data.src_node_ids == src, evaluate_data.dst_node_ids == dst)

        real_src_ids, real_dst_ids, real_times, real_edge_ids = \
            evaluate_data.src_node_ids[inds], evaluate_data.dst_node_ids[inds], \
                evaluate_data.node_interact_times[inds], evaluate_data.edge_ids[inds]
        src_embeds, dst_embeds = \
            query_pred_edge_batch(model_name=model_name, model=model, src_node_ids=real_src_ids,
                                  dst_node_ids=real_dst_ids,
                                  node_interact_times=real_times, edge_ids=real_edge_ids, edges_are_positive=True,
                                  num_neighbors=num_neighbors, time_gap=time_gap)
        real_predicted_probs = model[1](input_1=src_embeds, input_2=dst_embeds).squeeze(dim=-1).sigmoid()

    # Combine fake and real times, allowing plotting as a single graph
    all_times = np.concatenate((times, real_times))
    all_preds = np.concatenate((predicted_probs, real_predicted_probs.detach().cpu().numpy()))

    sort = np.argsort(all_times)

    return all_times[sort], all_preds[sort], real_times


def main(save_all=False, run=0, neg_spacing=1):
    # get arguments
    args = get_link_prediction_args(is_evaluation=False)

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, dataset = \
        get_link_pred_data_TRANS_TGB(dataset_name=args.dataset_name)

    # initialize training neighbor sampler to retrieve temporal graph
    # train_neighbor_sampler = get_neighbor_sampler(data=train_data,
    #                                               sample_neighbor_strategy=args.sample_neighbor_strategy,
    #                                               time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # get data loaders
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))),
                                               batch_size=args.batch_size, shuffle=False)

    # Evaluatign with an evaluator of TGB
    evaluator = Evaluator(name=args.dataset_name)
    dataset.load_test_ns()
    negative_sampler = dataset.negative_sampler

    set_random_seed(seed=args.seed + run)
    args.save_model_name = f'{args.model_name}_{args.dataset_name}_seed_{args.seed}_run_{run}'

    # create model
    if args.model_name == 'TGAT':
        dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                neighbor_sampler=full_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                                num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                device=args.device)
    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
        # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
        src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
            compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids,
                                             train_data.node_interact_times)
        dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                       neighbor_sampler=full_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                                       model_name=args.model_name, num_layers=args.num_layers,
                                       num_heads=args.num_heads,
                                       dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift,
                                       src_node_std_time_shift=src_node_std_time_shift,
                                       dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
                                       dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
    elif args.model_name == 'CAWN':
        dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                neighbor_sampler=full_neighbor_sampler,
                                time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim,
                                walk_length=args.walk_length,
                                num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
    elif args.model_name == 'TCL':
        dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                               neighbor_sampler=full_neighbor_sampler,
                               time_feat_dim=args.time_feat_dim, num_layers=args.num_layers,
                               num_heads=args.num_heads,
                               num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
    elif args.model_name == 'GraphMixer':
        dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                      neighbor_sampler=full_neighbor_sampler,
                                      time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors,
                                      num_layers=args.num_layers, dropout=args.dropout, device=args.device)
    elif args.model_name == 'DyGFormer':
        dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                     neighbor_sampler=full_neighbor_sampler,
                                     time_feat_dim=args.time_feat_dim,
                                     channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                     num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                     max_input_sequence_length=args.max_input_sequence_length, device=args.device)
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")

    link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                hidden_dim=node_raw_features.shape[1], output_dim=1)
    model = nn.Sequential(dynamic_backbone, link_predictor)

    model = convert_to_gpu(model, device=args.device)

    save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
    early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                   save_model_name=args.save_model_name, logger=logging.getLogger(), model_name=args.model_name)
    early_stopping.load_checkpoint(model)
    model.eval()

    if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        model[0].set_neighbor_sampler(full_neighbor_sampler)
    if args.model_name in ['JODIE', 'DyRep', 'TGN']:
        # reinitialize memory of memory-based models at the start of each epoch
        model[0].memory_bank.__init_memory_bank__()

    test_pairs = list(zip(test_data.src_node_ids.tolist(), test_data.dst_node_ids.tolist()))
    counts = {i: test_pairs.count(i) for i in set(test_pairs)}
    biggest = sorted(set(test_pairs), key=lambda i:counts[i], reverse=True)

    if save_all:
        D = {}
        pairs = tqdm(biggest)
    else:
        pairs = biggest[100:500:50]
        print(biggest[:500:50])

    for src, dst in pairs:
        count = counts[(src, dst)]

        # For some reason this one produces an error
        if src == 2400 and dst == 8723:
            continue

        if count == 1 and save_all:
            break

        times, probs, _ = \
            get_probabilities_by_time(model_name=args.model_name, model=model, neighbor_sampler=full_neighbor_sampler,
                                      evaluate_data=test_data, src=src, dst=dst, spacing=neg_spacing, num_neighbors=args.num_neighbors,
                                      time_gap=args.time_gap)

        hop0, hop1, hop2 = get_temporal_edge_times(dataset, src-1, dst-1, 2, mask=dataset.test_mask)

        # Sanity check
        # A, B, C = set(hop0.tolist()), set(hop1.tolist()), set(hop2.tolist())
        # print(len(A),len(B),len(C))
        # print(len(A&B),len(A&C),len(B&C))

        plt.rcParams["figure.figsize"] = (8, 4)

        if save_all:
            D[(src - 1, dst - 1)] = (times.astype(np.float32), probs, etimes.astype(np.float32))
        else:
            plt.title(f"Edge {src}->{dst}, count {count}")
            plt.plot(times, probs)
            plt.xlabel("Time")
            plt.ylabel("Predicted link probability")
            plt.ylim(-0.01, 1.01)

            for etime in hop0:
                plt.axvline(x=etime, color="C1", ls="--", linewidth=2, alpha=0.8)
            for etime in hop1:
                plt.axvline(x=etime, color="C2", ls="--", linewidth=2, alpha=0.8)
            for etime in hop2:
                plt.axvline(x=etime, color="C3", ls="--", linewidth=2, alpha=0.8)
            plt.show()

    if save_all:
        np.save(f"saved_models/graphmixer_preds_{run}", D)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    for run in range(4):
        main(save_all=False, neg_spacing=1, run=run)
