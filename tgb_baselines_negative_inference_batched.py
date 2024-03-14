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

from plot_utils import (get_temporal_edge_times, calculate_average_step_difference,
                        calculate_average_step_difference_full_range, total_variation_per_unit_time)
from plot_link_predicted_prob import query_pred_edge_batch


def get_probabilities_by_time_batched(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, dataset,
                                      evaluate_data: Data, srcs: np.array, dsts: np.array, time_step: int = 10,
                                      num_neighbors: int = 20, time_gap: int = 2000, batch: int = 512):
    """Returns (times, predicted probabilities, event times)."""
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)
    model.eval()

    t_min = evaluate_data.node_interact_times.min().item()
    t_max = evaluate_data.node_interact_times.max().item()
    times = np.arange(t_min, t_max, step=time_step)
    time_count = times.shape[0]

    metric = {}

    metric["TV"] = []
    metric["TV/s"] = []

    # for i in range(4):
    #     metric[f"TV-{i}"] = []
    #     metric[f"TV/s-{i}"] = []

    with torch.no_grad():

        # Batch across negative samples
        for edge_idx in tqdm(range(srcs.shape[0])):
            all_pred_probs = []

            src, dst = srcs[edge_idx], dsts[edge_idx]
            src_ids = np.array([src] * time_count)
            dst_ids = np.array([dst] * time_count)

            # Iterate over all times
            for start in range(0, time_count, batch):
                end = min(start + batch, time_count)

                src_embeds, dst_embeds = \
                    query_pred_edge_batch(model_name=model_name, model=model, src_node_ids=src_ids[start:end], dst_node_ids=dst_ids[start:end],
                                          node_interact_times=times[start:end], edge_ids=None, edges_are_positive=False,
                                          num_neighbors=num_neighbors, time_gap=time_gap)

                predicted_probs = model[1](input_1=src_embeds, input_2=dst_embeds).squeeze(dim=-1).sigmoid()
                all_pred_probs.append(predicted_probs.detach().cpu().numpy())

            predicted_probs = np.concatenate(all_pred_probs, axis=0)

            # plt.plot(times, predicted_probs)
            # plt.show()

            totvar, totvar_per_sec = total_variation_per_unit_time([], predicted_probs, times)
            metric["TV"].append(totvar)
            metric["TV/s"].append(totvar_per_sec)

            # Calculate metric
            # hops = get_temporal_edge_times(dataset, src-1, dst-1, num_hops=2, mask=dataset.test_mask)  # hop0,1,2
            # for hop_threshold in range(4):
            #     totvar, totvar_per_sec = total_variation_per_unit_time(hops[:hop_threshold], predicted_probs, times)
            #     # print(f"TotalVar-{hop_threshold} = {totvar}")
            #     # print(f"TotalVar/s-{hop_threshold} = {totvar_per_sec}")
            #
            #     metric[f"TV-{hop_threshold}"].append(totvar)
            #     metric[f"TV/s-{hop_threshold}"].append(totvar_per_sec)

    for k in list(metric):
        metric[k] = sum(metric[k]) / len(metric[k])

    return metric


MASK = None


def main(time_encoder, run=0):
    global MASK
    set_random_seed(seed=args.seed + run)
    args.time_encoder = time_encoder

    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, dataset = \
        get_link_pred_data_TRANS_TGB(dataset_name=args.dataset_name)

    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    dataset.load_test_ns()

    args.save_model_name = f"{args.model_name}_{args.dataset_name}_seed_{args.seed}_run_{run}_{args.time_encoder}_{args.mul}"

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
                                      num_layers=args.num_layers, dropout=args.dropout, device=args.device,
                                      time_encoder=args.time_encoder, time_multiplier=args.mul)
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

    srcs = test_data.src_node_ids
    dsts = test_data.dst_node_ids
    edges = np.concatenate((np.expand_dims(srcs, -1), np.expand_dims(dsts, -1)), axis=1)  # N x 2
    unique_edges = np.unique(edges, axis=0)
    srcs = unique_edges[:, 0]
    dsts = unique_edges[:, 1]

    if MASK is None:
        count = int(FILTER_PROPORTION * srcs.shape[0])
        MASK = np.random.permutation(srcs.shape[0])[:count]
        print(f"Generated mask, filtering from {srcs.shape[0]} -> {count} unique edges.")
        print(MASK[:10])  # sanity check for determinacy

    srcs = srcs[MASK]
    dsts = dsts[MASK]

    metric = get_probabilities_by_time_batched(model_name=args.model_name, model=model, dataset=dataset,
                                               neighbor_sampler=full_neighbor_sampler, evaluate_data=test_data,
                                               srcs=srcs, dsts=dsts, time_step=TIME_STEP,
                                               num_neighbors=args.num_neighbors, time_gap=args.time_gap, batch=512)
    return metric


if __name__ == "__main__":
    TIME_STEP = 50

    args = get_link_prediction_args(is_evaluation=False)

    # proportion of unique edges to keep for evaluation (chosen randomly, but constant between seeds)
    FILTER_PROPORTION = 0.2

    warnings.filterwarnings('ignore')

    save_data = {}
    save_path = f"GraphMixer_metric_{args.dataset_name}"
    print("Save path:", save_path)

    print(f"Calculating metric for GraphMixer on dataset {args.dataset_name}")

    for time_encoder in ["graph_mixer", "learned_cos", "scaled_fixed_id", "fixed_gaussian", "decay_amp_gm"]:
        print(f"Time encoder: {time_encoder}")
        ms = []
        for i in range(3):
            ms.append(main(run=i, time_encoder=time_encoder))

        print(f"Results for time encoder {time_encoder}")

        combined = {}
        for key in sorted(ms[0]):
            data = np.array([m[key] for m in ms])
            combined[key] = (data.sum(), data.std(), data)
            print(f"{key}: {data.sum()} +- {data.std()}")

        save_data[time_encoder] = combined

        # Save after each time encoder in case program terminates before completion
        np.save(save_path, save_data)
        print(f"Saved to {save_path}")
        print()
