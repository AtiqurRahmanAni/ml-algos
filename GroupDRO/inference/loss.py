import os
import torch
from torch import nn
import torch.nn.functional as F
from utils import get_device


class LossComputer:
    def __init__(self,
                 criterion,
                 dataset):
        self.criterion = criterion

        self.device = get_device()

        self.n_groups = dataset.n_groups
        self.group_counts = dataset.group_counts().to(self.device)
        self.group_frac = self.group_counts/self.group_counts.sum()
        self.exp_avg_loss = torch.zeros(self.n_groups).to(self.device)

        self.reset_stats()

    def loss(self, yhat, y, group_idx=None):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(
            per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg(
            (torch.argmax(yhat, 1) == y).float(), group_idx)

        actual_loss = per_sample_losses.mean()

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count)

        return actual_loss


    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(
            self.n_groups).unsqueeze(1).long().to(self.device)).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()  # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count


    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).to(self.device)
        self.update_data_counts = torch.zeros(self.n_groups).to(self.device)
        self.update_batch_counts = torch.zeros(self.n_groups).to(self.device)
        self.avg_group_loss = torch.zeros(self.n_groups).to(self.device)
        self.avg_group_acc = torch.zeros(self.n_groups).to(self.device)
        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.

    def update_stats(self, actual_loss, group_loss, group_acc, group_count):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom
        self.avg_group_loss = prev_weight * self.avg_group_loss + curr_weight * group_loss

        # avg group acc
        self.avg_group_acc = prev_weight * self.avg_group_acc + curr_weight * group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count / denom) * \
            self.avg_actual_loss + (1 / denom) * actual_loss

        # counts
        self.processed_data_counts += group_count
        self.update_data_counts += group_count
        self.update_batch_counts += (group_count > 0).float()
        self.batch_count += 1

        # avg per-sample quantities
        group_frac = self.processed_data_counts / self.processed_data_counts.sum()
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc


    def get_stats(self):
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f'avg_loss_group:{idx}'] = self.avg_group_loss[idx].item(
            )
            stats_dict[f'exp_avg_loss_group:{idx}'] = self.exp_avg_loss[idx].item(
            )
            stats_dict[f'avg_acc_group:{idx}'] = self.avg_group_acc[idx].item()
            stats_dict[f'processed_data_count_group:{idx}'] = self.processed_data_counts[idx].item(
            )
            stats_dict[f'update_data_count_group:{idx}'] = self.update_data_counts[idx].item(
            )
            stats_dict[f'update_batch_count_group:{idx}'] = self.update_batch_counts[idx].item(
            )

        stats_dict['avg_actual_loss'] = self.avg_actual_loss.item()
        stats_dict['avg_per_sample_loss'] = self.avg_per_sample_loss.item()
        stats_dict['avg_acc'] = self.avg_acc.item()

        return stats_dict

    def log_stats(self, logger):
        logger.write(
            f'Average incurred loss: {self.avg_per_sample_loss.item():.3f}  \n')
        logger.write(
            f'Average sample loss: {self.avg_actual_loss.item():.3f}  \n')
        logger.write(f'Average acc: {self.avg_acc.item():.3f}  \n')
        for group_idx in range(self.n_groups):
            # f'  {self.group_str(group_idx)}  '
            logger.write(
                f'[n = {int(self.processed_data_counts[group_idx])}]:\t'
                f'loss = {self.avg_group_loss[group_idx]:.3f}  '
                f'acc = {self.avg_group_acc[group_idx]:.3f}\n')
        logger.flush()
