# ------------------------- Important Packages ---------------------------------------
import os
import time
import math
import optuna
import pickle as pkl
import sqlite3
import multiprocessing
import tensorflow_graphics.image.transformer as tfg_transformer
from os import listdir
from os.path import isfile, join

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchmetrics import MeanMetric
from torchmetrics.functional.classification import binary_average_precision

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from tqdm import tqdm
from joblib import Parallel, delayed

from collections import defaultdict

from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.utils import occupancy_flow_data, occupancy_flow_grids


# ---------------------------- Model Configuration ------------------------------------

# coding=utf-8
"""Hyperparameters of the structured occupancy prediction models."""

class ConfigDict(dict):
  """A dictionary whose keys can be accessed as attributes."""

  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)

  def __setattr__(self, name, value):
    self[name] = value

  def get(self, key, default=None):
    """Allows to specify defaults when accessing the config."""
    if key not in self:
      return default
    return self[key]

def get_config():

    """Default values for all hyperparameters."""
    cfg = ConfigDict()

    # Directories:
    cfg.DATASET_FOLDER = '/data/horse/ws/thdi929f-Waymo'
    # TFRecord dataset.
    cfg.TRAIN_FILES    = f'{cfg.DATASET_FOLDER}/tf_example/training/training_tfexample.tfrecord*'
    cfg.VAL_FILES      = f'{cfg.DATASET_FOLDER}/tf_example/validation/validation_tfexample.tfrecord*'
    cfg.TEST_FILES     = f'{cfg.DATASET_FOLDER}/tf_example/testing/testing_tfexample.tfrecord*'
    #cfg.SAMPLE_FILES   = f'{cfg.DATASET_FOLDER}/tf_example/sample/training_tfexample.tfrecord*'

    cfg.TRAIN_PKL_FOLDER = f'{cfg.DATASET_FOLDER}/pkl_example/training/'
    cfg.VAL_PKL_FOLDER  = f'{cfg.DATASET_FOLDER}/pkl_example/validation/'
    cfg.TEST_PKL_FOLDER  = f'{cfg.DATASET_FOLDER}/pkl_example/testing/'
    # Text files containing validation and test scenario IDs for this challenge.
    cfg.VAL_SCENARIO_IDS_FILE  = f'{cfg.DATASET_FOLDER}/occupancy_flow_challenge/validation_scenario_ids.txt'
    cfg.TEST_SCENARIO_IDS_FILE = f'{cfg.DATASET_FOLDER}/occupancy_flow_challenge/testing_scenario_ids.txt'

    # Architecture:
    cfg.INPUT_SIZE                   = 23
    cfg.NUM_CLASSES                  = 32
            
    # Optimization:            
    cfg.OPTIMIZER                    = 'adamw'
    cfg.SCHEDULER                    = 'CosineAnnealingLR' 
    cfg.TRAIN_BATCH_SIZE             = 8
    cfg.VAL_BATCH_SIZE               = 1
    cfg.WEIGHT_DECAY                 = 0.007
    cfg.EPOCHS                       = 5
    cfg.LR                           = 0.0005
    cfg.MOMENTUM                     = 0.8

    # Grid sequence parameters:
    cfg.NUM_PRED_CHANNELS            = 4
    cfg.num_past_steps               = 10
    cfg.num_future_steps             = 80
    cfg.NUM_WAYPOINTS                = 8
    cfg.cumulative_waypoints         = False
    cfg.normalize_sdc_yaw            = True
    cfg.grid_height_cells            = 256
    cfg.grid_width_cells             = 256
    cfg.sdc_y_in_grid                = 192
    cfg.sdc_x_in_grid                = 128
    cfg.pixels_per_meter             = 3.2
    cfg.agent_points_per_side_length = 48
    cfg.agent_points_per_side_width  = 16

    # Train configs
    cfg.WANDB_MODE                   = "online"  # {'run', 'online', 'offline', 'dryrun', 'disabled'}

    return cfg

# ------------------------------  Loss Function  ---------------------------------------------

cfg = get_config()

cfg.NUM_PRED_CHANNELS            = 4
cfg.NUM_WAYPOINTS                = 8

def Occupancy_Flow_Loss(true_waypoints, pred_waypoint_logits):
    """Loss function.

    Args:
      config: OccupancyFlowTaskConfig proto message.
      true_waypoints: Ground truth labels.
      pred_waypoint_logits: Predicted occupancy logits and flows.

    Returns:
      A dict containing different loss tensors:
        observed_xe: Observed occupancy cross-entropy loss.
        occluded_xe: Occluded occupancy cross-entropy loss.
        flow: Flow loss.
    """
    loss_dict = {}
    # Store loss tensors for each waypoint and average at the end.
    loss_dict['observed_xe'] = []
    loss_dict['occluded_xe'] = []
    loss_dict['flow'] = []

    # Iterate over waypoints.
    for k in range(cfg.NUM_WAYPOINTS):
        # Occupancy cross-entropy loss.
        pred_observed_occupancy_logit = (pred_waypoint_logits['vehicles']['observed_occupancy'][k])
        pred_occluded_occupancy_logit = (pred_waypoint_logits['vehicles']['occluded_occupancy'][k])
        true_observed_occupancy = true_waypoints['vehicles']['observed_occupancy'][k]
        true_occluded_occupancy = true_waypoints['vehicles']['occluded_occupancy'][k]

        # Accumulate over waypoints.
        loss_dict['observed_xe'].append(_sigmoid_xe_loss(true_occupancy=true_observed_occupancy, pred_occupancy=pred_observed_occupancy_logit))
        loss_dict['occluded_xe'].append(_sigmoid_xe_loss(true_occupancy=true_occluded_occupancy, pred_occupancy=pred_occluded_occupancy_logit))

        # Flow loss.
        pred_flow = pred_waypoint_logits['vehicles']['flow'][k]
        true_flow = true_waypoints['vehicles']['flow'][k]
        loss_dict['flow'].append((k + 1) * _flow_loss(pred_flow, true_flow))

    # Mean over waypoints.
    loss_dict['observed_xe'] = (sum(loss_dict['observed_xe']) / cfg.NUM_WAYPOINTS)
    loss_dict['occluded_xe'] = (sum(loss_dict['occluded_xe']) / cfg.NUM_WAYPOINTS)
    loss_dict['flow']        = sum(loss_dict['flow']) / cfg.NUM_WAYPOINTS

    return loss_dict


def _sigmoid_xe_loss(true_occupancy, pred_occupancy, loss_weight: float = 1000):
    """Computes sigmoid cross-entropy loss over all grid cells."""
    # Since the mean over per-pixel cross-entropy values can get very small,
    # we compute the sum and multiply it by the loss weight before computing
    # the mean.
    xe_sum = F.binary_cross_entropy_with_logits(input=torch.flatten(pred_occupancy), target=torch.flatten(true_occupancy), reduction='mean')
    # Return mean.
    return xe_sum #loss_weight * xe_sum / list(torch.flatten(pred_occupancy).size())[0] # torch.shape(pred_occupancy, out_type=torch.float32)   


def _flow_loss(true_flow, pred_flow, loss_weight: float = 2):
    """Computes L1 flow loss."""
    diff = true_flow - pred_flow
    # Ignore predictions in areas where ground-truth flow is zero.
    # [batch_size, height, width, 1], [batch_size, height, width, 1]
    (true_flow_dx, true_flow_dy) = torch.split(true_flow, true_flow.size(-1) // 2, dim=-1)
    # [batch_size, height, width, 1]
    flow_exists = torch.logical_or(torch.not_equal(true_flow_dx, 0.0), torch.not_equal(true_flow_dy, 0.0))
    flow_exists = flow_exists.type(torch.float32)
    diff = diff * flow_exists
    diff_norm = torch.linalg.norm(diff, ord=1, axis=-1)  # L1 norm.
    mean_diff = torch.div(torch.sum(diff_norm), torch.sum(flow_exists) / 2)  # / 2 since (dx, dy) is counted twice.
    return loss_weight * mean_diff

# ----------------------------------- Hyperparameter  -----------------------------------------


def make_model_inputs(
    timestep_grids: occupancy_flow_grids.TimestepGrids,
    vis_grids: occupancy_flow_grids.VisGrids,
    ) -> tf.Tensor:

    """Concatenates all occupancy grids over past, current to a single tensor."""

    model_inputs = tf.concat(
        [
            vis_grids.roadgraph,
            timestep_grids.vehicles.past_occupancy,
            timestep_grids.vehicles.current_occupancy,
            tf.clip_by_value(
                timestep_grids.pedestrians.past_occupancy +
                timestep_grids.cyclists.past_occupancy, 0, 1),
            tf.clip_by_value(
                timestep_grids.pedestrians.current_occupancy +
                timestep_grids.cyclists.current_occupancy, 0, 1),
        ],
        axis=-1,
    )
    return model_inputs

def get_pred_waypoint_logits(model_outputs):
    
    """Slices model predictions into occupancy and flow grids."""

    pred_waypoint_logits = defaultdict(dict)
    model_outputs = torch.permute(model_outputs, (0, 2, 3, 1))  
    pred_waypoint_logits['vehicles']['observed_occupancy'] = []
    pred_waypoint_logits['vehicles']['occluded_occupancy'] = []
    pred_waypoint_logits['vehicles']['flow'] = []

    # Slice channels into output predictions.
    for k in range(cfg.NUM_WAYPOINTS):
        index = k * cfg.NUM_PRED_CHANNELS
        waypoint_channels = model_outputs[:, :, :, index:index + cfg.NUM_PRED_CHANNELS]
        pred_observed_occupancy = waypoint_channels[:, :, :, :1]
        pred_occluded_occupancy = waypoint_channels[:, :, :, 1:2]
        pred_flow = waypoint_channels[:, :, :, 2:]
        pred_waypoint_logits['vehicles']['observed_occupancy'].append(pred_observed_occupancy)
        pred_waypoint_logits['vehicles']['occluded_occupancy'].append(pred_occluded_occupancy)
        pred_waypoint_logits['vehicles']['flow'].append(pred_flow)
    return pred_waypoint_logits

# -------------------------------------- Dataset Loader ---------------------------------------


class WaymoOccupancyFlowDataset(Dataset):
    def __init__(self, data_dir) -> None:
        super().__init__()
        
        self.data_dir = data_dir
        self.scenes = [f for f in listdir(self.data_dir) if isfile(join(self.data_dir, f))]
        self.config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
        text_format.Parse(open('/data/horse/ws/thdi929f-Waymo/config.txt').read(), self.config)

    def __len__(self):
        return len(self.scenes)   

    def __getitem__(self, idx):

        scene = self.scenes[idx]
        inputs_pkl = open(self.data_dir + scene, 'rb')
        inputs = pkl.load(inputs_pkl)

        ID = inputs['scenario/id'].numpy()[0].decode("utf-8")
        inputs = occupancy_flow_data.add_sdc_fields(inputs)

        timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(inputs=inputs, config=self.config)
        true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(timestep_grids=timestep_grids, config=self.config)
        vis_grids      = occupancy_flow_grids.create_ground_truth_vis_grids(inputs=inputs, timestep_grids=timestep_grids, config=self.config)

        # Agent_trails = vis_grids.agent_trails.numpy()
        # 'agent_trails': torch.from_numpy(Agent_trails[0])
        
        model_inputs = make_model_inputs(timestep_grids, vis_grids).numpy()

        grid = torch.tensor(model_inputs[0])
        grid = torch.permute(grid, (2, 0, 1))

        waypoint = defaultdict(dict)
        waypoint['vehicles']['observed_occupancy']    = [torch.from_numpy(wp[0].numpy()) for wp in true_waypoints.vehicles.observed_occupancy]    # (256, 256, 1) * 8
        waypoint['vehicles']['occluded_occupancy']    = [torch.from_numpy(wp[0].numpy()) for wp in true_waypoints.vehicles.occluded_occupancy]    # (256, 256, 1) * 8
        waypoint['vehicles']['flow']                  = [torch.from_numpy(wp[0].numpy()) for wp in true_waypoints.vehicles.flow]                  # (256, 256, 2) * 8
        waypoint['vehicles']['flow_origin_occupancy']                  = [torch.from_numpy(wp[0].numpy()) for wp in true_waypoints.vehicles.flow_origin_occupancy]                  # (256, 256, 2) * 8


        sample = {'grids': grid, 'waypoints': waypoint, 'index': idx ,'scenario/id': ID }

        return sample
    
# ----------------------------------------- Metrics ------------------------------------------

"""Occupancy and flow metrics."""


def compute_occupancy_flow_metrics(config, true_waypoints, pred_waypoints):
    """Computes occupancy (observed, occluded) and flow metrics.

    Args:
        config: OccupancyFlowTaskConfig proto message.
        true_waypoints: Set of num_waypoints ground truth labels.
        pred_waypoints: Predicted set of num_waypoints occupancy and flow topdowns.

    Returns:
        OccupancyFlowMetrics proto message containing mean metric values averaged
        over all waypoints.
    """
    # Accumulate metric values for each waypoint and then compute the mean.
    metrics_dict = {
        'vehicles_observed_auc': [],
        'vehicles_occluded_auc': [],
        'vehicles_observed_iou': [],
        'vehicles_occluded_iou': [],
        'vehicles_flow_epe': [],
        'vehicles_flow_warped_occupancy_auc': [],
        'vehicles_flow_warped_occupancy_iou': [],
    }

    # Warp flow-origin occupancies according to predicted flow fields.
    warped_flow_origins = _flow_warp(config=config, true_waypoints=true_waypoints, pred_waypoints=pred_waypoints)

    # Iterate over waypoints.
    for k in range(config.NUM_WAYPOINTS):
        true_observed_occupancy = true_waypoints['vehicles']['observed_occupancy'][k]
        pred_observed_occupancy = pred_waypoints['vehicles']['observed_occupancy'][k]
        true_occluded_occupancy = true_waypoints['vehicles']['occluded_occupancy'][k]
        pred_occluded_occupancy = pred_waypoints['vehicles']['occluded_occupancy'][k]
        true_flow = true_waypoints['vehicles']['flow'][k]
        pred_flow = pred_waypoints['vehicles']['flow'][k]

        # Compute occupancy metrics.
        metrics_dict['vehicles_observed_auc'].append(_compute_occupancy_auc(true_observed_occupancy, pred_observed_occupancy))
        metrics_dict['vehicles_occluded_auc'].append(_compute_occupancy_auc(true_occluded_occupancy, pred_occluded_occupancy))
        metrics_dict['vehicles_observed_iou'].append(_compute_occupancy_soft_iou(true_observed_occupancy, pred_observed_occupancy))
        metrics_dict['vehicles_occluded_iou'].append(_compute_occupancy_soft_iou(true_occluded_occupancy, pred_occluded_occupancy))

        # Compute flow metrics.
        metrics_dict['vehicles_flow_epe'].append(_compute_flow_epe(true_flow, pred_flow))

        # Compute flow-warped occupancy metrics.
        # First, construct ground-truth occupancy of all observed and occluded
        # vehicles.
        true_all_occupancy = torch.clamp(true_observed_occupancy + true_occluded_occupancy, 0, 1)
        # # Construct predicted version of same value.
        pred_all_occupancy = torch.clamp(pred_observed_occupancy + pred_occluded_occupancy, 0, 1)
        # # We expect to see the same results by warping the flow-origin occupancies.
        #flow_warped_origin_occupancy = warped_flow_origins[k]

        flow_warped_origin_occupancy = torch.from_numpy(
            warped_flow_origins[k].numpy()
        ).to(pred_all_occupancy.device)
        
        # # Construct quantity that requires both prediction paths to be correct.
        flow_grounded_pred_all_occupancy = (pred_all_occupancy * flow_warped_origin_occupancy)
        # # Now compute occupancy metrics between this quantity and ground-truth.
        metrics_dict['vehicles_flow_warped_occupancy_auc'].append(_compute_occupancy_auc(flow_grounded_pred_all_occupancy, true_all_occupancy))
        metrics_dict['vehicles_flow_warped_occupancy_iou'].append(_compute_occupancy_soft_iou(flow_grounded_pred_all_occupancy, true_all_occupancy))


    # Compute means and return as proto message.
    metrics = {} #occupancy_flow_metrics_pb2.OccupancyFlowMetrics()
    metrics['vehicles_observed_auc'] = _mean(metrics_dict['vehicles_observed_auc'])
    metrics['vehicles_occluded_auc'] = _mean(metrics_dict['vehicles_occluded_auc'])
    metrics['vehicles_observed_iou'] = _mean(metrics_dict['vehicles_observed_iou'])
    metrics['vehicles_occluded_iou'] = _mean(metrics_dict['vehicles_occluded_iou'])
    metrics['vehicles_flow_epe'] = _mean(metrics_dict['vehicles_flow_epe'])
    metrics['vehicles_flow_warped_occupancy_auc'] = _mean(metrics_dict['vehicles_flow_warped_occupancy_auc'])
    metrics['vehicles_flow_warped_occupancy_iou'] = _mean(metrics_dict['vehicles_flow_warped_occupancy_iou'])
    return metrics


def _mean(tensor_list):
    """
    Compute mean value from a list of scalar tensors.
    """
    num_tensors = len(tensor_list)
    sum_tensors = sum(tensor_list).detach().cpu().numpy()
    return sum_tensors / num_tensors


def _compute_occupancy_auc(
    true_occupancy: torch.Tensor,
    pred_occupancy: torch.Tensor,
) -> torch.Tensor:
  """Computes the AUC between the predicted and true occupancy grids.

  Args:
    true_occupancy: float32 [batch_size, height, width, 1] tensor in [0, 1].
    pred_occupancy: float32 [batch_size, height, width, 1] tensor in [0, 1].

  Returns:
    AUC: float32 scalar.
  """
  return binary_average_precision(preds=pred_occupancy, target=true_occupancy.to(torch.int8), thresholds=100)



def _compute_occupancy_soft_iou(true_occupancy, pred_occupancy):
    """Computes the soft IoU between the predicted and true occupancy grids.

    Args:
        true_occupancy: float32 [batch_size, height, width, 1] tensor in [0, 1].
        pred_occupancy: float32 [batch_size, height, width, 1] tensor in [0, 1].

    Returns:
        Soft IoU score: float32 scalar.
    """
    true_occupancy = torch.reshape(true_occupancy, [-1])
    pred_occupancy = torch.reshape(pred_occupancy, [-1])

    intersection = torch.mean(torch.multiply(pred_occupancy, true_occupancy))
    true_sum = torch.mean(true_occupancy)
    pred_sum = torch.mean(pred_occupancy)
    # Scenes with empty ground-truth will have a score of 0.
    score = torch.div(intersection, pred_sum + true_sum - intersection)
    return score


def _compute_flow_epe(true_flow, pred_flow):
    """Computes average end-point-error between predicted and true flow fields.

    Flow end-point-error measures the Euclidean distance between the predicted and
    ground-truth flow vector endpoints.

    Args:
        true_flow: float32 Tensor shaped [batch_size, height, width, 2].
        pred_flow: float32 Tensor shaped [batch_size, height, width, 2].

    Returns:
        EPE averaged over all grid cells: float32 scalar.
    """
    # [batch_size, height, width, 2]
    diff = true_flow - pred_flow
    # [batch_size, height, width, 1], [batch_size, height, width, 1]
    true_flow_dx, true_flow_dy = torch.split(true_flow, true_flow.size(-1) // 2, dim=-1)
    # [batch_size, height, width, 1]
    flow_exists = torch.logical_or(torch.not_equal(true_flow_dx, 0.0), torch.not_equal(true_flow_dy, 0.0))
    flow_exists = flow_exists.float() # tf.cast(flow_exists, tf.float32)

    # Check shapes.
    # assert true_flow_dx.shape == true_flow_dy.shape
    # tf.debugging.assert_shapes([
    #     (true_flow_dx, ['batch_size', 'height', 'width', 1]),
    #     (true_flow_dy, ['batch_size', 'height', 'width', 1]),
    #     (diff, ['batch_size', 'height', 'width', 2]),
    # ])

    diff = diff * flow_exists
    # [batch_size, height, width, 1]
    epe = torch.linalg.norm(diff, ord=2, axis=-1, keepdims=True)
    # Scalar.
    sum_epe = torch.sum(epe)
    # Scalar.
    sum_flow_exists = torch.sum(flow_exists)
    # Scalar.
    mean_epe = torch.div(sum_epe, sum_flow_exists)

    # tf.debugging.assert_shapes([
    #     (epe, ['batch_size', 'height', 'width', 1]),
    #     (sum_epe, []),
    #     (mean_epe, []),
    # ])

    return mean_epe

def _flow_warp(config, true_waypoints, pred_waypoints):
    """Warps ground-truth flow-origin occupancies according to predicted flows."""
    h = torch.arange(0, config.grid_height_cells, dtype=torch.float32)
    w = torch.arange(0, config.grid_width_cells, dtype=torch.float32)
    h_idx, w_idx = torch.meshgrid(h, w)
    
    # Create identity indices for warping
    identity_indices = torch.stack(
        (w_idx.T, h_idx.T), dim=-1
    )  # [height, width, 2]
    
    warped_flow_origins = []
    for k in range(config.NUM_WAYPOINTS):
        # Extract true flow-origin occupancy and predicted flow
        flow_origin_occupancy = true_waypoints['vehicles']['flow_origin_occupancy'][k]
        pred_flow = pred_waypoints['vehicles']['flow'][k]
        
        # Compute warped indices
        warped_indices = identity_indices + pred_flow
        warped_indices = warped_indices.detach().cpu().numpy()  # Detach and convert to NumPy
        
        # Pad flow_origin_occupancy to avoid artifacts at boundaries
        flow_origin_occupancy = torch.nn.functional.pad(
            flow_origin_occupancy, (0, 0, 1, 1, 1, 1)
        ).detach().cpu().numpy()  # Detach and convert to NumPy
        
        # Warp the flow origin using TensorFlow Graphics
        warped_origin = tfg_transformer.sample(
            image=flow_origin_occupancy,
            warp=warped_indices,
            pixel_type=tfg_transformer.PixelType.INTEGER,
        )
        warped_flow_origins.append(warped_origin)
    
    return warped_flow_origins


# ------------------------------------------------ Model --------------------------------------


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.shape[2] - x1.shape[2]])
        diffX = torch.tensor([x2.shape[3] - x1.shape[3]])

        x1 = F.pad(x1, [torch.div(diffX, 2, rounding_mode='floor'), diffX - torch.div(diffX, 2, rounding_mode='floor'),
                        torch.div(diffY, 2, rounding_mode='floor'), diffY - torch.div(diffY, 2, rounding_mode='floor')])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, with_head, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.with_head = with_head

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(768, 256, bilinear)
        self.up2 = Up(384, 128, bilinear)
        self.up3 = Up(192, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        if self.with_head:
            self.head_in_ch = 32
            self.observed_head = sepHead(ch_in=self.head_in_ch, ch_out=8)
            self.occluded_head = sepHead(ch_in=self.head_in_ch, ch_out=8)
            self.flow_dx_head = sepHead(ch_in=self.head_in_ch, ch_out=8)
            self.flow_dy_head = sepHead(ch_in=self.head_in_ch, ch_out=8)

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)

        if self.with_head:
            out1 = self.observed_head(logits)
            out2 = self.occluded_head(logits)
            out3 = self.flow_dx_head(logits)
            out4 = self.flow_dy_head(logits)

            logits = torch.cat([out1, out2, out3, out4], dim=1)

        return logits


class sepHead(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(sepHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


def main():
    model = UNet(n_channels=23, n_classes=32, with_head=True).to(device)

    for i in range(10):
        inputs = torch.rand((3, 23, 256, 256)).to(device)
        t = time.time()

        output = model(inputs)

        print("inf time =", time.time() - t)
    print(output.shape)


if __name__ == "__main__":
    main()

# ------------------------------ Training Loop --------------------------------------


class Args:
    save_dir = "/data/horse/ws/thdi929f-Waymo/Models/Train_UNet V1"
    file_dir = "/data/horse/ws/thdi929f-Waymo/pkl_Folder"
    model_path = None  # Set path to a checkpoint file if resuming
    batch_size = 16
    epochs = 15
    lr = 1e-03
    wandb = False
    title = "UNet V1 | B- 8 | LR - 1e-03 | pkl_Folder"


args = Args()

SAVE_DIR = args.save_dir
FILES_DIR = args.file_dir
CHECKPOINT_PATH = args.model_path
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr

print("___________________________")

print(args.title)

print("___________________________")
# loss weights
ogm_weight  = 1.0
occ_weight  = 1.0
flow_weight = 1.0
flow_origin_weight = 1000.0


def apply_sigmoid_to_occupancy_logits(pred_waypoint_logits):

    """Converts occupancy logits with probabilities."""
    _SIGMOID = torch.nn.Sigmoid()
    pred_waypoints =  defaultdict(dict)
    pred_waypoints['vehicles']['observed_occupancy'] = [_SIGMOID(x) for x in pred_waypoint_logits['vehicles']['observed_occupancy']]
    pred_waypoints['vehicles']['occluded_occupancy'] = [_SIGMOID(x) for x in pred_waypoint_logits['vehicles']['occluded_occupancy']]
    pred_waypoints['vehicles']['flow'] = pred_waypoint_logits['vehicles']['flow']
    return pred_waypoints

model = UNet(n_channels=23, n_classes=32, with_head=True).to(device)

optimizer = torch.optim.NAdam(model.parameters(), lr=LR) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.5)

def get_dataloader():
    """
    Get training and validation dataloaders
    """

    train_dataset = WaymoOccupancyFlowDataset(data_dir = FILES_DIR + '/training/')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
        #num_workers=os.cpu_count()
    )

    val_dataset = WaymoOccupancyFlowDataset(data_dir = FILES_DIR + '/validation/')

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
        #num_workers=os.cpu_count()
    
    )

    return train_loader,val_loader



def train_epoch(model, train_loader, optimizer, scheduler, epoch, device):
    model.train()
    train_loss = MeanMetric().to(device)
    train_loss_occ = MeanMetric().to(device)
    train_loss_flow = MeanMetric().to(device)

    loop = tqdm(enumerate(train_loader), total=math.ceil(len(train_loader.dataset) / BATCH_SIZE), desc=f"Epoch {epoch + 1} Training")
    for batch, data in loop:
        grids = data['grids']  
        true_waypoints = data['waypoints'] 

        # forward pass
        model_outputs = model(grids)
        
        # compute loss
        pred_waypoint_logits = get_pred_waypoint_logits(model_outputs)
        loss_dict = Occupancy_Flow_Loss(true_waypoints, pred_waypoint_logits)
        loss_value = torch.sum(sum(loss_dict.values()))

        # Backward Pass
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        # Update losses
        train_loss.update(loss_dict['observed_xe'])
        train_loss_occ.update(loss_dict['occluded_xe'])
        train_loss_flow.update(loss_dict['flow'])

    obs_loss = train_loss.compute() / ogm_weight
    occ_loss = train_loss_occ.compute() / occ_weight
    flow_loss = train_loss_flow.compute() / flow_weight

    print(f"Epoch {epoch + 1} Training Losses:")
    print(f"Observed: {obs_loss:.4f}, Occluded: {occ_loss:.4f}, Flow: {flow_loss:.4f}")

    scheduler.step()
    return model.state_dict()  # Return model state dict for saving/checkpointing

def validate_epoch(model, val_loader, optimizer, epoch, device):
    model.eval()
    valid_loss = MeanMetric().to(device)
    valid_loss_occ = MeanMetric().to(device)
    valid_loss_flow = MeanMetric().to(device)

    loop = tqdm(enumerate(val_loader), total=math.ceil(len(val_loader.dataset) / BATCH_SIZE), desc=f"Epoch {epoch + 1} Validation")
    for batch, data in loop:
        grids = data['grids']  
        true_waypoints = data['waypoints'] 

        # forward pass
        model_outputs = model(grids)
            
        # compute loss
        pred_waypoint_logits = get_pred_waypoint_logits(model_outputs)
        loss_dict = Occupancy_Flow_Loss(true_waypoints, pred_waypoint_logits)
        loss_value = torch.sum(sum(loss_dict.values()))

        # update losses
        valid_loss.update(loss_dict['observed_xe'])
        valid_loss_occ.update(loss_dict['occluded_xe'])
        valid_loss_flow.update(loss_dict['flow'])

        pred_waypoints = apply_sigmoid_to_occupancy_logits(pred_waypoint_logits)
        metrics = compute_occupancy_flow_metrics(config=cfg, true_waypoints=true_waypoints, pred_waypoints=pred_waypoints)

    obs_loss = valid_loss.compute() / ogm_weight
    occ_loss = valid_loss_occ.compute() / occ_weight
    flow_loss = valid_loss_flow.compute() / flow_weight

    print(f"Epoch {epoch + 1} Validation Losses:")
    print(f"Observed: {obs_loss:.4f}, Occluded: {occ_loss:.4f}, Flow: {flow_loss:.4f}")

    print(f"Epoch {epoch + 1} Metrics:")
    print(metrics)

    return model.state_dict(), loss_value  # Return model state dict for saving/checkpointing

def train_and_validate_epoch(epoch, model, optimizer, scheduler, train_loader, val_loader, device):
    # Train for one epoch
    model_state = train_epoch(model, train_loader, optimizer, scheduler, epoch, device)
    
    # Validate the model after the epoch
    model_state, val_loss = validate_epoch(model, val_loader, optimizer, epoch, device)
    
    return model_state, val_loss

def model_training_parallel():
    """
    Model training and validation using parallelization
    """
    model, optimizer, scheduler
    train_loader, val_loader = get_dataloader()

    # Use multiprocessing to parallelize the epochs
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(train_and_validate_epoch)(epoch, model, optimizer, scheduler, train_loader, val_loader, device) 
        for epoch in range(EPOCHS)
    )

    # Save checkpoints after training
    for epoch, (model_state, _) in enumerate(results):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'{SAVE_DIR}/epoch_{epoch + 1}.pt')

# Ensure the directory exists for saving
os.path.exists(SAVE_DIR) or os.makedirs(SAVE_DIR)

if __name__ == "__main__":
    model_training_parallel()
