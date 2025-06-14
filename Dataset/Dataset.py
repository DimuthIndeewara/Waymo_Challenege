# Necessary imports
import os
import pickle as pkl
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from google.protobuf import text_format
from os.path import isfile, join, listdir

# Import Waymo-specific modules
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
from waymo_open_dataset.utils.occupancy_flow_data import make_model_inputs

class WaymoOccupancyFlowDataset(Dataset):
    def __init__(self, data_dir) -> None:
        """
        Dataset class for Waymo Occupancy Flow data.
        
        Args:
            data_dir (str): Path to directory containing occupancy flow scene .pkl files.
        """
        super().__init__()

        self.data_dir = data_dir
        
        # Load list of scene files (only files, not directories)
        self.scenes = [f for f in listdir(self.data_dir) if isfile(join(self.data_dir, f))]

        # Load occupancy flow task config from text format protobuf
        self.config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
        config_path = '/data/horse/ws/thdi929f-Waymo/config.txt'  # Make this a parameter for better flexibility
        with open(config_path, 'r') as config_file:
            text_format.Parse(config_file.read(), self.config)

    def __len__(self):
        """
        Return the number of scenes in the dataset.
        """
        return len(self.scenes)

    def __getitem__(self, idx):
        """
        Load and process the data for a single scene.
        
        Args:
            idx (int): Index of the scene to load.
        
        Returns:
            dict: Dictionary containing:
                - grids: Input grid tensor (C, H, W)
                - waypoints: Ground truth waypoints dict
                - index: Index of the sample
                - scenario/id: ID of the scenario
        """
        scene = self.scenes[idx]
        
        # Load the .pkl file
        with open(join(self.data_dir, scene), 'rb') as inputs_pkl:
            inputs = pkl.load(inputs_pkl)

        # Extract scenario ID
        ID = inputs['scenario/id'].numpy()[0].decode("utf-8")

        # Add fields required for processing
        inputs = occupancy_flow_data.add_sdc_fields(inputs)

        # Create ground truth grids from the inputs
        timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(inputs=inputs, config=self.config)
        true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(timestep_grids=timestep_grids, config=self.config)
        vis_grids = occupancy_flow_grids.create_ground_truth_vis_grids(inputs=inputs, timestep_grids=timestep_grids, config=self.config)

        # Convert model inputs to numpy and prepare torch tensor
        model_inputs = make_model_inputs(timestep_grids, vis_grids).numpy()
        grid = torch.tensor(model_inputs[0])  # Shape: (H, W, C)
        grid = torch.permute(grid, (2, 0, 1))  # Convert to (C, H, W) for PyTorch

        # Process ground truth waypoints into PyTorch tensors
        waypoint = defaultdict(dict)
        waypoint['vehicles']['observed_occupancy'] = [
            torch.from_numpy(wp[0].numpy()) for wp in true_waypoints.vehicles.observed_occupancy
        ]
        waypoint['vehicles']['occluded_occupancy'] = [
            torch.from_numpy(wp[0].numpy()) for wp in true_waypoints.vehicles.occluded_occupancy
        ]
        waypoint['vehicles']['flow'] = [
            torch.from_numpy(wp[0].numpy()) for wp in true_waypoints.vehicles.flow
        ]
        waypoint['vehicles']['flow_origin_occupancy'] = [
            torch.from_numpy(wp[0].numpy()) for wp in true_waypoints.vehicles.flow_origin_occupancy
        ]

        # Pack everything into a dictionary and return
        sample = {
            'grids': grid,
            'waypoints': waypoint,
            'index': idx,
            'scenario/id': ID
        }

        return sample
