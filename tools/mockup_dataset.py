import numpy as np
import torch
from torch.utils.data import Dataset
import random
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmcv.parallel import DataContainer as DC

import random
from mmcv.parallel import DataContainer as DC
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet.datasets.pipelines.formating import to_tensor
import copy

try:
    from projects.mmdet3d_plugin.datasets.nuscenes_vad_dataset import LiDARInstanceLines
except ImportError:
    # Define a simplified version if import fails
    class LiDARInstanceLines:
        def __init__(self, instance_list, *args, **kwargs):
            self.instance_list = instance_list if instance_list else []
            
        def __len__(self):
            return len(self.instance_list)


class MockupSSRDataset(Dataset):
    """A mockup dataset for SSR model testing without real data"""
    
    def __init__(self, num_samples=100, batch_size=1):
        self.num_samples = num_samples
        self.batch_size = batch_size
        
        # Define basic parameters matching the config
        self.bev_h = 100
        self.bev_w = 100
        self.queue_length = 1  # Set to 1 for testing to avoid batch size issues
        self.num_classes = 10  # nuScenes classes
        self.map_num_classes = 3  # map classes
        self.pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        
        # Class names
        self.class_names = [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
            'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]
        
        self.map_classes = ['divider', 'ped_crossing', 'boundary']
        
        # Image parameters
        self.img_h = 900
        self.img_w = 1600
        self.num_cameras = 6
        
    def __len__(self):
        return self.num_samples
    
    def _generate_random_images(self):
        """Generate random multi-view images"""
        imgs = []
        for _ in range(self.queue_length):
            queue_imgs = []
            for _ in range(self.num_cameras):
                # Generate random image (C, H, W)
                img = torch.randn(3, self.img_h, self.img_w, dtype=torch.float32)
                queue_imgs.append(img)
            imgs.append(torch.stack(queue_imgs))  # [num_cameras, C, H, W]
        return torch.stack(imgs)  # [queue_length, num_cameras, C, H, W]
    
    def _generate_random_boxes_3d(self):
        """Generate random 3D bounding boxes"""
        num_boxes = random.randint(10, 15)  # Ensure we have enough boxes
        
        # Generate random box centers within point cloud range
        centers_x = np.random.uniform(self.pc_range[0], self.pc_range[3], num_boxes)
        centers_y = np.random.uniform(self.pc_range[1], self.pc_range[4], num_boxes)
        centers_z = np.random.uniform(self.pc_range[2], self.pc_range[5], num_boxes)
        
        # Generate random dimensions (l, w, h)
        dims = np.random.uniform([1.0, 1.0, 1.0], [6.0, 3.0, 3.0], (num_boxes, 3))
        
        # Generate random yaw angles
        yaws = np.random.uniform(-np.pi, np.pi, num_boxes)
        
        # Generate random velocities
        velocities = np.random.uniform(-10.0, 10.0, (num_boxes, 2))
        
        # Combine into box tensor [x, y, z, l, w, h, yaw, vx, vy]
        boxes = np.column_stack([
            centers_x, centers_y, centers_z,
            dims[:, 0], dims[:, 1], dims[:, 2],  # l, w, h
            yaws,
            velocities[:, 0], velocities[:, 1]
        ])
        
        # Generate random labels
        labels = np.random.randint(0, self.num_classes, num_boxes)
        
        # Generate attribute labels (trajectory-related features)
        # Based on the VAD dataset and the planning metric expecting category at index 27:
        # The format should match the VAD annotation structure exactly
        T = 6
        
        # Create feature vector with category at the correct position (index 27)
        # Minimum size should be 28 to have index 27
        min_size = 28
        traj_dim = max(min_size, T*2 + T + 1 + 256 + T)  # Ensure we have at least 28 elements
        attr_labels = np.random.randn(num_boxes, traj_dim).astype(np.float32)
        
        # Set future masks to 1.0 (valid) for simplicity  
        if traj_dim > T*2+T:
            attr_labels[:, T*2:T*2+T] = 1.0  # fut_masks
        
        # Set category indices at index 27 (this is what the planning metric expects)
        for i in range(num_boxes):
            attr_labels[i, 27] = float(labels[i])  # Use the object class as category
            
        return boxes, labels, attr_labels
    
    def _generate_ego_trajectory_data(self):
        """Generate ego vehicle trajectory data"""
        # Historical trajectories (batch_size=1, past 4 frames, 2D)
        ego_his_trajs = np.random.randn(1, 4, 2).astype(np.float32)
        
        # Future trajectories (batch_size=1, 1, 6 future steps, 2D)  
        ego_fut_trajs = np.random.randn(1, 1, 6, 2).astype(np.float32)
        
        # Future masks (batch_size=1, 6 future steps)
        ego_fut_masks = np.ones((1, 6), dtype=np.float32)
        
        # Future command (navigation command) - should be 4D for model access cmd[0, 0, 0]
        # Shape: (batch_size, queue_length, 1, num_commands) -> after cmd[0, 0, 0] -> (num_commands,)
        ego_fut_cmd = np.zeros((1, 1, 1, 3), dtype=np.int64)  # One-hot encoded command (3 commands: 0, 1, 2)
        ego_fut_cmd[0, 0, 0, np.random.randint(0, 3)] = 1  # Random command: straight, left, right
        
        # LCF features (batch_size=1, 256-dim feature)
        ego_lcf_feat = np.random.randn(1, 256).astype(np.float32)
        
        return ego_his_trajs, ego_fut_trajs, ego_fut_masks, ego_fut_cmd, ego_lcf_feat
    
    def _generate_lidar_points(self):
        """Generate random LiDAR point cloud data"""
        # Generate random points within the point cloud range
        num_points = random.randint(10000, 50000)
        
        # Generate random coordinates
        x = np.random.uniform(self.pc_range[0], self.pc_range[3], num_points)
        y = np.random.uniform(self.pc_range[1], self.pc_range[4], num_points)
        z = np.random.uniform(self.pc_range[2], self.pc_range[5], num_points)
        
        # Generate random intensity and ring values
        intensity = np.random.uniform(0, 255, num_points)
        ring = np.random.randint(0, 32, num_points)
        
        # Stack to create points array [N, 5] (x, y, z, intensity, ring)
        points = np.column_stack([x, y, z, intensity, ring]).astype(np.float32)
        
        return torch.tensor(points, dtype=torch.float32)
    
    def _generate_map_data(self):
        """Generate random map data"""
        num_vectors = random.randint(5, 20)
        
        if num_vectors == 0:
            # Return empty data if no vectors
            map_labels = np.array([], dtype=np.int64)
            # Create a simple mock object with the required attribute
            class MockMapData:
                def __init__(self, data):
                    self.fixed_num_sampled_points = data
            mock_map_data = MockMapData(torch.empty((0, 20, 2), dtype=torch.float32))
            return mock_map_data, map_labels
        
        # Generate random map vector labels
        map_labels = np.random.randint(0, self.map_num_classes, num_vectors)
        
        # Generate random vector points (simplified as line segments)
        # Each vector has fixed number of points as per config
        fixed_ptsnum_per_line = 20
        map_pts = []
        
        for _ in range(num_vectors):
            # Generate a random line within the point cloud range
            start_x = np.random.uniform(self.pc_range[0], self.pc_range[3])
            start_y = np.random.uniform(self.pc_range[1], self.pc_range[4])
            end_x = np.random.uniform(self.pc_range[0], self.pc_range[3])
            end_y = np.random.uniform(self.pc_range[1], self.pc_range[4])
            
            # Interpolate points along the line
            x_points = np.linspace(start_x, end_x, fixed_ptsnum_per_line)
            y_points = np.linspace(start_y, end_y, fixed_ptsnum_per_line)
            
            line_pts = np.column_stack([x_points, y_points])
            map_pts.append(line_pts)
        
        # Create a mock LiDARInstanceLines object with required attributes
        if map_pts:
            map_tensor = torch.tensor(np.array(map_pts), dtype=torch.float32)  # [num_vectors, fixed_ptsnum_per_line, 2]
        else:
            map_tensor = torch.empty((0, fixed_ptsnum_per_line, 2), dtype=torch.float32)
            
        # Create a simple mock object with the required attribute
        class MockMapData:
            def __init__(self, data):
                self.fixed_num_sampled_points = data
                
        mock_map_data = MockMapData(map_tensor)
            
        return mock_map_data, map_labels
    
    def _generate_camera_metadata(self):
        """Generate camera metadata"""
        camera_metas = {}
        
        # Intrinsic matrices for each camera
        camera_intrinsics = []
        camera2ego = []
        lidar2img = []
        lidar2cam = []
        
        for i in range(self.num_cameras):
            # Random camera intrinsics
            intrinsic = np.array([
                [1200.0, 0.0, 800.0],
                [0.0, 1200.0, 450.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            
            # Random camera to ego transformation
            cam2ego = np.eye(4, dtype=np.float32)
            # Add some random rotation and translation
            angle = i * np.pi / 3  # Distribute cameras around the vehicle
            cam2ego[0, 3] = np.cos(angle) * 1.5  # x translation
            cam2ego[1, 3] = np.sin(angle) * 1.5  # y translation
            cam2ego[2, 3] = 1.5  # z translation (height)
            
            # Random lidar to camera transformation
            lidar2cam_rt = np.eye(4, dtype=np.float32)
            lidar2cam_rt[:3, :3] = np.random.randn(3, 3) * 0.1 + np.eye(3)
            lidar2cam_rt[:3, 3] = np.random.randn(3) * 0.5
            
            # Lidar to image transformation
            viewpad = np.eye(4, dtype=np.float32)
            viewpad[:3, :3] = intrinsic
            lidar2img_rt = viewpad @ lidar2cam_rt
            
            camera_intrinsics.append(viewpad)
            camera2ego.append(cam2ego)
            lidar2cam.append(lidar2cam_rt)
            lidar2img.append(lidar2img_rt)
        
        return {
            'camera_intrinsics': camera_intrinsics,
            'camera2ego': camera2ego,
            'lidar2cam': lidar2cam,
            'lidar2img': lidar2img,
            'img_shape': [(900, 1600) for _ in range(self.num_cameras)]  # (H, W) for each camera
        }
    
    def _generate_sample_metadata(self, idx):
        """Generate sample metadata"""
        # Random transformations
        lidar2ego = np.eye(4, dtype=np.float32)
        lidar2ego[:3, 3] = np.random.randn(3) * 0.1  # Small random translation
        
        ego2global = np.eye(4, dtype=np.float32)
        ego2global[:3, 3] = np.random.randn(3) * 10.0  # Larger translation for global position
        
        # CAN bus data (vehicle state)
        can_bus = np.zeros(18, dtype=np.float32)
        can_bus[:3] = ego2global[:3, 3]  # global position
        can_bus[7] = np.random.uniform(-5.0, 5.0)  # velocity
        can_bus[-1] = np.random.uniform(0, 360)  # heading angle
        
        return {
            'sample_idx': f'sample_{idx:06d}',
            'timestamp': 1000000.0 * idx,  # microseconds
            'lidar2ego': lidar2ego,
            'ego2global': ego2global,
            'can_bus': can_bus,
            'scene_token': f'scene_{idx // 10:03d}',  # 10 samples per scene
            'frame_idx': idx % 10,
            'prev_bev': idx % 10 > 0,
            'fut_valid_flag': True
        }
    
    def __getitem__(self, idx):
        """Get a single sample"""
        # Generate all the required data
        imgs = self._generate_random_images()
        points = self._generate_lidar_points()
        boxes, labels, attr_labels = self._generate_random_boxes_3d()
        ego_his_trajs, ego_fut_trajs, ego_fut_masks, ego_fut_cmd, ego_lcf_feat = self._generate_ego_trajectory_data()
        map_pts, map_labels = self._generate_map_data()
        camera_meta = self._generate_camera_metadata()
        sample_meta = self._generate_sample_metadata(idx)
        
        # Convert to LiDARInstance3DBoxes
        gt_bboxes_3d = LiDARInstance3DBoxes(
            torch.tensor(boxes, dtype=torch.float32),
            box_dim=boxes.shape[-1],
            origin=(0.5, 0.5, 0.5)
        )
        
        # Create img_metas - matching expected format for both head and transformer
        # Head expects img_metas[0][0] for current frame
        # Transformer expects to iterate over img_metas[0] for all frames
        frame_metadata_list = []
        for i in range(self.queue_length):
            frame_meta = {
                'sample_idx': sample_meta['sample_idx'],
                'timestamp': sample_meta['timestamp'] + i * 0.5,
                'can_bus': sample_meta['can_bus'].copy(),
                'lidar2ego': sample_meta['lidar2ego'],
                'ego2global': sample_meta['ego2global'], 
                'scene_token': sample_meta['scene_token'],
                'frame_idx': sample_meta['frame_idx'] + i,
                'prev_bev': sample_meta['prev_bev'] and i > 0,
                'fut_valid_flag': sample_meta['fut_valid_flag'],
                **camera_meta
            }
            # Adjust can_bus for relative motion like VAD dataset
            if i == 0:
                frame_meta['can_bus'][:3] = 0  # Reset position for first frame
                frame_meta['can_bus'][-1] = 0  # Reset angle for first frame
            frame_metadata_list.append(frame_meta)
        
        # img_metas should be structured so that:
        # - img_metas[0] gives the list of frame metadata (for transformer iteration)
        # - img_metas[0][0] gives the current frame metadata (for head access)
        img_metas_formatted = frame_metadata_list
        
        # Pack data in the expected format
        data = {
            'img': DC(imgs, cpu_only=False, stack=True),
            'points': DC(points, cpu_only=False),
            'gt_bboxes_3d': DC([gt_bboxes_3d], cpu_only=True),
            'gt_labels_3d': DC([torch.tensor(labels, dtype=torch.long)], cpu_only=False),
            'gt_attr_labels': DC([torch.tensor(attr_labels, dtype=torch.float32)], cpu_only=False),
            'ego_his_trajs': DC(torch.tensor(ego_his_trajs, dtype=torch.float32), cpu_only=False),
            'ego_fut_trajs': DC(torch.tensor(ego_fut_trajs, dtype=torch.float32), cpu_only=False),
            'ego_fut_masks': DC(torch.tensor(ego_fut_masks, dtype=torch.float32), cpu_only=False),
            'ego_fut_cmd': DC(torch.tensor(ego_fut_cmd, dtype=torch.long), cpu_only=False),
            'ego_lcf_feat': DC(torch.tensor(ego_lcf_feat, dtype=torch.float32), cpu_only=False),
            'map_gt_labels_3d': DC(torch.tensor(map_labels, dtype=torch.long), cpu_only=False),
            'map_gt_bboxes_3d': DC(map_pts, cpu_only=True),
            'img_metas': DC(frame_metadata_list, cpu_only=True),  # Direct list of frame metadata
            'fut_valid_flag': DC(torch.tensor([True]), cpu_only=False)  # Make it 1-d tensor
        }
        
        return data


def create_mockup_dataloader(batch_size=1, num_workers=0, num_samples=100):
    """Create a mockup dataloader for testing"""
    from torch.utils.data import DataLoader
    from mmcv.parallel import collate
    from functools import partial
    
    dataset = MockupSSRDataset(num_samples=num_samples, batch_size=batch_size)
    
    # Use MMDetection's collate function which properly handles DataContainer
    collate_fn = partial(collate, samples_per_gpu=batch_size)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # Force single-threaded to avoid shared memory issues
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return dataloader
