# ---------------------------------------------
# Quick CPU vs GPU Verification Tool for SSR
# Simple PCC comparison for key outputs
# ---------------------------------------------

import sys
sys.path.append('')
import numpy as np
import argparse
import torch
import warnings
import os
from scipy.stats import pearsonr
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
import time

warnings.filterwarnings("ignore")

def extract_key_tensors(output):
    """Extract key numerical tensors from model output for comparison."""
    key_tensors = {}
    
    if isinstance(output, list) and len(output) > 0:
        result = output[0]  # Take first result
        
        # Handle extra nesting level in CPU output
        if isinstance(result, list) and len(result) > 0:
            result = result[0]  # CPU output has extra nesting: output[0][0]
        
        if isinstance(result, dict) and 'pts_bbox' in result:
            pts_bbox = result['pts_bbox']
            
            # Handle different types of pts_bbox
            if isinstance(pts_bbox, dict):
                # Extract key prediction tensors
                if 'ego_fut_preds' in pts_bbox:
                    key_tensors['ego_fut_preds'] = pts_bbox['ego_fut_preds']
                
                if 'ego_fut_cmd' in pts_bbox:
                    key_tensors['ego_fut_cmd'] = pts_bbox['ego_fut_cmd']
                    
            # Handle pts_bbox as an object with attributes
            elif hasattr(pts_bbox, '__dict__'):
                if hasattr(pts_bbox, 'ego_fut_preds'):
                    key_tensors['ego_fut_preds'] = pts_bbox.ego_fut_preds
                
                if hasattr(pts_bbox, 'ego_fut_cmd'):
                    key_tensors['ego_fut_cmd'] = pts_bbox.ego_fut_cmd
            
            # Extract bounding box predictions if available
            if isinstance(pts_bbox, dict):
                if 'boxes_3d' in pts_bbox:
                    key_tensors['boxes_3d'] = pts_bbox['boxes_3d'].tensor if hasattr(pts_bbox['boxes_3d'], 'tensor') else pts_bbox['boxes_3d']
                if 'scores_3d' in pts_bbox:
                    key_tensors['scores_3d'] = pts_bbox['scores_3d']
                if 'labels_3d' in pts_bbox:
                    key_tensors['labels_3d'] = pts_bbox['labels_3d']
    
    return key_tensors

def quick_pcc(cpu_output, gpu_output):
    """Quick PCC computation for key tensors."""
    cpu_tensors = extract_key_tensors(cpu_output)
    gpu_tensors = extract_key_tensors(gpu_output)
    
    results = {}
    
    for key in cpu_tensors:
        if key in gpu_tensors:
            cpu_data = cpu_tensors[key].detach().cpu().numpy().flatten()
            gpu_data = gpu_tensors[key].detach().cpu().numpy().flatten()
            
            if cpu_data.shape == gpu_data.shape and len(cpu_data) > 1:
                # Use numpy for comparison
                if np.var(cpu_data) > 1e-10 and np.var(gpu_data) > 1e-10:
                    # Compute PCC using scipy
                    corr, p_val = pearsonr(cpu_data, gpu_data)
                    
                    # Use numpy.allclose for robust comparison
                    are_close = np.allclose(cpu_data, gpu_data, rtol=1e-5, atol=1e-6)
                    
                    # Comprehensive comparison metrics
                    mse = np.mean((cpu_data - gpu_data)**2)
                    mae = np.mean(np.abs(cpu_data - gpu_data))
                    max_abs_diff = np.max(np.abs(cpu_data - gpu_data))
                    
                    results[key] = {
                        'allclose': are_close,
                        'correlation': corr,
                        'p_value': p_val,
                        'pcc_threshold_0999': corr >= 0.999,
                        'allclose_and_pcc': are_close and corr >= 0.999,
                        'shape': cpu_data.shape,
                        'cpu_mean': np.mean(cpu_data),
                        'gpu_mean': np.mean(gpu_data),
                        'mse': mse,
                        'mae': mae,
                        'max_abs_diff': max_abs_diff,
                        'status': 'excellent' if (are_close and corr >= 0.999) else 'good' if corr >= 0.99 else 'poor'
                    }
                else:
                    # For constant arrays, use numpy.allclose for comparison
                    are_close = np.allclose(cpu_data, gpu_data, rtol=1e-5, atol=1e-6)
                    results[key] = {
                        'allclose': are_close,
                        'allclose_and_pcc': are_close,  # For constant arrays, allclose is sufficient
                        'status': 'identical_constant' if are_close else 'different_constant'
                    }
            else:
                results[key] = {'status': 'shape_mismatch', 'cpu_shape': cpu_data.shape, 'gpu_shape': gpu_data.shape}
        else:
            results[key] = {'status': 'missing_in_gpu'}
    
    # Check for keys only in GPU
    for key in gpu_tensors:
        if key not in cpu_tensors:
            results[key] = {'status': 'missing_in_cpu'}
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Quick CPU vs GPU verification')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Quick CPU vs GPU Verification Tool")
    print("="*50)
    
    # Load config and setup
    cfg = Config.fromfile(args.config)
    
    # Import plugins
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir).split('/')
            _module_path = '.'.join(_module_dir)
            importlib.import_module(_module_path)
    
    set_random_seed(args.seed, deterministic=True)
    
    # Create single test sample
    from tools.mockup_dataset import create_mockup_dataloader
    data_loader = create_mockup_dataloader(batch_size=1, num_workers=0, num_samples=1)
    test_data = next(iter(data_loader))
    
    # Build models
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    
    print("Setting up models...")
    
    # CPU model
    cpu_model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(cpu_model, args.checkpoint, map_location='cpu')
    cpu_model = cpu_model.cpu().eval()
    
    # GPU model (if available)
    if not torch.cuda.is_available():
        print("CUDA not available - CPU only test")
        return
    
    gpu_model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(gpu_model, args.checkpoint, map_location='cpu')
    gpu_model = MMDataParallel(gpu_model, device_ids=[0]).eval()
    
    print("Running inference...")
    
    # CPU inference
    from tools.test_cpu import single_gpu_test_cpu
    start_time = time.time()
    with torch.no_grad():
        # Create a proper data loader from the single test data
        from torch.utils.data import DataLoader, TensorDataset
        
        # Use the CPU test function helper for data handling
        from mmcv.parallel import DataContainer
        
        def scatter_kwargs(kwargs):
            """Extract data from DataContainer objects for CPU execution."""
            scattered_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, DataContainer):
                    scattered_kwargs[key] = value.data[0] if isinstance(value.data, list) else value.data
                else:
                    scattered_kwargs[key] = value
            return scattered_kwargs
        
        cpu_model.eval()
        scattered_data = scatter_kwargs(test_data)
        cpu_output = [cpu_model(return_loss=False, rescale=True, **scattered_data)]
    cpu_time = time.time() - start_time
    
    # GPU inference
    start_time = time.time()
    with torch.no_grad():
        gpu_output = gpu_model(return_loss=False, rescale=True, **test_data)
        gpu_output = [gpu_output] if not isinstance(gpu_output, list) else gpu_output
    gpu_time = time.time() - start_time
    
    print(f"CPU time: {cpu_time:.3f}s")
    print(f"GPU time: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x (GPU faster)")
    
    # Extract and compare tensors
    print("\nExtracting and comparing key outputs...")
    comparison = quick_pcc(cpu_output, gpu_output)
    
    print("="*50)
    print("COMPARISON RESULTS:")
    print("="*50)
    
    all_good = True
    for key, result in comparison.items():
        if 'correlation' in result:
            pcc = result['correlation']
            p_val = result['p_value']
            mse = result['mse']
            mae = result['mae']
            max_diff = result['max_abs_diff']
            status = result['status']
            allclose = result['allclose']
            meets_threshold = result['pcc_threshold_0999']
            allclose_and_pcc = result['allclose_and_pcc']
            
            print(f"{key}:")
            print(f"  np.allclose: {'✅' if allclose else '❌'} {allclose}")
            print(f"  PCC (scipy): {pcc:.6f} (p-value: {p_val:.2e})")
            print(f"  PCC ≥ 0.999: {'✅' if meets_threshold else '❌'} {meets_threshold}")
            print(f"  allclose AND pcc: {'✅' if allclose_and_pcc else '❌'} {allclose_and_pcc}")
            print(f"  Status: {status}")
            print(f"  MSE: {mse:.2e}")
            print(f"  MAE: {mae:.2e}")
            print(f"  Max absolute diff: {max_diff:.2e}")
            print(f"  Shape: {result['shape']}")
            
            if not allclose_and_pcc:
                print(f"  ⚠️  Failed allclose AND PCC ≥ 0.999 test!")
                all_good = False
            else:
                print(f"  ✅ Passed allclose AND PCC ≥ 0.999 test")
        elif 'allclose' in result:
            # For constant arrays
            allclose = result['allclose']
            allclose_and_pcc = result['allclose_and_pcc']
            print(f"{key}:")
            print(f"  np.allclose: {'✅' if allclose else '❌'} {allclose}")
            print(f"  Status: {result['status']}")
            if not allclose_and_pcc:
                all_good = False
        else:
            print(f"{key}: {result['status']}")
            if result['status'] not in ['identical_constant']:
                all_good = False
    
    print("="*50)
    if all_good:
        print("✅ OVERALL: CPU implementation looks correct!")
    else:
        print("⚠️  OVERALL: Some differences detected")
    print("="*50)

if __name__ == '__main__':
    main()
