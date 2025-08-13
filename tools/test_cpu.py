# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
#  Modified for CPU execution
# ---------------------------------------------
import sys
sys.path.append('')
import numpy as np
import argparse
import mmcv
import os
import copy
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings
from torch.profiler import profile, record_function, ProfilerActivity
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, save_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
# from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from projects.mmdet3d_plugin.SSR.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import json

import warnings
warnings.filterwarnings("ignore")

def single_gpu_test_with_profiler_cpu(model, data_loader, profiler_args):
    """Test model with single cpu and profiler enabled.
    
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        profiler_args: Arguments containing profiler configuration.
        
    Returns:
        list[dict]: The prediction results.
    """
    from mmcv.parallel import DataContainer
    
    def scatter_kwargs(kwargs):
        """Scatter kwargs for CPU execution (extract from DataContainer)."""
        scattered_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, DataContainer):
                # Extract data from DataContainer
                scattered_kwargs[key] = value.data[0] if isinstance(value.data, list) else value.data
            else:
                scattered_kwargs[key] = value
        return scattered_kwargs
    
    # Create profiler output directory
    os.makedirs(profiler_args.profiler_output_dir, exist_ok=True)
    
    # Configure profiler
    profiler_schedule = torch.profiler.schedule(
        wait=profiler_args.profiler_wait,
        warmup=profiler_args.profiler_warmup,
        active=profiler_args.profiler_active,
        repeat=profiler_args.profiler_repeat
    )
    
    def trace_handler(prof):
        # Save Chrome trace
        chrome_trace_path = os.path.join(profiler_args.profiler_output_dir, f"trace_{prof.step_num}.json")
        prof.export_chrome_trace(chrome_trace_path)
        print(f"Profiler trace saved to: {chrome_trace_path}")
        
        # Save detailed profiler table
        table_path = os.path.join(profiler_args.profiler_output_dir, f"profiler_table_{prof.step_num}.txt")
        with open(table_path, 'w') as f:
            # CPU profiling
            f.write("CPU time sorted by CPU time:\n")
            f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
            f.write("\n\nMemory sorted by self CPU memory:\n")
            f.write(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=50))
            
            # CUDA profiling (should be empty for CPU execution)
            f.write("\n\nCUDA time sorted by CUDA time (should be empty for CPU execution):\n")
            cuda_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=50)
            f.write(cuda_table)
            
            # Check if any CUDA operations were detected
            key_averages = prof.key_averages()
            cuda_ops = [item for item in key_averages if item.cuda_time_total > 0]
            
            if cuda_ops:
                f.write(f"\n\n⚠️  WARNING: {len(cuda_ops)} CUDA operations detected! This indicates GPU usage.\n")
                f.write("CUDA operations found:\n")
                for op in cuda_ops[:10]:  # Show first 10 CUDA operations
                    f.write(f"  - {op.key}: {op.cuda_time_total:.2f}μs\n")
            else:
                f.write("\n\n✅ SUCCESS: No CUDA operations detected. Running purely on CPU.\n")
        
        print(f"Profiler table saved to: {table_path}")
        
        # Print immediate feedback about CUDA usage
        key_averages = prof.key_averages()
        cuda_ops = [item for item in key_averages if item.cuda_time_total > 0]
        
        if cuda_ops:
            print(f"⚠️  WARNING: {len(cuda_ops)} CUDA operations detected in step {prof.step_num}!")
        else:
            print(f"✅ Step {prof.step_num}: No CUDA operations detected - pure CPU execution confirmed.")
    
    print("Running inference with profiler enabled on CPU (monitoring both CPU and CUDA to verify no GPU usage)...")
    # Create profiler context and run inference with step tracking (CPU + CUDA monitoring)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # Monitor both CPU and CUDA
        schedule=profiler_schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        model.eval()
        results = []
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))
        
        for i, data in enumerate(data_loader):
            with record_function("inference_step"):
                with torch.no_grad():
                    # Handle DataContainer conversion for CPU
                    scattered_data = scatter_kwargs(data)
                    result = model(return_loss=False, rescale=True, **scattered_data)
            
            results.extend(result)
            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()
            
            # Step the profiler
            prof.step()
        
        return results

def single_gpu_test_cpu(model, data_loader, show=False, show_dir=None):
    """Test model with single CPU (modified from single_gpu_test).
    
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to show results.
        show_dir (str): Directory to save visualization results.
        
    Returns:
        list[dict]: The prediction results.
    """
    from mmcv.parallel import DataContainer
    
    def scatter_kwargs(kwargs):
        """Scatter kwargs for CPU execution (extract from DataContainer)."""
        scattered_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, DataContainer):
                # Extract data from DataContainer
                scattered_kwargs[key] = value.data[0] if isinstance(value.data, list) else value.data
            else:
                scattered_kwargs[key] = value
        return scattered_kwargs
    
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # Handle DataContainer conversion for CPU
            scattered_data = scatter_kwargs(data)
            result = model(return_loss=False, rescale=True, **scattered_data)
        
        results.extend(result)
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model on CPU')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--json_dir', help='json parent dir name file') # NOTE: json file parent folder name
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--enable-profiler',
        action='store_true',
        help='Enable torch profiler for performance analysis')
    parser.add_argument(
        '--profiler-output-dir',
        type=str,
        default='./profiler_results',
        help='Directory to save profiler results')
    parser.add_argument(
        '--profiler-wait',
        type=int,
        default=1,
        help='Number of steps to wait before profiling')
    parser.add_argument(
        '--profiler-warmup',
        type=int,
        default=1,
        help='Number of warmup steps for profiler')
    parser.add_argument(
        '--profiler-active',
        type=int,
        default=3,
        help='Number of active profiling steps')
    parser.add_argument(
        '--profiler-repeat',
        type=int,
        default=1,
        help='Number of profiling cycles to repeat')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # Disable CUDNN for CPU execution
    # if cfg.get('cudnn_benchmark', False):
    #     torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # Force non-distributed execution for CPU
    distributed = False

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # Create mockup dataset and dataloader
    from tools.mockup_dataset import create_mockup_dataloader
    
    # Create mockup dataloader with a small number of samples for testing
    data_loader = create_mockup_dataloader(
        batch_size=samples_per_gpu,
        num_workers=0,  # Force single-threaded to avoid shared memory issues
        num_samples=10  # Small number for quick testing
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Disable fp16 for CPU execution
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    
    # Load checkpoint with CPU mapping
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        # Use default nuScenes classes for mockup
        model.CLASSES = [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
            'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        # No PALETTE for mockup dataset
        model.PALETTE = None

    # CPU execution - no MMDataParallel device_ids
    print("Running inference on CPU...")
    model = model.cpu()  # Ensure model is on CPU
    
    # Set up profiler if enabled
    if args.enable_profiler:
        outputs = single_gpu_test_with_profiler_cpu(model, data_loader, args)
    else:
        outputs = single_gpu_test_cpu(model, data_loader, args.show, args.show_dir)

    tmp = {}
    tmp['bbox_results'] = outputs
    outputs = tmp
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            if isinstance(outputs, list):
                mmcv.dump(outputs, args.out)
            else:
                mmcv.dump(outputs['bbox_results'], args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            # For mockup dataset, skip format_results since we don't have a real dataset
            print("Skipping format_results for mockup dataset")

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            # For mockup dataset, skip evaluation since we don't have ground truth
            print("Skipping evaluation for mockup dataset - no ground truth available")
        
        # Print profiler summary if enabled
        if args.enable_profiler:
            print(f"\n{'='*60}")
            print("TORCH PROFILER SUMMARY (CPU with CUDA monitoring)")
            print(f"{'='*60}")
            print(f"Profiler results saved to: {args.profiler_output_dir}")
            print("Files generated:")
            print("  - trace_*.json: Chrome trace files (can be viewed in chrome://tracing)")
            print("  - profiler_table_*.txt: Detailed performance tables with CUDA verification")
            print("\nCUDA Usage Verification:")
            print("  Check the profiler tables for CUDA operations.")
            print("  If no CUDA operations are found, CPU-only execution is confirmed.")
            print(f"{'='*60}")

if __name__ == '__main__':
    main()
