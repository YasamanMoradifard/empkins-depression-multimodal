#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""
#----------------------------------------------------------------
# imports
#----------------------------------------------------------------
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import logging
logging.getLogger("speechbrain").setLevel(logging.WARNING)
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import argparse
import numpy as np
import random
import yaml
import wandb
import torch
import gc 
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import sys

# Get script directory and parent directory (MMFformer_ root)
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent

# Add parent directory to path (relative to this script's location)
sys.path.insert(0, str(parent_dir))

from models import MultiModalDepDet
# DepMamba is imported conditionally when needed (requires mamba_ssm)
from datasets_process import get_dvlog_dataloader, get_lmvd_dataloader, get_eks_dataloader
from train_val.utils import EarlyStopping, LOG_INFO, adjust_learning_rate
from train_val.train_val import train_epoch, val
from train_val.losses import CombinedLoss
from train_val.plotting import (
    plot_train_val_curves, 
    plot_confusion_matrix, 
    get_data_statistics, 
    format_statistics_string
)

# Seed 
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2025)

# Get absolute path to config file (relative to this script's location)
CONFIG_PATH = "/home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/configs/config.yaml"

def str_to_bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v is None:
        return True  # When flag is used without value, const=True is used, but handle None just in case
    if isinstance(v, str):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')

def parse_args():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="Train and test a model."
    )

    ## arguments whose default values are in config.yaml
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_gender", type=str)
    parser.add_argument("--test_gender", type=str)
    parser.add_argument(
        "-m", "--model", type=str,
    )
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-bs", "--batch_size", type=int)
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument('--optimizer', default='AdamW', type=str, 
                        help='Adam or AdamW or SGD or RMSProp')
    parser.add_argument('--lr_scheduler', default='cos', type=str, 
                        help='cos or StepLR or Plateau')
    parser.add_argument('--amsgrad', default=0, type=int, 
                        help='Adam amsgrad')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--lr_steps', default=[100, 200, 300, 400, 550, 700], type=float, nargs="+", metavar='LRSteps',
                        help='epochs to decay learning rate by 10')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--lr_patience', default=10, type=int,
                        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument("-ds", "--dataset", type=str)
    parser.add_argument("--condition", type=str, default="all",
                        help="For d02: condition filter {CR, CRADK, ADK, SHAM, all}")
    parser.add_argument("--phase", type=str, default="all",
                        help="For d02: phase filter {training_pos, training_neg, induction1, induction2, all}")
    parser.add_argument("--modalities", type=str, default="av",
                        help="For d02: 'av' (audio+video), 'video', or 'audio'")
    parser.add_argument("-g", "--gpu", type=str)
    parser.add_argument("-wdb", "--if_wandb", type=str_to_bool, default=False)
    parser.add_argument("-tqdm", "--tqdm_able", type=str_to_bool, default=False)
    parser.add_argument("-tr", "--train", type=str_to_bool, default=False,
                        help='Whether you want to training or not! (true/false)')
    parser.add_argument("--cross_infer", type=str_to_bool, default=False,
                        help="Exchange the dataset name and model")
    parser.add_argument("-d", "--device", type=str, nargs="*")
    parser.add_argument('-n_h', '--num_heads', default=1, type=int, 
                        help='number of heads, in the paper 1 or 4')
    parser.add_argument('-fus', '--fusion', default='ia', type=str, 
                        help='fusion type: lt | it | ia | MT | audio | video')
    parser.add_argument('--begin_epoch', default=1, type=int,
                        help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')
    
    # ========== NEW: Regularization Arguments (Added to tackle overfitting) ==========
    # Dropout rates - control how many neurons are randomly disabled during training
    parser.add_argument('--fusion_dropout', default=0.5, type=float,
                        help='Dropout rate for fusion layer (default: 0.5). Higher = more regularization.')
    parser.add_argument('--audio_dropout', default=0.5, type=float,
                        help='Dropout rate for audio features (default: 0.5). Higher = more regularization.')
    parser.add_argument('--visual_dropout', default=0.5, type=float,
                        help='Dropout rate for visual features (default: 0.5). Higher = more regularization.')
    parser.add_argument('--classifier_dropout', default=0.0, type=float,
                        help='Dropout rate before final classifier (default: 0.0). Add this for extra regularization.')
    parser.add_argument('--attention_dropout', default=0.1, type=float,
                        help='Dropout rate in attention mechanisms (default: 0.1). Controls attention overfitting.')
    parser.add_argument('--transformer_dropout', default=0.1, type=float,
                        help='Dropout rate in transformer encoder layers (default: 0.1). Regularizes transformer processing.')
    
    # Loss function regularization - controls penalty on model weights
    parser.add_argument('--lambda_reg', default=1e-5, type=float,
                        help='L2 regularization coefficient in loss (default: 1e-5). Increase to 1e-3 or 1e-2 for stronger regularization.')
    parser.add_argument('--focal_weight', default=0.5, type=float,
                        help='Weight for focal loss component (default: 0.5). Balances focal loss vs other components.')
    parser.add_argument('--l2_weight', default=0.5, type=float,
                        help='Weight for L2 regularization in loss (default: 0.5). Controls strength of weight penalty.')
    parser.add_argument('--label_smoothing', default=0.1, type=float,
                        help='Label smoothing epsilon (default: 0.1). Softens hard labels to reduce overfitting.')
    
    # Early stopping - prevents training too long when not improving
    parser.add_argument('--early_stopping_patience', default=5, type=int,
                        help='Early stopping patience (default: 5). Number of epochs to wait before stopping.')
    parser.add_argument('--early_stopping_delta', default=0.0, type=float,
                        help='Early stopping minimum delta improvement (default: 0.0). Minimum improvement to reset patience.')
    # ========== END NEW: Regularization Arguments ==========
    
    parser.set_defaults(**config)
    
    # Parse arguments with better error handling to show unrecognized args
    import sys
    
    # First, print all received arguments for debugging
    print("\n" + "="*80, file=sys.stderr)
    print("DEBUG: All command-line arguments received:", file=sys.stderr)
    print("="*80, file=sys.stderr)
    for i, arg in enumerate(sys.argv[1:], 1):
        print(f"  [{i}] {repr(arg)}", file=sys.stderr)
    print("="*80 + "\n", file=sys.stderr)
    
    # Try to parse and catch the error to show unrecognized arguments
    try:
        args, unknown = parser.parse_known_args()
        if unknown:
            print(f"\n{'='*80}", file=sys.stderr)
            print("ERROR: Unrecognized arguments found:", file=sys.stderr)
            print(f"{'='*80}", file=sys.stderr)
            for arg in unknown:
                print(f"  - {repr(arg)}", file=sys.stderr)
            print(f"\nTotal unrecognized arguments: {len(unknown)}", file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    except SystemExit as e:
        # If there's a SystemExit, it means argparse found an error
        # Let's try to get more info
        if hasattr(e, 'code') and e.code != 0:
            print(f"\nSystemExit with code {e.code}", file=sys.stderr)
        raise

    return args

def main():
    args = parse_args()
    
    # Convert data_dir to absolute path if it's relative
    if args.data_dir and not os.path.isabs(args.data_dir):
        # Resolve relative to the config file's directory (parent_dir)
        args.data_dir = os.path.join(parent_dir, args.data_dir.lstrip('./'))
        # Normalize the path (remove .. and .)
        args.data_dir = os.path.normpath(args.data_dir)
    
    # For d02 dataset, use d02_npy folder name instead of d02
    if args.dataset == 'd02_video_npy':
        args.data_dir = os.path.join(args.data_dir, 'd02_video_npy')
    elif args.dataset == 'd02_video_npy_downsampled_30':
        args.data_dir = os.path.join(args.data_dir, 'd02_video_npy_downsampled_30')
    elif args.dataset == 'd02_video_npy_downsampled_30_savgol':
        args.data_dir = os.path.join(args.data_dir, 'd02_video_npy_downsampled_30_savgol')
    elif args.dataset == 'd02_npy_downsampled_filtered':
        args.data_dir = os.path.join(args.data_dir, 'd02_npy_downsampled_filtered')
    else:
        args.data_dir = os.path.join(args.data_dir, args.dataset)
    
    # Convert to absolute path and verify it exists
    args.data_dir = os.path.abspath(args.data_dir)
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    print(f"Using data directory: {args.data_dir}")

    # prepare the data
    if args.dataset=='dvlog-dataset':
        train_loader = get_dvlog_dataloader(
                args.data_dir, "train", args.batch_size, args.train_gender
            )
        val_loader = get_dvlog_dataloader(
                args.data_dir, "valid", args.batch_size, args.test_gender
            )
        test_loader = get_dvlog_dataloader(
                args.data_dir, "test", args.batch_size, args.test_gender
            )

    elif args.dataset=='lmvd-dataset':
        train_loader = get_lmvd_dataloader(
            args.data_dir, "train", args.batch_size, args.train_gender
        )
        val_loader = get_lmvd_dataloader(
            args.data_dir, "valid", args.batch_size, args.test_gender
        )
        test_loader = get_lmvd_dataloader(
            args.data_dir, "test", args.batch_size, args.test_gender
        )
    elif args.dataset in ['d02_video_npy', 'd02_video_npy_downsampled_30', 'd02_video_npy_downsampled_30_savgol', 'd02_npy_downsampled_filtered']:
        # EKSpression / d02 dataset: use EKS dataloader with condition/phase/modalities filters
        train_loader = get_eks_dataloader(
            args.data_dir,
            "train",
            args.batch_size,
            gender=args.train_gender,
            condition=args.condition,
            phase=args.phase,
            modalities=args.modalities,
        )
        val_loader = get_eks_dataloader(
            args.data_dir,
            "validation",
            args.batch_size,
            gender=args.test_gender,
            condition=args.condition,
            phase=args.phase,
            modalities=args.modalities,
        )
        test_loader = get_eks_dataloader(
            args.data_dir,
            "test",
            args.batch_size,
            gender=args.test_gender,
            condition=args.condition,
            phase=args.phase,
            modalities=args.modalities,
        )

    if args.if_wandb:
        wandb_run_name = f"{args.model}-{args.train_gender}-{args.test_gender}"
        wandb.init(
            project="Multi-Modal Depression Model", config=args, name=wandb_run_name,
        )
        args = wandb.config
    print(args)

    if args.cross_infer:
        # Automatically switch dataset
        if args.dataset == "dvlog-dataset":
            args.dataset = "lmvd-dataset"
        elif args.dataset == "lmvd-dataset":
            args.dataset = "dvlog-dataset"
    
    # Build Save Dir
    os.makedirs(f"{args.save_dir}/{args.dataset}_{args.model}_{args.fusion}", exist_ok=True)
    os.makedirs(f"{args.save_dir}/{args.dataset}_{args.model}_{args.fusion}/samples", exist_ok=True)
    os.makedirs(f"{args.save_dir}/{args.dataset}_{args.model}_{args.fusion}/checkpoints", exist_ok=True)
    
    # Create runlog folder for d02 dataset
    runlog_dir = None
    runlog_file = None
    if args.dataset in ['d02_video_npy', 'd02_video_npy_downsampled_30', 'd02_video_npy_downsampled_30_savgol', 'd02_npy_downsampled_filtered']:
        # Create runlog folder: runlog_{condition}_{phase}_date_time
        condition_str = args.condition if hasattr(args, 'condition') else 'all'
        phase_str = args.phase if hasattr(args, 'phase') else 'all'
        date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        runlog_folder_name = f"runlog_{condition_str}_{phase_str}_{date_time_str}"
        runlog_dir = Path(f"{args.save_dir}/{args.dataset}_{args.model}_{args.fusion}") / runlog_folder_name
        runlog_dir.mkdir(parents=True, exist_ok=True)
        (runlog_dir / "plots").mkdir(exist_ok=True)
        
        # Create runlog.txt file
        runlog_file = runlog_dir / "runlog.txt"
        
        # Log data statistics
        train_stats = get_data_statistics(
            train_loader, 
            dataset_name=args.dataset,
            condition=condition_str,
            phase=phase_str,
            modalities=args.modalities if hasattr(args, 'modalities') else 'av'
        )
        val_stats = get_data_statistics(
            val_loader,
            dataset_name=args.dataset,
            condition=condition_str,
            phase=phase_str,
            modalities=args.modalities if hasattr(args, 'modalities') else 'av'
        )
        test_stats = get_data_statistics(
            test_loader,
            dataset_name=args.dataset,
            condition=condition_str,
            phase=phase_str,
            modalities=args.modalities if hasattr(args, 'modalities') else 'av'
        )
        
        with open(runlog_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRAINING RUN LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Fusion: {args.fusion}\n")
            f.write(f"Condition: {condition_str}\n")
            f.write(f"Phase: {phase_str}\n")
            f.write(f"Modalities: {args.modalities if hasattr(args, 'modalities') else 'av'}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Learning Rate: {args.learning_rate}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write("\n")
            f.write("TRAIN SET:\n")
            f.write(format_statistics_string(train_stats) + "\n\n")
            f.write("VALIDATION SET:\n")
            f.write(format_statistics_string(val_stats) + "\n\n")
            f.write("TEST SET:\n")
            f.write(format_statistics_string(test_stats) + "\n\n")
            f.write("=" * 80 + "\n")
            f.write("TRAINING PROGRESS\n")
            f.write("=" * 80 + "\n")
        
        LOG_INFO(f"Runlog directory created: {runlog_dir}", 'green')
        LOG_INFO(f"Data statistics logged to: {runlog_file}", 'green')

    # construct the model
    if args.model == "DepMamba":
        # Import DepMamba only when needed (requires mamba_ssm)
        from models import DepMamba
        if args.dataset=='lmvd-dataset':
            net = DepMamba(**args.mmmamba_lmvd)
        elif args.dataset=='dvlog-dataset':
            net = DepMamba(**args.mmmamba)
        elif args.dataset in ['d02_video_npy', 'd02_video_npy_downsampled_30', 'd02_video_npy_downsampled_30_savgol', 'd02_npy_downsampled_filtered']:
            # For now reuse LMVD Mamba config (adjust if needed)
            net = DepMamba(**args.mmmamba_lmvd)
    elif args.model == "MultiModalDepDet":
        if args.dataset=='lmvd-dataset':
            # ========== NEW: Pass dropout parameters to model (Added for regularization control) ==========
            net = MultiModalDepDet(
                **args.lmvd, 
                fusion=args.fusion, 
                num_heads=args.num_heads,
                fusion_dropout=args.fusion_dropout,
                audio_dropout=args.audio_dropout,
                visual_dropout=args.visual_dropout,
                attention_dropout=args.attention_dropout,
                transformer_dropout=args.transformer_dropout,
                classifier_dropout=args.classifier_dropout
            )
            # ========== END NEW ==========
        elif args.dataset=='dvlog-dataset':
            # ========== NEW: Pass dropout parameters to model (Added for regularization control) ==========
            net = MultiModalDepDet(
                **args.dvlog, 
                fusion=args.fusion, 
                num_heads=args.num_heads,
                fusion_dropout=args.fusion_dropout,
                audio_dropout=args.audio_dropout,
                visual_dropout=args.visual_dropout,
                attention_dropout=args.attention_dropout,
                transformer_dropout=args.transformer_dropout,
                classifier_dropout=args.classifier_dropout
            )
            # ========== END NEW ==========
        elif args.dataset in ['d02_video_npy', 'd02_video_npy_downsampled_30', 'd02_video_npy_downsampled_30_savgol', 'd02_npy_downsampled_filtered']:
            # ========== NEW: Pass dropout parameters to model (Added for regularization control) ==========
            net = MultiModalDepDet(
                **args.d02, 
                fusion=args.fusion, 
                num_heads=args.num_heads,
                fusion_dropout=args.fusion_dropout,
                audio_dropout=args.audio_dropout,
                visual_dropout=args.visual_dropout,
                attention_dropout=args.attention_dropout,
                transformer_dropout=args.transformer_dropout,
                classifier_dropout=args.classifier_dropout
            )
            # ========== END NEW ==========
    else:
        raise NotImplementedError(f"The {args.model} method has not been implemented by this repo")
    
    ### model check
    # Handle device argument - it can be a list (from nargs="*") or string
    if isinstance(args.device, list):
        device_str = args.device[0] if len(args.device) > 0 else 'cpu'
        use_multiple_devices = len(args.device) > 1
    else:
        device_str = args.device if args.device else 'cpu'
        use_multiple_devices = False
    
    if device_str != 'cpu':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    args.device = device_str
    print(f'Final choice of computing device: {args.device}')
    net = net.to(args.device)
    
    # Check if multiple devices were requested (for DataParallel)
    if use_multiple_devices:
        # net = torch.nn.DataParallel(net, device_ids=args.device)
        net = torch.nn.DataParallel(net, device_ids=None)

        pytorch_total_params = sum(p.numel() for p in net.parameters() if
                                p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)
        pytorch_total_params_ = sum(p.numel() for p in net.parameters())
        print("Total number of parameters: ", pytorch_total_params_)
    
    # ========== NEW: Pretrained Weights Verification (Added to verify fine-tuning setup) ==========
    print("\n" + "="*80)
    print("PRETRAINED WEIGHTS VERIFICATION")
    print("="*80)
    
    # Get the actual model (unwrap DataParallel if needed)
    model_to_check = net.module if hasattr(net, 'module') else net
    
    # Check audio model (if it exists)
    if hasattr(model_to_check, 'audio_model'):
        audio_params = sum(p.numel() for p in model_to_check.audio_model.parameters())
        audio_trainable = sum(p.numel() for p in model_to_check.audio_model.parameters() if p.requires_grad)
        audio_frozen = audio_params - audio_trainable
        print(f"Audio Model (AudioSet):")
        print(f"  - Total parameters: {audio_params:,}")
        print(f"  - Trainable parameters: {audio_trainable:,} ({100*audio_trainable/audio_params:.1f}%)")
        print(f"  - Frozen parameters: {audio_frozen:,} ({100*audio_frozen/audio_params:.1f}%)")
        print(f"  - Status: {'✓ Fine-tuning enabled' if audio_trainable > 0 else '✗ All frozen'}")
    else:
        print("Audio Model: Not found (video-only mode)")
    
    # Check visual model (if it exists)
    if hasattr(model_to_check, 'visual_model'):
        visual_params = sum(p.numel() for p in model_to_check.visual_model.parameters())
        visual_trainable = sum(p.numel() for p in model_to_check.visual_model.parameters() if p.requires_grad)
        visual_frozen = visual_params - visual_trainable
        print(f"\nVisual Model (VisualMAE):")
        print(f"  - Total parameters: {visual_params:,}")
        print(f"  - Trainable parameters: {visual_trainable:,} ({100*visual_trainable/visual_params:.1f}%)")
        print(f"  - Frozen parameters: {visual_frozen:,} ({100*visual_frozen/visual_params:.1f}%)")
        print(f"  - Status: {'✓ Fine-tuning enabled' if visual_trainable > 0 else '✗ All frozen'}")
    else:
        print("Visual Model: Not found (audio-only mode)")
    
    # Check total model
    total_params = sum(p.numel() for p in model_to_check.parameters())
    total_trainable = sum(p.numel() for p in model_to_check.parameters() if p.requires_grad)
    total_frozen = total_params - total_trainable
    print(f"\nTotal Model:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {total_trainable:,} ({100*total_trainable/total_params:.1f}%)")
    print(f"  - Frozen parameters: {total_frozen:,} ({100*total_frozen/total_params:.1f}%)")
    print(f"  - Status: {'✓ Ready for fine-tuning' if total_trainable > 0 else '✗ All parameters frozen'}")
    
    # Verify pretrained weights are loaded (check if weights are non-zero/random)
    if hasattr(model_to_check, 'audio_model') and hasattr(model_to_check.audio_model, 'v'):
        # Check a sample weight from audio transformer
        sample_audio_weight = next(model_to_check.audio_model.v.blocks[0].parameters())
        weight_std = sample_audio_weight.std().item()
        print(f"\nAudio Model Weight Check:")
        print(f"  - Sample weight std: {weight_std:.6f}")
        print(f"  - Status: {'✓ Weights loaded (non-random)' if weight_std > 0.01 else '⚠ Weights may be random (check loading)'}")
    
    if hasattr(model_to_check, 'visual_model') and hasattr(model_to_check.visual_model, 'visual_model'):
        # Check a sample weight from visual transformer
        if hasattr(model_to_check.visual_model.visual_model, 'blocks'):
            sample_visual_weight = next(model_to_check.visual_model.visual_model.blocks[0].parameters())
            weight_std = sample_visual_weight.std().item()
            print(f"\nVisual Model Weight Check:")
            print(f"  - Sample weight std: {weight_std:.6f}")
            print(f"  - Status: {'✓ Weights loaded (non-random)' if weight_std > 0.01 else '⚠ Weights may be random (check loading)'}")
    
    print("="*80 + "\n")
    # ========== END NEW: Pretrained Weights Verification ==========

    # set other training components
    # ========== NEW: Use regularization arguments from command line (Changed from hardcoded values) ==========
    loss_fn = CombinedLoss(
        lambda_reg=args.lambda_reg,      # NEW: Use args.lambda_reg instead of hardcoded 1e-5
        focal_weight=args.focal_weight,  # NEW: Use args.focal_weight instead of hardcoded 0.5
        l2_weight=args.l2_weight,        # NEW: Use args.l2_weight instead of hardcoded 0.5
        smooth_eps=args.label_smoothing  # NEW: Use args.label_smoothing instead of default 0.1
    )
    # ========== END NEW ==========
    
    # Setting the optimizer for model training
    assert args.optimizer in ["RMSprop", "SGD", "Adam", "AdamW"]
    if args.optimizer=="SGD":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()),     # Parameters passed into the model
            lr=args.learning_rate,                                   # Learning rate
            momentum=args.momentum,                                  # Momentum factor (optional)
            dampening=args.dampening,
            weight_decay=args.weight_decay,                          # Weight decay (L2 regularization)
            nesterov=False)
    elif args.optimizer=="Adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),  
            lr=args.learning_rate,  
            betas=(0.9,0.999), 
            weight_decay=args.weight_decay,  
            amsgrad=args.amsgrad
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, net.parameters()),  
            lr=args.learning_rate, 
            betas=(0.9, 0.999),  
            weight_decay=args.weight_decay,  
            amsgrad=args.amsgrad  
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, net.parameters()),  
            lr=args.learning_rate, 
            alpha=0.99,  
            eps=1e-8,  
            weight_decay=args.weight_decay,  
            momentum=0.9,  
        )
    
    ## learning scheluder
    if args.lr_scheduler == "cos":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs // 5, eta_min=args.learning_rate / 20
        )
    elif args.lr_scheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.00001
        )
    elif args.lr_scheduler == "Plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=args.lr_patience
        )
    else:
        lr_scheduler = None

    # ========== NEW: Use early stopping arguments from command line (Changed from hardcoded values) ==========
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,  # NEW: Use args.early_stopping_patience instead of hardcoded 5
        delta=args.early_stopping_delta,        # NEW: Use args.early_stopping_delta instead of default 0.0
        verbose=True, 
        save_path=f"{args.save_dir}/{args.dataset}_{args.model}_{args.fusion}/checkpoints/best_model.pt"
    )
    # ========== END NEW ==========
    
    best_val_acc = -1.0
    best_test_acc = -1.0

    # Check for fold-specific best model
    fold_best_model_path = f"{args.save_dir}/{args.dataset}_{args.model}_{args.fusion}/checkpoints/best_model.pt"
    print(fold_best_model_path)
    if os.path.exists(fold_best_model_path) and not args.resume_path:  # Only check fold-specific if no resume_path
        print(f"Resuming from fold-specific checkpoint: {fold_best_model_path}")
        checkpoint = torch.load(fold_best_model_path, map_location=args.device, weights_only=False)
        # checkpoint = torch.load(fold_best_model_path, weights_only=False)
        assert args.model == checkpoint['arch']
        best_val_acc = checkpoint['best_val_acc']
        print(f"Loaded Model Best Val Acc: {best_val_acc}")
        args.begin_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        print(f"Ended Epoch: {checkpoint['epoch']} and Begining Epoch: {args.begin_epoch}")
        # print(checkpoint['state_dict'])
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    elif args.resume_path:
        print('loading checkpoint {}'.format(args.resume_path))
        checkpoint = torch.load(args.resume_path, map_location=args.device, weights_only=False)
        
        # Check if it's a full checkpoint or just weights
        if isinstance(checkpoint, dict) and 'arch' in checkpoint:
            # Full checkpoint format
            assert args.model == checkpoint['arch']
            best_val_acc = checkpoint.get('best_val_acc', -1.0)
            print(f"Loaded Model Best Val Acc: {best_val_acc}")
            args.begin_epoch = checkpoint.get('epoch', 0) + 1
            print(f"Ended Epoch: {checkpoint.get('epoch', 0)} and Beginning Epoch: {args.begin_epoch}")
            net.load_state_dict(checkpoint['state_dict'])
        else:
            # Just weights (e.g., VisualMAE pretrained model)
            print("Loading pretrained weights (not a full checkpoint)...")
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                weights = checkpoint['state_dict']
            else:
                weights = checkpoint
            
            # Try to load into visual_model if it's VisualMAE weights
            if args.model == 'MultiModalDepDet' and hasattr(net, 'visual_model'):
                print("Attempting to load VisualMAE weights into visual_model...")
                try:
                    # Try loading directly into visual_model
                    net.visual_model.load_state_dict(weights, strict=False)
                    print("Successfully loaded VisualMAE weights into visual_model")
                except Exception as e:
                    print(f"Could not load into visual_model directly: {e}")
                    print("Attempting to load with prefix matching...")
                    # Try with 'visual_model.' prefix if weights don't have it
                    if not any(k.startswith('visual_model.') for k in weights.keys()):
                        visual_weights = {f'visual_model.{k}': v for k, v in weights.items()}
                    else:
                        visual_weights = weights
                    # Try loading into full model with strict=False
                    missing_keys, unexpected_keys = net.load_state_dict(visual_weights, strict=False)
                    if missing_keys:
                        print(f"Missing keys (will use random init): {len(missing_keys)} keys")
                    if unexpected_keys:
                        print(f"Unexpected keys (ignored): {len(unexpected_keys)} keys")
                    print("Loaded VisualMAE weights with partial matching")
            else:
                # Try loading into full model
                missing_keys, unexpected_keys = net.load_state_dict(weights, strict=False)
                if missing_keys:
                    print(f"Missing keys (will use random init): {len(missing_keys)} keys")
                if unexpected_keys:
                    print(f"Unexpected keys (ignored): {len(unexpected_keys)} keys")
                print("Loaded pretrained weights with partial matching")
            
            args.begin_epoch = 1
            best_val_acc = -1.0
    else:
        args.begin_epoch = 1
        best_val_acc = -1.0

    # Initialize training history tracking for d02
    train_loss_history = []
    train_acc_history = []
    train_f1_history = []
    val_loss_history = []
    val_acc_history = []
    val_f1_history = []
    
    print(f"Training: {args.train}")
    if args.train:
        for epoch in range(args.begin_epoch, args.epochs+1):
            adjust_learning_rate(optimizer, epoch, args)

            train_results = train_epoch(
                net, train_loader, loss_fn, optimizer, lr_scheduler,
                args.device, epoch, args.epochs, args.tqdm_able
            )
            val_results = val(net, val_loader, loss_fn, args.device, args.tqdm_able, msg='additional metrics', cross_infer=args.cross_infer)

            # Track history for plotting
            if args.dataset in ['d02_video_npy', 'd02_video_npy_downsampled_30', 'd02_video_npy_downsampled_30_savgol', 'd02_npy_downsampled_filtered']:
                train_loss_history.append(train_results["loss"])
                train_acc_history.append(train_results["acc"])
                train_f1_history.append(train_results["f1"])
                val_loss_history.append(val_results["loss"])
                val_acc_history.append(val_results["acc"])
                val_f1_history.append(val_results["f1"])
                
                # Log to runlog file
                if runlog_file:
                    with open(runlog_file, 'a') as f:
                        f.write(f"\nEpoch {epoch}/{args.epochs}:\n")
                        f.write(f"  Train Loss: {train_results['loss']:.4f}, Train Acc: {train_results['acc']:.4f}, Train F1: {train_results['f1']:.4f}\n")
                        f.write(f"  Val Loss: {val_results['loss']:.4f}, Val Acc: {val_results['acc']:.4f}\n")
                        f.write(f"  Val Precision: {val_results['precision']:.4f}, Recall: {val_results['recall']:.4f}, F1: {val_results['f1']:.4f}\n")

            # val_acc = (val_results["acc"] + val_results["precision"]+ val_results["recall"]+ val_results["f1"])/4.0
            val_acc = val_results["acc"] 
            if val_acc > best_val_acc:
                best_val_acc = val_acc

                state = {
                    'epoch': epoch,
                    'arch': args.model,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'model': net
                }
                save_path = f"{args.save_dir}/{args.dataset}_{args.model}_{args.fusion}/checkpoints/best_model.pt"
                torch.save(state, save_path)
                LOG_INFO(f"[{args.model}_{args.fusion}]: Model saved at epoch {state['epoch']}: {save_path}  | best_val_acc: {best_val_acc}", 'green')
                
                # Save validation confusion matrix when best model is found
                if args.dataset in ['d02_video_npy', 'd02_video_npy_downsampled_30', 'd02_video_npy_downsampled_30_savgol', 'd02_npy_downsampled_filtered'] and runlog_dir and 'true_labels' in val_results:
                    plot_confusion_matrix(
                        val_results['true_labels'],
                        val_results['predicted_labels'],
                        runlog_dir / "plots" / "confusion_matrix_val.png",
                        split='validation'
                    )

            if early_stopping.early_stop: ## Early stop when it found increase loss or satuated
                LOG_INFO("Early stopping triggered", 'red')
                if runlog_file:
                    with open(runlog_file, 'a') as f:
                        f.write("\nEarly stopping triggered!\n")
                break

            if args.if_wandb:
                wandb.log({
                    "loss/train": train_results["loss"],
                    "acc/train": train_results["acc"],
                    "f1/train": train_results["f1"],
                    "loss/val": val_results["loss"],
                    "acc/val": val_results["acc"],
                    "precision/val": val_results["precision"],
                    "recall/val": val_results["recall"],
                    "f1/val": val_results["f1"]
                })
        
    # print(f"resume_path: {args.resume_path}")

    # upload the best model to wandb website
    # load the best model for testing
    with torch.no_grad():
        if not args.resume_path:
            # print("not resume_path")
            best_state_path = f"{args.save_dir}/{args.dataset}_{args.model}_{args.fusion}/checkpoints/best_model.pt"
            LOG_INFO(f"best_state_path: {best_state_path}")
            checkpoint = torch.load(best_state_path, map_location=args.device, weights_only=False)
            net.load_state_dict(
                checkpoint['state_dict']
            )

        net.eval()
        test_results = val(net, test_loader, loss_fn, args.device,args.tqdm_able, msg='additional metrics', cross_infer=args.cross_infer)
        LOG_INFO(f"[{args.dataset}_{args.model}_{args.fusion}] Test results:")
        LOG_INFO(f"Test Result: {test_results}", "magenta")
        color_map = {
            "loss": "red",
            "acc": "cyan",
            "precision": "magenta",
            "recall": "yellow",
            "f1": "green",
            "weighted_accuracy": "blue",
            "unweighted_accuracy": "light_blue",
            "weighted_precision": "light_magenta",
            "unweighted_precision": "light_yellow",
            "weighted_recall": "light_cyan",
            "unweighted_recall": "light_green",
            "weighted_f1": "white",
            "unweighted_f1": "light_grey",
        }
        for key, value in test_results.items():
            if key not in ['true_labels', 'predicted_labels']:
                color = color_map.get(key, 'blue')  # Default to blue if key not found
                LOG_INFO(f"{key}: {value:.4f}", color) # Format float values

        # Save plots and confusion matrices for d02
        if args.dataset in ['d02_video_npy', 'd02_video_npy_downsampled_30', 'd02_video_npy_downsampled_30_savgol', 'd02_npy_downsampled_filtered'] and runlog_dir:
            # Save training curves
            if len(train_loss_history) > 0 and len(val_loss_history) > 0:
                plot_train_val_curves(
                    train_loss_history,
                    val_loss_history,
                    runlog_dir / "plots" / "train_val_loss.png",
                    metric='loss'
                )
                plot_train_val_curves(
                    train_acc_history,
                    val_acc_history,
                    runlog_dir / "plots" / "train_val_acc.png",
                    metric='acc'
                )
                plot_train_val_curves(
                    train_f1_history,
                    val_f1_history,
                    runlog_dir / "plots" / "train_val_f1.png",
                    metric='f1'
                )
                LOG_INFO(f"Training curves saved to {runlog_dir / 'plots'}", 'green')
            
            # Save test confusion matrix
            if 'true_labels' in test_results and 'predicted_labels' in test_results:
                plot_confusion_matrix(
                    test_results['true_labels'],
                    test_results['predicted_labels'],
                    runlog_dir / "plots" / "confusion_matrix_test.png",
                    split='test'
                )
                LOG_INFO(f"Test confusion matrix saved to {runlog_dir / 'plots'}", 'green')
            
            # Log test results to runlog
 # Log test results to runlog
            if runlog_file:
                with open(runlog_file, 'a') as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("TEST RESULTS\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"Test Loss: {test_results['loss']:.4f}\n")
                    f.write(f"Test Accuracy: {test_results['acc']:.4f}\n")
                    f.write(f"Test Precision: {test_results['precision']:.4f}\n")
                    f.write(f"Test Recall: {test_results['recall']:.4f}\n")
                    f.write(f"Test F1: {test_results['f1']:.4f}\n")
                    # Add weighted and unweighted metrics
                    f.write(f"\nWeighted Metrics:\n")
                    f.write(f"  Weighted Accuracy: {test_results['weighted_accuracy']:.4f}\n")
                    f.write(f"  Weighted Precision: {test_results['weighted_precision']:.4f}\n")
                    f.write(f"  Weighted Recall: {test_results['weighted_recall']:.4f}\n")
                    f.write(f"  Weighted F1: {test_results['weighted_f1']:.4f}\n")
                    f.write(f"\nUnweighted Metrics:\n")
                    f.write(f"  Unweighted Accuracy: {test_results['unweighted_accuracy']:.4f}\n")
                    f.write(f"  Unweighted Precision: {test_results['unweighted_precision']:.4f}\n")
                    f.write(f"  Unweighted Recall: {test_results['unweighted_recall']:.4f}\n")
                    f.write(f"  Unweighted F1: {test_results['unweighted_f1']:.4f}\n")
                    f.write(f"\nBest Validation Accuracy: {best_val_acc:.4f}\n")
                    f.write(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 80 + "\n")

        results_file = parent_dir / "results" / f"{args.dataset}_{args.model}_{args.fusion}.txt"
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:    
            test_result_str = f'Accuracy:{test_results["acc"]}, Precision:{test_results["precision"]}, Recall:{test_results["recall"]}, F1:{test_results["f1"]},\
                    Avg:{(test_results["acc"] + test_results["precision"]+ test_results["recall"]+ test_results["f1"])/4.0},\
                    WA:{test_results["weighted_accuracy"]}, UA:{test_results["unweighted_accuracy"]},\
                            WP:{test_results["weighted_precision"]}, UP:{test_results["unweighted_precision"]},\
                            WR:{test_results["weighted_recall"]}, UR:{test_results["unweighted_recall"]},\
                                WF:{test_results["weighted_f1"]}, UF:{test_results["unweighted_f1"]}'
            f.write(test_result_str)         

    if args.if_wandb:
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(f"{args.save_dir}/{args.dataset}_{args.model}_{args.fusion}/checkpoints/best_model.pt")

        wandb.run.summary["acc/best_val_acc"] = best_val_acc
        wandb.log_artifact(artifact)
        wandb.run.summary["acc/test_acc"] = test_results["acc"]
        wandb.run.summary["loss/test_loss"] = test_results["loss"]
        wandb.run.summary["precision/test_precision"] = test_results["precision"]
        wandb.run.summary["recall/test_recall"] = test_results["recall"]
        wandb.run.summary["f1/test_f1"] = test_results["f1"]

        wandb.finish()

    del net 
    del optimizer
    del lr_scheduler
    del early_stopping
    del loss_fn 

    del train_loader
    del val_loader
    del test_loader

    gc.collect
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    

if __name__ == '__main__':
    main()