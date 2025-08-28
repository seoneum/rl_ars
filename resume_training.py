#!/usr/bin/env python3
"""
Resume training from checkpoint with flexible options
"""

import os
import sys
import argparse
import subprocess
import numpy as np

def check_checkpoint(path):
    """Check checkpoint and return info"""
    if not os.path.exists(path):
        return None
    
    try:
        data = np.load(path, allow_pickle=True)
        info = {
            'iterations': int(data['iter']) if 'iter' in data else 0,
            'obs_dim': int(data['obs_dim']),
            'act_dim': int(data['act_dim']),
        }
        
        # Try to get metadata if available
        if 'meta' in data:
            meta = data['meta'].item()
            if meta:
                info['meta'] = meta
        
        return info
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('checkpoint', help='Checkpoint file to resume from')
    parser.add_argument('--iterations', type=int, default=100, 
                       help='Additional iterations to train')
    parser.add_argument('--new-save-path', type=str, default=None,
                       help='New save path (default: overwrite original)')
    parser.add_argument('--step-size', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--noise-std', type=float, default=None,
                       help='Override exploration noise')
    parser.add_argument('--show-info', action='store_true',
                       help='Just show checkpoint info and exit')
    
    args = parser.parse_args()
    
    # Check checkpoint
    info = check_checkpoint(args.checkpoint)
    if info is None:
        print(f"Error: Cannot load checkpoint '{args.checkpoint}'")
        sys.exit(1)
    
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Current iterations: {info['iterations']}")
    print(f"Network dimensions: obs_dim={info['obs_dim']}, act_dim={info['act_dim']}")
    
    if 'meta' in info and info['meta']:
        print("\nSaved hyperparameters:")
        for key, value in info['meta'].items():
            if key != 'xml_path':
                print(f"  {key}: {value}")
    
    if args.show_info:
        print("=" * 60)
        sys.exit(0)
    
    print(f"\nResuming training for {args.iterations} more iterations...")
    print(f"Total iterations after training: {info['iterations'] + args.iterations}")
    print("=" * 60)
    
    # Build command
    cmd = [
        "python", "mjx_ars_train.py",
        "--xml", "quadruped.xml",
        "--save-path", args.new_save_path or args.checkpoint,
        "--resume",
        "--iterations", str(args.iterations),
    ]
    
    # Add overrides if specified
    if args.step_size is not None:
        cmd.extend(["--step-size", str(args.step_size)])
        print(f"Overriding step-size: {args.step_size}")
    
    if args.noise_std is not None:
        cmd.extend(["--noise-std", str(args.noise_std)])
        print(f"Overriding noise-std: {args.noise_std}")
    
    # Use saved hyperparameters if available
    if 'meta' in info and info['meta']:
        meta = info['meta']
        # Important parameters to preserve
        preserve_params = [
            'num_envs', 'num_dirs', 'top_dirs', 'episode_length', 'action_repeat',
            'crouch_init_ratio', 'knee_band_low', 'knee_band_high', 'target_z_low', 'target_z_high'
        ]
        
        for param in preserve_params:
            if param in meta:
                param_flag = '--' + param.replace('_', '-')
                cmd.extend([param_flag, str(meta[param])])
    
    print("\nExecuting command:")
    print(" ".join(cmd))
    print("")
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("✓ Training resumed successfully!")
        
        # Check new checkpoint
        new_info = check_checkpoint(args.new_save_path or args.checkpoint)
        if new_info:
            print(f"New total iterations: {new_info['iterations']}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n✗ Training interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()