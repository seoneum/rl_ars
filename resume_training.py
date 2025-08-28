#!/usr/bin/env python3
"""
Resume training from a checkpoint with flexible options.
This script is safe for the A100/JAX environment as it doesn't set
any environment variables, relying on the shell to do so.
"""
import os
import sys
import argparse
import subprocess
import numpy as np

def check_checkpoint(path):
    """Loads a checkpoint and returns its metadata."""
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        info = {
            'iterations': int(data['iter']) if 'iter' in data else 0,
            'obs_dim': int(data['obs_dim']),
            'act_dim': int(data['act_dim']),
            'meta': None,
        }
        if 'meta' in data:
            meta_data = data['meta'].item()
            if isinstance(meta_data, dict):
                info['meta'] = meta_data
        return info
    except Exception as e:
        print(f"Error: Could not load or parse checkpoint '{path}': {e}")
        return None

def main():
    """Main function to parse arguments and resume training."""
    parser = argparse.ArgumentParser(
        description='Resume ARS training from a checkpoint.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('checkpoint', help='Checkpoint file (.npz) to resume from.')
    parser.add_argument('--iterations', type=int, default=100, help='Number of additional iterations to train.')
    parser.add_argument('--new-save-path', type=str, default=None, help='New path to save the updated checkpoint. Defaults to overwriting the original.')
    parser.add_argument('--step-size', type=float, default=None, help='Override the learning rate (step size).')
    parser.add_argument('--noise-std', type=float, default=None, help='Override the exploration noise standard deviation.')
    parser.add_argument('--show-info', action='store_true', help='Display checkpoint info and exit without training.')
    args = parser.parse_args()

    info = check_checkpoint(args.checkpoint)
    if info is None:
        sys.exit(1)

    print("=" * 60)
    print(f"Checkpoint Information: {args.checkpoint}")
    print(f"  - Current Iterations: {info['iterations']}")
    print(f"  - Network Dimensions: obs_dim={info['obs_dim']}, act_dim={info['act_dim']}")
    if info['meta']:
        print("\n  Saved Hyperparameters:")
        for k, v in info['meta'].items():
            if k != 'xml_path':
                print(f"    - {k}: {v}")
    print("=" * 60)

    if args.show_info:
        sys.exit(0)

    print(f"\nResuming training for {args.iterations} more iterations...")
    print(f"Total iterations after this run: {info['iterations'] + args.iterations}")
    print("-" * 60)

    # Base command
    cmd = [
        "python", "mjx_ars_train.py",
        "--xml", "quadruped.xml",
        "--save-path", args.new_save_path or args.checkpoint,
        "--resume",
        "--iterations", str(args.iterations),
    ]

    # Override hyperparameters if provided
    if args.step_size is not None:
        cmd.extend(["--step-size", str(args.step_size)])
        print(f"Overriding step-size to: {args.step_size}")
    if args.noise_std is not None:
        cmd.extend(["--noise-std", str(args.noise_std)])
        print(f"Overriding noise-std to: {args.noise_std}")

    # Preserve key hyperparameters from the checkpoint
    if info['meta']:
        meta = info['meta']
        preserve_keys = [
            'num_envs', 'num_dirs', 'top_dirs', 'episode_length', 'action_repeat',
            'crouch_init_ratio', 'knee_band_low', 'knee_band_high',
            'target_z_low', 'target_z_high'
        ]
        print("Preserving hyperparameters from checkpoint:")
        for key in preserve_keys:
            if key in meta:
                arg_key = "--" + key.replace("_", "-")
                cmd.extend([arg_key, str(meta[key])])
                print(f"  {arg_key}: {meta[key]}")

    print("\nExecuting command:\n" + " ".join(cmd) + "\n")

    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("✓ Training resumed and completed successfully!")
        final_path = args.new_save_path or args.checkpoint
        new_info = check_checkpoint(final_path)
        if new_info:
            print(f"New total iterations in '{final_path}': {new_info['iterations']}")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code {e.returncode}.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n✗ Training interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()
