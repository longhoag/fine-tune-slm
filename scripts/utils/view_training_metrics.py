#!/usr/bin/env python3
"""
View training metrics from TensorBoard logs stored in S3.

This script:
1. Lists available training runs in S3
2. Downloads TensorBoard event files for selected run
3. Extracts and visualizes training/validation metrics
4. Generates plots: loss curves, learning rate schedule, etc.

Usage:
    # View latest training metrics
    poetry run python scripts/utils/view_training_metrics.py
    
    # View specific training run
    poetry run python scripts/utils/view_training_metrics.py --timestamp 20251111_022951
    
    # Save plots to file instead of displaying
    poetry run python scripts/utils/view_training_metrics.py --save-plots
    
    # Export metrics to CSV
    poetry run python scripts/utils/view_training_metrics.py --export-csv metrics.csv
"""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.utils.config import load_all_configs
from src.utils.aws_helpers import AWSClient, S3Manager


def list_training_runs(s3_manager: S3Manager, bucket: str, prefix: str) -> List[Dict]:
    """List available training runs in S3."""
    logger.info(f"Listing training runs in s3://{bucket}/{prefix}/")
    
    try:
        response = s3_manager.client.s3.list_objects_v2(
            Bucket=bucket,
            Prefix=f"{prefix}/",
            Delimiter='/'
        )
        
        runs = []
        if 'CommonPrefixes' in response:
            for prefix_info in response['CommonPrefixes']:
                full_prefix = prefix_info['Prefix']
                parts = full_prefix.rstrip('/').split('/')
                timestamp = parts[-1]
                
                # Verify timestamp format
                if '_' in timestamp and len(timestamp) == 15:
                    from datetime import datetime
                    try:
                        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                        
                        # Check if logs exist for this run
                        logs_prefix = f"{full_prefix}logs/"
                        logs_response = s3_manager.client.s3.list_objects_v2(
                            Bucket=bucket,
                            Prefix=logs_prefix,
                            MaxKeys=1
                        )
                        
                        if 'Contents' in logs_response:
                            runs.append({
                                'timestamp': timestamp,
                                'datetime': dt,
                                's3_prefix': full_prefix.rstrip('/'),
                                'logs_prefix': logs_prefix,
                                'formatted_date': dt.strftime("%Y-%m-%d %H:%M:%S")
                            })
                    except ValueError:
                        pass
        
        runs.sort(key=lambda x: x['datetime'], reverse=True)
        return runs
        
    except Exception as e:
        logger.error(f"Failed to list training runs: {e}")
        return []


def download_logs(s3_manager: S3Manager, bucket: str, logs_prefix: str, local_dir: Path) -> bool:
    """Download TensorBoard logs from S3."""
    logger.info(f"Downloading logs from s3://{bucket}/{logs_prefix}")
    
    try:
        response = s3_manager.client.s3.list_objects_v2(
            Bucket=bucket,
            Prefix=logs_prefix
        )
        
        if 'Contents' not in response:
            logger.error("No log files found in S3")
            return False
        
        local_dir.mkdir(parents=True, exist_ok=True)
        
        for obj in response['Contents']:
            s3_key = obj['Key']
            rel_path = s3_key.replace(logs_prefix, '')
            
            if not rel_path:
                continue
            
            local_file = local_dir / rel_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.debug(f"Downloading: {rel_path}")
            s3_manager.client.s3.download_file(bucket, s3_key, str(local_file))
        
        logger.success(f"✅ Downloaded {len(response['Contents'])} log files")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download logs: {e}")
        return False


def parse_tensorboard_logs(log_dir: Path) -> Dict[str, List]:
    """Parse TensorBoard event files and extract metrics."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        logger.error("tensorboard not installed. Install with: poetry add tensorboard --group dev")
        return {}
    
    logger.info("Parsing TensorBoard event files...")
    
    # Find event files
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        logger.error("No TensorBoard event files found")
        return {}
    
    logger.info(f"Found {len(event_files)} event file(s)")
    
    # Load events
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    
    # Extract only important scalar metrics (filter out noise)
    important_metrics = {
        'train/loss', 'eval/loss', 'train/learning_rate', 'train/epoch'
    }
    
    metrics = {}
    for tag in ea.Tags()['scalars']:
        # Only keep important metrics
        if tag in important_metrics:
            events = ea.Scalars(tag)
            # Sort by step to ensure chronological order (important when reading from multiple event files)
            sorted_events = sorted(events, key=lambda e: e.step)
            metrics[tag] = {
                'steps': [e.step for e in sorted_events],
                'values': [e.value for e in sorted_events],
                'wall_times': [e.wall_time for e in sorted_events]
            }
    
    logger.success(f"✅ Extracted {len(metrics)} metric(s)")
    logger.info(f"   Metrics: {', '.join(metrics.keys())}")
    return metrics


def plot_metrics(metrics: Dict[str, List], save_path: Path = None):
    """Plot training metrics with combined loss plot."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.error("matplotlib not installed. Install with: poetry add matplotlib --group dev")
        return
    
    if not metrics:
        logger.warning("No metrics to plot")
        return
    
    # Separate loss metrics from other metrics
    train_loss = None
    eval_loss = None
    other_metrics = {}
    
    for name, data in metrics.items():
        if 'train/loss' in name or name == 'loss':
            train_loss = (name, data)
        elif 'eval/loss' in name or 'eval_loss' in name:
            eval_loss = (name, data)
        else:
            other_metrics[name] = data
    
    # Determine number of plots needed
    n_plots = len(other_metrics)
    if train_loss or eval_loss:
        n_plots += 1  # Combined loss plot
    if train_loss and eval_loss:
        n_plots += 1  # Gap plot
    
    if n_plots == 0:
        logger.warning("No metrics to plot")
        return
    
    # Create figure with subplots
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot combined training and eval loss
    if train_loss or eval_loss:
        ax = axes[plot_idx]
        plot_idx += 1
        
        # Plot training loss (BLUE)
        if train_loss:
            name, data = train_loss
            steps = data['steps']
            values = data['values']
            
            # Data is already sorted by step at extraction time
            ax.plot(steps, values, linewidth=2, color='#2E86DE',
                   label='Training Loss', alpha=0.8)
            
            # Annotate minimum training loss (move further up to avoid overlap with val final)
            min_val = min(values)
            min_idx = values.index(min_val)
            ax.plot(steps[min_idx], min_val, 'o', color='#2E86DE', markersize=8)
            ax.annotate(f'Train Min: {min_val:.4f}',
                       xy=(steps[min_idx], min_val),
                       xytext=(10, 35), textcoords='offset points',
                       fontsize=9, color='#2E86DE',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='#2E86DE', alpha=0.9))
            
            # Annotate final training loss (bottom-right)
            final_val = values[-1]
            ax.annotate(f'Train Final: {final_val:.4f}',
                       xy=(steps[-1], final_val),
                       xytext=(-10, -35), textcoords='offset points',
                       fontsize=9, color='#2E86DE', ha='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='#2E86DE', alpha=0.9))
        
        # Plot eval loss (RED)
        if eval_loss:
            name, data = eval_loss
            steps = data['steps']
            values = data['values']
            
            # Data is already sorted by step at extraction time
            ax.plot(steps, values, linewidth=2.5, color='#E74C3C',
                   label='Validation Loss', marker='o', markersize=5, alpha=0.9)
            
            # Annotate minimum eval loss (top-right position)
            min_val = min(values)
            min_idx = values.index(min_val)
            ax.plot(steps[min_idx], min_val, 'o', color='#E74C3C', markersize=10)
            ax.annotate(f'Val Min: {min_val:.4f}',
                       xy=(steps[min_idx], min_val),
                       xytext=(-10, 25), textcoords='offset points',
                       fontsize=9, color='#E74C3C', ha='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='#E74C3C', alpha=0.9))
            
            # Annotate final eval loss (top-right)
            final_val = values[-1]
            ax.annotate(f'Val Final: {final_val:.4f}',
                       xy=(steps[-1], final_val),
                       xytext=(-10, 15), textcoords='offset points',
                       fontsize=9, color='#E74C3C', ha='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='#E74C3C', alpha=0.9))
        
        ax.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax.set_title('Training and Validation Loss', fontsize=13, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
        
        # Add improvement percentage if both losses exist
        if train_loss and eval_loss:
            train_improvement = ((train_loss[1]['values'][0] - train_loss[1]['values'][-1]) / 
                               train_loss[1]['values'][0]) * 100
            eval_improvement = ((eval_loss[1]['values'][0] - eval_loss[1]['values'][-1]) / 
                              eval_loss[1]['values'][0]) * 100
            
            ax.text(0.02, 0.50, 
                   f'Train: {train_improvement:.1f}%↓\nVal: {eval_improvement:.1f}%↓',
                   transform=ax.transAxes, fontsize=10, fontweight='bold',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # Calculate and plot train-val loss gap
    if train_loss and eval_loss:
        ax = axes[plot_idx]
        plot_idx += 1
        
        # Get eval steps and values
        eval_steps = np.array(eval_loss[1]['steps'])
        eval_values = np.array(eval_loss[1]['values'])
        
        # Filter out initial evaluation steps (often at steps 1-10 during initialization)
        # Keep only steps >= 100 for cleaner visualization (regular eval intervals)
        mask = eval_steps >= 100
        eval_steps_filtered = eval_steps[mask]
        eval_values_filtered = eval_values[mask]
        
        # Interpolate train loss at eval steps
        train_steps = np.array(train_loss[1]['steps'])
        train_values = np.array(train_loss[1]['values'])
        train_at_eval_steps = np.interp(eval_steps_filtered, train_steps, train_values)
        
        # Calculate gap (eval - train)
        loss_gap = eval_values_filtered - train_at_eval_steps
        
        # Plot gap as line chart with markers (better for time series)
        colors = ['#27AE60' if gap > 0 else '#E74C3C' for gap in loss_gap]
        ax.plot(eval_steps_filtered, loss_gap, linewidth=2, color='#9B59B6',
                marker='o', markersize=6, alpha=0.8, label='Loss Gap')
        # Color individual markers
        for step, gap, color in zip(eval_steps_filtered, loss_gap, colors):
            ax.plot(step, gap, 'o', color=color, markersize=6, alpha=0.9)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Training Steps', fontsize=10, fontweight='bold')
        ax.set_ylabel('Loss Gap (Eval - Train)', fontsize=10)
        ax.set_title('Train-Validation Loss Gap', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add interpretation annotation
        mean_gap = np.mean(loss_gap)
        final_gap = loss_gap[-1]
        trend = "stable" if abs(final_gap - loss_gap[0]) < 0.1 else ("increasing" if final_gap > loss_gap[0] else "decreasing")
        
        ax.annotate(f'Mean: {mean_gap:.4f}\nFinal: {final_gap:.4f}\nTrend: {trend}',
                   xy=(0.02, 0.98), xycoords='axes fraction',
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    
    # Plot other metrics
    # Clean metric names for better display
    metric_display_names = {
        'train/learning_rate': 'Learning Rate Schedule',
        'train/epoch': 'Training Epoch Progress'
    }
    
    for name, data in other_metrics.items():
        ax = axes[plot_idx]
        plot_idx += 1
        
        steps = data['steps']
        values = data['values']
        
        # Data is already sorted by step at extraction time
        # Choose color and style based on metric type
        if 'learning_rate' in name:
            # Line plot for learning rate (show all data points for detail)
            color = '#F39C12'
            ax.plot(steps, values, linewidth=2, color=color, alpha=0.8)
            ax.fill_between(steps, values, alpha=0.3, color=color)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Show peak learning rate
            final_val = values[-1]
            max_val = max(values)
            ax.text(0.98, 0.98, f'Peak: {max_val:.2e}\nFinal: {final_val:.2e}',
                   transform=ax.transAxes, fontsize=9, ha='right', va='top',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
        else:
            # Line graph for epoch progress
            color = '#3498DB'
            ax.plot(steps, values, linewidth=2, color=color, alpha=0.8)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # For epoch, show final value
            final_val = values[-1]
            ax.text(0.98, 0.98, f'Final: {final_val:.1f} epochs',
                   transform=ax.transAxes, fontsize=9, ha='right', va='top',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
        
        ax.set_xlabel('Training Steps', fontsize=10, fontweight='bold')
        
        # Use clean display name
        display_name = metric_display_names.get(name, name)
        ax.set_ylabel(display_name, fontsize=10)
        ax.set_title(display_name, fontsize=12, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.success(f"✅ Saved plot to: {save_path}")
    else:
        logger.info("Displaying plot (close window to continue)...")
        plt.show()


def export_to_csv(metrics: Dict[str, List], output_path: Path):
    """Export metrics to CSV file."""
    import csv
    
    logger.info(f"Exporting metrics to {output_path}")
    
    # Collect all unique steps
    all_steps = set()
    for data in metrics.values():
        all_steps.update(data['steps'])
    all_steps = sorted(all_steps)
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['step'] + list(metrics.keys())
        writer.writerow(header)
        
        # Data rows
        for step in all_steps:
            row = [step]
            for name, data in metrics.items():
                # Find value for this step
                try:
                    idx = data['steps'].index(step)
                    row.append(data['values'][idx])
                except ValueError:
                    row.append('')  # Missing value
            writer.writerow(row)
    
    logger.success(f"✅ Exported to: {output_path}")


def print_summary(metrics: Dict[str, List]):
    """Print summary statistics."""
    logger.info("\n" + "="*60)
    logger.info("Training Metrics Summary")
    logger.info("="*60)
    
    for name, data in metrics.items():
        values = data['values']
        if not values:
            continue
        
        import statistics
        
        min_val = min(values)
        max_val = max(values)
        final_val = values[-1]
        mean_val = statistics.mean(values)
        
        logger.info(f"\n{name}:")
        logger.info(f"  Final:   {final_val:.6f}")
        logger.info(f"  Min:     {min_val:.6f}")
        logger.info(f"  Max:     {max_val:.6f}")
        logger.info(f"  Mean:    {mean_val:.6f}")
        
        if 'loss' in name.lower() and len(values) > 1:
            improvement = ((values[0] - final_val) / values[0]) * 100
            logger.info(f"  Improvement: {improvement:.1f}%")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="View training metrics from TensorBoard logs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Configuration directory"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        help="Specific training run timestamp (e.g., '20251111_022951')"
    )
    parser.add_argument(
        "--save-plots",
        type=str,
        metavar="PATH",
        help="Save plots to file instead of displaying"
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        metavar="PATH",
        help="Export metrics to CSV file"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available training runs and exit"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("View Training Metrics")
    logger.info("="*60)
    
    try:
        # Load configs
        configs = load_all_configs(args.config_dir, use_ssm=True)
        
        region = configs.get_aws('aws.region')
        s3_bucket = configs.get_training('output.s3_bucket')
        s3_prefix = configs.get_training('output.s3_prefix')
        
        # Initialize AWS clients
        aws_client = AWSClient(region=region)
        s3_manager = S3Manager(aws_client)
        
        # List training runs
        logger.info("\n📦 Finding training runs...")
        runs = list_training_runs(s3_manager, s3_bucket, s3_prefix)
        
        if not runs:
            logger.error("No training runs with logs found in S3")
            logger.info(f"Expected: s3://{s3_bucket}/{s3_prefix}/YYYYMMDD_HHMMSS/logs")
            sys.exit(1)
        
        logger.success(f"Found {len(runs)} training run(s) with logs:")
        for i, run in enumerate(runs, 1):
            marker = "← LATEST" if i == 1 else ""
            logger.info(f"  {i}. {run['timestamp']} ({run['formatted_date']}) {marker}")
        
        if args.list:
            sys.exit(0)
        
        # Select run
        if args.timestamp:
            selected = None
            for run in runs:
                if run['timestamp'] == args.timestamp:
                    selected = run
                    break
            
            if not selected:
                logger.error(f"Timestamp not found: {args.timestamp}")
                sys.exit(1)
        else:
            selected = runs[0]
            logger.info(f"\n✅ Using latest: {selected['timestamp']}")
        
        # Download logs
        temp_dir = Path(tempfile.mkdtemp(prefix="tb_logs_"))
        logger.info(f"\n📥 Downloading logs...")
        
        success = download_logs(
            s3_manager=s3_manager,
            bucket=s3_bucket,
            logs_prefix=selected['logs_prefix'],
            local_dir=temp_dir
        )
        
        if not success:
            sys.exit(1)
        
        # Parse logs
        metrics = parse_tensorboard_logs(temp_dir)
        
        if not metrics:
            logger.error("Failed to extract metrics from logs")
            sys.exit(1)
        
        # Print summary
        print_summary(metrics)
        
        # Export CSV if requested
        if args.export_csv:
            export_to_csv(metrics, Path(args.export_csv))
        
        # Plot metrics
        if args.save_plots or not args.export_csv:
            save_path = Path(args.save_plots) if args.save_plots else None
            plot_metrics(metrics, save_path)
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        logger.success("\n✅ Complete!")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
