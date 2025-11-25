"""
Evaluation and inference script for PhaseNet
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import seisbench.models as sbm
from obspy import read


def load_model(checkpoint_path: str):
    """
    Load trained PhaseNet model from checkpoint
    
    Args:
        checkpoint_path: path to model checkpoint
    
    Returns:
        Loaded model
    """
    from model import PhaseNetLightning
    
    model = PhaseNetLightning.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    return model


def predict_on_waveform(model, waveform: np.ndarray, sampling_rate: int = 100):
    """
    Predict phase picks on a single waveform
    
    Args:
        model: trained PhaseNet model
        waveform: numpy array of shape (3, n_samples) for Z, N, E
        sampling_rate: sampling rate in Hz
    
    Returns:
        predictions: dict with P and S pick probabilities
    """
    # Convert to tensor
    waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        predictions = model(waveform_tensor)
    
    # Convert to numpy
    predictions = predictions.squeeze(0).cpu().numpy()
    
    return {
        'P': predictions[0, :],
        'S': predictions[1, :],
        'N': predictions[2, :]
    }


def find_picks(probabilities: np.ndarray, threshold: float = 0.5, 
               min_distance: int = 50) -> List[int]:
    """
    Find phase picks from probability array
    
    Args:
        probabilities: 1D array of phase probabilities
        threshold: minimum probability threshold
        min_distance: minimum distance between picks in samples
    
    Returns:
        List of pick indices
    """
    from scipy.signal import find_peaks
    
    peaks, properties = find_peaks(
        probabilities,
        height=threshold,
        distance=min_distance
    )
    
    return peaks.tolist()


def plot_predictions(waveform: np.ndarray, predictions: Dict[str, np.ndarray],
                     picks: Dict[str, List[int]] = None, 
                     sampling_rate: int = 100,
                     save_path: str = None):
    """
    Plot waveform with predicted probabilities and picks
    
    Args:
        waveform: numpy array of shape (3, n_samples)
        predictions: dict with P, S, N probabilities
        picks: dict with P and S pick indices
        sampling_rate: sampling rate in Hz
        save_path: path to save figure
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    
    time = np.arange(waveform.shape[1]) / sampling_rate
    
    # Plot waveforms
    channels = ['Z', 'N', 'E']
    for i in range(3):
        axes[0].plot(time, waveform[i, :], label=channels[i], alpha=0.7)
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Waveforms')
    axes[0].grid(True, alpha=0.3)
    
    # Plot P probability
    axes[1].plot(time, predictions['P'], 'b-', linewidth=2)
    axes[1].set_ylabel('P Probability')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)
    if picks and 'P' in picks:
        for pick in picks['P']:
            axes[1].axvline(pick / sampling_rate, color='b', linestyle='--', alpha=0.7)
    
    # Plot S probability
    axes[2].plot(time, predictions['S'], 'r-', linewidth=2)
    axes[2].set_ylabel('S Probability')
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3)
    if picks and 'S' in picks:
        for pick in picks['S']:
            axes[2].axvline(pick / sampling_rate, color='r', linestyle='--', alpha=0.7)
    
    # Plot Noise probability
    axes[3].plot(time, predictions['N'], 'g-', linewidth=2)
    axes[3].set_ylabel('Noise Probability')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylim([0, 1])
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_model(model, test_dataloader, device='cpu'):
    """
    Evaluate model on test dataset
    
    Args:
        model: trained model
        test_dataloader: DataLoader with test data
        device: device to run evaluation on
    
    Returns:
        metrics: dict with evaluation metrics
    """
    model.eval()
    model.to(device)
    
    all_losses = []
    all_accuracies = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            predictions = model(x)
            loss = torch.nn.functional.cross_entropy(predictions, y)
            
            preds = torch.argmax(predictions, dim=1)
            targets = torch.argmax(y, dim=1)
            acc = (preds == targets).float().mean()
            
            all_losses.append(loss.item())
            all_accuracies.append(acc.item())
    
    metrics = {
        'test_loss': np.mean(all_losses),
        'test_accuracy': np.mean(all_accuracies),
        'test_loss_std': np.std(all_losses),
        'test_accuracy_std': np.std(all_accuracies)
    }
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate PhaseNet model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to test data")
    parser.add_argument("--output", type=str, default="results/evaluation",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    print("Evaluation script - implement based on your specific needs")
