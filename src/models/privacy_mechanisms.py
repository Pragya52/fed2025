"""
Privacy Mechanisms: Gaussian Noise and Quantization
Implementation based on the methodology paper
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
import math

class GaussianNoiseMechanism:
    """
    Implements Gaussian noise addition for differential privacy
    Based on the Gaussian mechanism for (ε, δ)-differential privacy
    """
    
    def __init__(self, sigma: float, device: str = 'cuda'):
        """
        Args:
            sigma: Standard deviation of Gaussian noise
            device: Device to perform computations on
        """
        self.sigma = sigma
        self.device = device
        
    def add_noise(self, data: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to input data
        
        Args:
            data: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Noisy data tensor
        """
        if self.sigma <= 0:
            return data
            
        # Generate Gaussian noise with same shape as data
        noise = torch.randn_like(data, device=self.device) * self.sigma
        noisy_data = data + noise
        
        return noisy_data
    
    def compute_privacy_parameters(self, delta: float, sensitivity: float) -> float:
        """
        Compute epsilon for (ε, δ)-differential privacy
        
        Args:
            delta: Privacy parameter δ
            sensitivity: L2-sensitivity of the function
            
        Returns:
            Privacy parameter ε
        """
        if self.sigma <= 0:
            return float('inf')
            
        epsilon = (sensitivity * math.sqrt(2 * math.log(1.25 / delta))) / self.sigma
        return epsilon

class QuantizationMechanism:
    """
    Implements quantization for communication efficiency and additional privacy
    using the roundσ function.
    """
    
    def __init__(self, num_bits: int = 8, device: str = 'cuda', k: float = 10.0):
        """
        Args:
            num_bits: Number of bits for quantization (e.g., 8 for int8)
            device: Device to perform computations on
            k: Steepness parameter for sigmoid function in roundσ
        """
        self.num_bits = num_bits
        self.device = device
        self.k = k  # Steepness parameter for sigmoid
        self.max_value = 2**(num_bits - 1) - 1  # e.g., 127 for int8
        self.min_value = -2**(num_bits - 1)     # e.g., -128 for int8
        
    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute sigmoid function σ(x) = 1 / (1 + e^(-x))
        """
        return torch.sigmoid(x)
    
    def quantize(self, data: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Quantize floating-point data to integers using roundσ function
        
        Args:
            data: Input tensor to quantize
            
        Returns:
            Tuple of (quantized_data, scaling_parameters)
        """
        # Compute min and max values for scaling
        data_min = torch.min(data)
        data_max = torch.max(data)
        
        # Avoid division by zero
        if torch.abs(data_max - data_min) < 1e-8:
            data_max = data_min + 1e-8
        
        # Scale data to [0, 2^b - 1] range
        scaled_data = (data - data_min) / (data_max - data_min) * (2**self.num_bits - 1)
        
        # Apply roundσ(x, k) = Σ_n n [σ(k(x - (n - 0.5))) - σ(k(x - (n + 0.5)))]
        n = torch.arange(0, 2**self.num_bits, device=self.device)
        n = n.view(1, -1)  # Shape: (1, 2^num_bits)
        
        # Compute terms for roundσ
        x = scaled_data.unsqueeze(-1)  # Add dimension for broadcasting
        term1 = self._sigmoid(self.k * (x - (n)))
        term2 = self._sigmoid(self.k * (x - (n + 1)))
        quantized_data = torch.sum(n * (term1 - term2), dim=-1)
        
        # Clamp to valid range and convert to integer
        quantized_data = torch.round(quantized_data).clamp(0, 2**self.num_bits - 1)
        
        # Convert to appropriate integer type
        if self.num_bits == 8:
            quantized_data = quantized_data.to(torch.int8) + self.min_value
        elif self.num_bits == 16:
            quantized_data = quantized_data.to(torch.int16) + self.min_value
        else:
            quantized_data = quantized_data.to(torch.int32) + self.min_value
        
        # Store scaling parameters for dequantization
        scaling_params = {
            'data_min': data_min,
            'data_max': data_max,
            'scale': (data_max - data_min) / (2**self.num_bits - 1)
        }
        
        return quantized_data, scaling_params
    
    def dequantize(self, quantized_data: torch.Tensor, scaling_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Dequantize integer data back to floating-point
        
        Args:
            quantized_data: Quantized integer tensor
            scaling_params: Scaling parameters from quantization
            
        Returns:
            Dequantized floating-point tensor
        """
        # Convert back to float and remove offset
        float_data = quantized_data.float() - self.min_value
        
        # Scale back to original range
        data_min = scaling_params['data_min']
        data_max = scaling_params['data_max']
        
        dequantized_data = (float_data / (2**self.num_bits - 1)) * (data_max - data_min) + data_min
        
        return dequantized_data
    
    def compute_quantization_error(self, original: torch.Tensor, dequantized: torch.Tensor) -> Dict[str, float]:
        """
        Compute quantization error metrics
        
        Args:
            original: Original floating-point tensor
            dequantized: Dequantized tensor
            
        Returns:
            Dictionary of error metrics
        """
        mse = torch.mean((original - dequantized) ** 2).item()
        mae = torch.mean(torch.abs(original - dequantized)).item()
        max_error = torch.max(torch.abs(original - dequantized)).item()
        
        # Signal-to-noise ratio
        signal_power = torch.mean(original ** 2).item()
        snr = 10 * math.log10(signal_power / (mse + 1e-8))
        
        return {
            'mse': mse,
            'mae': mae,
            'max_error': max_error,
            'snr_db': snr
        }

class PrivacyPreservingTransmission:
    """
    Combines Gaussian noise and quantization for privacy-preserving transmission
    """
    
    def __init__(self, sigma: float = 0.1, num_bits: int = 8, device: str = 'cuda'):
        """
        Args:
            sigma: Standard deviation for Gaussian noise
            num_bits: Quantization bits
            device: Device to perform computations on
        """
        self.gaussian_mechanism = GaussianNoiseMechanism(sigma, device)
        self.quantization_mechanism = QuantizationMechanism(num_bits, device)
        self.device = device
        
    def prepare_transmission(self, data: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Prepare data for transmission: add noise + quantize
        
        Args:
            data: Original smashed data from client
            
        Returns:
            Tuple of (quantized_noisy_data, scaling_parameters)
        """
        # Step 1: Add Gaussian noise
        noisy_data = self.gaussian_mechanism.add_noise(data)
        
        # Step 2: Quantize noisy data
        quantized_data, scaling_params = self.quantization_mechanism.quantize(noisy_data)
        
        return quantized_data, scaling_params
    
    def reconstruct_transmission(self, quantized_data: torch.Tensor, 
                               scaling_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct data from transmission: dequantize
        Note: Noise cannot be removed, providing privacy protection
        
        Args:
            quantized_data: Received quantized data
            scaling_params: Scaling parameters for dequantization
            
        Returns:
            Reconstructed (noisy) data
        """
        # Dequantize the received data
        reconstructed_data = self.quantization_mechanism.dequantize(quantized_data, scaling_params)
        
        return reconstructed_data
    
    def compute_communication_cost(self, original_shape: Tuple[int, ...]) -> Dict[str, int]:
        """
        Compute communication cost in bits
        
        Args:
            original_shape: Shape of original tensor
            
        Returns:
            Dictionary with communication costs
        """
        num_elements = np.prod(original_shape)
        
        # Original cost (32-bit floats)
        original_bits = num_elements * 32
        
        # Quantized cost
        quantized_bits = num_elements * self.quantization_mechanism.num_bits
        
        # Scaling parameters cost (min, max values as 32-bit floats)
        scaling_bits = 2 * 32  # data_min and data_max
        
        total_quantized_bits = quantized_bits + scaling_bits
        
        compression_ratio = original_bits / total_quantized_bits
        
        return {
            'original_bits': original_bits,
            'quantized_bits': total_quantized_bits,
            'compression_ratio': compression_ratio,
            'bandwidth_savings': 1 - (total_quantized_bits / original_bits)
        }
    
    def analyze_privacy_utility_tradeoff(self, original_data: torch.Tensor, 
                                       reconstructed_data: torch.Tensor,
                                       delta: float = 1e-5) -> Dict[str, float]:
        """
        Analyze privacy-utility tradeoff
        
        Args:
            original_data: Original data before privacy mechanisms
            reconstructed_data: Data after noise + quantization + dequantization
            delta: Privacy parameter δ
            
        Returns:
            Dictionary with privacy and utility metrics
        """
        # Utility metrics (quantization + noise error)
        utility_metrics = self.quantization_mechanism.compute_quantization_error(
            original_data, reconstructed_data
        )
        
        # Privacy metrics
        sensitivity = torch.norm(original_data, p=2).item()  # L2 sensitivity
        epsilon = self.gaussian_mechanism.compute_privacy_parameters(delta, sensitivity)
        
        # Privacy budget per round
        privacy_metrics = {
            'epsilon': epsilon,
            'delta': delta,
            'sigma': self.gaussian_mechanism.sigma,
            'sensitivity': sensitivity
        }
        
        return {**utility_metrics, **privacy_metrics}

class DifferentialPrivacyAccountant:
    """
    Tracks privacy budget consumption across training rounds
    """
    
    def __init__(self, total_epsilon: float, delta: float):
        """
        Args:
            total_epsilon: Total privacy budget
            delta: Privacy parameter δ
        """
        self.total_epsilon = total_epsilon
        self.delta = delta
        self.consumed_epsilon = 0.0
        self.round_epsilons = []
        
    def add_round_epsilon(self, round_epsilon: float) -> bool:
        """
        Add epsilon consumption for current round
        
        Args:
            round_epsilon: Epsilon consumed in current round
            
        Returns:
            True if within budget, False if budget exceeded
        """
        if self.consumed_epsilon + round_epsilon > self.total_epsilon:
            return False
            
        self.consumed_epsilon += round_epsilon
        self.round_epsilons.append(round_epsilon)
        return True
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget"""
        return self.total_epsilon - self.consumed_epsilon
    
    def compute_advanced_composition(self, num_rounds: int, round_epsilon: float) -> float:
        """
        Compute total epsilon using advanced composition theorem
        
        Args:
            num_rounds: Number of training rounds
            round_epsilon: Epsilon per round
            
        Returns:
            Total epsilon under advanced composition
        """
        if round_epsilon == 0:
            return 0.0
            
        # Advanced composition bound
        term1 = math.sqrt(2 * num_rounds * math.log(1/self.delta)) * round_epsilon
        term2 = num_rounds * round_epsilon * (math.exp(round_epsilon) - 1) / 2
        
        return term1 + term2
    
    def get_privacy_analysis(self) -> Dict[str, float]:
        """Get comprehensive privacy analysis"""
        num_rounds = len(self.round_epsilons)
        
        if num_rounds == 0:
            return {
                'total_epsilon': 0.0,
                'consumed_epsilon': 0.0,
                'remaining_epsilon': self.total_epsilon,
                'num_rounds': 0
            }
        
        # Basic composition
        basic_composition = sum(self.round_epsilons)
        
        # Advanced composition (if uniform epsilon per round)
        if len(set(self.round_epsilons)) == 1:  # All rounds have same epsilon
            round_epsilon = self.round_epsilons[0]
            advanced_composition = self.compute_advanced_composition(num_rounds, round_epsilon)
        else:
            advanced_composition = basic_composition  # Fallback to basic
        
        return {
            'total_epsilon_budget': self.total_epsilon,
            'consumed_epsilon_basic': basic_composition,
            'consumed_epsilon_advanced': advanced_composition,
            'remaining_epsilon': self.total_epsilon - self.consumed_epsilon,
            'num_rounds': num_rounds,
            'average_epsilon_per_round': np.mean(self.round_epsilons),
            'privacy_exhausted': self.consumed_epsilon >= self.total_epsilon
        }

# Test functions
def test_privacy_mechanisms():
    """Test privacy mechanisms with sample data"""
    
    # Create sample data
    batch_size, seq_len, embed_dim = 4, 10, 256
    sample_data = torch.randn(batch_size, seq_len, embed_dim)
    
    # Initialize privacy mechanism
    privacy_mechanism = PrivacyPreservingTransmission(sigma=0.1, num_bits=8)
    
    # Test transmission
    print("Testing Privacy-Preserving Transmission...")
    quantized_data, scaling_params = privacy_mechanism.prepare_transmission(sample_data)
    reconstructed_data = privacy_mechanism.reconstruct_transmission(quantized_data, scaling_params)
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Quantized shape: {quantized_data.shape}")
    print(f"Reconstructed shape: {reconstructed_data.shape}")
    
    # Analyze privacy-utility tradeoff
    analysis = privacy_mechanism.analyze_privacy_utility_tradeoff(sample_data, reconstructed_data)
    print("\nPrivacy-Utility Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value:.6f}")
    
    # Communication cost analysis
    comm_cost = privacy_mechanism.compute_communication_cost(sample_data.shape)
    print(f"\nCommunication Cost Analysis:")
    for key, value in comm_cost.items():
        print(f"  {key}: {value:.4f}")
    
    # Test privacy accountant
    print("\nTesting Privacy Accountant...")
    accountant = DifferentialPrivacyAccountant(total_epsilon=1.0, delta=1e-5)
    
    # Simulate multiple rounds
    for round_num in range(10):
        round_epsilon = analysis['epsilon'] / 10  # Divide by 10 rounds
        success = accountant.add_round_epsilon(round_epsilon)
        if not success:
            print(f"Privacy budget exhausted at round {round_num}")
            break
    
    privacy_analysis = accountant.get_privacy_analysis()
    print("Privacy Budget Analysis:")
    for key, value in privacy_analysis.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_privacy_mechanisms()
