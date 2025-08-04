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
    """
    
    def __init__(self, num_bits: int = 8, device: str = 'cuda'):
        """
        Args:
            num_bits: Number of bits for quantization (e.g., 8 for int8)
            device: Device to perform computations on
        """
        self.num_bits = num_bits
        self.device = device
        self.max_value = 2**(num_bits - 1) - 1  # e.g., 127 for int8
        self.min_value = -2**(num_bits - 1)     # e.g., -128 for int8
        
    def quantize(self, data: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Quantize floating-point data to integers
        
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
        
        # Round to nearest integer and clamp to valid range
        quantized_data = torch.round(scaled_data).clamp(0, 2**self.num_bits - 1)
        
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
