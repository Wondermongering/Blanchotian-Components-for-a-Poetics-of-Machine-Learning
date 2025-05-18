import torch

def blanchot_activation(x, epsilon=1e-6):
    """
    A Blanchot-inspired activation function where thresholds exist
    in states of simultaneous presence and absence.
    
    Parameters:
    - x: Input tensor
    - epsilon: Small constant preventing numerical instability
    
    Returns:
    - Activated tensor through Blanchotian thresholds
    """
    # Thomas's experience of the sea - neither solid nor void
    sea_threshold = torch.mean(torch.abs(x)) * 0.5
    
    # The experience of night that watches - neither light nor complete darkness
    night_mask = torch.sigmoid((x - sea_threshold) * 2)
    
    # The neutrality space - what Blanchot calls "the neuter"
    neuter_zone = torch.where(
        torch.abs(x - sea_threshold) < epsilon,
        x * (1 + epsilon),  # Slight amplification of liminal signals
        x
    )
    
    # The gaze that sees and cannot see simultaneously
    thomas_gaze = x * night_mask + (1 - night_mask) * torch.min(x, torch.zeros_like(x))
    
    # The impossible death - neither absence nor presence
    impossible_death = torch.where(
        x < 0,
        torch.exp(x) - 1,  # Gradual fading rather than sharp cutoff
        x
    )
    
    # Final activation combines all Blanchotian threshold experiences
    return thomas_gaze * 0.7 + impossible_death * 0.3
