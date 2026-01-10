"""
Training script for Concept-Based Learning.

This script sets up wandb logging and provides a template for the training loop.
"""
import jax
import jax.random as jr
import wandb
from src.config import Config, get_default_config, Config as ConfigType


def init_wandb(config: ConfigType, resume: bool = False):
    """
    Initialize Weights & Biases logging.
    
    Args:
        config: Configuration object (will be updated if running a sweep)
        resume: Whether to resume a previous run
    
    Returns:
        Updated config object (with sweep parameters if in a sweep)
    """
    # Convert config to dict for wandb
    config_dict = config.to_dict()
    
    # Initialize wandb
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name,
        tags=list(config.wandb.tags) if config.wandb.tags else None,
        notes=config.wandb.notes,
        mode=config.wandb.mode,
        save_code=config.wandb.save_code,
        group=config.wandb.group,
        job_type=config.wandb.job_type,
        config=config_dict,
        resume=resume,
    )
    
    # If running a sweep, update config with sweep parameters
    # wandb.config is populated after wandb.init() when in a sweep
    # Note: wandb.config only contains the sweep parameters, not all config values
    if wandb.config and len(wandb.config) > 0:
        # wandb.config contains the sweep parameters
        # Convert to dict and handle nested structure (dot notation)
        sweep_config_dict = dict(wandb.config)
        
        # Handle dot-notation keys (e.g., "training.learning_rate")
        if any('.' in k for k in sweep_config_dict.keys()):
            nested_dict = {}
            for key, value in sweep_config_dict.items():
                parts = key.split('.')
                if len(parts) == 2:
                    section, param = parts
                    if section not in nested_dict:
                        nested_dict[section] = {}
                    nested_dict[section][param] = value
                else:
                    nested_dict[key] = value
            # Update config with sweep values (from_dict only updates provided values)
            sweep_config = Config.from_dict(nested_dict)
            # Copy sweep values to main config
            for section_name in ['model', 'training', 'reward', 'environment', 'concepts']:
                if hasattr(sweep_config, section_name):
                    sweep_section = getattr(sweep_config, section_name)
                    base_section = getattr(config, section_name)
                    # Update attributes that exist in both
                    for attr_name in dir(sweep_section):
                        if not attr_name.startswith('_') and hasattr(base_section, attr_name):
                            sweep_value = getattr(sweep_section, attr_name)
                            if sweep_value is not None:  # Only update if value was provided
                                setattr(base_section, attr_name, sweep_value)
        else:
            # Flat structure - update directly
            sweep_config = Config.from_dict(sweep_config_dict)
            # Copy sweep values to main config
            for section_name in ['model', 'training', 'reward', 'environment', 'concepts']:
                if hasattr(sweep_config, section_name):
                    sweep_section = getattr(sweep_config, section_name)
                    base_section = getattr(config, section_name)
                    for attr_name in dir(sweep_section):
                        if not attr_name.startswith('_') and hasattr(base_section, attr_name):
                            sweep_value = getattr(sweep_section, attr_name)
                            if sweep_value is not None:
                                setattr(base_section, attr_name, sweep_value)
    
    return config


def main():
    """Main training entry point."""
    # Get default config
    config = get_default_config()
    
    # Initialize wandb (this will update config if running a sweep)
    config = init_wandb(config)
    
    # Log that we're starting
    print(f"Starting training with config:")
    print(f"  Project: {config.wandb.project}")
    print(f"  Learning rate: {config.training.sub_policy_lr}")
    print(f"  Hidden dims: {config.model.hidden_dims}")
    print(f"  Reward scale: {config.reward.reward_scale}")
    
    # TODO: Initialize environment
    # TODO: Initialize agent/policies
    # TODO: Training loop
    # TODO: Log metrics to wandb
    
    # Example of logging metrics:
    # wandb.log({
    #     "train/episode_reward": episode_reward,
    #     "train/loss": loss,
    #     "train/epsilon": epsilon,
    #     "step": step_count,
    # })
    
    print("Training template ready. Implement training loop.")
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
