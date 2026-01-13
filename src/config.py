"""
Centralized configuration for Concept-Based Learning.

This module provides a structured configuration system that:
- Organizes all hyperparameters and settings
- Enables easy integration with wandb
- Supports hyperparameter sweeps
- Can be loaded from YAML/JSON files
"""
from dataclasses import dataclass, field
from typing import Tuple, Optional
from pathlib import Path
import jax.numpy as jnp

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    # Sub-policy network architecture
    hidden_dims: Tuple[int, ...] = (64, 64)
    
    # Meta-policy configuration (placeholder for future)
    meta_hidden_dims: Tuple[int, ...] = (128, 128)
    
    # Instance selector configuration (placeholder for future)
    selector_hidden_dims: Tuple[int, ...] = (64, 64)


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    # Random seed
    seed: int = 42
    
    # Learning rates
    sub_policy_lr: float = 1e-3
    meta_policy_lr: float = 1e-3
    selector_lr: float = 1e-3
    
    # Q-learning hyperparameters
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Polyak averaging coefficient for target network updates
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 100000
    
    # Training loop
    batch_size: int = 32
    buffer_size: int = 100000
    train_frequency: int = 4  # Train every N steps
    target_update_frequency: int = 1000  # Update target network every N steps
    learning_starts: int = 1000  # Steps before training begins
    
    # PQN-specific hyperparameters
    rollout_len: int = 32  # Steps per update loop (rollout length)
    epochs: int = 3  # Number of times to reuse collected data (like PPO/PQN)
    
    # Hysteresis / Stickiness
    # How much bonus to add to the previously selected option's Q-value/Logit
    # A value of 0.5 means the new option must be >0.5 better to switch.
    meta_stickiness: float = 0.5
    selector_stickiness: float = 1.0  # Higher stickiness for targets to prevent jitter
    
    # Training duration
    total_steps: int = 1000000
    eval_frequency: int = 10000
    eval_episodes: int = 5
    exploration_fraction: float = 0.1  # Fraction of total steps for exploration
    
    # Parallel environments for vectorization
    num_envs: int = 256  # Number of parallel environments to run
    
    # Gradient clipping
    max_grad_norm: Optional[float] = 10.0


@dataclass
class RewardConfig:
    """Configuration for reward shaping."""
    # Reward scaling - Boosted to prevent target collapse
    # Scale distance delta (approx 0.05 per frame) by 100 -> +5.0 reward per frame for approaching
    reward_scale: float = 100.0  # High scale to generate healthy gradients
    bonus_success: float = 1000.0  # Make success huge (like a jackpot) to dominate sparse signals
    penalty_failure: float = -10.0  # Collision
    aim_bonus_scale: float = 0.5  # Penalty for drifting away from target
    
    # Global score reward
    global_score_scale: float = 0.01  # Scale for game score difference


@dataclass
class EnvironmentConfig:
    """Configuration for environment settings."""
    # Game settings
    game_name: str = "seaquest"
    
    # Atari wrapper settings
    frame_stack_size: int = 4
    frame_skip: int = 4
    sticky_actions: bool = True
    episodic_life: bool = True
    first_fire: bool = True
    noop_reset: int = 30
    clip_reward: bool = False
    max_pooling: bool = False
    
    # Screen dimensions
    screen_width: float = 160.0
    screen_height: float = 210.0
    
    # Object capacity
    max_objs: int = 35
    n_global_feats: int = 10
    n_obj_feats: int = 4  # x, y, w, h
    
    # Action space
    # Number of actions available in the environment
    # Different games have different action spaces (e.g., Seaquest: 6, full Atari: 18)
    # Can also be set from environment.action_space if needed
    n_actions: int = 6
    
    # Object indices (Seaquest-specific, but kept here for now)
    idx_sharks_start: int = 1
    idx_subs_start: int = 13
    idx_divers_start: int = 25
    idx_emissiles_start: int = 29
    idx_ssub: int = 33
    idx_player_missile: int = 34


@dataclass
class ConceptConstants:
    """Semantic labels and constants (environment-agnostic concepts)."""
    # Labels
    label_player: int = 0
    label_player_tool: int = 1
    label_collect: int = 2
    label_avoid: int = 3
    label_destroy: int = 4
    label_monitor_statistic: int = 5
    label_high_level_concept: int = 6
    
    # Events
    event_triggered: int = 1
    event_not_triggered: int = 0
    
    # Capabilities
    cap_can_fire: int = 0
    n_capabilities: int = 4
    
    # Physics/Rules
    shooting_vector: jnp.ndarray = field(default_factory=lambda: jnp.array([1.0, 0.0]))


@dataclass
class WandBConfig:
    """Configuration for Weights & Biases logging."""
    project: str = "concept-based-learning"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: Tuple[str, ...] = ()
    notes: Optional[str] = None
    mode: str = "online"  # "online", "offline", "disabled"
    save_code: bool = True
    group: Optional[str] = None
    job_type: Optional[str] = None


@dataclass
class Config:
    """Main configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    concepts: ConceptConstants = field(default_factory=ConceptConstants)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    
    def to_dict(self):
        """Convert config to dictionary for wandb."""
        def convert_value(v):
            if isinstance(v, jnp.ndarray):
                return v.tolist()
            elif isinstance(v, tuple):
                return list(v)
            elif hasattr(v, '__dict__'):
                # Handle dataclass instances
                result = {}
                for k in v.__dict__:
                    if not k.startswith('_'):
                        result[k] = convert_value(getattr(v, k))
                return result
            return v
        
        return {
            'model': convert_value(self.model),
            'training': convert_value(self.training),
            'reward': convert_value(self.reward),
            'environment': convert_value(self.environment),
            'concepts': convert_value(self.concepts),
            'wandb': convert_value(self.wandb),
        }
    
    @classmethod
    def from_dict(cls, d):
        """Create config from dictionary (e.g., from wandb sweep)."""
        config = cls()
        
        if 'model' in d:
            for k, v in d['model'].items():
                if hasattr(config.model, k):
                    if isinstance(v, list) and k.endswith('_dims'):
                        setattr(config.model, k, tuple(v))
                    else:
                        setattr(config.model, k, v)
        
        if 'training' in d:
            for k, v in d['training'].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)
        
        if 'reward' in d:
            for k, v in d['reward'].items():
                if hasattr(config.reward, k):
                    setattr(config.reward, k, v)
        
        if 'environment' in d:
            for k, v in d['environment'].items():
                if hasattr(config.environment, k):
                    setattr(config.environment, k, v)
        
        if 'concepts' in d:
            for k, v in d['concepts'].items():
                if hasattr(config.concepts, k):
                    if k == 'shooting_vector' and isinstance(v, list):
                        setattr(config.concepts, k, jnp.array(v))
                    else:
                        setattr(config.concepts, k, v)
        
        if 'wandb' in d:
            for k, v in d['wandb'].items():
                if hasattr(config.wandb, k):
                    if k == 'tags':
                        # Handle tags: convert list to tuple, None to empty tuple
                        if v is None:
                            setattr(config.wandb, k, ())
                        elif isinstance(v, list):
                            setattr(config.wandb, k, tuple(v))
                        else:
                            setattr(config.wandb, k, v)
                    else:
                        setattr(config.wandb, k, v)
        
        return config
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        Create config from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Config object loaded from YAML
            
        Raises:
            ImportError: If PyYAML is not installed
            FileNotFoundError: If YAML file doesn't exist
        """
        if yaml is None:
            raise ImportError(
                "PyYAML is required to load config from YAML files. "
                "Install it with: pip install pyyaml"
            )
        
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            yaml_dict = yaml.safe_load(f)
        
        # Handle empty or None YAML files
        if yaml_dict is None:
            yaml_dict = {}
        
        return cls.from_dict(yaml_dict)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def load_config_from_file(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file or return default config.
    
    Args:
        config_path: Optional path to YAML configuration file.
                    If None, returns default config.
    
    Returns:
        Config object
    """
    if config_path is None:
        return get_default_config()
    
    return Config.from_yaml(config_path)
