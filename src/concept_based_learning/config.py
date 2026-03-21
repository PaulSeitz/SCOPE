"""
Centralized configuration for Concept-Based Learning.

This module provides a structured configuration system that:
- Organizes all hyperparameters and settings
- Enables easy integration with wandb
- Supports hyperparameter sweeps
- Can be loaded from YAML/JSON files
"""
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
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
    
    # Instance selector configuration
    selector_hidden_dims: Tuple[int, ...] = (64, 64)
    
    # Normalization (PQN parameters)
    norm_input: bool = False  # Whether to normalize inputs (PQN parameter)
    norm_type: str = "layer_norm"  # Normalization type: "layer_norm" or "batch_norm" (PQN parameter)


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    # Random seed
    seed: int = 42
    
    # Learning rates (High-throughput defaults for 1024 envs)
    sub_policy_lr: float = 0.0005
    meta_policy_lr: float = 0.0005
    selector_lr: float = 0.0005
    
    # Q-learning hyperparameters
    gamma: float = 0.99  # Discount factor
    lambda_: float = 0.65  # Q(lambda) return parameter (PQN parameter, blends multi-step returns)
    tau: float = 0.005  # Polyak averaging coefficient for target network updates
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01  # Lower epsilon_end for high-throughput (enough samples to risk exploration)
    epsilon_decay: float = 0.1  # Epsilon decay rate (PQN parameter, used if custom schedules disabled)
    
    # Gradient clipping (PQN parameter)
    max_grad_norm: float = 5.0  # Tighter clip because 1024 envs can sum to huge gradients
    
    # Learning rate decay (PQN parameter)
    lr_linear_decay: bool = False  # Whether to linearly decay learning rate over training
    
    # PQN-specific hyperparameters (High-throughput defaults for 1024 envs)
    rollout_len: int = 32  # Steps per update loop (rollout length) - shorter for more frequent updates
    num_minibatches: int = 64  # Number of minibatches per epoch - keeps batch size ~1024 (32*1024/64)
    epochs: int = 4  # Number of times to reuse collected data (like PPO/PQN)
    
    # Hysteresis / Stickiness
    # How much bonus to add to the previously selected option's Q-value/Logit
    # A value of 0.5 means the new option must be >0.5 better to switch.
    meta_stickiness: float = 0.5
    selector_stickiness: float = 1.0  # Higher stickiness for targets to prevent jitter
    
    # Training duration (High-throughput defaults for 1024 envs)
    total_steps: int = 100_000_000  # 100M steps ensures enough optimization steps (~1,500-2,000 updates)
    
    # Parallel environments for vectorization
    num_envs: int = 1024  # High-throughput: massive parallelism for stable PQN online targets
    
    # Phased training: freeze parts of the network for certain steps
    # Phase 1 (0 to phase_1_steps): Only Sub-Policy trains
    # Phase 2 (phase_1_steps to phase_2_steps): Sub-Policy + Selector train
    # Phase 3 (after phase_2_steps): All components train (Sub-Policy + Selector + Meta-Policy)
    phase_1_steps: int = 20_000_000  # Scaled up for high-throughput curriculum
    phase_2_steps: int = 50_000_000  # Scaled up for high-throughput curriculum
    
    # Forced concept training mode: permanently force one concept and only train that
    # When enabled, the agent will always select the specified concept regardless of phase
    # Only the sub-policy for that concept will be trained (meta and selector are frozen)
    force_concept_mode: bool = False  # Enable forced concept training
    force_concept_id: Optional[int] = None  # Concept ID to force (0=COLLECT, 1=AVOID, 2=DESTROY, 3=DEFAULT)


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
    # Game mods: list of strings that can be applied to the environment
    # See jaxatari documentation for available mods
    game_mods: List[str] = field(default_factory=list)
    
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
    # High-level concept label (used to determine number of concepts)
    # Should be set to the highest concept label index (e.g., LABEL_DEFAULT = 3 for Seaquest)
    # num_concepts = label_high_level_concept + 1
    label_high_level_concept: int = 3
    
    # Capabilities
    n_capabilities: int = 4


@dataclass
class WandBConfig:
    """Weights & Biases logging. Pass to init_wandb(config) for one-line setup."""
    project: str = "concept-based-learning"
    mode: str = "online"  # "online" | "offline" | "disabled"
    name: Optional[str] = None
    log_interval: int = 10


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
                    setattr(config.concepts, k, v)
        
        if 'wandb' in d:
            for k, v in d['wandb'].items():
                if hasattr(config.wandb, k):
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


def init_wandb(config: Config, **kwargs):
    """
    Initialize Weights & Biases from config. Call once at the start of training.
    Any extra kwargs are passed through to wandb.init() (e.g. entity, group, tags).
    """
    try:
        import wandb
    except ImportError:
        return
    w = config.wandb
    wandb.init(
        project=w.project,
        mode=w.mode,
        name=w.name,
        config=config.to_dict(),
        **kwargs,
    )

