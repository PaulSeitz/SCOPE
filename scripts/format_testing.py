import jax
import src.environments.seaquest.seaquest_wrapper as wrapper
from jaxatari.wrappers import AtariWrapper
from jaxatari import make as make_atari_env
import src.environments.seaquest.seaquest_rewards as rewards
from src.config import get_default_config

if __name__ == "__main__":
    config = get_default_config()
    env = wrapper.SeaquestConceptWrapper(AtariWrapper(make_atari_env('seaquest'), frame_stack_size=1), config=config)
    key = jax.random.key(0)
    env.reset(key)

    print("=== Observation Space ===")
    print(env.observation_space())

    print("\n=== Reset ===")
    obs, state = env.reset(key)
    print("Observation:", obs)
    print("State:", state)

    print("\n=== First Step ===")
    action = 0
    prev_obs = obs
    prev_state = state
    obs, state, reward, done, info = env.step(state, action)
    print("Observation:", obs)
    print("State:", state)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)
    
    print("\n=== Computed Rewards ===")
    multi_reward = rewards.compute_rewards(prev_obs, obs, prev_state, state, config)
    print("Global Reward:", multi_reward.global_reward)
    print("Concept Rewards Shape:", multi_reward.concept_rewards.shape)
    print("Concept Rewards (first few concepts and objects):")
    # Handle both unbatched [Num_Concepts, Max_Objs] and batched [Batch, Num_Concepts, Max_Objs]
    if multi_reward.concept_rewards.ndim == 2:
        print(multi_reward.concept_rewards[:5, :5])  # [Num_Concepts, Max_Objs]
    else:
        print(multi_reward.concept_rewards[0, :5, :5])  # [Batch, Num_Concepts, Max_Objs]