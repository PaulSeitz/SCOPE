import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from concept_based_learning.train_agent import QNetwork
from concept_based_learning.pqn_visualizer import load_base_config, load_params_from_file


SKILL_NAME_TO_INDEX = {
    "destroy": 0,
    "collect": 1,
    "avoid": 2,
}


def infer_input_dim_from_params(skill_params_tree) -> int:
    """
    Infer the Q-network input dimension from a single skill's parameter tree.

    We look for the first 2D kernel (Dense layer weight) and take its input size.
    """
    leaves = jax.tree_util.tree_leaves(skill_params_tree)
    for leaf in leaves:
        arr = np.asarray(leaf)
        if arr.ndim == 2:
            return int(arr.shape[0])
    raise ValueError("Could not infer input dimension from params (no 2D Dense kernel found).")


def build_ego_grid_inputs(
    ego_feat_dim: int,
    max_norm: float,
    num_points: int,
    orientation_rad: float = 0.0,
):
    """
    Construct a batch of ego feature vectors for a 2D sweep over (dx_ego, dy_ego).

    The underlying training features come from `get_ego_obs`, which produces
    a flattened stack of:
        [dx_ego / 160, dy_ego / 210, sin(ori), cos(ori)]
    for each frame in the stack.

    We approximate the network's inputs by:
    - Filling all features with zeros
    - Setting ONLY the last frame's 4 features:
        [-4] = dx_ego (already normalized)
        [-3] = dy_ego (already normalized)
        [-2] = sin(orientation_rad)
        [-1] = cos(orientation_rad)

    This lets us sweep over the most recent (dx_ego, dy_ego) while holding the
    rest of the history fixed.
    """
    dx_vals = jnp.linspace(-max_norm, max_norm, num_points)
    dy_vals = jnp.linspace(-max_norm, max_norm, num_points)
    DX, DY = jnp.meshgrid(dx_vals, dy_vals, indexing="xy")

    flat_dx = DX.ravel()
    flat_dy = DY.ravel()

    batch_size = flat_dx.shape[0]
    ego_batch = jnp.zeros((batch_size, ego_feat_dim), dtype=jnp.float32)

    ego_batch = ego_batch.at[:, -4].set(flat_dx)
    ego_batch = ego_batch.at[:, -3].set(flat_dy)

    sin_o = jnp.sin(orientation_rad)
    cos_o = jnp.cos(orientation_rad)
    ego_batch = ego_batch.at[:, -2].set(sin_o)
    ego_batch = ego_batch.at[:, -1].set(cos_o)

    return DX, DY, ego_batch


def compute_q_surface(
    skill_network: QNetwork,
    skill_params_tree,
    ego_feat_dim: int,
    max_norm: float,
    num_points: int,
):
    """
    Evaluate max_a Q(s, a) on a 2D grid over (dx_ego, dy_ego) for a single skill.
    """
    DX, DY, ego_batch = build_ego_grid_inputs(
        ego_feat_dim=ego_feat_dim,
        max_norm=max_norm,
        num_points=num_points,
        orientation_rad=0.0,  # fixed facing direction
    )

    q_vals = skill_network.apply(
        {"params": skill_params_tree},
        ego_batch,
        train=False,
    )  # [batch, action_dim]

    q_scalar = jnp.max(q_vals, axis=-1)  # max over ego actions
    Z = np.array(q_scalar).reshape(DX.shape)
    return DX, DY, Z


def plot_q_surface(DX, DY, Z, skill_name: str, output_path: Path):
    """
    Save a heatmap of the Q-surface over (dx_ego, dy_ego).
    """
    plt.figure(figsize=(6, 5))
    extent = [
        float(DX.min()),
        float(DX.max()),
        float(DY.min()),
        float(DY.max()),
    ]
    im = plt.imshow(
        Z,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="viridis",
    )
    plt.colorbar(im, label="max_a Q(dx_ego, dy_ego, a)")
    plt.xlabel("dx_ego (normalized)")
    plt.ylabel("dy_ego (normalized)")
    plt.title(f"Q-Surface for {skill_name.upper()} Skill")
    plt.axhline(0.0, color="red", linestyle="--", linewidth=1.0, label="dy_ego = 0")
    plt.legend(loc="upper right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the Q-surface of a trained skill over (dx_ego, dy_ego).\n"
            "For a well-trained DESTROY skill, you should see a ridge of high value near dy_ego = 0."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to .safetensors file containing the trained skill Q-network parameters.",
    )
    parser.add_argument(
        "--skill",
        type=str,
        choices=["destroy", "collect", "avoid"],
        default="destroy",
        help="Which skill to visualize (DESTROY=0, COLLECT=1, AVOID=2).",
    )
    parser.add_argument(
        "--max-norm",
        type=float,
        default=1.0,
        help="Maximum |dx_ego| and |dy_ego| (already normalized) to sweep over.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=201,
        help="Number of points per axis in the grid.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path for the Q-surface plot. Defaults to q_surface_<skill>.png in the project root.",
    )

    args = parser.parse_args()

    base_config = load_base_config()

    num_skills = base_config["NUM_SKILLS"]
    ego_action_dim = base_config["EGO_ACTION_DIM"]
    hidden_size = base_config["HIDDEN_SIZE"]
    num_layers = base_config["NUM_LAYERS"]

    skill_index = SKILL_NAME_TO_INDEX[args.skill]
    if not (0 <= skill_index < num_skills):
        raise ValueError(
            f"Skill index {skill_index} for '{args.skill}' is out of range for NUM_SKILLS={num_skills}."
        )

    params_tree = load_params_from_file(args.model)

    # Extract params for the selected skill (leading skill dimension).
    skill_params_tree = jax.tree_util.tree_map(
        lambda p: p[skill_index],
        params_tree,
    )

    ego_feat_dim = infer_input_dim_from_params(skill_params_tree)

    skill_network = QNetwork(
        action_dim=ego_action_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

    DX, DY, Z = compute_q_surface(
        skill_network=skill_network,
        skill_params_tree=skill_params_tree,
        ego_feat_dim=ego_feat_dim,
        max_norm=args.max_norm,
        num_points=args.num_points,
    )

    project_root = Path(__file__).resolve().parents[2]
    if args.output is None:
        output_path = project_root / f"q_surface_{args.skill}.png"
    else:
        output_path = Path(args.output)

    plot_q_surface(DX, DY, Z, args.skill, output_path)
    print(f"Saved Q-surface plot for skill '{args.skill}' to: {output_path}")


if __name__ == "__main__":
    main()

