"""
Setup script for Concept-Based Learning project
"""
from setuptools import setup, find_packages

setup(
    name="concept-based-learning",
    version="0.1.0",
    description="Concept-based hierarchical reinforcement learning",
    packages=find_packages(where="src", include=["concept_based_learning*"]),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "jax",
        "jaxlib",
        "flax",
        "chex",
        "numpy",
        "pygame",
        "wandb",
        "pyyaml",
        "rtpt",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
        ],
    },
)
