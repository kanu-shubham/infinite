from setuptools import find_packages, setup

setup(
    name="rl_project",
    version="0.1.0",
    description="End-to-end Reinforcement Learning project: DQN + PPO on CartPole and a real-world inventory management environment.",
    author="You",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)
