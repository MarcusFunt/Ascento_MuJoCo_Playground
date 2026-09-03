"""Conventional RSL-RL PPO settings for balance."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

AscentoBalanceRlCfg = RslRlOnPolicyRunnerCfg(
  actor=RslRlModelCfg(
    hidden_dims=(256, 256, 256),
    activation="elu",
    obs_normalization=False,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.7, "std_type": "scalar"},
  ),
  critic=RslRlModelCfg(hidden_dims=(256, 256, 256), activation="elu", obs_normalization=False),
  algorithm=RslRlPpoAlgorithmCfg(
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=3.0e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    entropy_coef=0.005,
    max_grad_norm=1.0,
    class_name="ascento_mjlab.ppo:InstrumentedPPO",
  ),
  logger="tensorboard",
  experiment_name="ascento_balance",
  wandb_project="ascento_mjlab",
  upload_model=False,
  num_steps_per_env=24,
  save_interval=250,
  max_iterations=10_000,
  clip_actions=1.0,
)
