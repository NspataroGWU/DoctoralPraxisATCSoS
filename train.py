# -*- coding: utf-8 -*-
"""
=============================================================================
 train.py
 PPO Agent Training Script — ATC Conflict Management
 Northeast Corridor (NEC) Airspace Simulation

 Nicholas D. Spataro, D.Eng. Candidate
 George Washington University, 2026

 Part of: AI-Assisted Air Traffic Management in the Northeast Corridor
 D.Eng. Praxis — Appendix B

 Description:
   Trains a Proximal Policy Optimization (PPO) agent using Stable-Baselines3
   on the custom ATCEnv Gymnasium environment. Training runs for 1,000,000
   timesteps with vectorized normalization. The trained model and normalization
   statistics are saved to results_ppo/ for use by evaluate.py.

 Outputs (saved to results_ppo/):
   ppo_atc_model_final.zip    Trained PPO model weights
   vecnormalize.pkl           VecNormalize observation/reward statistics
   eval_baseline_aircraft.csv Fixed baseline aircraft states (seed=42)

 Usage:
   python train.py

 Dependencies:
   stable-baselines3, gymnasium, numpy, pandas
=============================================================================
"""

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from atc_env import ATCEnv, MAX_AIRCRAFT, build_gradual_approach_baseline

os.makedirs("results_ppo", exist_ok=True)

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
TOTAL_TIMESTEPS  = 1_000_000   # Total training timesteps (per praxis methodology)
N_ENVS           = 4           # Number of parallel environments
LEARNING_RATE    = 3e-4        # PPO learning rate
N_STEPS          = 2048        # Steps per rollout per environment
BATCH_SIZE       = 64          # Minibatch size
N_EPOCHS         = 10          # PPO optimization epochs per update
GAMMA            = 0.99        # Discount factor
GAE_LAMBDA       = 0.95        # GAE lambda
CLIP_RANGE       = 0.2         # PPO clip range
ENT_COEF         = 0.01        # Entropy coefficient (encourages exploration)
SEED             = 42          # Random seed for reproducibility

CHECKPOINT_FREQ  = 100_000     # Save checkpoint every N timesteps
EVAL_FREQ        = 50_000      # Run evaluation every N timesteps
EVAL_EPISODES    = 5           # Episodes per evaluation callback


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
def train():
    print("\n" + "=" * 65)
    print("  SPATARO PRAXIS — PPO TRAINING")
    print("  Nicholas D. Spataro, D.Eng., GWU 2026")
    print("=" * 65)
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel envs   : {N_ENVS}")
    print(f"  Aircraft        : {MAX_AIRCRAFT}")
    print(f"  Observation dim : {MAX_AIRCRAFT * 5}")
    print(f"  Action dim      : {MAX_AIRCRAFT * 3}")
    print("=" * 65 + "\n")

    # ------------------------------------------------------------------
    # Build fixed baseline aircraft states (seed=42) and cache to disk
    # This ensures evaluate.py uses the identical starting conditions
    # ------------------------------------------------------------------
    _csv = "results_ppo/eval_baseline_aircraft.csv"
    if not os.path.exists(_csv):
        baseline = build_gradual_approach_baseline(MAX_AIRCRAFT, seed=SEED)
        baseline.to_csv(_csv, index=False)
        print(f"[BASELINE SAVED] eval_baseline_aircraft.csv ({MAX_AIRCRAFT} aircraft, seed={SEED})")
    else:
        print(f"[BASELINE EXISTS] Using existing eval_baseline_aircraft.csv")

    # ------------------------------------------------------------------
    # Create vectorized, normalized training environment
    # ------------------------------------------------------------------
    print("\nInitializing training environment...")
    vec_env = make_vec_env(ATCEnv, n_envs=N_ENVS, seed=SEED)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=GAMMA,
    )

    # Separate eval environment (not normalized with training stats)
    eval_env = make_vec_env(ATCEnv, n_envs=1, seed=SEED + 1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # ------------------------------------------------------------------
    # Define PPO agent
    # Policy: MlpPolicy (fully connected, 2 hidden layers of 64 units)
    # ------------------------------------------------------------------
    print("Building PPO agent...")
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        verbose=1,
        seed=SEED,
        tensorboard_log="results_ppo/tensorboard/",
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ // N_ENVS,
        save_path="results_ppo/checkpoints/",
        name_prefix="ppo_atc",
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="results_ppo/best_model/",
        log_path="results_ppo/eval_logs/",
        eval_freq=EVAL_FREQ // N_ENVS,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\nStarting training ({TOTAL_TIMESTEPS:,} timesteps)...\n")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # ------------------------------------------------------------------
    # Save final model and normalization statistics
    # ------------------------------------------------------------------
    model.save("results_ppo/ppo_atc_model_final")
    vec_env.save("results_ppo/vecnormalize.pkl")

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print("  Saved: results_ppo/ppo_atc_model_final.zip")
    print("  Saved: results_ppo/vecnormalize.pkl")
    print("  Run evaluate.py to reproduce Chapter 4 results.")
    print("=" * 65 + "\n")

    vec_env.close()
    eval_env.close()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    train()