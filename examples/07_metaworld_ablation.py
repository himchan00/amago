"""
Metaworld ablation: feed-forward (memory-free) TrajEncoder baseline.

Uses FFTrajEncoder — each timestep is processed independently with no
cross-timestep memory.  Comparing this against 07_metaworld.py (Transformer)
and 07_metaworld_mate.py (MATE) isolates the contribution of memory itself.

Run example (same flags as 07_metaworld.py):
    python 07_metaworld_ablation.py \
        --run_name metaworld_ml45_ff \
        --benchmark ml45 \
        --buffer_dir /path/to/buffer \
        --parallel_actors 40 \
        --timesteps_per_epoch 1501 \
        --agent_type multitask \
        --max_seq_len 256 \
        --memory_size 320 \
        --memory_layers 3 \
        --dset_max_size 25000 \
        --epochs 5000 \
        --val_interval 40
"""
from argparse import ArgumentParser

import wandb

from amago.envs.builtin.metaworld_ml import Metaworld
from amago.nets.tstep_encoders import FFTstepEncoder
from amago.nets.traj_encoders import FFTrajEncoder
from amago import cli_utils


def add_cli(parser):
    parser.add_argument(
        "--benchmark",
        type=str,
        default="reach-v2",
        help="`name-v2` for ML1, or `ml10`/`ml45`",
    )
    parser.add_argument("--k", type=int, default=3, help="K-Shots")
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument(
        "--hide_rl2s",
        action="store_true",
        help="hides the 'rl2 info' (previous actions, rewards)",
    )
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    cli_utils.add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    config = {
        "amago.nets.tstep_encoders.FFTstepEncoder.hide_rl2s": args.hide_rl2s,
        "amago.nets.actor_critic.NCriticsTwoHot.min_return": -100.0,
        "amago.nets.actor_critic.NCriticsTwoHot.max_return": 5000 * args.k,
        "amago.nets.actor_critic.NCriticsTwoHot.output_bins": 96,
    }

    # Hardcode FFTrajEncoder (no memory).
    # memory_size -> d_model, memory_layers -> n_layers, matching the other scripts.
    ff_config_prefix = f"{FFTrajEncoder.__module__}.{FFTrajEncoder.__name__}"
    config[f"{ff_config_prefix}.d_model"] = args.memory_size
    config[f"{ff_config_prefix}.n_layers"] = args.memory_layers

    traj_encoder_type = FFTrajEncoder

    agent_type = cli_utils.switch_agent(
        config, args.agent_type, reward_multiplier=1.0, num_critics=4
    )
    exploration_type = cli_utils.switch_exploration(
        config, "bilevel", steps_anneal=2_000_000, rollout_horizon=args.k * 500
    )
    cli_utils.use_config(config, args.configs)

    make_train_env = lambda: Metaworld(args.benchmark, "train", k_episodes=args.k)
    make_test_env = lambda: Metaworld(args.benchmark, "test", k_episodes=args.k)

    group_name = (
        f"{args.run_name}_metaworld_{args.benchmark}_K_{args.k}_L_{args.max_seq_len}_ff"
    )
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = cli_utils.create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=min(500 * args.k + 1, args.max_seq_len * 4),
            group_name=group_name,
            run_name=run_name,
            tstep_encoder_type=FFTstepEncoder,
            traj_encoder_type=traj_encoder_type,
            agent_type=agent_type,
            val_timesteps_per_epoch=15 * args.k * 500 + 1,
            learning_rate=5e-4,
            grad_clip=2.0,
            exploration_wrapper_type=exploration_type,
        )

        experiment = cli_utils.switch_async_mode(experiment, args.mode)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(make_test_env, timesteps=20_000, render=False)
        experiment.delete_buffer_from_disk()
        wandb.finish()
