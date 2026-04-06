from argparse import ArgumentParser

import wandb

from amago.envs.builtin.metaworld_ml import Metaworld
from amago.nets.tstep_encoders import FFTstepEncoder
from amago.nets.traj_encoders import MateTrajEncoder
from amago.envs.exploration import BilevelEpsilonGreedy
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

    # MateTrajEncoder uses d_model / n_layers (not memory_size / n_layers).
    # switch_traj_encoder's else-branch maps memory_size -> "memory_size" key,
    # which doesn't exist on MateTrajEncoder, so we set the correct gin keys here.
    config = {
        "amago.nets.tstep_encoders.FFTstepEncoder.hide_rl2s": args.hide_rl2s,
        # delete the next three lines to use the paper settings
        "amago.nets.actor_critic.NCriticsTwoHot.min_return": -100.0,
        "amago.nets.actor_critic.NCriticsTwoHot.max_return": 5000 * args.k,
        "amago.nets.actor_critic.NCriticsTwoHot.output_bins": 96,
    }

    # Use MateTrajEncoder directly and bind d_model / n_layers correctly.
    #
    # --- Parameter count reference (d_model=320, d_ff=1280) ---
    # MATE now uses the same FFN sub-layer as Transformer (SigmaReparam + NormFormer
    # norms), so parameter counts per block match between the two architectures.
    # The only difference is that MATE has *no* attention sub-layer per depth.
    #
    #   Transformer n_layers=3 : ~3.80 M  (attention ~412K + FFN ~827K per layer)
    #   MATE        n_layers=3 : ~2.57 M  (FFN ~827K per layer, no attention)
    #   MATE        n_layers=4 : ~3.39 M  (closest param match, slightly under)
    #   MATE        n_layers=5 : ~4.22 M  (slightly over)
    #
    # => For a fair param-matched comparison use --memory_layers 4.
    #    Using --memory_layers 3 isolates the effect of attention vs. cumsum
    #    at the same depth, but with ~33% fewer parameters.
    mate_config_prefix = (
        f"{MateTrajEncoder.__module__}.{MateTrajEncoder.__name__}"
    )
    config[f"{mate_config_prefix}.d_model"] = args.memory_size
    config[f"{mate_config_prefix}.n_layers"] = args.memory_layers
    config[f"{mate_config_prefix}.proj"] = "hyper"

    traj_encoder_type = MateTrajEncoder

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
        f"{args.run_name}_metaworld_{args.benchmark}_K_{args.k}_L_{args.max_seq_len}_mate"
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
            full_transition=True,
        )

        experiment = cli_utils.switch_async_mode(experiment, args.mode)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(make_test_env, timesteps=20_000, render=False)
        experiment.delete_buffer_from_disk()
        wandb.finish()
