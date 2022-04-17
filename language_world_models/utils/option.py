import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser("Experiments in Language World Model")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--batch_episode", type=int, default=4, help="number of episode per batch")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--beta_dim", type=int, default=16, help="beta dim")
    parser.add_argument("--m_dim", type=int, default=64, help="m dim")
    parser.add_argument("--z_dim", type=int, default=64, help="z dim")
    parser.add_argument("--pred_z_steps", type=int, default=10, help="z dim")
    parser.add_argument("--num_episodes", type=int, default=200000, help="number of episodes")
    parser.add_argument("--num_epochs", type=int, default=10000, help="number of epochs")
    parser.add_argument("--num_data", type=int, default=10240, help="number of data")
    # Checkpointing, logging and saving
    #parser.add_argument("--verbose", action="store_true", default=False, help="prints out more info during training")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--print_freq", type=int, default=100, help="how frequently log is printed")
    parser.add_argument("--save_freq", type=int, default=None, help="how frequently model is saved")
    parser.add_argument("--test_freq", type=int, default=100, help="how frequently model is tested")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints/", help="directory where model and results are saved")
    parser.add_argument("--episode_dir", type=str, default="./episodes/", help="directory where model and results are saved")
    parser.add_argument("--load_dir", type=str, default=None, help="directory where model is loaded from")
    parser.add_argument('--results_dir', type=str, default='./results/', help='directory where results are saved.')

    args = parser.parse_args()

    args.save_freq = args.num_episodes if args.save_freq is None else args.save_freq

    os.makedirs(args.episode_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    return args
