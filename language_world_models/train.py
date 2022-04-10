import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import torch
import torch.optim as optim
from torchvision import transforms
import torchvision.utils as vutils

from envs import create_ChoosePathGridDefaultEnv
from models.vae import VAE 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser("Experiments in Language World Model")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--z-dim", type=int, default=64, help="z dim")
    # Checkpointing, logging and saving
    #parser.add_argument("--verbose", action="store_true", default=False, help="prints out more info during training")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--test-freq", type=int, default=100, help="how frequently model is tested")
    parser.add_argument("--print-freq", type=int, default=100, help="how frequently log is printed")
    parser.add_argument("--save-freq", type=int, default=None, help="how frequently model is saved")
    parser.add_argument("--log-saving", action="store_true", default=False, help="save params on log-ish scale")
    parser.add_argument("--num-episodes", type=int, default=200000, help="number of episodes")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="directory where model and results are saved")
    parser.add_argument("--load-dir", type=str, default="", help="directory where model is loaded from")
    parser.add_argument("--exp-name", type=str, default=None, help="type of coordination game being played")

    args = parser.parse_args()

    args.save_freq = args.num_episodes if args.save_freq is None else args.save_freq

    return args

transform_speaker = transforms.Compose([
    transforms.Resize(128),
    # [0, 255] -> [0.0, 1.0]; (H, W, C) -> (C, H, W)
    transforms.ToTensor(), 
])

transform_listener = transforms.Compose([
    transforms.Resize(32),
    # [0, 255] -> [0.0, 1.0]; (H, W, C) -> (C, H, W)
    transforms.ToTensor(), 
])


def get_obs_speaker(env):
    obs_speaker = env.grid.render(tile_size=11).astype(np.uint8)
    obs_speaker = transform_speaker(Image.fromarray(obs_speaker))
    return obs_speaker

def get_obs_listener(obs_agent):
    obs_listener = obs_agent[0].astype(np.uint8) # 1人目のエージェントのobs
    obs_listener = transform_listener(Image.fromarray(obs_listener))
    return obs_listener

# 1エピソードの実行
def play_one_episode(env, policy=None):
    obs_listener_ep, obs_speaker_ep, reward_ep, done_ep = [], [], [], []

    obs_agent = env.reset() # TODO
    obs_speaker = get_obs_speaker(env)
    obs_listener = get_obs_listener(obs_agent)

    done = False
    while not done:
        # 行動を決定（とりあえずランダム）
        act = None
        if policy is None:
            act = env.action_space.sample()

        # ゲーム環境に入力
        obs_agent, reward, done, _ = env.step(act)

        # 画像用配列への変換
        obs_speaker = get_obs_speaker(env)
        obs_listener = get_obs_listener(obs_agent)
        
        obs_listener_ep.append(obs_listener)
        obs_speaker_ep.append(obs_speaker)
        reward_ep.append(reward)
        done_ep.append(done)
 
    return obs_listener_ep, obs_speaker_ep, reward_ep, done_ep

@torch.no_grad()
def test(args, env, model_vae):
    # 1エピソードの実行
    obs_listener_ep, obs_speaker_ep, reward_ep, done_ep = play_one_episode(env)
    
    # VAEのテスト
    x = torch.stack(obs_listener_ep).to(device)
    
    recon_x, z_mean, z_logstd = model_vae(x)
    # Plot the real images
    fig = plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Input")
    plt.imshow(np.transpose(vutils.make_grid(x.to(device)[:16], normalize=True, nrow=4).cpu(),(1,2,0)))

    # Plot the reconstruction images
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Reconstruction")
    plt.imshow(np.transpose(vutils.make_grid(recon_x.to(device)[:16], normalize=True, nrow=4).cpu(),(1,2,0)))
    vutils.save_image(recon_x.to(device)[:16], "vae_recon.png", normalize=True, nrow=4)
    
    fig.savefig("vae_test.png")
    plt.close(fig)



def train(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs_listener_list, obs_speaker_list, reward_list, done_list = [], [], [], [],

    # ゲーム環境のインスタンスの取得
    env = create_ChoosePathGridDefaultEnv()
    
    # モデルの定義
    model_vae = VAE(args.z_dim).to(device)
    print(model_vae)
    optimizer_vae = optim.Adam(model_vae.parameters())
    test(args, env, model_vae)

    losses_vae = []

    last_time = time.time()

    # Main training loop
    for i_episode in range(args.num_episodes):
        # 1エピソードの実行
        obs_listener_ep, obs_speaker_ep, reward_ep, done_ep = play_one_episode(env)

        obs_listener_list.extend(obs_listener_ep)
        obs_speaker_list.extend(obs_speaker_ep)
        reward_list.extend(reward_ep)
        done_list.extend(done_ep)

        # バッチサイズ分のデータが集まったらパラメータの更新
        if len(obs_listener_list) >= args.batch_size:
            # VAE の更新
            model_vae.train()
            model_vae.zero_grad()

            x = torch.stack(obs_listener_list).to(device)
            recon_x, z_mean, z_logstd = model_vae(x)
            loss_vae, KL_loss, recon_loss = model_vae.loss(x, recon_x, z_mean, z_logstd)
            loss_vae.backward()
            optimizer_vae.step()

            losses_vae.append(loss_vae.cpu().detach().numpy())
        
            obs_listener_list.clear()
            obs_speaker_list.clear()
            reward_list.clear() # TODO
            done_list.clear()   # TODO
        
        if (i_episode+1) % args.test_freq == 0:
            test(args, env, model_vae)

        if (i_episode+1) % args.print_freq == 0:
            print('episode: %d / %d (%d sec)    Train Lower Bound: %lf  (KL loss : %lf,  Reconstruction loss : %lf)' %
                  (i_episode+1, args.num_episodes, time.time()-last_time, np.average(losses_vae),  KL_loss, recon_loss))
            last_time = time.time()
            losses_vae = []


if __name__ == "__main__":
    args = parse_args()
    train(args)
