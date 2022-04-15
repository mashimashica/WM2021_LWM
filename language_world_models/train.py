import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import time
import torch
import torch.optim as optim
from torchvision import transforms
import torchvision.utils as vutils

from envs import create_ChoosePathGridDefaultEnv
from models.vae import VAE 
from utils import option, util


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def test_vae(args, model_vae, i_episode, x):
    model_vae.eval()
    _, _, _, recon_x = model_vae(x)

    # 入力画像の表示
    fig = plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Input")
    plt.imshow(np.transpose(vutils.make_grid(x.to(device)[:16], normalize=True, nrow=4).cpu(),(1,2,0)))

    # 再構成画像の表示
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Reconstruction")
    plt.imshow(np.transpose(vutils.make_grid(recon_x.to(device)[:16], normalize=True, nrow=4).cpu(),(1,2,0)))
    #vutils.save_image(recon_x.to(device)[:16], "vae_recon.png", normalize=True, nrow=4)
    
    result_vae_path = os.path.join(args.results_dir, "train_vae_%06d.png" % (i_episode+1))
    fig.savefig(result_vae_path)

    plt.close(fig)


@torch.no_grad()
def test(args, env, model_vae, i_episode):
    # 1エピソードの実行
    obs_listener_ep, obs_speaker_ep, reward_ep, done_ep = util.play_one_episode(env)
    
    # VAEのテスト
    x = torch.stack(obs_listener_ep).to(device)
    test_vae(args, model_vae, i_episode, x)


# VAE の更新
def train_batch_vae(model_vae, optimizer_vae, x):
    model_vae.train()
    model_vae.zero_grad()

    _, z_mean, z_logstd, recon_x = model_vae(x)
    vae_loss, kl_loss, recon_loss = model_vae.loss(x, recon_x, z_mean, z_logstd)
    vae_loss.backward()
    optimizer_vae.step()
    
    return vae_loss.cpu().detach().numpy(), kl_loss.cpu().detach().numpy(), recon_loss.cpu().detach().numpy()


def train(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs_listener_list, obs_speaker_list = [], []
    losses = {'vae' : {'loss':[], 'kl_loss':[], 'recon_loss':[]}}

    # ゲーム環境のインスタンスの取得
    env = create_ChoosePathGridDefaultEnv()
    
    # Listenerの定義
    # VAEの定義
    model_vae = VAE(args.z_dim).to(device)
    print(model_vae)
    optimizer_vae = optim.Adam(model_vae.parameters())

    test(args, env, model_vae, -1)

    last_time = time.time()

    # 訓練ループ
    for i_episode in range(args.num_episodes):
        # 1エピソードの実行
        obs_listener_ep, obs_speaker_ep, reward_ep, done_ep = util.play_one_episode(env)

        obs_listener_list.extend(obs_listener_ep)
        obs_speaker_list.extend(obs_speaker_ep)

        # バッチサイズ分のデータが集まったらパラメータの更新
        if len(obs_listener_list) >= args.batch_size:
            # VAE の更新
            x = torch.stack(obs_listener_list[:args.batch_size]).to(device)
            loss, kl_loss, recon_loss = train_batch_vae(model_vae, optimizer_vae, x)
            losses['vae']['loss'].append(loss)
            losses['vae']['kl_loss'].append(kl_loss)
            losses['vae']['recon_loss'].append(recon_loss)
       
            del obs_listener_list[:args.batch_size]
            del obs_speaker_list[:args.batch_size]
        
        if (i_episode+1) % args.print_freq == 0:
            print('episode: %d / %d (%d sec)' % (i_episode+1, args.num_episodes, time.time()-last_time))
            last_time = time.time()
            
            # VAE について表示
            print(losses['vae']['loss'])
            print('\tVAE : Train Lower Bound: %lf  (KL loss : %lf,  Reconstruction loss : %lf)' %
                  (np.average(losses['vae']['loss']),  np.average(losses['vae']['kl_loss']), np.average(losses['vae']['recon_loss'])))
            losses['vae']['loss'].clear()
            losses['vae']['kl_loss'].clear()
            losses['vae']['recon_loss'].clear()

        if (i_episode+1) % args.save_freq == 0:
            model_vae_path = os.path.join(args.checkpoints_dir, "model_vae_%06d.pth" % (i_episode+1))
            torch.save(model_vae.state_dict(), model_vae_path)

        if (i_episode+1) % args.test_freq == 0:
            test(args, env, model_vae, i_episode)


if __name__ == "__main__":
    args = option.parse_args()
    train(args)
