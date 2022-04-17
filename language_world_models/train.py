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
from models.speaker import Speaker
from models.vae import VAE
from utils import misc, option


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def test_speaker(args, model_speaker, i_episode, x):
    model_speaker.eval()
    recon_x, _ = model_speaker(x)

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
    
    result_vae_path = os.path.join(args.results_dir, "train_speaker_%06d.png" % (i_episode+1))
    fig.savefig(result_vae_path)

    plt.close(fig)

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
def test(args, env, model_vae, model_speaker, i_episode):
    # 1エピソードの実行
    obs_listener_ep, obs_speaker_ep, reward_ep, done_ep = misc.play_one_episode(env)
     
    # Speaker のテスト
    x = torch.stack(obs_speaker_ep).to(device)
    test_speaker(args, model_speaker, i_episode, x)

    # VAE のテスト
    x = torch.stack(obs_listener_ep).to(device)
    test_vae(args, model_vae, i_episode, x)


# Speaker の更新
def train_batch_speaker(model_speaker, optimizer_speaker, x):
    model_speaker.train()
    model_speaker.zero_grad()

    recon_x, p_soft = model_speaker(x)
    loss, n_entropy, recon_loss = model_speaker.loss(x, recon_x, p_soft)
    loss.backward()
    optimizer_speaker.step()
    
    return loss.item(), n_entropy.item(), recon_loss.item()

# VAE の更新
def train_batch_vae(model_vae, optimizer_vae, x):
    model_vae.train()
    model_vae.zero_grad()

    _, z_mean, z_logstd, recon_x = model_vae(x)
    vae_loss, kl_loss, recon_loss = model_vae.loss(x, recon_x, z_mean, z_logstd)
    vae_loss.backward()
    optimizer_vae.step()
    
    return vae_loss.item(), kl_loss.item(), recon_loss.item()

"""LBF の更新
m_eps   : [EPS_NUM, SEQ_LEN, m_dim]
z_eps   : [EPS_NUM, SEQ_LEN, z_dim]
real_zs : [EPS_NUM, SEQ_LEN, pred_z_steps, z_dim]
pred_zs : [EPS_NUM, SEQ_LEN, pred_z_steps, z_dim]
"""
def train_lbf_step(model_lbf, optimizer_lbf, input_seq_len, m_eps, z_eps):
    model_lbf.train()
    model_lbf.zero_grad()
     
    pred_zs, beta, beta_mean, beta_logstd = model_lbf(m_eps, z_eps)

    real_zs[i] = z_eps[:, i:i+pred_z_steps]
    loss_lbf, KL_loss, pred_loss = model_lbf.loss(correct_zs, pred_zs, beta_mean, beta_logstd)
    loss_lbf.backward()
    optimizer_lbf.step()
    return loss_lbf, KL_loss, pred_loss


def train(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs_listener_list, obs_speaker_list = [], []
    losses = {'speaker' : {'loss':[], 'n_entropy':[], 'recon_loss':[]}, 
              'vae' : {'loss':[], 'kl_loss':[], 'recon_loss':[]}}

    # ゲーム環境のインスタンスの取得
    env = create_ChoosePathGridDefaultEnv()

    # Speakerの定義
    model_speaker = Speaker(args.m_dim).to(device)
    print(model_speaker)
    optimizer_speaker = optim.Adam(model_speaker.parameters())
   
    # Listenerの定義
    # VAEの定義
    model_vae = VAE(args.z_dim).to(device)
    print(model_vae)
    optimizer_vae = optim.Adam(model_vae.parameters())

    # LBFの定義
    model_lbf = LBF(args.beta_dim, args.m_dim, args.z_dim, args.pred_z_steps).to(device)
    print(model_lbf)
    optimizer_lbf = optim.Adam(model_lbf.parameters())

    test(args, env, model_vae, model_speaker, -1)

    last_time = time.time()

    # 訓練ループ
    for i_episode in range(args.num_episodes):
        # 1エピソードの実行
        obs_listener_ep, obs_speaker_ep, reward_ep, done_ep = misc.play_one_episode(env)

        obs_listener_eps.append(obs_listener_ep)
        obs_speaker_eps.append(obs_speaker_ep)

        obs_listener_list.extend(obs_listener_ep)
        obs_speaker_list.extend(obs_speaker_ep)

        # バッチサイズ分のデータが集まったらパラメータの更新
        if len(obs_listener_list) >= args.batch_size:
            # Speaker の更新
            x = torch.stack(obs_speaker_list[:args.batch_size]).to(device)
            loss, n_entropy, recon_loss = train_batch_speaker(model_speaker, optimizer_speaker, x)
            losses['speaker']['loss'].append(loss)
            losses['speaker']['n_entropy'].append(n_entropy)
            losses['speaker']['recon_loss'].append(recon_loss)
 
            # VAE の更新
            x = torch.stack(obs_listener_list[:args.batch_size]).to(device)
            loss, kl_loss, recon_loss = train_batch_vae(model_vae, optimizer_vae, x)
            losses['vae']['loss'].append(loss)
            losses['vae']['kl_loss'].append(kl_loss)
            losses['vae']['recon_loss'].append(recon_loss)
       
            del obs_listener_list[:args.batch_size]
            del obs_speaker_list[:args.batch_size]

         # 一定数のエピソード分のデータが集まったらパラメータの更新
        if len(obs_listener_eps) >= args.batch_episode:
            # LBF の更新
            model_vae.eval()
            model_speaker.eval()

            m_eps = None
            z_eps = None
            with torch.no_grad:
                for obs_l_ep in obs_listener_eps:
                    obs_l_ep = torch.stack(obs_l_ep).to(device)
                    z_ep, _, _, _ = model_vae(obs_l_ep)
                    # TODO z_epの足りない部分をz_ep[-1]で補完する
                    if len(z_ep) < seq_len:
                        z_ep.extend([z_ep[-1]] * (seq_len- len(z_ep)))
                    z_eps.append(z_ep)

            m_eps = torch.stack(m_eps).to(device)
            z_eps = torch.stack(z_eps).to(device)
            loss_lbf, KL_loss, pred_loss = train_lbf_step(model_lbf, optimizer_lbf, m_eps, z_eps)
            
            losses_lbf.append(loss_lbf.cpu().detach().numpy())
        
            del obs_listener_eps
            del obs_speaker_eps

        
        if (i_episode+1) % args.print_freq == 0:
            print('episode: %d / %d (%d sec)' % (i_episode+1, args.num_episodes, time.time()-last_time))
            last_time = time.time()
             
            # Speaker について表示
            print('\tSpeaker : Loss: %lf  (Negative entropy : %lf,  Reconstruction loss : %lf)' %
                  (np.average(losses['speaker']['loss']),  np.average(losses['speaker']['n_entropy']), np.average(losses['speaker']['recon_loss'])))
            losses['speaker']['loss'].clear()
            losses['speaker']['n_entropy'].clear()
            losses['speaker']['recon_loss'].clear()
           
            # VAE について表示
            print('\tVAE : Train Lower Bound: %lf  (KL loss : %lf,  Reconstruction loss : %lf)' %
                  (np.average(losses['vae']['loss']),  np.average(losses['vae']['kl_loss']), np.average(losses['vae']['recon_loss'])))
            losses['vae']['loss'].clear()
            losses['vae']['kl_loss'].clear()
            losses['vae']['recon_loss'].clear()

        if (i_episode+1) % args.save_freq == 0:
            model_speaker_path = os.path.join(args.checkpoints_dir, "model_speaker_%06d.pth" % (i_episode+1))
            torch.save(model_speaker.state_dict(), model_speaker_path)

            model_vae_path = os.path.join(args.checkpoints_dir, "model_vae_%06d.pth" % (i_episode+1))
            torch.save(model_vae.state_dict(), model_vae_path)

        if (i_episode+1) % args.test_freq == 0:
            test(args, env, model_vae, model_speaker, i_episode)


if __name__ == "__main__":
    args = option.parse_args()
    train(args)
