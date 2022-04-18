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
from models.lbf import LBF
from utils import misc, option
from utils.test import test_speaker, test_vae
from utils.train import train_speaker_batch, train_vae_batch, train_lbf_batch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def test(args, env, model_vae, model_speaker, i_episode):
    # Speaker のテスト
    x = torch.stack(misc.get_many_head_frame(env, 16)).to(device)
    test_speaker(args, model_speaker, i_episode, x)

    # VAE のテスト
    obs_listener_ep, _, _, _ = misc.play_one_episode(env)
    x = torch.stack(obs_listener_ep).to(device)
    test_vae(args, model_vae, i_episode, x)

@torch.no_grad()
def generate_blf_train_data_from_obs(env, input_seq_len, pred_z_steps, 
                                     model_speaker, model_vae,
                                     obs_speaker_eps, obs_listener_eps):
    # target_z_seqs   : (SEQ_NUM, input_seq_len, pred_z_steps, z_dim)
    input_m_seqs, input_z_seqs, target_z_seqs = [], [], []

    for obs_s_ep in obs_speaker_eps:
        model_speaker.eval()
        if len(obs_s_ep) < input_seq_len:
            # obs_s_epの足りない部分をobs_s_ep[-1]で補完する
            obs_s_ep_ep.extend([obs_s_ep[-1]] * (input_seq_len- len(obs_s_ep)))
        obs_s_ep = torch.stack(obs_s_ep).to(device)
        m_ep, _, _ = model_speaker(obs_s_ep)
        input_m_seqs.append(m_ep[:input_seq_len, :])
 
    for obs_l_ep in obs_listener_eps:
        model_vae.eval()
        seq_len = input_seq_len + pred_z_steps - 1
        if len(obs_l_ep) < seq_len:
            # obs_l_epの足りない部分をobs_l_ep-1]で補完する
            obs_l_ep.extend([obs_l_ep[-1]] * (seq_len- len(obs_l_ep)))
        obs_l_ep = torch.stack(obs_l_ep).to(device)
        z_ep, _, _, _ = model_vae(obs_l_ep)

        input_z_seqs.append(z_ep[:input_seq_len, :])
        
        target_z_seq = [] # (input_seq_len, pred_z_steps, z_dim)
        for i in range(input_seq_len):
            target_z_seq.append(z_ep[i:(i+pred_z_steps), :])
        target_z_seqs.append(torch.stack(target_z_seq))

    return input_m_seqs, input_z_seqs, target_z_seqs


def train(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs_listener_list, obs_speaker_list = [], []
    obs_listener_eps, obs_speaker_eps = [], []

    losses = {'speaker' : {'loss':[], 'n_entropy':[], 'recon_loss':[]}, 
              'vae' : {'loss':[], 'kl_loss':[], 'recon_loss':[]},
              'lbf' : {'loss':[], 'kl_loss':[], 'pred_loss':[]}}

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
            loss, n_entropy, recon_loss = train_speaker_batch(model_speaker, optimizer_speaker, x)
            losses['speaker']['loss'].append(loss)
            losses['speaker']['n_entropy'].append(n_entropy)
            losses['speaker']['recon_loss'].append(recon_loss)
 
            # VAE の更新
            x = torch.stack(obs_listener_list[:args.batch_size]).to(device)
            loss, kl_loss, recon_loss = train_vae_batch(model_vae, optimizer_vae, x)
            losses['vae']['loss'].append(loss)
            losses['vae']['kl_loss'].append(kl_loss)
            losses['vae']['recon_loss'].append(recon_loss)
       
            del obs_listener_list[:args.batch_size]
            del obs_speaker_list[:args.batch_size]

         # 一定数のエピソード分のデータが集まったらパラメータの更新
        if len(obs_speaker_eps) >= args.batch_episode:
            # LBF の更新
            input_m_seqs, input_z_seqs, target_z_seqs = \
                    generate_blf_train_data_from_obs(env, 20, args.pred_z_steps, 
                                                     model_speaker, model_vae,
                                                     obs_speaker_eps, obs_listener_eps)
            input_m_seqs = torch.stack(input_m_seqs).to(device)
            input_z_seqs = torch.stack(input_z_seqs).to(device)
            target_z_seqs = torch.stack(target_z_seqs).to(device)
            loss, kl_loss, pred_loss = train_lbf_batch(model_lbf, optimizer_lbf,
                                                       input_m_seqs, input_z_seqs, target_z_seqs)
            losses['lbf']['loss'].append(loss)
            losses['lbf']['kl_loss'].append(kl_loss)
            losses['lbf']['pred_loss'].append(pred_loss)
            
            obs_listener_eps.clear()
            obs_speaker_eps.clear()

        
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

            # LBF について表示
            print('\tLBF : Train Lower Bound: %lf  (KL loss : %lf,  Prediction loss : %lf)' %
                  (np.average(losses['lbf']['loss']),  np.average(losses['lbf']['kl_loss']), np.average(losses['lbf']['pred_loss'])))
            losses['lbf']['loss'].clear()
            losses['lbf']['kl_loss'].clear()
            losses['lbf']['pred_loss'].clear()


        if (i_episode+1) % args.save_freq == 0:
            model_speaker_path = os.path.join(args.checkpoints_dir, "model_speaker_%06d.pth" % (i_episode+1))
            torch.save(model_speaker.state_dict(), model_speaker_path)

            model_vae_path = os.path.join(args.checkpoints_dir, "model_vae_%06d.pth" % (i_episode+1))
            torch.save(model_vae.state_dict(), model_vae_path)

            model_lbf_path = os.path.join(args.checkpoints_dir, "model_lbf_%06d.pth" % (i_episode+1))
            torch.save(model_lbf.state_dict(), model_lbf_path)

        if (i_episode+1) % args.test_freq == 0:
            test(args, env, model_vae, model_speaker, i_episode)


if __name__ == "__main__":
    args = option.parse_args()
    train(args)
