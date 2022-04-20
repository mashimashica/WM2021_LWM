import copy
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
from models.reinforce import REINFORCE
from utils import misc, option
from utils.test import test_speaker, test_vae
from utils.train import train_speaker_batch, train_vae_batch, train_lbf_batch, train_reinforce_batch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def test(args, env, model_speaker, model_vae, model_lbf, model_controller, i):
    # Speaker のテスト
    x = torch.stack(misc.get_many_head_frame(env, 16)).to(device)
    test_speaker(args, model_speaker, i, x)

    # VAE のテスト
    obs_listener_ep, obs_speaker_ep, act_ep, reward_ep, m_ep, z_ep, beta_ep, success = \
            misc.play_one_episode(env, model_speaker, model_vae, model_lbf, model_controller)
    x = torch.stack(obs_listener_ep).to(device)
    test_vae(args, model_vae, i, x)


@torch.no_grad()
def generate_blf_rnn_train_data_from_obs(env, input_seq_len, pred_z_steps, 
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

def generate_blf_train_data_from_obs(pred_z_steps, m_eps, z_eps):
    # m_eps     : (EP_NUM, EP_STEP, m_dim)
    # z_eps     : (EP_NUM, EP_STEP, z_dim)
    # input_m   : (B_SIZE, m_dim)
    # input_z   : (B_SIZE, z_dim)
    # target_z_seq  : (B_SIZE, pred_z_steps, z_dim)
    input_m_b, input_z_b, target_z_seq_b = [], [], []
    for m_ep, z_ep in zip(m_eps, z_eps):
        input_m = m_ep[:-pred_z_steps+1]
        input_z = z_ep[:-pred_z_steps+1]
        input_m_b.extend(input_m)
        input_z_b.extend(input_z)
        for i in range(len(input_z)):
            # z_i, .., z_{i+pred_z_steps-1}
            target_z_seq = z_ep[i:(i+pred_z_steps)]
            target_z_seq = torch.stack(target_z_seq).to(device)
            target_z_seq_b.append(target_z_seq)
    return input_m_b, input_z_b, target_z_seq_b


def train(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs_listener_list, obs_speaker_list = [], []
    m_eps, z_eps = [], []

    logs = {'entire' : {'success':[], 'reward':[]}, 
            'speaker' : {'loss':[], 'n_entropy':[], 'recon_loss':[]}, 
            'vae' : {'loss':[], 'kl_loss':[], 'recon_loss':[]},
            'lbf' : {'loss':[], 'kl_loss':[], 'pred_loss':[]},
            'controller' : {'loss':[], 'a_loss':[], 'n_ent_a':[], 'value_loss':[]}
           } 
    tmp_logs = copy.deepcopy(logs)


    # ゲーム環境のインスタンスの取得
    env = create_ChoosePathGridDefaultEnv(max_steps=args.explore_max_steps)

    # Speakerの定義
    model_speaker = Speaker(args.m_dim).to(device)
    print(model_speaker)
    optimizer_speaker = optim.Adam(model_speaker.parameters(), args.lr)
   
    # Listenerの定義
    # VAEの定義
    model_vae = VAE(args.z_dim).to(device)
    print(model_vae)
    optimizer_vae = optim.Adam(model_vae.parameters(), args.lr)

    # LBFの定義
    model_lbf = LBF(args.beta_dim, args.m_dim, args.z_dim, args.pred_z_steps).to(device)
    print(model_lbf)
    optimizer_lbf = optim.Adam(model_lbf.parameters(), args.lr)

    # REINFORCEの定義
    model_controller = REINFORCE(args.z_dim, args.beta_dim, n_steps=1).to(device)
    print(model_controller)
    optimizer_controller = optim.Adam(model_controller.parameters(), args.lr)


    test(args, env, model_speaker, model_vae, model_lbf, model_controller, -1)

    last_time = time.time()

    # 訓練ループ
    for i_episode in range(args.num_episodes):
        # 1エピソードの実行
        max_steps = 30
        if i_episode < args.explore_episodes:
            # 最初のいくつかのエピソードでは，制限時間を長くして訓練する
            max_steps = args.explore_max_steps
        obs_listener_ep, obs_speaker_ep, act_ep, reward_ep, m_ep, z_ep, beta_ep, success = \
                misc.play_one_episode(env, model_speaker, model_vae, model_lbf, model_controller, max_steps=max_steps)

        tmp_logs['entire']['success'].append(success)
        tmp_logs['entire']['reward'].append(np.average(reward_ep))

        m_eps.append(m_ep)
        z_eps.append(z_ep)

        obs_listener_list.extend(obs_listener_ep)
        obs_speaker_list.extend(obs_speaker_ep)

        # バッチサイズ分のデータが集まったらパラメータの更新
        while len(obs_listener_list) >= args.batch_size:
            # Speaker の更新
            x = torch.stack(obs_speaker_list[:args.batch_size]).to(device)
            loss, n_entropy, recon_loss = train_speaker_batch(model_speaker, optimizer_speaker, x)
            tmp_logs['speaker']['loss'].append(loss)
            tmp_logs['speaker']['n_entropy'].append(n_entropy)
            tmp_logs['speaker']['recon_loss'].append(recon_loss)
 
            # VAE の更新
            x = torch.stack(obs_listener_list[:args.batch_size]).to(device)
            loss, kl_loss, recon_loss = train_vae_batch(model_vae, optimizer_vae, x)
            tmp_logs['vae']['loss'].append(loss)
            tmp_logs['vae']['kl_loss'].append(kl_loss)
            tmp_logs['vae']['recon_loss'].append(recon_loss)
       
            del obs_listener_list[:args.batch_size]
            del obs_speaker_list[:args.batch_size]

         # 一定数のエピソード分のデータが集まったらパラメータの更新
        if len(m_eps) >= args.batch_episode:
            # LBF の更新
            input_m, input_z, target_z_seq = \
                    generate_blf_train_data_from_obs(args.pred_z_steps, m_eps, z_eps)
            input_m = torch.stack(input_m).to(device)
            input_z = torch.stack(input_z).to(device)
            target_z_seq = torch.stack(target_z_seq).to(device)
            loss, kl_loss, pred_loss = train_lbf_batch(model_lbf, optimizer_lbf,
                                                       input_m, input_z, target_z_seq)
            tmp_logs['lbf']['loss'].append(loss)
            tmp_logs['lbf']['kl_loss'].append(kl_loss)
            tmp_logs['lbf']['pred_loss'].append(pred_loss)
            
            m_eps.clear()
            z_eps.clear()
        
        # Controller の更新
        if i_episode > args.train_encoder_episodes:
            act_ep = torch.stack(act_ep).to(device)
            reward_ep = torch.tensor(np.array(reward_ep)).to(device)
            z_ep = torch.stack(z_ep).to(device)
            beta_ep = torch.stack(beta_ep).to(device)

            loss, a_loss, n_ent_a, value_loss = \
                    train_reinforce_batch(model_controller, optimizer_controller, act_ep, reward_ep, z_ep, beta_ep)
            tmp_logs['controller']['loss'].append(loss)
            tmp_logs['controller']['a_loss'].append(a_loss)
            tmp_logs['controller']['n_ent_a'].append(n_ent_a)
            tmp_logs['controller']['value_loss'].append(value_loss)

        
        if (i_episode+1) % args.print_freq == 0:
            print('episode: %d / %d (%d sec)' % (i_episode+1, args.num_episodes, time.time()-last_time))
            last_time = time.time()
             
            # ログの処理
            for model, d in tmp_logs.items():
                for metric, v in d.items():
                    # 平均を取る
                    logs[model][metric].append(np.average(v))
                    v.clear()

            # 全体について表示
            print('\tEntire : Success Rate : %lf, Reward : %lf' %
                  (logs['entire']['success'][-1], logs['entire']['reward'][-1]))

            # Speaker について表示
            print('\tSpeaker : Loss: %lf  (Negative entropy : %lf,  Reconstruction loss : %lf)' %
                  (logs['speaker']['loss'][-1], logs['speaker']['n_entropy'][-1], logs['speaker']['recon_loss'][-1]))
           
            # VAE について表示
            print('\tVAE : Train Lower Bound: %lf  (KL loss : %lf,  Reconstruction loss : %lf)' %
                  (logs['vae']['loss'][-1], logs['vae']['kl_loss'][-1], logs['vae']['recon_loss'][-1]))

            # LBF について表示
            print('\tLBF : Train Lower Bound: %lf  (KL loss : %lf,  Prediction loss : %lf)' %
                  (logs['lbf']['loss'][-1], logs['lbf']['kl_loss'][-1], logs['lbf']['pred_loss'][-1]))

            # REINFORCE について表示
            print('\tController : Loss: %lf  (Action loss : %lf,  Negative action entropy : %lf, Value loss : %lf)' %
                  (logs['controller']['loss'][-1], \
                   logs['controller']['a_loss'][-1], logs['controller']['n_ent_a'][-1], logs['controller']['value_loss'][-1]))



        if (i_episode+1) % args.save_freq == 0:
            model_speaker_path = os.path.join(args.checkpoints_dir, "model_speaker.pth")
            torch.save(model_speaker.state_dict(), model_speaker_path)

            model_vae_path = os.path.join(args.checkpoints_dir, "model_vae.pth")
            torch.save(model_vae.state_dict(), model_vae_path)

            model_lbf_path = os.path.join(args.checkpoints_dir, "model_lbf.pth")
            torch.save(model_lbf.state_dict(), model_lbf_path)

            model_controller_path = os.path.join(args.checkpoints_dir, "model_controller.pth")
            torch.save(model_controller.state_dict(), model_controller_path)

            logs_path = os.path.join(args.results_dir, "logs.npy")
            np.save(logs_path, logs)


        if (i_episode+1) % args.test_freq == 0:
            test(args, env, model_speaker, model_vae, model_lbf, model_controller, i_episode)


if __name__ == "__main__":
    args = option.parse_args()
    train(args)
