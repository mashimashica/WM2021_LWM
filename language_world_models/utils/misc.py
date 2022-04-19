import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import torch
from torchvision import transforms

# TODO
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_speaker = transforms.Compose([
    transforms.Resize(32),
    # [0, 255] -> [0.0, 1.0]; (H, W, C) -> (C, H, W)
    transforms.ToTensor(), 
])

transform_listener = transforms.Compose([
    transforms.Resize(32),
    # [0, 255] -> [0.0, 1.0]; (H, W, C) -> (C, H, W)
    transforms.ToTensor(), 
])

def get_obs_speaker(env):
    obs_speaker = env.grid.render(tile_size=1).astype(np.uint8)
    obs_speaker = transform_speaker(Image.fromarray(obs_speaker))
    return obs_speaker

def get_obs_listener(obs_agent):
    obs_listener = obs_agent[0].astype(np.uint8) # 1人目のエージェントのobs
    obs_listener = transform_listener(Image.fromarray(obs_listener))
    return obs_listener

# 1エピソードの実行
def get_many_head_frame(env, num):
    obs_speaker_list = []

    for i in range(num):
        obs_agent = env.reset()
        obs_speaker = get_obs_speaker(env)
        obs_speaker_list.append(obs_speaker)
 
    return obs_speaker_list

# 1エピソードの実行
def play_one_episode_random(env):
    obs_listener_ep, obs_speaker_ep, reward_ep, done_ep = [], [], [], []

    obs_agent = env.reset()
    obs_speaker = get_obs_speaker(env)
    obs_listener = get_obs_listener(obs_agent)

    done = False
    while not done:
        # ゲーム環境のステップ実行
        act = env.action_space.sample()
        obs_agent, reward, done, _ = env.step(act)

        # 画像用配列への変換
        obs_speaker = get_obs_speaker(env)
        obs_listener = get_obs_listener(obs_agent)
        
        obs_listener_ep.append(obs_listener)
        obs_speaker_ep.append(obs_speaker)
        reward_ep.append(reward)
        done_ep.append(done)
 
    return obs_listener_ep, obs_speaker_ep, reward_ep, done_ep

def _encode_obs(env, model_speaker, model_vae, model_lbf, obs_agent):
    obs_speaker = get_obs_speaker(env)
    obs_listener = get_obs_listener(obs_agent)

    m, _, _ = model_speaker(obs_speaker.unsqueeze(0).to(device))
    z, _, _, _ = model_vae(obs_listener.unsqueeze(0).to(device))
    beta, _, _, _ = model_lbf(m, z)

    return obs_listener, obs_speaker, m.squeeze(0), z.squeeze(0), beta.squeeze(0)


# 1エピソードの実行
@torch.no_grad()
def play_one_episode(env, model_speaker, model_vae, model_lbf, model_controller):
    obs_listener_ep, obs_speaker_ep, act_ep, reward_ep, m_ep, z_ep, beta_ep = \
            [], [], [], [], [], [], []
    success = 0

    obs_agent = env.reset()
    obs_listener, obs_speaker, m, z, beta = \
            _encode_obs(env, model_speaker, model_vae, model_lbf, obs_agent)
    done = False

    while not done:
        # ゲーム環境のステップ実行
        act = model_controller.act(z.unsqueeze(0), beta.unsqueeze(0))
        obs_agent, reward, done, _ = env.step(act)

        if reward > 0:
            success = 1

        # 観測のエンコード
        obs_listener, obs_speaker, m, z, beta = \
                _encode_obs(env, model_speaker, model_vae, model_lbf, obs_agent)
        
        obs_listener_ep.append(obs_listener)
        obs_speaker_ep.append(obs_speaker)
        act_ep.append(act)
        reward_ep.append(reward)
        m_ep.append(m)
        z_ep.append(z)
        beta_ep.append(beta)
 
    return obs_listener_ep, obs_speaker_ep, act_ep, reward_ep, m_ep, z_ep, beta_ep, success


# 複数エピソードの実行
def play_many_episodes(env, n_episodes, policy=None):
    obs_listener_eps, obs_speaker_eps, reward_eps, done_eps = [], [], [], []

    for i in range(n_episodes):
        obs_listener_ep, obs_speaker_ep, reward_ep, done_ep = play_one_episode(env, policy)

        obs_listener_eps.append(obs_listener_eps)
        obs_speaker_eps.append(obs_speaker_eps)
        reward_eps.append(reward_eps)
        done_eps.append(done_eps)
 
    return obs_listener_eps, obs_speaker_eps, reward_eps, done_eps


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())

    fig = plt.figure(figsize=(20,20))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    fig.savefig("debug.png")

    plt.close(fig)
