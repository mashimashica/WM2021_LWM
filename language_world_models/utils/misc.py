import numpy as np
from PIL import Image
from torchvision import transforms

transform_speaker = transforms.Compose([
    transforms.Resize(11),
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
def play_one_episode(env, max_steps=30, policy=None):
    obs_listener_ep, obs_speaker_ep, reward_ep, done_ep = [], [], [], []

    obs_agent = env.reset()
    obs_speaker = get_obs_speaker(env)
    obs_listener = get_obs_listener(obs_agent)

    done = False
    while not done:
        act = None
        if policy is None:
            # 行動を決定（とりあえずランダム）
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
