'''
ゲーム環境の初期状態の可視化
'''
import matplotlib.pyplot as plt
import numpy as np

from envs import create_ChoosePathGridDefaultEnv


# 環境の初期化
env = create_ChoosePathGridDefaultEnv()
agent_obs_list = env.reset()

agent_obs = agent_obs_list[0].astype(np.uint8)
global_obs = env.grid.render(tile_size=11).astype(np.uint8)

# 画像として保存
plt.imsave('agent_obs.jpeg', agent_obs)
plt.imsave('global_obs.jpeg', global_obs)
