from marlgrid.utils.video import GridRecorder

from envs import create_ChoosePathGridDefaultEnv

env = create_ChoosePathGridDefaultEnv()
env = GridRecorder(env, save_root="./", render_kwargs={"tile_size": 100})

obs = env.reset()
env.recording = True

count = 0
done = False

while not done:
    act = env.action_space.sample()
    obs, rew, done, _ = env.step(act)
    count += 1

env.export_video("test_env.mp4")
