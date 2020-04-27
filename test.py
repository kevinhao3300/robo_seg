import numpy as np
import robosuite as suite
import os
import argparse



def play_traj(traj):

	env = suite.make("SawyerLift", has_renderer=True)
	env.viewer.set_camera(camera_id=0)
	env.reset()

	for state in traj:
		env.set_robot_joint_positions(state)
		o,_,_,_ = env.step()
		print(obs["image"][::-1])
		env.render()

if __name__ == "__main__":
	# parser = argparse.ArgumentParser()
	# parser.add_argument("--file", type=str)
	# args = parser.parse_args()
	# trajs = np.load(args.file)
	# for coord in trajs:
	# 	for demonstration in coord: 
	# 		play_traj(demonstration)
	# play_traj(trajs)

	m = np.load('traj_data.npy',allow_pickle='TRUE', encoding='bytes').item()
	for k in m:
		print(k, len(m[k]), m[k][0].shape)

# def record(traj):
# 	writer = imageio.get_writer('video.mp4', fps=20)