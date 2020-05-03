import numpy as np
import robosuite as suite
import os
import argparse
import imageio

from ProMP import *


def record(traj,name):
	writer = imageio.get_writer(name, fps=60)

	traj = np.array(traj).T
	env = suite.make("SawyerLift", has_renderer=False)
	env.reset()

	for state in traj:
		env.set_robot_joint_positions(state)
		obs,_,_,_ = env.step([0]*env.dof)
		i = obs["image"][::-1]
		writer.append_data(i)

def play_traj(traj):
	traj = np.array(traj).T
	env = suite.make("SawyerLift", has_renderer=True)
	env.reset()

	for state in traj:
		env.viewer.set_camera(camera_id=0)
		env.set_robot_joint_positions(state)
		env.render()

if __name__ == "__main__":
	grouped_traj_map = np.load('only_train.npy',allow_pickle='TRUE').item()
	promp_map = {}
	for k,v in grouped_traj_map.items():
		if k != 'cube_data/HD':
			promp_map[k] = make_ProMP(v)
	# for promp in promp_map.values():
	# 	s = sample(promp)
	# 	play_traj(s)


	np.save('promp_map',promp_map)


