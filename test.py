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
	promp_map = np.load('promp_map.npy',allow_pickle='TRUE').item()
	# how to generate sample trajectory
	# this goes through and samples a single trajectory for each type of demonstration
	for k,v in promp_map.items():
		play_traj(sample(*v))
	# start with 100
	# data loading 


