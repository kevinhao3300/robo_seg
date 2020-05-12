import numpy as np
import robosuite as suite
import os
import argparse
import imageio
import time
import matplotlib.pyplot as plt
import random

from ProMP import *


def record(traj,name, env):
	writer = imageio.get_writer(name, fps=60)

	traj = np.array(traj).T
	# env = suite.make("SawyerLift", has_renderer=False)
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
	print(traj[0])
	print(traj[-1])
	for state in traj:
		env.viewer.set_camera(camera_id=0)
		env.set_robot_joint_positions(state)
		env.render()

def generate(map, size, name, env):
	keys = list(map.keys())
	for count in range(5072,5372):
		i = random.randint(0,len(keys)-1)
		# print('data/' + name + '/' + keys[i] + str(count) + '.mp4')
		record(sample(*map[keys[i]]), 'data/' + name + '/' + keys[i] + str(count) + '.mp4', env)


if __name__ == "__main__":
	# promp_map = np.load('promp_map.npy',allow_pickle='TRUE').item()
	# trajs = np.load('only_train.npy',allow_pickle='TRUE').item()
	# t = trajs['cube_data/HD']
	# del t[8]
	# promp_map['cube_data/HD'] = make_ProMP(t)
	# np.save('promp_map', promp_map)
	# promp_map = np.load('promp_map.npy',allow_pickle='TRUE').item()

	# how to generate sample trajectory
	# this goes through and samples a single trajectory for each type of demonstration
	# for k,v in promp_map.items():
	# 	for i in range(1):
	# 		record(sample(*v), 'data/cube' + k[9:] + str(i) + '.mp4')
	# trajs = np.load('only_train.npy',allow_pickle='TRUE').item()
	# rev = trajs['cube_data/HG'][0]
	# for i in range(len(rev)):
	# 	rev[i] = rev[i][::-1]
	# # play_traj(rev)
	# n = [norm,norm]
	# play_traj(sample(*make_ProMP(n))) 

	# trajs = np.load('only_train.npy',allow_pickle='TRUE').item()
	# AGH_map = {}
	# nodes = ['A','G','H']
	# for i in range(len(nodes)):
	# 	for j in range(len(nodes)):
	# 		if i != j:
	# 			AGH_map[nodes[i] + nodes[j]] = make_ProMP(trajs['cube_data/' + nodes[i] + nodes[j]])
	# np.save('AGH_map', AGH_map)


	AGH_map = np.load('AGH_map.npy',allow_pickle='TRUE').item()
	for k,v in AGH_map.items():
		play_traj(sample(*v))
	# env = suite.make("SawyerLift", has_renderer=True)
	# generate(AGH_map, 10000, 'large', env)

	
