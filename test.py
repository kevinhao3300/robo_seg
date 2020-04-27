import numpy as np
import robosuite as suite
import os
import argparse
import imageio




def play_traj(traj, name):
	writer = imageio.get_writer(name, fps=60)

	traj = np.array(traj).T
	env = suite.make("SawyerLift", has_renderer=False)
	env.reset()

	for state in traj:
		# env.viewer.set_camera(camera_id=0)
		env.set_robot_joint_positions(state)
		obs,_,_,_ = env.step([0]*env.dof)
		i = obs["image"][::-1]
		writer.append_data(i)
		# env.render()

if __name__ == "__main__":
	m = np.load('traj_data.npy',allow_pickle='TRUE', encoding='bytes').item()
	play_traj(m[b'cube_data/AB'][0], 'test.mp4')
	# for i,d in enumerate(m[b'cube_data/AB']):	
	# 	print(i)
	# 	play_traj(d,'AB' + '_' + str(i) + '.mp4')
	# for k in m:
	# 	print(k, len(m[k]), m[k][0].shape)

