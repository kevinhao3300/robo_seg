This repository utilises robosuite to create demonstrations of certain movements using human input. It then uses a combination of CNNs and RNNs to learn the classifcation of those trajectories.

myHuman.py - feed in a directory, such as --directory=data, saves human demonstration as a hdf5 file

playback.py - feed in the directory that you want to play hdf5 files from, such as --folder=data/1587919591_164246, will choose one at random and play it

recording.py - generates an mp4 file in local director, right now it just generates random movements
