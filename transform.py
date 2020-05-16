import os
import imageio
from PIL import Image

os.mkdir('data/mediumjpg')
for file in os.listdir('data/medium'):
	name = 'data/mediumjpg/' + file[:-4]
	os.mkdir(name)
	reader = imageio.get_reader('data/medium/' + file)
	for i,im in enumerate(reader):
		image = Image.fromarray(im)
		image.save(name + '/' + str(i) + '.jpg')

