import matplotlib.pyplot as plt

with open('test_log/100smalljpg.txt','r') as f:
	plt.figure()
	x = []
	arr = []
	for line in f:
		epoch, acc = line.split()
		arr.append(float(acc))
		x.append(int(epoch))
	print(x)
	print(arr)
	plt.plot(x,arr)
	plt.show()