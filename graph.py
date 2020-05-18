import matplotlib.pyplot as plt
import numpy as np

# with open('test_log/100smalljpg.txt','r') as f:
# 	x = []
# 	arr = []
# 	for line in f:
# 		epoch, acc = line.split()
# 		arr.append(float(acc))
# 		x.append(int(epoch))
# 	print(arr)
# 	fig = plt.figure()
# 	ax1 = fig.add_subplot()
# 	ax1.set_title('Test Set Loss with 100 Demonstrations')
# 	ax1.set_ylabel('cross entropy loss')
# 	ax1.set_xlabel('epoch #')

# 	ax1.plot(x,arr, lw=2)

# 	plt.show()

acc = np.load('sureness.npy')
acc /= np.max(acc)
x = list(range(0,100))
# # plt.figure()
# # plt.xlabel = 'cross entropy loss'

# # plt.plot(x,acc)
# # plt.show()


fig = plt.figure()
ax1 = fig.add_subplot()
ax1.set_title('Sureness with 100 Demonstrations')
ax1.set_ylabel('cross entropy loss')
ax1.set_xlabel('epoch #')

ax1.plot(x,acc, lw=2)

plt.show()