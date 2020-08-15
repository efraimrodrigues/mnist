import numpy as np

import utils

images = utils.training_images()
labels = utils.training_labels()

samples = 100

x_trans = []
y = []

for i in range(0, samples):
    x_trans.append(np.matrix.flatten(images[i]/255))
    
    cod = [0] * 10
    cod[labels[i]] = 1
    y.append(cod)

x = np.transpose(x_trans)
y = np.transpose(y)

yx = np.dot(y, x_trans)

xx_inv = np.linalg.pinv(np.dot(x, x_trans))

w = np.dot(yx, xx_inv) 



#import matplotlib.pyplot as plt
#plt.imshow(images[2], cmap='gray')
#plt.show()