import numpy as np
import matplotlib.pyplot as plt

plt.ion()
np.set_printoptions(precision=4, suppress=True)

w = np.load('w.npy')
NBCELLS = w.shape[0]
SIDE = np.ceil(np.sqrt(NBCELLS )).astype(int)
RFSIZE = np.sqrt(w.shape[1])

fig = plt.figure()

for nn in range(NBCELLS):
    ax = plt.subplot(SIDE, SIDE, nn+1)
    ax.matshow(np.reshape(w[nn,:], (RFSIZE, RFSIZE)), cmap='Greys_r')
    ax.set_axis_off()

plt.draw()
plt.savefig('RFs.png', bbox_inches='tight')



