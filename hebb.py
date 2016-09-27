# Super-simple, super-fast Hebbian learning of V1-like receptive fields
# Only takes a couple seconds
# Even nicer/smoother RFs if you run for longer with a smaller ETA
# Uses the Instar Hebbian rule, softmax competition, and adaptive thresholds to drive firing rates towards 1/NBCELLS (note that actual firing rates vary quite a lot!)
# Resulting firing is sparse, and often only once cell fires, but sometimes several cells have significant firing (rarely more than 3 though). 
# Requires patches of z-scored natural images as inputs, stored as linearized vectors in patches.npy (see makepatches.py)
# To plot the resulting RFs, run plotRFs.py

import numpy as np
np.set_printoptions(precision=4, suppress=True)

patches = np.load('patches.npy') / 19.0 # Because makepatches.py stores patches with standard dev. 19 (to expand the range before conversion to integers)
NBCELLS = 100   
SIZE = patches.shape[1]         # Size of each input patch (= RFWIDTH * RFHEIGHT)
ETA = .01                       # Learning rate
nbframes = patches.shape[0]     # Total number of patches in the file

# Weights initialization (all cells habe initially z-scored input weights)
w = np.random.randn(NBCELLS, SIZE).astype(float)
w -= np.mean(w, axis=1)[:,None]
w /= (1e-8 + np.std(w, axis=1)[:,None])
allps = []
firingrates = np.zeros(NBCELLS)
thres = np.zeros(NBCELLS)
ALPHATRACEFIRING = .99
TARGETRATE = 1.0 / NBCELLS

for nn in range(100000):  
    if nn % 100 == 0:
        print nn
    inp = patches[np.random.randint(nbframes), :]
    outs = w.dot(inp)
    outs -= thres
    outs[outs<0] = 0        # Thresholding
    p = np.exp(outs) / (1e-8 + np.sum(np.exp(outs)))    # Softmax creates competition between active cells. This approximates Foldiak's rule and LCA (very roughly).
    w += ETA * (inp[None, :] -  w) * p[:, None]         # Instar rule, from Steve Grossberg. See e.g. Vasilkoski et al. IJCNN 2011 for discussion and comparison with other rules.
    firingrates = ALPHATRACEFIRING * firingrates + (1-ALPHATRACEFIRING) * p     # We keep an exponential running average of the firing rate for each cell...
    thres +=  (firingrates - TARGETRATE)                                        # ...and use it to adjust threshold to drive firing rates towards TARGETRATE.
    #allps.append(p)
np.save('w.npy', w)


