![alt text](https://raw.githubusercontent.com/ThomasMiconi/BasicV1/master/RFs.png "Receptive fields")


This code implements super-simple, super-fast Hebbian learning of nice V1-like simple-cell receptive fields.

25 lines of Python, runs in a couple seconds (though the RF look even smoother if you run it for longer with a smaller learning rate).


This code uses the "instar" Hebbian rule (see e.g. [Vasilkoski et al. IJCNN 2011](http://cns-pc62.bu.edu/cn510/Papers/stability.pdf)), SoftMax competition between output cells, and adaptive thresholds to drive firing rates towards 1/NBCELLS (though the actual firing rates vary quite a lot).


The resulting firing is sparse, and most often only one cell fires. But sometimes several cells have significant firing (rarely more than 3 though). 

The main program is `hebb.py`, which can be run as-is. `makepatches.py` generates the input patches (it requires a directory with many natural images in it - see code). `plotRFs.py` plots the learned receptive fields. 

This repository also includes pre-computed patches of z-scored natural images as inputs, stored as linearized vectors in patches.npy (see `makepatches.py`).

To plot the resulting RFs, after running `hebb.py`, run `plotRFs.py`.

