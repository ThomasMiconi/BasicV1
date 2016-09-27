
import glob
import numpy as np
import scipy as sp
from scipy import ndimage

RFSIZE = 13
CROPMARGIN = 13
CROPSIZE = RFSIZE + 2* CROPMARGIN
NBFRAMES = 100000

fnames = glob.glob('../stdp/images/ImageNet2/images/*')
images = []
imageso = []

nbimages = 0
while 1:
    #print nbimages, 
    if nbimages % 10 == 0:
        print nbimages
    while 1:
        try:
            fname = np.random.choice(fnames)
            im = sp.ndimage.imread(fname)
            im = im.astype(float)
        except:
            print "Unloadable image "+fname
            continue
        break   # If no exception raised, exit the loop

    if im.ndim == 3:
        im = np.mean(im, axis=2)  # i.e. along 3rd dimension (Color)
        
    if im.shape[0] < CROPSIZE +1 or im.shape[1] < CROPSIZE +1:
        print "Too small image"
        continue

    xpos = np.random.randint(im.shape[0]-CROPSIZE-1)
    ypos = np.random.randint(im.shape[1]-CROPSIZE-1)
    imcrop = im[xpos:xpos+CROPSIZE, ypos:ypos+CROPSIZE]
    #imageso.append(imcrop)
    imfilt = sp.ndimage.filters.gaussian_filter(imcrop, 1.0) - sp.ndimage.filters.gaussian_filter(imcrop, 2.0) 
    imfilt = imfilt[CROPMARGIN:CROPMARGIN+RFSIZE, CROPMARGIN:CROPMARGIN+RFSIZE].flatten()
    imfilt -= np.mean(imfilt)
    if np.std(imfilt) < 1.0:
        print "Not enough variation in image."
        continue
    imfilt /= np.std(imfilt)
    imfilt *= 19.0
    imfilt = np.rint(np.clip(imfilt, -127, 127)).astype('int8')
    images.append(imfilt)
    #print
    nbimages += 1
    if nbimages == NBFRAMES:
        break

aimages = np.array(images) 
#trimmin = np.percentile(aimages, 1)
#trimmax = np.percentile(aimages, 99)
#np.clip(aimages, trimmin, trimmax, out=aimages)  # alters the mean and variance...
np.save('patches.npy', aimages)


