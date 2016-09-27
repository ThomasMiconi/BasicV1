
import glob
import numpy as np
import scipy as sp
from scipy import ndimage

RFSIZE = 13
CROPMARGIN = 13
CROPSIZE = RFSIZE + 2* CROPMARGIN
NBFRAMES = 100000

fnames = glob.glob('../stdp/images/ImageNet2/images/*')   # Replace with your own directory containing many natural images.
images = []
imageso = []

nbimages = 0
while 1:
    #print nbimages, 
    if nbimages % 10 == 0:
        print nbimages
    while 1:

        # Let's try to load an image
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

    # We extract a small patch from the image, of size CROPSIZE * CROPSIZE
    # CROPSIZE is larger than RFSIZE, because we want ton include some margin around the patch to avoid edge effects during the filtering.
    xpos = np.random.randint(im.shape[0]-CROPSIZE-1)
    ypos = np.random.randint(im.shape[1]-CROPSIZE-1)
    imcrop = im[xpos:xpos+CROPSIZE, ypos:ypos+CROPSIZE]
    #imageso.append(imcrop)
    # Difference-of-Gaussians filtering emulates the processing of retinal / LGN cells, decorrelates nearby pixels, and also flattens the spectrum.
    imfilt = sp.ndimage.filters.gaussian_filter(imcrop, 1.0) - sp.ndimage.filters.gaussian_filter(imcrop, 2.0) 
    imfilt = imfilt[CROPMARGIN:CROPMARGIN+RFSIZE, CROPMARGIN:CROPMARGIN+RFSIZE].flatten()

    # Z-scoring!
    imfilt -= np.mean(imfilt)
    if np.std(imfilt) < 1.0:
        print "Not enough variation in image."
        continue
    imfilt /= np.std(imfilt)

    # We want to convert this to 8-bit signed integers, to save space
    imfilt *= 19.0    # Expand the range as much as possible while keeping the vast majority of the pixels  within the -127:127 interval
    imfilt = np.rint(np.clip(imfilt, -127, 127)).astype('int8')   # There should be very little clipping
    images.append(imfilt)
    #print
    nbimages += 1
    if nbimages == NBFRAMES:
        break

aimages = np.array(images) 
np.save('patches.npy', aimages)


