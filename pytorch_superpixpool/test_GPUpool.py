from __future__ import print_function
import torch
from suppixpool_layer import SupPixPool, SupPixUnpool
import torch.nn as nn
import numpy as np
import time
from skimage.segmentation import slic

if __name__ == "__main__":

    GPU = torch.device("cuda:0")
    batch_size = 1
    n_channels = 16
    xSize = 256
    ySize = 512

    X = torch.randn((batch_size,n_channels,xSize,ySize), dtype=torch.float32, device=GPU)
    spx = np.array([np.arange(xSize*ySize).reshape(xSize,ySize)]*batch_size)
    # spx = np.zeros((batch_size, xSize, ySize))
    spx = torch.from_numpy(spx) 
    spx = spx.to(GPU)
    # X + X 
    print ("INPUT ARRAY  ----------------- \n", X) 
    pool = SupPixPool()
    pld = pool(X, spx)
    print ("POOLED ARRAY ----------------- \n", pld)
    print ("Shape of pooled array: ", pld.size())
    unpool = SupPixUnpool()
    unpld = unpool(pld, spx)
    print ("Unpooling back to original: ", np.all(unpld == X))

    res = torch.autograd.gradcheck(pool, (X, spx), raise_exception=False)
    resUnpool = torch.autograd.gradcheck(unpool, (pld, spx), raise_exception=False) 

    print ("Gradients of pooling are {}.".format("correct" if res else "wrong")) # res should be True if the gradients are correct.
    print ("Gradients of unpooling are {}.".format("correct" if resUnpool else "wrong"))