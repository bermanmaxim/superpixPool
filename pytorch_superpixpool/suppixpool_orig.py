from __future__ import print_function 
import chainer
import chainer.functions as F
import chainer.links as L
import util.nameDef as names
import numpy as np
import chainer
import math
from util.utilsGPU import read_code, load_kernel

from scipy import ndimage
from numba import jit
import sys
from os import path

GPU_KERNEL = path.join(path.dirname(path.abspath(__file__)), 'supervoxel_pooling.cu')
CUDA_MAX_THREADS = 256

cp = chainer.cuda.cupy

# ==========================================================================================
# AVERAGE POOLING 
# ==========================================================================================

# ----- 
# NUMBA
# -----

@jit
def average_numba(A, labels, K):

    sums = np.zeros(K, dtype=A.dtype)
    counts = np.zeros(K, dtype=np.int32)

    for pix in np.ndindex(labels.shape): # pix is an (x,y,z)-tuple.
        label = labels[pix]
        sums[label] += A[pix]
        counts[label] += 1
    sums = np.divide(sums, counts, where=counts != 0)
    return sums, counts

class SupVoxPoolNumba_avg(chainer.Function):
    """
    Supervoxel-wise average pooling.

    - own implementation using numba
    - assumes that there are nClasses classes, so the pooling 
    is done once for every slice along the second axis (every class score). 
    assumed shape of the image: (batch_size, nClasses, *, *)
    """

    def initialize_arrays(self, img, dimension, batch_size, K, args):

        if dimension == 3: 
            innerloop = self._innerloopGT
            outputs = np.zeros((batch_size, K), dtype=img.dtype)
            self.counts = np.zeros((batch_size, K), dtype=np.int32)

        else:
            innerloop = self._innerloopFull
            args["nClasses"] = img.shape[1]
            outputs = np.zeros((batch_size, img.shape[1], K), dtype=img.dtype)
            self.counts = np.zeros((batch_size, img.shape[1], K), dtype=np.int32)

        args["output"] = outputs
        return innerloop 

    def forward_cpu(self, inputs):
        img, labels = inputs

        K = np.max(labels)+1
        batch_size = img.shape[0]

        args = {"labels": labels, 
                "img": img,
                "K": K}

        dimension = len(img.shape)
        innerloop = self.initialize_arrays(img, dimension, batch_size, K, args)

        for batch in xrange(batch_size):
            innerloop(batch, **args)

        return args["output"],

    def _innerloopFull(self, batchIx, img, nClasses, labels, output, K): 
        for classIx in xrange(nClasses): 
            output[batchIx, classIx, :], self.counts[batchIx, classIx, :] = average_numba(img[batchIx, classIx, :, :, :], labels[batchIx, :, :, :], K)

    def _innerloopGT(self, labels, output, K):
        output[batchIx, :], self.counts[batchIx, :] = average_numba(img[batchIx, :, :, :], K)

    # @jit 
    def backward_cpu(self, inputs, grad_outputs):
        img, labels = inputs
        grad_in = np.zeros_like(img)
        if len(img.shape) == 5: 
            for batch in xrange(img.shape[0]):
                bCounts = self.counts[batch, :,:]
                blabels = labels[batch, :,:,:]
                bOuts = grad_outputs[0][batch, :,:]
                grad_in[batch, :,:,:] = (1.0/(bCounts[:, blabels])*bOuts[:, blabels]).astype(img.dtype)
                # print(self.counts[batch, :, labels[batch,:,:,:]].shape)
                # grad_in[batch, :,:,:,:] = (np.divide(1.0, self.counts[batch, :, labels[batch,:,:,:]], where=self.counts[batch,:, labels[batch,:,:,:]] != 0)*grad_outputs[0][batch,:, labels[batch, :,:,:]]).astype(img.dtype)
        else: 
            for batch in xrange(img.shape[0]):
                grad_in = (np.divide(1.0, self.counts[batch, labels[batch,:,:,:]], where=self.counts[batch, labels[batch,:,:,:]] != 0)*grad_outputs[0][batch, labels]).astype(img.dtype)
        return grad_in, np.zeros_like(labels) # Second argument needs to be returned to match shapes of arguments in forward and backward passes. 

# -------
# NDIMAGE 
# -------

class SupVoxPoolNdImage_avg(chainer.Function):
    """
    Supervoxel-wise average pooling.

    PRELIMINARY VERSION -- 
    - own implementation using numba
    - assumes that there are 4 classes, so the pooling 
    is done once for every slice along the first axis (every class score). 
    assumed shape of the image: (4, *, *, *)
    """
    def forward_cpu(self, inputs): 
        img, labels = inputs
        outputs = []

        for batch in xrange(img.shape[0]):
            batchOut = []
            for classIx in xrange(img.shape[1]):
                # outputs.append(ndimage.maximum(img[classIx, :,:,:], labels=labels, index= np.unique(labels)))

                batchOut.append(ndimage.mean(img[batch, classIx, :,:,:], labels=labels[batch,:,:,:], index=range(labels[batch,:,:,:].max()+1)).astype(img.dtype))
            outputs.append(batchOut)
        return np.array(outputs),

    def backward_cpu(self, inputs, grad_outputs):
        img, labels = inputs

        grad_in = np.zeros_like(img)
        for batch in xrange(img.shape[0]):
            counts = np.tile(np.bincount(np.ravel(labels[batch,:,:,:])), (img.shape[1],1))
            blabels = labels[batch,:,:,:]
            bgrad_out = grad_outputs[0][batch,:,:]
            grad_in[batch, :, :, :] = (1.0/counts[:,blabels])*bgrad_out[:, blabels].astype(img.dtype)
            # grad_in[batch, :,:,:] = (1.0/(counts[:, labels[batch,:,:,:]])*grad_outputs[0][batch, :, labels[batch,:,:,:]]).astype(img.dtype)
        return grad_in, np.zeros_like(labels) 

# ---
# GPU
# ---

class SupVoxPoolGPU_avg(chainer.Function): 
    """
    Another version of the GPU-based pooling layer. This version should be more efficient in one sense
    (that it's more-or-less K-independent). It does benefit less from parallellization.

    shape assumptions
    -----------------

    - the first axis of both image and labels contains the different samples in the minibatch
    - if image has 5 axes, the second axis denotes the class (/channel) 
    - the last 3 axes are always x,y and z.
    """

    def __init__(self):
        super(SupVoxPoolGPU_avg, self).__init__()
        self.divide = True 

    def initialize_arrays(self, img, dimension, K):  

        batch_size = img.shape[0]

        if dimension == 5: # multiple classes 
            n_classes = img.shape[1]
            outputs = cp.zeros((batch_size, n_classes, K), dtype=img.dtype)
            expand_axis = 1 

        elif dimension == 4: # GT image 
            n_classes = 1 
            outputs = cp.zeros((batch_size, K), dtype=img.dtype)
            expand_axis = None

        counts = cp.zeros((batch_size,K), dtype=cp.int32)

        return batch_size, n_classes, outputs, counts, expand_axis

    def forward_gpu(self, inputs):
        img, labels = inputs

        # ------------------
        # INPUT VERIFICATION
        # ------------------
        assert img.flags["C_CONTIGUOUS"]
        assert len(labels.shape)>=4
        assert img.dtype == cp.float32 or img.dtype == cp.int32
        assert(labels.flags["C_CONTIGUOUS"])

        # ----------
        # INITIALIZE
        # ----------

        volumeSize = np.prod(img.shape[-3:])
        blockSize = np.min((CUDA_MAX_THREADS, volumeSize))
        nbPixPerThread = int(math.ceil(volumeSize/float(blockSize)))

        K = int(labels.max()+1)
        
        # -------------------------------
        # FIGURE OUT MEANING OF EACH AXIS
        # -------------------------------
        dimension = len(img.shape)
        batch_size, n_classes, outputs, counts, expand_axis = self.initialize_arrays(img, dimension, K)
        self.counts = counts

        self.code = read_code(GPU_KERNEL) # TODO: Should be able to be moved outside this function.
        
        # # ---
        # # PERFORM AVERAGE ON GPU
        # # ---
        
        summation = load_kernel('avg_pooling', self.code)

        # print("labels: ", cp.ravel(labels))
        args = (img, labels.astype(cp.int32), outputs, self.counts, 
                volumeSize, n_classes, batch_size, nbPixPerThread, K)

        block = (blockSize,)  # block size = size of one volume (one block per class) 
        grid = (np.prod(img.shape[:-3]), batch_size) # 1 block for each class 
        summation(grid, block, args)
        
        if self.divide: 
            if expand_axis is not None: 
                outputs /= cp.repeat(cp.expand_dims(self.counts, expand_axis), n_classes, expand_axis)
            else:
                outputs /= self.counts # TODO maybe write kernel for this if it seems that cupy doesn't parallellize this. 
                                       # If it does, a new call to kernel might cause too much overhead. 
        return outputs,

    def backward_gpu(self, inputs, grad_outputs):

        # print("backprop running")
        # print("number of gradients: ", len(grad_outputs))
        # for gr in grad_outputs:
        #     print("output grads to propagate: ", cp.where(gr != 0))

        img, labels = inputs
        assert grad_outputs[0].dtype == cp.float32
        # print(img)
        grad_in = cp.zeros_like(img)

        K = int(labels.max()+1)
        volumeSize = np.prod(img.shape[-3:])

        dimension = len(img.shape)
        batch_size, n_classes, _, _, _ = self.initialize_arrays(img, dimension, K)

        # print("forward pass -------------")
        # print("batch_size: ", batch_size)
        # print("n_classes: ", n_classes)
        # print("outputs: \n", outputs)
        # print("tileCounts: ", tileCounts)
        # print("counts: ", self.counts)

        blockSizeX = 32
        blockSizeY = min(CUDA_MAX_THREADS/32, n_classes)
        blockSizeZ = 1

        nbBlocksX = int(math.ceil(volumeSize/float(blockSizeX)))
        nbBlocksY = int(math.ceil(n_classes/float(blockSizeY)))
        nbBlocksZ = int(math.ceil(batch_size/float(blockSizeZ)))

        kern = load_kernel('bw_avg_pooling', self.code)

        args = (grad_in, self.counts, labels.astype(cp.int32), grad_outputs[0], K, volumeSize, n_classes, batch_size, chainer.config.train)
        block = (blockSizeX, blockSizeY, blockSizeZ)
        grid = (nbBlocksX, nbBlocksY, nbBlocksZ)
        kern(grid, block, args=args)
        return grad_in, cp.zeros_like(labels) # Second argument needs to be returned to match shapes of arguments in forward and backward passes. 

# ==========================================================================================
# MAX POOLING 
# ==========================================================================================

# ----- 
# NUMBA 
# -----
@jit
def maximum_numba(A, labels):

    # xp = np #chainer.cuda.get_array_module(A) 

    n = labels.max() + 1
    values = -np.inf * np.ones(n, dtype=A.dtype)
    indices = np.zeros((n, len(A.shape)), dtype=np.int) 
    for pix in np.ndindex(labels.shape): # pix is an (x,y,z)-tuple.
        candidate = A[pix]
        if candidate > values[labels[pix]]: 
            values[labels[pix]] = candidate
            indices[labels[pix],:] = pix
        # values[:, labels[pix]] = np.maximum(A[:,pix], values[:, labels[ix]])
    return values, indices

class SupVoxPoolNumba(chainer.Function):
    """
    Supervoxel-wise max/average pooling.
    """

    @jit
    def forward_cpu(self, inputs):
        img, labels = inputs
        outputs = []
        self.max_indices = []
        for batch in xrange(img.shape[0]):
            batchOut = []
            batchIdx = []
            for classIx in xrange(img.shape[1]):
                values, indices = maximum_numba(img[batch, classIx,:,:,:], labels[batch,:,:,:])
                batchIdx.append(indices)
                batchOut.append(values)
            self.max_indices.append(batchIdx)
            outputs.append(batchOut)

        return np.array(outputs),

    @jit 
    def backward_cpu(self, inputs, grad_outputs):
        # xp = np #chainer.cuda.get_array_module(*inputs) 
        img, labels = inputs
        grad_in = np.zeros_like(img)

        for batchIdx in xrange(img.shape[0]):
            for classIx in xrange(img.shape[1]):
                grad_in[batchIdx, classIx, self.max_indices[batchIdx][classIx][:,0], self.max_indices[batchIdx][classIx][:,1], self.max_indices[batchIdx][classIx][:,2]] = \
                grad_outputs[0][batchIdx, classIx, :] # grad_outputs should have the same length as the list of selected indices.
                
                # _, max_indices = maximum_numba(img[classIx,:,:,:], labels)
                # grad_in[classIx, max_indices[0], max_indices[1], max_indices[2]] = \
                # grad_outputs[0][classIx,:] # grad_outputs should have the same length as the list of selected indices.
        return grad_in, np.zeros_like(labels) # Second argument needs to be returned to match shapes of arguments in forward and backward passes. 

# ------------- 
# GPU version 1
# ------------- 

class SupVoxPoolGPU(chainer.Function):  

    def forward_gpu(self, inputs):
        img, labels = inputs
        self.max_indices = cp.zeros(img.size, dtype=cp.int32)

        volumeSize = np.prod(img.shape[1:])
        blockSizeX = np.min((64, volumeSize))
        blockSizeY = 1 
        blockSizeZ = 1

        nbBlocksX = int(math.ceil(volumeSize/float(blockSizeX)))

        K = int(labels.max()+1)


        outputs = (-np.inf*cp.ones((img.shape[0], K))).astype(img.dtype)
        self.max_indices = -cp.ones(outputs.shape, dtype=cp.int32) # Initialize as -1 so negative values can be ignored in backward pass.
                                                                   # This is a bit wasteful, only saving the ones that matter is better, TODO: look at this later  

        self.code = read_code(GPU_KERNEL) # TODO: Should be able to be moved outside this function. But it needs the information in config ... 
        kern = load_kernel('max_pooling', self.code)

        args = (img, labels, self.max_indices, 
                volumeSize, img.shape[0], K)

        block = (blockSizeX, blockSizeY, blockSizeZ)  # block size = size of one volume (one block per class) 
        grid = (nbBlocksX, img.shape[0], K)

        # print("indices before: ", self.max_indices)
        kern(grid, block, shared_mem = blockSizeX, args=args)
        fill_vals = load_kernel('fill_values', self.code)
        blockSizeX = 16
        blockSizeY = 16
        nbBlocksX = int(math.ceil(img.shape[0]/float(blockSizeX)))
        nbBlocksY = int(math.ceil(K/float(blockSizeY)))
        block = (blockSizeX, blockSizeY)
        grid = (nbBlocksX, nbBlocksY)

        args = (img, self.max_indices, K, img.shape[0], outputs)
        fill_vals(grid, block, args=args)
        # print("indices after: ", self.max_indices)
        return outputs,

    def backward_gpu(self, inputs, grad_outputs):

        img, labels = inputs
        # print(img)
        grad_in = cp.zeros_like(img)

        K = int(labels.max()+1)
        blockSizeX = 32
        blockSizeY = min(CUDA_MAX_THREADS/32, img.shape[0])
        nbBlocksX = int(math.ceil(K/float(blockSizeX)))
        nbBlocksY = int(math.ceil(img.shape[0]/float(blockSizeY)))

        kern = load_kernel('bw_max_pooling', self.code)
        # print("before bw: ", self.max_indices)
        args = (grad_in, img, self.max_indices, K*img.shape[0], grad_outputs[0], K)
        block = (blockSizeX, blockSizeY) # block size = size of one row in the labels volume
        grid = (nbBlocksX, nbBlocksY) 
        kern(grid, block, args=args)
        return grad_in, cp.zeros_like(labels) # Second argument needs to be returned to match shapes of arguments in forward and backward passes. 

# -------------
# GPU version 2
# -------------

class SupVoxPoolGPU_v2(chainer.Function): 
    """
    Another version of the GPU-based pooling layer. This version should be more efficient in one sense
    (that it's more-or-less K-independent). It does benefit less from parallellization. 
    """

    def initialize_arrays(self, img, dimension, K):  

        batch_size = img.shape[0]

        if dimension == 5: # multiple classes 
            n_classes = img.shape[1]
            outputs = cp.zeros((batch_size, n_classes, K), dtype=img.dtype)

        elif dimension == 4: # GT image 
            n_classes = 1 
            outputs = cp.zeros((batch_size, K), dtype=img.dtype)

        return batch_size, n_classes, outputs

    def forward_gpu(self, inputs):

        img, labels = inputs

        # ------------------
        # INPUT VERIFICATION
        # ------------------
        assert img.dtype == cp.float32 or img.dtype == cp.int32
        assert labels.dtype == cp.int32 or labels.dtype == cp.int64
        assert len(labels.shape)>=4

        labels = labels.astype(cp.int32)
        # ----------
        # INITIALIZE
        # ----------

        volumeSize = np.prod(img.shape[-3:])
        blockSize = np.min((CUDA_MAX_THREADS, volumeSize)) 
        nbPixPerThread = int(math.ceil(volumeSize/float(blockSize)))

        K = int(labels.max()+1)

        # -------------------------------
        # FIGURE OUT MEANING OF EACH AXIS
        # -------------------------------
        
        dimension = len(img.shape)

        batch_size, n_classes, outputs = self.initialize_arrays(img, dimension, K)
        self.max_indices = -cp.ones(outputs.shape, dtype=cp.int32) # Initialize as -1 so negative values can be ignored in backward pass.
                                                                   # This is a bit wasteful, only saving the ones that matter is better, TODO: look at this later  
        self.code = read_code(GPU_KERNEL) # TODO: Should be able to be moved outside this function. But it needs the information in config ... 

        # ---
        # PERFORM ARG MAX ON GPU
        # ---
        kern = load_kernel('max_pooling_v2', self.code)
        args = (img, labels.astype(cp.int32), self.max_indices, 
                volumeSize, n_classes, batch_size, nbPixPerThread, K)

        block = (blockSize,)  # block size = size of one volume (one block per class) 
        grid = (np.prod(img.shape[:-3]), batch_size) # 1 block for each class 

        kern(grid, block, args)
        
        # print("max_indices: ", self.max_indices)
        # print("corresponding labels: ", cp.ravel(labels)[self.max_indices])
        # ---
        # FILL IN CORRESPONDING VALUES
        # ---

        fill_vals = load_kernel('fill_values', self.code)
        blockSizeX = 16
        blockSizeY = CUDA_MAX_THREADS/blockSizeX
        nbBlocksX = int(math.ceil(n_classes/float(blockSizeX)))
        nbBlocksY = int(math.ceil(K/float(blockSizeY)))
        block = (blockSizeX, blockSizeY)
        grid = (nbBlocksX, nbBlocksY, batch_size)

        args = (img, self.max_indices, K, n_classes, batch_size, outputs)
        fill_vals(grid, block, args=args)

        return outputs,

    def backward_gpu(self, inputs, grad_outputs):

        img, labels = inputs
        # print(img)
        grad_in = cp.zeros_like(img)

        K = int(labels.max()+1)
        dimension = len(img.shape)
        batch_size, n_classes, _ = self.initialize_arrays(img, dimension, K)

        blockSizeX = 32
        blockSizeY = min(CUDA_MAX_THREADS/32, n_classes)
        nbBlocksX = int(math.ceil(K/float(blockSizeX)))
        nbBlocksY = int(math.ceil(n_classes/float(blockSizeY)))

        kern = load_kernel('bw_max_pooling', self.code)
        args = (grad_in, img, self.max_indices, n_classes, grad_outputs[0], K, batch_size)
        block = (blockSizeX, blockSizeY) # block size = size of one row in the labels volume
        grid = (nbBlocksX, nbBlocksY, batch_size) 
        kern(grid, block, args=args)
        return grad_in, cp.zeros_like(labels) # Second argument needs to be returned to match shapes of arguments in forward and backward passes. 

# -------
# ndimage
# -------

class SupVoxPoolNdImage(chainer.Function): 
    """
    Supervoxel-wise max/average pooling.

    PRELIMINARY VERSION -- 
    - Based on scipy ndimage
    - assumes that there are 4 classes, so the max pooling 
    is done once for every slice along the first axis (every class score). 
    assumed shape of the image: (4, *, *, *)
    - CPU-only, no GPU-compatibility built-in yet. 
    """

    def forward_cpu(self, inputs): 
        img, labels = inputs
        outputs = []
        for batch in xrange(img.shape[0]):
            batchOut = []
            for classIx in xrange(img.shape[1]):
                # outputs.append(ndimage.maximum(img[classIx, :,:,:], labels=labels, index= np.unique(labels)))
                batchOut.append(ndimage.maximum(img[batch, classIx, :,:,:], labels=labels[batch,:,:,:], index=range(labels[batch,:,:,:].max()+1)))
            outputs.append(batchOut)
        return np.array(outputs),

    def backward_cpu(self, inputs, grad_outputs):
        img, labels = inputs
        grad_in = np.zeros_like(img)
        for batch in xrange(img.shape[0]):
            for classIx in xrange(img.shape[1]):
            # indices = np.array(ndimage.maximum_position(img[classIx, :,:,:], labels=labels, index=np.unique(labels)))
                indices = np.array(ndimage.maximum_position(img[batch, classIx, :,:,:], labels=labels[batch,:,:,:], index=range(labels[batch,:,:,:].max()+1)))
                grad_in[batch, classIx, indices[:, 0], indices[:, 1], indices[:, 2]] = grad_outputs[0][batch, classIx,:] # grad_outputs should have the same length as the list of selected indices.
        return grad_in, np.zeros_like(labels) # Second argument needs to be returned to match shapes of arguments in forward and backward passes. 

# ==========================================================================================
# MAJORITY VOTE POOLING (GT) 
# ==========================================================================================

# class SupVoxPoolMajority(chainer.Function): 
#     """
#     Used to format GT labels into the supervoxel labels
#     """

#     def forward_cpu(self, inputs):

#         img, labels = inputs

#         # ------------------
#         # INPUT VERIFICATION
#         # ------------------
#         assert img.dtype == np.int32
#         assert labels.dtype == np.int32 or labels.dtype == np.int64
#         assert len(labels.shape)>=4

#         labels = labels.astype(np.int32)
     
#         # ----------
#         # INITIALIZE
#         # ----------

#         # volumeSize = np.prod(img.shape[-3:])
#         # blockSize = np.min((CUDA_MAX_THREADS, volumeSize)) 
#         # nbPixPerThread = int(math.ceil(volumeSize/float(blockSize)))

#         K = int(labels.max()+1)
#         n_classes = int(img.max()+1)

#         # -------------------------------
#         # FIGURE OUT MEANING OF EACH AXIS
#         # -------------------------------
    
#         outputs = np.zeros((n_classes, K))
#         dimension = len(img.shape)

#         batch_size, n_classes, outputs = self.initialize_arrays(img, dimension, K)
#         self.max_indices = -cp.ones(outputs.shape, dtype=cp.int32) # Initialize as -1 so negative values can be ignored in backward pass.
#                                                                    # This is a bit wasteful, only saving the ones that matter is better, TODO: look at this later  
#         self.code = read_code(GPU_KERNEL) # TODO: Should be able to be moved outside this function. But it needs the information in config ... 

#         # ---
#         # PERFORM ARG MAX ON GPU
#         # ---
#         kern = load_kernel('max_pooling_v2', self.code)
#         args = (img, labels.astype(cp.int32), self.max_indices, 
#                 volumeSize, n_classes, batch_size, nbPixPerThread, K)

#         block = (blockSize,)  # block size = size of one volume (one block per class) 
#         grid = (np.prod(img.shape[:-3]), batch_size) # 1 block for each class 

#         kern(grid, block, args)
        
#         # print("max_indices: ", self.max_indices)
#         # print("corresponding labels: ", cp.ravel(labels)[self.max_indices])
#         # ---
#         # FILL IN CORRESPONDING VALUES
#         # ---

#         fill_vals = load_kernel('fill_values', self.code)
#         blockSizeX = 16
#         blockSizeY = CUDA_MAX_THREADS/blockSizeX
#         nbBlocksX = int(math.ceil(n_classes/float(blockSizeX)))
#         nbBlocksY = int(math.ceil(K/float(blockSizeY)))
#         block = (blockSizeX, blockSizeY)
#         grid = (nbBlocksX, nbBlocksY, batch_size)

#         args = (img, self.max_indices, K, n_classes, batch_size, outputs)
#         fill_vals(grid, block, args=args)

#         return outputs,