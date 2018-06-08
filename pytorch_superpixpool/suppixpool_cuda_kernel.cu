#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm> //min
#include <math.h> // ceil
#include <stdio.h>

template <typename scalar_t> 
__device__
float atomicMaxIndex(int* address, int newCandidate, const scalar_t* image)
{
    int* address_as_int =(int*)address;
    int old = *address_as_int, assumed;

    while (old < 0) 
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, newCandidate);
    }
    // old is definitely initialized and therefore a valid index. 
    // printf("Unsigned %lu \n", old);
    // printf("Signed %ld \n", (int) old);

    while (image[newCandidate] > image[old]) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, newCandidate);
        }
    return old;
}

template <typename scalar_t> 
__global__
void fill_values(
    const scalar_t* image,
    const int* maxIdx,
    const int K, 
    const int nClasses,
    const int batchSize,
    scalar_t* outVals)
{ 
    int batch = blockIdx.z*blockDim.z + threadIdx.z; 
    int channel = blockIdx.x*blockDim.x + threadIdx.x; 
    int relevantLabel = blockIdx.y*blockDim.y+threadIdx.y;
    int outIdx = batch*nClasses*K + (K*channel) + relevantLabel;

    if (outIdx<K*nClasses*batchSize && outIdx>=0){
        int maximizer = maxIdx[outIdx];
        if (maximizer >= 0){
            outVals[outIdx] = image[maximizer];
        }
    }
}

// -------
// KERNELS 
// ------- 

template <typename scalar_t> 
__global__
void spx_max_pooling_forward_kernel(
    const scalar_t* __restrict__ image,
    const int* __restrict__ labels,
    int* outIdx,
    const int imWidth,
    const int imHeight, 
    const int threadW, 
    const int threadH,
    const int nClasses, 
    const int batchSize,
    const int K)
{   
    // extern __shared__ int sharedMem[];

    int batch = blockIdx.y;
    int channel = blockIdx.x;
    int x = threadIdx.x*threadW; 
    int y = threadIdx.y*threadH; 
    int imgStartIdx = batch*nClasses*imWidth*imHeight+
                      channel*imWidth*imHeight+
                      y*imWidth+
                      x;
    
    int labelStartIdx = batch*imWidth*imHeight +
                        y*imWidth+
                        x; 

    if (x < imWidth && y < imHeight && channel < nClasses && batch < batchSize)
    {
        int imgIndex = imgStartIdx;
        int labelIndex = labelStartIdx;
        int label;
        int outIndex;
        // int runningIdx;

        for (int idY=0; idY < threadH; idY++)
        {
            imgIndex = imgStartIdx + idY*imWidth;
            labelIndex = labelStartIdx + idY*imWidth;
            if (y+idY < imHeight)
            {
                for (int idX=0; idX<threadW; idX++)
                {
                    if (x + idX < imWidth){
                        label = labels[labelIndex];
                        outIndex = batch*nClasses*K + K*channel + label;
                        atomicMaxIndex(&outIdx[outIndex], imgIndex, image);
                        imgIndex += 1;
                        labelIndex += 1; 
                    }
                    else{break;}
                }
            }else{break;}
        }
    }
}



// template <typename scalar_t> 
// __global__
// void spx_max_pooling_forward_kernel(
//     const scalar_t* __restrict__ image,
//     const int* __restrict__ labels,
//     int* outIdx,
//     const int volumeSize, 
//     const int nClasses, 
//     const int batchSize, 
//     const int nbPixPerThread,
//     const int K)
// {   
//     // extern __shared__ int sharedMem[];

//     int batch = blockIdx.y;
//     int channel = blockIdx.x;
//     int x = threadIdx.x; 
//     int imgStartIdx = batch*volumeSize*nClasses+
//                       channel*volumeSize+
//                       x*nbPixPerThread;
    
//     int labelStartIdx = volumeSize*batch + x*nbPixPerThread; 

//     if (x*nbPixPerThread < volumeSize && channel < nClasses && batch < batchSize)
//     {
//         int imgIndex = imgStartIdx;
//         int labelIndex = labelStartIdx;
//         int label;
//         int outIndex;
//         int runningIdx;

//         // else{printf("%li \n", (labelIndex-batch*volumeSize+(nbPixPerThread-1)));}
//         for (int i=0; i<nbPixPerThread; i++)
//         {
//             imgIndex = imgStartIdx+i;
//             labelIndex = labelStartIdx+i;
        
//             runningIdx = labelIndex-batch*volumeSize;
//             if (runningIdx < volumeSize)
//             {   
//                 // if (runningIdx == volumeSize-1){
//                 //     printf("imgIndex: %li \n", imgIndex);
//                 //     printf("Image value: %f \n", image[imgIndex]);
//                 // }
//                 // candidate = image[imgIndex];
//                 label = labels[labelIndex]; 
//                 outIndex = batch*nClasses*K + K*channel + label;
//                 atomicMaxIndex(&outIdx[outIndex], imgIndex, image);
//             }
//             else
//             {
//                 return;
//             }
//         }
//     }
// }

template <typename scalar_t> 
__global__
void spx_max_pooling_backward_kernel(
    scalar_t* grad_in, 
    const scalar_t* img,
    const int* indices, 
    const int nClasses,
    const scalar_t* grad_outputs,
    const int K,
    const int batchSize)
{
    int batch = blockIdx.z*blockDim.z + threadIdx.z;
    int label = blockIdx.x*blockDim.x + threadIdx.x;
    int channel = blockIdx.y*blockDim.y + threadIdx.y;
    int lstIdx = batch*nClasses*K+ channel*K + label;
    int lstSize = batchSize*nClasses*K;
    if (lstIdx<lstSize)
    {       
        int imgIndex = indices[lstIdx];
        if (imgIndex >= 0)
        {
            grad_in[imgIndex] = grad_outputs[lstIdx];
        }
    }
}

// ---------
// Wrappers
// --------- 

std::vector<at::Tensor> suppixpool_max_cuda_forward(
    at::Tensor img,
    at::Tensor spx_labels,
    at::Tensor output,
    at::Tensor outIdx,
    const int K)   
{
    /* 
    Shape assumptions: 
    - image: [nBatch, nChannel, x, y]
    - spx_labels: [nBatch, x, y]
    */

    const int batch_size = img.size(0);
    const int channels_size = img.size(1);

    const int imW = img.size(3);
    const int imH = img.size(2); 
    // const int nPixels = img.size(2)*img.size(3);

    int blockSizeX = std::min(32, imW);
    const int threadW    = ceil(imW/(float)blockSizeX);

    int blockSizeY = std::min(32, imH);
    const int threadH    = ceil(imH/(float)blockSizeY);

    // const int nbPixPerThread = ceil(nPixels/((float)blockSize));

    const dim3 blocks(channels_size, batch_size);
    const dim3 threads(blockSizeX, blockSizeY);

    AT_DISPATCH_FLOATING_TYPES(img.type(), "spx_max_pooling_forward_cuda", ([&] {
    spx_max_pooling_forward_kernel<scalar_t><<<blocks, threads>>>(
        img.data<scalar_t>(),
        spx_labels.data<int>(),
        outIdx.data<int>(),
        imW,
        imH,
        threadW,
        threadH, 
        channels_size, 
        batch_size,
        K);
    }));

    // fill in values at max positions (second kernel)
    blockSizeX = 16;
    blockSizeY = 1024/blockSizeX;
    const int nbBlocksX = ceil(channels_size/((float)blockSizeX));
    const int nbBlocksY = ceil(K/((float)blockSizeY));
    const dim3 blocksFill(nbBlocksX, nbBlocksY, batch_size);
    const dim3 threadsFill(blockSizeX, blockSizeY);

    AT_DISPATCH_FLOATING_TYPES(img.type(), "fill_max_values", ([&] {
    fill_values<scalar_t><<<blocksFill, threadsFill>>>(
        img.data<scalar_t>(),
        outIdx.data<int>(),
        K, 
        channels_size, 
        batch_size, 
        output.data<scalar_t>()
        );
    }));
    return {output, outIdx};
}

std::vector<at::Tensor> suppixpool_max_cuda_backward(
    at::Tensor grad_outputs,
    at::Tensor img,
    at::Tensor spx_labels,
    at::Tensor max_indices, 
    const int K)   
{
    /* 
    Shape assumptions: 
    - image: [nBatch, nChannel, x, y]
    - spx_labels: [nBatch, x, y]
    */

    const int batch_size = img.size(0);
    const int channels_size = img.size(1);
    const int nPixels = (int)img.size(2)*img.size(3);

    auto grad_in = at::zeros_like(img);

    const int blockSizeX = 32;
    const int blockSizeY = std::min(1024/blockSizeX, channels_size);
    const int nbBlocksX = ceil(K/((float)blockSizeX));
    const int nbBlocksY = ceil(channels_size/((float)blockSizeY));
    
    const dim3 blocks(nbBlocksX, nbBlocksY);
    const dim3 threads(blockSizeX, blockSizeY);

    AT_DISPATCH_FLOATING_TYPES(img.type(), "spx_max_pooling_backward_cuda", ([&] {
    spx_max_pooling_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_in.data<scalar_t>(),
        img.data<scalar_t>(),
        max_indices.data<int>(),
        channels_size,
        grad_outputs.data<scalar_t>(),
        K, 
        batch_size);
    }));
    return {grad_in};
    // return {grad_outputs};
}
