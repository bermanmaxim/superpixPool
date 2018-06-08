extern "C" __device__
float atomicMaxIndex(int* address, int newCandidate, const float* image)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;

    while (old < 0) 
    {
        assumed = old; 
        old = atomicCAS(address_as_int, assumed, newCandidate);
    }
    // old is definitely initialized and therefore a valid index. 
    while (image[newCandidate] > image[old]) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, newCandidate);
        }
    return old;
}

extern "C" __global__
void fill_values(
    const float* image,
    const int* maxIdx,
    const int K, 
    const int nClasses,
    const int batchSize,
    float* outVals)
{ 
    int batch = blockIdx.z*blockDim.z + threadIdx.z; 
    int channel = blockIdx.x*blockDim.x + threadIdx.x; 
    int relevantLabel = blockIdx.y*blockDim.y+threadIdx.y;
    
    int outIdx = batch*nClasses*K + K*channel+ relevantLabel;

    if (outIdx<K*nClasses*batchSize)
        outVals[outIdx] = image[maxIdx[outIdx]];
}

extern "C" __global__
void max_pooling(
    const float* image,
    const int* labels,
    int* outIdx,
    const int volumeSize, 
    const int nClasses, 
    const int K)
{   
    extern __shared__ int sharedIx[]; // contains index of max

    int tid = threadIdx.x; 
    int channel = blockIdx.y; 
    int relevantLabel = blockIdx.z;
    int volIndex = blockIdx.x*blockDim.x + tid;

    int outputIndex = K*channel+relevantLabel;

    int imgIndex = volumeSize*channel+volIndex;

    sharedIx[tid] = -1;

    if (volIndex<volumeSize && labels[volIndex]==relevantLabel)
    {
        sharedIx[tid] = imgIndex;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x/2; s>0; s>>=1)
    {
        if (tid<s && volIndex < volumeSize && sharedIx[tid+s] > 0)
        {
            if (image[sharedIx[tid+s]] > image[sharedIx[tid]])
                sharedIx[tid] = sharedIx[tid+s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        if (sharedIx[0] > 0){
            atomicMaxIndex(&outIdx[outputIndex], sharedIx[0], image);
        }
    }
}

extern "C" __global__
void max_pooling_v2(
    const float* image,
    const int* labels,
    int* outIdx,
    const int volumeSize, 
    const int nClasses, 
    const int batchSize, 
    const int nbPixPerThread,
    const int K)
{
    int batch = blockIdx.y;
    int numel = volumeSize*nClasses*batchSize;
    int channel = blockIdx.x;
    int x = threadIdx.x; 
    int imgStartIdx = batch*volumeSize*nClasses+
                      channel*volumeSize+
                      x*nbPixPerThread;
    
    int labelStartIdx = volumeSize*batch + x*nbPixPerThread; 

    if (x*nbPixPerThread < volumeSize && channel < nClasses && batch < batchSize)
    {
        int imgIndex = imgStartIdx;
        int labelIndex = labelStartIdx; 
        float candidate;
        int label;
        int outIndex;
        int runningIdx;

        for (int i=0; i<nbPixPerThread; i++)
        {
            imgIndex = imgStartIdx+i;
            labelIndex = labelStartIdx+i;
        
            runningIdx = labelIndex-batch*volumeSize;
            if (runningIdx < volumeSize)
            {
                // candidate = image[imgIndex];
                label = labels[labelIndex]; 
                outIndex = batch*nClasses*K + K*channel + label;
                atomicMaxIndex(&outIdx[outIndex], imgIndex, image);
            }
            else
            {
                return;
            }
        }
    }
}

extern "C" __global__
void bw_max_pooling(
    float* grad_in, 
    const float* img,
    const int* indices, 
    const int nClasses,
    const float* grad_outputs,
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


extern "C" __global__
void avg_pooling(
    const float* image,
    const int* labels,
    float* outVals,
    int* outCounts,
    const int volumeSize, 
    const int nClasses,
    const int batchSize,  
    const int nbPixPerThread,
    const int K)
{ 

    int batch = blockIdx.y; 
    int numel = volumeSize*nClasses*batchSize; 
    int channel = blockIdx.x;
    int x = threadIdx.x; 
    int imgStartIdx = batch*volumeSize*nClasses+ 
                      channel*volumeSize+
                      x*nbPixPerThread;

    int labelStartIdx = batch*volumeSize + x*nbPixPerThread;

    if (x*nbPixPerThread < volumeSize && channel < nClasses && batch < batchSize)
    {
        int imgIndex = imgStartIdx;
        int labelIndex = labelStartIdx; 
        float newAddition;
        int label;
        int outIndex;
        int runningIdx; 
        for (int i=0; i<nbPixPerThread; i++)
        {
            imgIndex = imgStartIdx+i;
            labelIndex = labelStartIdx+i;

            runningIdx = labelIndex-batch*volumeSize; 
            if (runningIdx < volumeSize)
            {
                newAddition = image[imgIndex];
                label = labels[labelIndex];

                outIndex = batch*nClasses*K+K*channel+label;
                atomicAdd(&outVals[outIndex], newAddition);
                if (channel == 0) 
                {
                    atomicAdd(&outCounts[batch*K+label], 1);
                }
            }
            else
            {
                return;
            }
        }
    }
}


extern "C" __global__
void bw_avg_pooling(
    float* grad_in,
    const int* counts,
    const int* labels, 
    const float* grad_outputs,
    const int K,
    const int volumeSize,
    const int nClasses,
    const int batchSize,
    const bool training)
{
    int batch = blockIdx.z*blockDim.z + threadIdx.z;
    int channel = blockIdx.y*blockDim.y + threadIdx.y;
    int volumeIdx = blockIdx.x*blockDim.x + threadIdx.x;
    int totalIdx = batch*volumeSize*nClasses+channel*volumeSize + volumeIdx;
    if (volumeIdx<volumeSize && channel<nClasses && batch<batchSize)
    {       
        int label = labels[batch*volumeSize+volumeIdx];
        int lstIdx = batch*K*nClasses + K*channel + label;
        if (training){
            int count = counts[batch*K+label];
            if (count == 0){
                grad_in[totalIdx] = 0;
            } else{
                grad_in[totalIdx] = grad_outputs[lstIdx]/count;
            }
        } else {
            grad_in[totalIdx] = grad_outputs[lstIdx]; 
        }
    }
}