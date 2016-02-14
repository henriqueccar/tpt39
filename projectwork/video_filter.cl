__kernel void gaussian_filter(
        __global float *input_a,
        __constant float *mask,
        __global float *blurredImage,
//        __private int maskSize
    ) {
 
    const int2 pos = {get_global_id(0), get_global_id(1)};
    int maskSize = 3
 
    // Collect neighbor values and multiply with Gaussian
    float sum = 0.0f;
    for(int a = -maskSize; a < maskSize+1; a++) {
        for(int b = -maskSize; b < maskSize+1; b++) {
            sum += mask[a+maskSize+(b+maskSize)*(maskSize*2+1)]
                *read_imagef(image, sampler, pos + (int2)(a,b)).x;
        }
    }
 
    blurredImage[pos.x+pos.y*get_global_size(0)] = sum;
}
