__kernel void filter(   __global float *A,
                        __global float *B,
                        const int M,
			const int K,    
			__global float *restrict C)

{
  
   int txm = get_global_id(0); 
   int ty = get_global_id(1);
   // value stores the element that is 
   // computed by the thread

    for (int im_row=0; im_row<IMGR; img_row++) {
        for (int im_col=0; im_col<N; im_col++) {
            float acc = 0.0f;
            for (int k_r=0; k_r<K; k_r++) {
		for (int k_c=0;k_c<K; k_c++)
                acc += A[k*M + m] * B[n*K + k];
            }
            C[n*M + m] = acc;
        }
    }

//printf("%f",*A);
   float value = 0;
   for (int k = 0; k < *N; k++)
   {
      C[tx * *N +ty] = C[tx * *N+ty]+ A[tx * *N + k]*B[k * *N + ty];
      }
   // Write the matrix to device memory each 
   // thread writes one element
//   C[tx * N + ty] += value;
//printf("%f",C);
//printf("%d",*N);
//source http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmopencl
}
