__kernel void matmul(   __global float *A,
                            __global float *B,
                            __global const unsigned *N,
			    __global float *restrict C)

{
  
   int tx = get_global_id(0); 
   int ty = get_global_id(1);
   // value stores the element that is 
   // computed by the thread
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
}
