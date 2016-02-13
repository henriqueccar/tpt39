__kernel void vector_add(__global const float *x, 
                        __global float * restrict z)
{
size_t i=get_global_id(0);
size_t l=get_global_size(0);
for(unsigned j = 0; j < l; ++j) {
*z=*z+x[j];
}
*z=*z/l;
//printf("the average is %f ",*z);
}
