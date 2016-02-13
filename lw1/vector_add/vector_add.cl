__kernel void vector_add(__global const float *x, 
                        __global const float *y, 
                        __global float *restrict z)
{
size_t i=get_global_id(0);
z[i]=x[i]+y[i];
//z[get_local_id(0)]=x[get_local_id(0)]+y[get_local_id(0)];
//z[g*4+i]=x[g*4+i]+y[g*4+i];
}
