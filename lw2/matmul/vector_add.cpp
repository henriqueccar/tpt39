#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <chrono>
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#define STRING_BUFFER_LEN 1024
using namespace std;
using ms=chrono::milliseconds;
using ns=chrono::nanoseconds;
using get_time = chrono::steady_clock;

//time code from  https://www.quora.com/What-is-the-easiest-way-to-calculate-the-time-elapsed-in-C++


void print_clbuild_errors(cl_program program,cl_device_id device)
	{
		cout<<"Program Build failed\n";
		size_t length;
		char buffer[2048];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
		cout<<"--- Build log ---\n "<<buffer<<endl;
		exit(1);
	}
unsigned char ** read_file(const char *name) {
  size_t size;
  unsigned char **output=(unsigned char **)malloc(sizeof(unsigned char *));
  FILE* fp = fopen(name, "rb");
  if (!fp) {
    printf("no such file:%s",name);
    exit(-1);
  }

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  *output = (unsigned char *)malloc(size);
  if (!*output) {
    fclose(fp);
    printf("mem allocate failure:%s",name);
    exit(-1);
  }

  if(!fread(*output, size, 1, fp)) printf("failed to read file\n");
  fclose(fp);
  return output;
}

void callback(const char *buffer, size_t length, size_t final, void *user_data)
{
     fwrite(buffer, 1, length, stdout);
}


void checkError(int status, const char *msg) {
	if(status!=CL_SUCCESS)	
		printf("%s\n",msg);
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
  return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

int main()
{
     char char_buffer[STRING_BUFFER_LEN];
     cl_platform_id platform;
     cl_device_id device;
     cl_context context;
     cl_context_properties context_properties[] =
     { 
          CL_CONTEXT_PLATFORM, 0,
          CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
          CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
          0
     };
     cl_command_queue queue;
     cl_program program;
     cl_kernel kernel;



//--------------------------------------------------------------------
const unsigned N = 50;
float *input_a=(float *) malloc(sizeof(float)*N*N);
float *input_b=(float *) malloc(sizeof(float)*N*N);
float *output=(float *) malloc(sizeof(float)*N*N);
float *ref_output=(float *) malloc(sizeof(float)*N*N);
//int *input_N=(int *) malloc(sizeof(int));
//float  ref_output=0.0;
cl_mem input_a_buf; // num_devices elements
cl_mem input_b_buf; // num_devices elements
cl_mem input_N_buf;// num_devices elements
cl_mem output_buf; // num_devices elements
int status;
int input_N=N;



	auto start = get_time::now();
	for(unsigned j = 0; j < N*N; ++j) {
//		for(unsigned i = 0; i < N; ++i){
	      input_a[j] = rand_float();
	      input_b[j] = rand_float();
//		}
	}
float test;
        for(unsigned i = 0; i < N;i++) {
                for(unsigned j = 0; j < N; j++){
//			ref_output[i*N+j]=0.0;
			test=0.0;
			for(unsigned k = 0 ; k<N; k++){
				test =test+  input_a[i*N+k]* input_b[k*N+j];
		} 
	ref_output[i*N+j]+=test;
	   }
	}
	auto end = get_time::now();
	auto diff =end-start;
	cout<<"CPU Elapsed time is :  "<< chrono::duration_cast<ns>(diff).count()<<" ns "<<endl;


     clGetPlatformIDs(1, &platform, NULL);
     clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
     clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
     clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

     context_properties[1] = (cl_context_properties)platform;
     clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
     context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
     queue = clCreateCommandQueue(context, device, 0, NULL);
	printf("%d",input_N);
     unsigned char **opencl_program=read_file("vector_add.cl");
     program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
     if (program == NULL)
	{
         printf("Program creation failed\n");
         return 1;
	}	
     int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	 if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
     kernel = clCreateKernel(program, "vector_add", NULL);
 // Input buffers.
    input_a_buf = clCreateBuffer(context,CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
       N*N* sizeof(float), input_a, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf = clCreateBuffer(context,CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
        N*N* sizeof(float), input_b, &status);
    checkError(status, "Failed to create buffer for input B");

    input_N_buf = clCreateBuffer(context,CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
       sizeof(int), &input_N, &status);
    checkError(status, "Failed to create buffer for input N");


    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY,
       N*N* sizeof(float), output, &status);
    checkError(status, "Failed to create buffer for output");


	auto nstart = get_time::now();
    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[2];
	cl_event kernel_event,finish_event;
    status = clEnqueueWriteBuffer(queue, input_a_buf, CL_TRUE,
        0, N*N* sizeof(float), input_a, 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input A");

    status = clEnqueueWriteBuffer(queue, input_b_buf, CL_TRUE,
        0, N*N* sizeof(float), input_b, 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input B");

    status = clEnqueueWriteBuffer(queue, input_N_buf, CL_TRUE,
        0, sizeof(int), &input_N, 0, NULL, &write_event[2]);
    checkError(status, "Failed to transfer input B");


    // Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_b_buf);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_N_buf);
    checkError(status, "Failed to set argument 3");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 4");

    const size_t global_work_size[] = {N,N};
//    const size_t local_work_size = N/256;


    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
        global_work_size, NULL, 2,  write_event, &kernel_event);
    checkError(status, "Failed to launch kernel");
    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE,
        0, N*N* sizeof(float), output, 1, &kernel_event, &finish_event);
   checkError(status, "Failed to READ kernel");

	auto nend  = get_time::now();
	auto ndiff = nend-nstart;
//	std::chrono::duration<double, std::milli> fp_ms = t2 - t1;

cout<<"\n GPU Elapsed time is :  "<< chrono::duration_cast<ns>(ndiff).count()<<" ns\n "<<endl;
auto res = diff-ndiff;
//cout<<"\n GPU is "<< chrono::duration_cast<ms>(res).count<<"ms\n"<<endl;
//printf("\n GPU is %f faster than CPU\n",(diff-ndiff));


// Verify results.
bool pass = true;

for(unsigned j = 0; j < N*N && pass; ++j){
      if(fabsf(output[j] - ref_output[j]) > 1.0e-5f) {
        printf("Failed verification @ index %d  Output: %f\nReference: %f\n",j,*output,*ref_output);
	 pass = false;
      }
	else 
	printf("exactly the same");
	pass = false;
}
// Release local events.
clReleaseEvent(write_event[0]);
clReleaseEvent(write_event[1]);
//clReleaseEvent(write_event[2]);
clReleaseKernel(kernel);
clReleaseCommandQueue(queue);
clReleaseMemObject(input_a_buf);
clReleaseMemObject(input_b_buf);
clReleaseMemObject(input_N_buf);
clReleaseMemObject(output_buf);
clReleaseProgram(program);
clReleaseContext(context);


//--------------------------------------------------------------------






     clFinish(queue);

     return 0;
}
