//Blurs the given image using the GPU
char pna_blur_gpu(char* imgname,uint32_t size,float sigma)
{
    uint32_t imgSize;
    float* matrix;
    cl_int ret;//the openCL error code/s
    //get the image
    ME_ImageBMP bmp;
    meImageBMP_Init(&bmp,imgname);
    imgSize = bmp.imgWidth*bmp.imgHeight*3;
    //create the gaussian kernel
    matrix = createGaussianKernel(size,sigma);
    //create the pointer that will hold the new (blurred) image data
    unsigned char* newData;
    RF_MALLOC(newData,imgSize);
    
    // Read in the kernel code into a c string
    FILE* f;
    char* kernelSource;
    size_t kernelSrcSize;
    if( (f = fopen("kernel.cl", "r")) == NULL)
    {
        fprintf(stderr, "Failed to load OpenCL kernel code.\n");
        return false;
    }
    RF_MALLOC(kernelSource,MAX_SOURCE_SIZE)
    kernelSrcSize = fread( kernelSource, 1, MAX_SOURCE_SIZE, f);
    fclose(f);
    
    //Get platform and device information
    cl_platform_id platformID;//will hold the ID of the openCL available platform
    cl_uint platformsN;//will hold the number of openCL available platforms on the machine
    cl_device_id deviceID;//will hold the ID of the openCL device
    cl_uint devicesN; //will hold the number of OpenCL devices in the system
    if(clGetPlatformIDs(1, &platformID, &platformsN) != CL_SUCCESS)
    {
        printf("Could not get the OpenCL Platform IDs\n");
        return false;
    }
    if(clGetDeviceIDs(platformID, CL_DEVICE_TYPE_DEFAULT, 1,&deviceID, &devicesN) != CL_SUCCESS)
    {
        printf("Could not get the system's OpenCL device\n");
        return false;
    }
    
    
    
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &deviceID, NULL, NULL, &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Could not create a valid OpenCL context\n");
        return false;
    }
    // Create a command queue
    cl_command_queue cmdQueue = clCreateCommandQueue(context, deviceID, 0, &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Could not create an OpenCL Command Queue\n");
        return false;
    }
    
    
    /// Create memory buffers on the device for the two images
    cl_mem gpuImg = clCreateBuffer(context,CL_MEM_READ_ONLY,imgSize,NULL,&ret);
    if(ret != CL_SUCCESS)
    {
        printf("Unable to create the GPU image buffer object\n");
        return false;
    }
    cl_mem gpuGaussian = clCreateBuffer(context,CL_MEM_READ_ONLY,size*size*sizeof(float),NULL,&ret);
    if(ret != CL_SUCCESS)
    {
        printf("Unable to create the GPU image buffer object\n");
        return false;
    }
    cl_mem gpuNewImg = clCreateBuffer(context,CL_MEM_WRITE_ONLY,imgSize,NULL,&ret);
    if(ret != CL_SUCCESS)
    {
        printf("Unable to create the GPU image buffer object\n");
        return false;
    }
    
    
    //Copy the image data and the gaussian kernel to the memory buffer
    if(clEnqueueWriteBuffer(cmdQueue, gpuImg, CL_TRUE, 0,imgSize,bmp.imgData, 0, NULL, NULL) != CL_SUCCESS)
    {
        printf("Error during sending the image data to the OpenCL buffer\n");
        return false;
    }
    if(clEnqueueWriteBuffer(cmdQueue, gpuGaussian, CL_TRUE, 0,size*size*sizeof(float),matrix, 0, NULL, NULL) != CL_SUCCESS)
    {
        printf("Error during sending the gaussian kernel to the OpenCL buffer\n");
        return false;
    }
    
    
        //Create a program object and associate it with the kernel's source code.
    cl_program program = clCreateProgramWithSource(context, 1,(const char **)&kernelSource, (const size_t *)&kernelSrcSize, &ret);
    free(kernelSource);
    if(ret != CL_SUCCESS)
    {
        printf("Error in creating an OpenCL program object\n");
        return false;
    }
    //Build the created OpenCL program
    if((ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL))!= CL_SUCCESS)
    {
        printf("Failed to build the OpenCL program\n");
        //create the log string and show it to the user. Then quit
        char* buildLog;
        RF_MALLOC(buildLog,MAX_LOG_SIZE);
        if(clGetProgramBuildInfo(program,deviceID,CL_PROGRAM_BUILD_LOG,MAX_LOG_SIZE,buildLog,NULL) != CL_SUCCESS)
        {
            printf("Could not get any Build info from OpenCL\n");
            free(buildLog);
            return false;
        }
        printf("**BUILD LOG**\n%s",buildLog);
        free(buildLog);
        return false;
    }
    
    // Create the OpenCL kernel. This is basically one function of the program declared with the __kernel qualifier
    cl_kernel kernel = clCreateKernel(program, "gaussian_blur", &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Failed to create the OpenCL Kernel from the built program\n");
        return false;
    }
    ///Set the arguments of the kernel
    if(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&gpuImg) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"gpuImg\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&gpuGaussian) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"gpuGaussian\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 2, sizeof(int), (void *)&bmp.imgWidth) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"imageWidth\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 3, sizeof(int), (void *)&bmp.imgHeight) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"imgHeight\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel,4,sizeof(int),(void*)&size) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"gaussian size\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&gpuNewImg) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"gpuNewImg\" argument\n");
        return false;
    }
    
    
    ///enqueue the kernel into the OpenCL device for execution
    size_t globalWorkItemSize = imgSize;//the total size of 1 dimension of the work items. Basically the whole image buffer size
    size_t workGroupSize = 64; //The size of one work group
    ret = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, &globalWorkItemSize, &workGroupSize,0, NULL, NULL);
    
    ///Read the memory buffer of the new image on the device to the new Data local variable
    ret = clEnqueueReadBuffer(cmdQueue, gpuNewImg, CL_TRUE, 0,imgSize, newData, 0, NULL, NULL);
    
    ///Clean up everything
    free(matrix);
    clFlush(cmdQueue);
    clFinish(cmdQueue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(gpuImg);
    clReleaseMemObject(gpuGaussian);
    clReleaseMemObject(gpuNewImg);
    clReleaseCommandQueue(cmdQueue);
    clReleaseContext(context);
    ///save the new image and return success
    bmp.imgData = newData;
    meImageBMP_Save(&bmp,"gpu_blur.bmp");
    return true;
