#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <fstream>
#include <time.h>
#include "opencv2/opencv.hpp"
#include <chrono>
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#define STRING_BUFFER_LEN 1024


using namespace cv;
using namespace std;
#define SHOW

//gpu support functions


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







int main(int, char**)
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




    VideoCapture camera("./bourne.mp4");
    if(!camera.isOpened())  // check if we succeeded
        return -1;

    const string NAME = "./output.avi";   // Form the new name with container
    int ex = static_cast<int>(CV_FOURCC('M','J','P','G'));
    Size S = Size((int) camera.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) camera.get(CV_CAP_PROP_FRAME_HEIGHT));
	//Size S =Size(1280,720);
	cout << "SIZE:" << S << endl;
	int w = (int) camera.get(CV_CAP_PROP_FRAME_WIDTH);
	int h = (int) camera.get(CV_CAP_PROP_FRAME_HEIGHT);
	

    VideoWriter outputVideo;                                        // Open the output
        outputVideo.open(NAME, ex, 25, S, true);

    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << NAME << endl;
        return -1;
    }
	time_t start,end;
	double diff,tot;
	tot=0.0;
	int count=0;
	const char *windowName = "filter";   // Name shown in the GUI window.
    #ifdef SHOW
    namedWindow(windowName); // Resizable window, might not work on Windows.
    #endif
    while (true) {
        Mat cameraFrame,displayframe;
		count=count+1;
		if(count > 299) break;
        camera >> cameraFrame;
		time (&start);
        Mat filterframe = Mat(cameraFrame.size(), CV_8UC3);
        Mat grayframe,edge_x,edge_y,edge;
	Mat floatFrame;
//    	float *floatFrame=(float *) malloc(sizeof(float)*w*h);
	cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);
	grayframe.convertTo(floatFrame, CV_32FC1);
//	int kernelSise=3;// 3x3 filter
	Mat ugauss =  getGaussianKernel(3,1,CV_32F);
	Mat gauss;
//	float *gaus=(float *) malloc(sizeof(float)*9);
	ugauss.convertTo(gauss,CV_32F);

//converting Mat to one line array source
//  https://stackoverflow.com/questions/26681713/convert-mat-to-array-vector-in-opencv/26685567
float *array = (float *)malloc( 3*sizeof(float)*h*w );
cv::MatConstIterator_<cv::Vec3f> it = floatFrame.begin<cv::Vec3f>();
for (unsigned i = 0; it != floatFrame.end<cv::Vec3f>(); it++ ) {
    for ( unsigned j = 0; j < 3; j++ ) {
        *(array + i ) = (*it)[j];
        i++;
    }
}

float *gaussarray = (float *)malloc( 3*3*sizeof(float));
cv::MatConstIterator_<cv::Vec3f> it2 = gauss.begin<cv::Vec3f>();
for (unsigned i = 0; it2 != gauss.end<cv::Vec3f>(); it++ ) {
    for ( unsigned j = 0; j < 3; j++ ) {
        *(gaussarray + i ) = (*it2)[j];
        i++;
    }
}


//printf("%f\n", *array);

//creating Sob matrices

//float Sobx = [(-1, 0, 1,
//          -2, 0, 2,
//          -1, 0, 1)];
//float Soby = [(-1, -2, -1,
//          0, 0, 0,
//          1, 2, 1)];

//float *input_a=(float *) malloc(sizeof(float)*n*n);
//float *input_b=(float *) malloc(sizeof(float)*N*N);
float *output=(float *) malloc(sizeof(float)*h*w);
//float *ref_output=(float *) malloc(sizeof(float)*N*N);
//int *input_N=(int *) malloc(sizeof(int));
//float  ref_output=0.0;
cl_mem input_a_buf; // num_devices elements
//cl_mem input_b_buf; // num_devices elements
//cl_mem input_N_buf;// num_devices elements
cl_mem output_buf; // num_devices elements
int status;




// get platform info


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


     unsigned char **opencl_program=read_file("filter.cl");
     program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
     if (program == NULL)
	{
         printf("Program creation failed\n");
         return 1;
	}	
     int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
	kernel = clCreateKernel(program, "filter", NULL);



 // Input buffers.
	input_a_buf = clCreateBuffer(context,CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
	 h*w* sizeof(float), array, &status);
	checkError(status, "Failed to create buffer for input A");






    	GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
    	GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
    	GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
		Scharr(grayframe, edge_x, CV_8U, 0, 1, 1, 0, BORDER_DEFAULT );
		Scharr(grayframe, edge_y, CV_8U, 1, 0, 1, 0, BORDER_DEFAULT );
		addWeighted( edge_x, 0.5, edge_y, 0.5, 0, edge );
	    threshold(edge, edge, 80, 255, THRESH_BINARY_INV);
		time (&end);
        cvtColor(edge, displayframe, CV_GRAY2BGR);
	// Clear the output image to black, so that the cartoon line drawings will be black (ie: not drawn).
    	memset((char*)displayframe.data, 0, displayframe.step * displayframe.rows);
		grayframe.copyTo(displayframe,edge);
	        cvtColor(displayframe, displayframe, CV_GRAY2BGR);
		outputVideo << displayframe;
	#ifdef SHOW
        imshow(windowName, displayframe);
	#endif
		diff = difftime (end,start);
		tot+=diff;
	}
	outputVideo.release();
	camera.release();
  	printf ("FPS %.2lf .\n", 299.0/tot );

    return EXIT_SUCCESS;

}
