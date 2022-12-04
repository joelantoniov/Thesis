#include <stdlib.h>
#include <string.h>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <CL/cl.h>
#include "LexRisLogicHeaders/LexRisLogic/include/LexRisLogic/FileStream.h" 
#include "LexRisLogicHeaders/LexRisLogic/include/LexRisLogic/StringSplitter.h"
#include "LexRisLogicHeaders/LexRisLogic/include/LexRisLogic/Convert.h"

const char* opencl_wave=""\
"float exact_solution(const float x1,const float t)"\
"{"\
"    float m1=(3 + 4*cosh(2*x1 - 8*t) + cosh(4*x1 - 64*t));"\
"    float m2 = (3*cosh(x1 - 28*t) + cosh(3*x1 - 36*t));"\
"    return (m1/(m2*m2))*12;"\
"}"\
"float funcion(float x1)                                                                                    \n"\
"{                                                                                                          \n"\
"   float m1 = (3 + 4*cosh(2*x1) + cosh(4*x1));                                                             \n"\
"   float m2 = (3*cosh(x1) + cosh(3*x1));                                                                   \n"\
"   m2 = m2*m2;                                                                                             \n"\
"   return ((m1/m2)*12);                                                                                    \n"\
"}                                                                                                          \n"\
"                                                                                                           \n"\
"float phi(const float y, const float c,const float num)                                                    \n"\
"{                                                                                                          \n"\
"  return (y-num)/(sqrt((y-num)*(y-num) + c*c));                                                            \n"\
"}                                                                                                          \n"\
"                                                                                                           \n"\
"float ux(const float h, const int a,__global float *glob, const int unsigned m, const float c,const int ini)\n"\
"{                                                                                              \n"\
"  int count = 1;                                                                               \n"\
"  float sum   = 0;                                                                               \n"\
"  float val1, val2;                                                                            \n"\
"  while(count <= m)                                                                            \n"\
"  {                                                                                            \n"\
"    val1 = phi(a, c, glob[count+ini]) - phi(a+h, c, glob[count+ini]);                                  \n"\
"    val2 = glob[count+ini+1]-glob[count+ini];                                                         \n"\
"    sum  += (val1/h)*(val2);                                                                   \n"\
"    ++count;                                                                                   \n"\
"  }                                                                                            \n"\
"  return (sum/2);                                                                              \n"\
"}                                                                                              \n"\
"                                                                                                           \n"\
"float uxxx(const float h, const int a,__global float *glob, const float m, const float c,const float current,const int ini)\n"\
"{                                                                                              \n"\
"  float val = (ux(h, a+h, glob, m, c,ini) - 2*current + ux(h, a-h, glob, m, c,ini));                 \n"\
"  return val/(h*h);                                                                            \n"\
"}                                                                                              \n"\
"                                                                                                           \n"\
"__kernel void __waves(__global float* result,__global float* real,__global float* delta,__global float* mu,__global float* tao, "\
"           __global int* from,__global int* to,                                                            "\
"           __global float* step,__global float* next,__global int* at_time,__global int* meshs_ini_index,  "\
"          const int ini_index,const int end_index)                                                         \n"\
"{                                                                                                          \n"\
"   int o=get_global_id(0)+ini_index;                                                                      \n"\
"   if(o<end_index)                                                                                        \n"\
"   {                                                                                                      \n"\
"       int m = (to[o]-from[o])/step[o];                                                                   \n"\
"       int start = 1;                                                                                     \n"\
"       result[meshs_ini_index[o]+1]=from[o];                                                              \n"\
"       int tmp_a = from[o];"\
"       while(start < m)"\
"       {"\
"          result[meshs_ini_index[o]+start+1] = funcion(from[o] + (step[o]*start));"\
"          start = start+1;"\
"       }"\
"       start = 1;"\
"       result[meshs_ini_index[o]+m+1]=0;"\
"       while(start < m)"\
"       {"\
"           tmp_a = from[o] + (start*step[o]);"\
"           start = start+1;"\
"           float val1 = result[meshs_ini_index[o]+start]*delta[o]*ux(step[o], tmp_a, result, m, next[o],meshs_ini_index[o]);"\
"           float val2 = mu[o]*(uxxx(step[o], tmp_a, result, m, next[o], val1,meshs_ini_index[o]));"\
"           result[meshs_ini_index[o]+start] = (result[meshs_ini_index[o]+start] - tao[o]*(val1 + val2));"\
"           real[meshs_ini_index[o]+start] = exact_solution(tmp_a,at_time[o]);"\
"           "\
"       }"\
"   }                                                                                                      \n"\
"}\n";

struct Device
{
    cl_platform_id p_id;
    cl_device_id id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    std::vector<cl_mem> memory;
    bool error=false;
    std::vector<int> meshs;
    int final_mesh_size;
    int final_mesh_bytes;
    float* data;
    float* real;
};

struct Plattform
{
    cl_platform_id id;
    cl_uint numDevices;
    cl_device_id* devices;
    bool error=false;
    Device* deviceData=nullptr;
    ~Plattform()
    {
        if(deviceData)
            delete(deviceData);
    }
};

cl_uint numPlatforms;
Plattform* plataformas=nullptr;

std::vector<Device*> available_device;
cl_int result;

void waves(float* delta,float* mu,float* tao,
           int* from,int* to,
           float* step,float* next,int* at_time,
           int ini_index,int end_index,
           int height,
           Device& device)
{
    device.final_mesh_size=0;
    device.meshs.clear();
    int* meshs=new int[end_index-ini_index];
    for(int i=ini_index;i<end_index;++i)
    {
        meshs[i]=device.final_mesh_size;
        device.meshs.push_back((to[i] - from[i])/step[i]+2);
        device.final_mesh_size += device.meshs[device.meshs.size()-1];
    }
    device.final_mesh_bytes=device.final_mesh_size*sizeof(float);
    device.memory.push_back(clCreateBuffer(device.context, CL_MEM_READ_WRITE, device.final_mesh_bytes, NULL, NULL));    //RESULTADO
    device.memory.push_back(clCreateBuffer(device.context, CL_MEM_READ_WRITE, device.final_mesh_bytes, NULL, NULL));    //REAL
    int height_size_float=height*sizeof(float);
    int height_size_int=height*sizeof(int);
    device.memory.push_back(clCreateBuffer(device.context, CL_MEM_READ_ONLY,height_size_float, NULL, NULL));            //DELTA
        clEnqueueWriteBuffer(device.queue,device.memory[device.memory.size()-1],CL_TRUE,0,height_size_float,delta,0,NULL,NULL);
    device.memory.push_back(clCreateBuffer(device.context, CL_MEM_READ_ONLY,height_size_float, NULL, NULL));            //MU
        clEnqueueWriteBuffer(device.queue,device.memory[device.memory.size()-1],CL_TRUE,0,height_size_float,mu,0,NULL,NULL);
    device.memory.push_back(clCreateBuffer(device.context, CL_MEM_READ_ONLY,height_size_float, NULL, NULL));            //TAO
        clEnqueueWriteBuffer(device.queue,device.memory[device.memory.size()-1],CL_TRUE,0,height_size_float,tao,0,NULL,NULL);
    device.memory.push_back(clCreateBuffer(device.context, CL_MEM_READ_ONLY,height_size_int, NULL, NULL));              //FROM
        clEnqueueWriteBuffer(device.queue,device.memory[device.memory.size()-1],CL_TRUE,0,height_size_int,from,0,NULL,NULL);
    device.memory.push_back(clCreateBuffer(device.context, CL_MEM_READ_ONLY,height_size_int, NULL, NULL));              //TO
        clEnqueueWriteBuffer(device.queue,device.memory[device.memory.size()-1],CL_TRUE,0,height_size_int,to,0,NULL,NULL);
    device.memory.push_back(clCreateBuffer(device.context, CL_MEM_READ_ONLY,height_size_float, NULL, NULL));            //STEP
        clEnqueueWriteBuffer(device.queue,device.memory[device.memory.size()-1],CL_TRUE,0,height_size_float,step,0,NULL,NULL);
    device.memory.push_back(clCreateBuffer(device.context, CL_MEM_READ_ONLY,height_size_float, NULL, NULL));            //NEXT
        clEnqueueWriteBuffer(device.queue,device.memory[device.memory.size()-1],CL_TRUE,0,height_size_float,next,0,NULL,NULL);
    device.memory.push_back(clCreateBuffer(device.context, CL_MEM_READ_ONLY,height_size_int, NULL, NULL));              //AT_TIME
        clEnqueueWriteBuffer(device.queue,device.memory[device.memory.size()-1],CL_TRUE,0,height_size_int,at_time,0,NULL,NULL);
    int size_end_ini=(end_index-ini_index)*sizeof(int);
    device.memory.push_back(clCreateBuffer(device.context, CL_MEM_READ_ONLY,size_end_ini, NULL, NULL));                 //MESHS INI INDEX
        clEnqueueWriteBuffer(device.queue,device.memory[device.memory.size()-1],CL_TRUE,0,size_end_ini,meshs,0,NULL,NULL);
    int total_waves=end_index-ini_index;
    unsigned int local_size=64;
    unsigned int global_size=ceil((total_waves)/(float)local_size)*local_size;
    for(unsigned int i=0;i<device.memory.size();++i)
        clSetKernelArg(device.kernel, i, sizeof(cl_mem), &device.memory[i]);
    clSetKernelArg(device.kernel, device.memory.size(), sizeof(int), &ini_index);                                       //INI INDEX
    clSetKernelArg(device.kernel, device.memory.size()+1, sizeof(int), &end_index);                                     //END INDEX
    clEnqueueNDRangeKernel(device.queue, device.kernel, 1, NULL, &global_size, &local_size,0, NULL, NULL);
    delete(meshs);
}

float allNumDevices=0;

void CL_CALLBACK onOpenCLError(const char *errinfo,  const void *private_info,
                               size_t cb, void *user_data)
{
    printf("Error while creating context or working in this context : %s", errinfo);
}

int main()
{
    ofstream myfile;
    myfile.open("waves_results.txt");
    {
        /***************************************************************************************/
        /*****                    Getting available platforms                              *****/
        /***************************************************************************************/
        //Declarations
        cl_uint             numEntries = 50;     //Max number of platform ids we want to get, this should be n
        cl_platform_id*     platforms;          //List of platforms IDs

        //Allocations
        platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*numEntries);
        //cl_uint waves_per_platform = height/(sizeof(platforms)/sizeof(platforms[0]));

        //We use the clGetPlatformIDs function
        result = clGetPlatformIDs(numEntries, platforms, &numPlatforms);
        if(result != CL_SUCCESS)
        {
            std::cout << "Error while getting available platforms" << std::endl;
            exit(1);
        }
        plataformas=new Plattform[numPlatforms];
        for(cl_uint plattform_iter=0;plattform_iter<numPlatforms;++plattform_iter)
        {
            plataformas[plattform_iter].id=platforms[plattform_iter];
            /****************************************************************************************/
            /*****                     Getting available GPU devices                            *****/
            /****************************************************************************************/
            //Declarations
            cl_uint             maxDevices = 50;     //Max number of devices we want to get
            cl_device_id*       deviceIDs;          //List of device IDs
            cl_uint             numDevices;         //The actual number of returned device IDs returned

            //Allocations
            deviceIDs = (cl_device_id*)malloc(maxDevices*sizeof(cl_device_id));

            //We use the clGetDeviceIDs function
            result = clGetDeviceIDs(platforms[plattform_iter], CL_DEVICE_TYPE_GPU, maxDevices,
                                    deviceIDs, &numDevices);
            if(result != CL_SUCCESS)
            {
                std::cout<<"Error while getting available devices for: "<<plattform_iter<<std::endl;
                plataformas[plattform_iter].error=true;
            }
            else
            {
                allNumDevices+=(plataformas[plattform_iter].numDevices=numDevices);
                plataformas[plattform_iter].devices=deviceIDs;
                plataformas[plattform_iter].deviceData=new Device[plataformas[plattform_iter].numDevices];
                for(cl_uint device_iter=0;device_iter<plataformas[plattform_iter].numDevices;++device_iter)
                {
                    Device& device=plataformas[plattform_iter].deviceData[device_iter];
                    device.id=plataformas[plattform_iter].devices[device_iter];
                    device.p_id=plataformas[plattform_iter].id;
                    /****************************************************************************************/
                    /*****                     Creation of the OpenCL context                           *****/
                    /****************************************************************************************/
                    //Declarations
                    cl_context_properties*  properties = 0;   //We don't use any property for now
                    cl_uint                 usedDevices = 1;  //The number of devices we want to use


                    //We use the clCreateContext function
                    device.context = clCreateContext(properties,
                                                     usedDevices,
                                                     &device.id,
                                                     &onOpenCLError, NULL, &result);
                    if(result != CL_SUCCESS)
                    {
                        std::cout << "Error while creating the OpenCL context" << std::endl;
                        device.error=true;
                    }
                    else
                    {
                        /****************************************************************************************/
                        /*****                     Creation of the command queue                            *****/
                        /****************************************************************************************/
                        //Declarations
                        cl_command_queue_properties commandQueueProperties = 0;     //No properties used for now

                        //Creation of the command queue using the context created above and the ID of the device
                        //we want to use. Since the beginning, we are looking for using only one device, so this
                        //is simply the first element of the deviceIDs array.
                        device.queue = clCreateCommandQueue(device.context,device.id, commandQueueProperties, &result);
                        if(result != CL_SUCCESS)
                        {
                            std::cout << "Error while creating the command queue" << std::endl;
                            device.error=true;
                            clReleaseContext(device.context);
                        }
                        else
                        {
                            device.program=clCreateProgramWithSource(device.context,1,&opencl_wave,NULL,&result);
                            clBuildProgram(device.program, 0, NULL, NULL, NULL, NULL);
                            device.kernel = clCreateKernel(device.program, "__waves", &result);
                            available_device.push_back(&device);
                        }
                    }
                }
            }
        }
    }
    for(cl_uint plattform_iter=0;plattform_iter<numPlatforms;++plattform_iter)
    {
        std::cout<<"Plattform: "<<plataformas[plattform_iter].id<<std::endl;
        if(plataformas[plattform_iter].error)
            std::cout<<"\t Error"<<std::endl;
        else
        {
            std::cout<<"\t GPU: "<<plataformas[plattform_iter].numDevices<<std::endl;
            for(cl_uint device_iter=0;device_iter<plataformas[plattform_iter].numDevices;++device_iter)
            {
                Device& device=plataformas[plattform_iter].deviceData[device_iter];
                std::cout<<"\t Device: "<<device.id<<std::endl;
                if(device.error)
                    std::cout<<"\t \t Error"<<std::endl;
                else
                {
                    std::cout<<"\t \t Context: "<<device.context<<std::endl;
                    std::cout<<"\t \t Queue: "<<device.queue<<std::endl;
                }
            }
        }
    }
    //system("pause");
    LL::FileStream file;
    file.set_path("waves.txt");
    if(file.load())
    {
        int height=file.size();
        float* delta=new float[height];
        float* mu=new float[height];
        float* tao=new float[height];
        int* from=new int[height];
        int* to=new int[height];
        float* step=new float[height];
        float* next=new float[height];
        int* at_time=new int[height];
        LL::StringSplitter splitter;
        for(int i=0;i<height;++i)
        {
            splitter.set_string(file[i]);
            splitter.split(' ');
            delta[i]=LL::to_float(splitter[0]);
            mu[i]=LL::to_float(splitter[1]);
            tao[i]=LL::to_float(splitter[2]);
            from[i]=LL::to_float(splitter[3]);
            to[i]=LL::to_float(splitter[4]);
            step[i]=LL::to_float(splitter[5]);
            next[i]=LL::to_float(splitter[6]);
            at_time[i]=LL::to_float(splitter[7]);
        }
        //OJO todos los GPUS estan ejecutando las mismas ondas, Ondas -> [0,height>
        for(unsigned int i=0; i < numPlatforms;++i)
        {
            for(unsigned int j=0; j< plataformas[i].numDevices;++j)
                waves(delta,mu,tao,from,to,step,next,at_time,0,height,height,plataformas[i].deviceData[j]);
        }
        for(unsigned int i=0; i < numPlatforms;++i)
        {
            for(unsigned int j=0; j< plataformas[i].numDevices;++j)
            {
                Device& device=plataformas[i].deviceData[j];
                clFinish(device.queue);
                device.data=new float[int(device.final_mesh_size)];
                device.real=new float[int(device.final_mesh_size)];
                clEnqueueReadBuffer(device.queue, device.memory[0], CL_TRUE, 0,device.final_mesh_bytes, device.data, 0, NULL, NULL);
                clEnqueueReadBuffer(device.queue, device.memory[1], CL_TRUE, 0,device.final_mesh_bytes, device.real, 0, NULL, NULL);
                for(unsigned int m=0;m<device.memory.size();++m)
                    clReleaseMemObject(device.memory[m]);
                device.memory.clear();
            }
        }
        for(unsigned int i=0; i < numPlatforms;++i)
        {
            for(unsigned int j=0; j< plataformas[i].numDevices;++j)
            {
                Device& device=plataformas[i].deviceData[j];
                int cont=0;
                for(unsigned int k=0;k<device.meshs.size();++k)
                {
                    std::cout<<"Onda: "<<k<<std::endl;
                    for(int l=0;l<device.meshs[k];++l,++cont)
                    {
                        //std::cout<<"Numerico: "<<device.data[cont]<<" Exacta: "<<device.real[cont]<<"\n";
                        myfile << device.data[cont] << std::endl;
                    }
                    myfile << std::endl;
                    //system("pause");
                }
                delete(device.data);
                delete(device.real);
            }
        }
        delete(delta);
        delete(mu);
        delete(tao);
        delete(from);
        delete(to);
        delete(step);
        delete(next);
        delete(at_time);
        myfile.close();
    }
    else
    {
        std::cout<<"Falta Archivo 'waves.txt'"<<std::endl;
        //system("pause");
    }
    for(unsigned int i=0; i < numPlatforms;++i)
    {
        for(unsigned int j=0; j< plataformas[i].numDevices;++j)
        {
            Device& device=plataformas[i].deviceData[j];
            clReleaseKernel(device.kernel);
            clReleaseProgram(device.program);
        }
    }
    delete(plataformas);
    return 0;
}
