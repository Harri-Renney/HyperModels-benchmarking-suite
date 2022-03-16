#pragma once
#ifndef OPENCL_WRAPPER_H
#define OPENCL_WRAPPER_H

#include <iostream>

struct OpenCL_Device
{
	uint32_t platform_id;
	std::string platform_name;
	uint32_t device_id;
	std::string device_name;
};

class OpenCL_Wrapper
{
private:
	//OpenGL Inter-operability//
	cl::Context contextOpenGL_;
	uint32_t vertexBufferObject_;
	uint32_t frameBufferObject_;
	uint32_t vertexArrayObject_;

	cl_int errorStatus_ = 0;
public:
	OpenCL_Wrapper()
	{
	}
	OpenCL_Wrapper(uint32_t aPlatformIdx, uint32_t aDeviceIdx, cl::Context& aContext, cl::Device& aDevice, cl::CommandQueue& aCommandQueue)
	{
		init(aPlatformIdx, aDeviceIdx, aContext, aDevice, aCommandQueue);
	}
	static void getDevices(std::vector <cl::Platform>& aPlatforms, std::vector <std::vector<cl::Device>>& aDevices)
	{
		cl::Platform::get(&aPlatforms);
		aDevices.resize(aPlatforms.size());

		for (uint32_t i = 0; i != aPlatforms.size(); ++i)
		{
			aPlatforms[i].getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &(aDevices[i]));

			//Create context context using platform for GPU device//
			//cl::Context context;
			//context = cl::Context(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, contextProperties.data());
		}
	}
	void init(uint32_t aPlatformIdx, uint32_t aDeviceIdx, cl::Context& aContext, cl::Device& aDevice, cl::CommandQueue& aCommandQueue)
	{
		/////////////////////////////////////
		//Step 1: Set up OpenCL environment//
		/////////////////////////////////////

		//Discover platforms//
		std::vector <cl::Platform> platforms;
		cl::Platform::get(&platforms);

		//intptr_t isOpenGL = (intptr_t)wglGetCurrentContext() == 0 ? false : true;

		//Create contex properties for first platform//
		std::vector<cl_context_properties > contextProperties;
		std::cout << "Creating OpenCL context without OpenGL Interop." << std::endl;
		contextProperties.push_back(CL_CONTEXT_PLATFORM);
		contextProperties.push_back((cl_context_properties)(platforms[aPlatformIdx])());
		contextProperties.push_back(0);

		//Create context context using platform for GPU device//
		aContext = cl::Context(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, contextProperties.data());

		//Get device list from context//
		std::vector<cl::Device> devices = aContext.getInfo<CL_CONTEXT_DEVICES>();
		aDevice = devices[aDeviceIdx];

		//Create command queue for first device - Profiling enabled//
		aCommandQueue = cl::CommandQueue(aContext, aDevice, CL_QUEUE_PROFILING_ENABLE, &errorStatus_);	//Need to specify device 1[0] of platform 3[2] for dedicated graphics - Harri Laptop.
		if (errorStatus_)
			std::cout << "ERROR creating command queue for device. Status code: " << errorStatus_ << std::endl;

		std::cout << "\t\tDevice Name: " << aDevice.getInfo<CL_DEVICE_NAME>() << std::endl;
	}

	void createKernelProgram(cl::Context& aContext, cl::Program& aKernelProgram, const std::string aSourcePath, const char options[])
	{
		//Read in program source - I'm going to go for reading from compiled object instead//
		std::ifstream sourceFileName(aSourcePath.c_str());
		std::string sourceFile(std::istreambuf_iterator<char>(sourceFileName), (std::istreambuf_iterator<char>()));

		//Create program source object from std::string source code//
		std::vector<std::string> programSources;
		programSources.push_back(sourceFile);
		cl::Program::Sources source(programSources);	//Apparently this takes a vector of strings as the program source.

		//Create program from source code//
		aKernelProgram = cl::Program(aContext, source, &errorStatus_);
		if (errorStatus_)
			std::cout << "ERROR creating program from source. Status code: " << errorStatus_ << std::endl;

		//Build program//
		aKernelProgram.build(options);
	}
	void createKernel(cl::Context& aContext, cl::Program& aKernelProgram, cl::Kernel& aKernel, std::string aKernelName)
	{
		//Create kernel program on device//
		aKernel = cl::Kernel(aKernelProgram, aKernelName.c_str(), &errorStatus_);
		if (errorStatus_)
			std::cout << "ERROR creating kernel. Status code: " << errorStatus_ << std::endl;
	}

	void createBuffer(cl::Context& aContext, cl::Buffer& aBuffer, int aMemFlags, unsigned int aBufferSize)
	{
		aBuffer = cl::Buffer(aContext, aMemFlags, aBufferSize);
	}
	void writeBuffer(cl::CommandQueue& aCommandQueue, cl::Buffer& aBuffer, unsigned int aBufferSize, void* input)
	{
		aCommandQueue.enqueueWriteBuffer(aBuffer, CL_TRUE, 0, aBufferSize, input);
	}
	void readBuffer(cl::CommandQueue& aCommandQueue, cl::Buffer& aBuffer, unsigned int aBufferSize, void* output)
	{
		aCommandQueue.enqueueReadBuffer(aBuffer, CL_TRUE, 0, aBufferSize, output);
	}
	void deleteBuffer(cl::Buffer& aBuffer)
	{
		clReleaseMemObject(aBuffer());
	}

	void setKernelArgument(cl::Kernel& aKernel, cl::Buffer& aBuffer, int aIndex, int aSize)
	{
		aKernel.setArg(aIndex, aSize, &aBuffer);
	}
	void setKernelArgument(cl::Kernel& aKernel, void* aValue, int aIndex, int aSize)
	{
		aKernel.setArg(aIndex, aSize, aValue);
	}

	void enqueueKernel(cl::CommandQueue& aCommandQueue, cl::Kernel& aKernel, cl::NDRange aGlobalSize, cl::NDRange aLocalSize)
	{
		aCommandQueue.enqueueNDRangeKernel(aKernel, cl::NullRange/*globaloffset*/, aGlobalSize, aLocalSize, NULL, NULL);
	}

	void enqueueKernel(cl::CommandQueue& aCommandQueue, cl::Kernel& aKernel, cl::NDRange aGlobalSize, cl::NDRange aLocalSize, cl::Event& aEvent)
	{
		aCommandQueue.enqueueNDRangeKernel(aKernel, cl::NullRange/*globaloffset*/, aGlobalSize, aLocalSize, NULL, &aEvent);
	}

	void enqueueCopyBuffer(cl::CommandQueue& aCommandQueue, cl::Buffer& aSrcBuffer, cl::Buffer& aDstBuffer, const uint32_t aSize)
	{
		aCommandQueue.enqueueCopyBuffer(aSrcBuffer, aDstBuffer, 0, 0, aSize, NULL, NULL);
	}

	void* mapMemory(cl::CommandQueue& aCommandQueue, cl::Buffer& aBuffer, cl_mem_flags aFlags, const uint32_t aSize)
	{
		return aCommandQueue.enqueueMapBuffer(aBuffer, true, aFlags, 0, aSize, NULL, NULL, &errorStatus_);
	}
	void unmapMemory(cl::CommandQueue& aCommandQueue, cl::Buffer& aBuffer, void* aPtr, const uint32_t aSize)
	{
		aCommandQueue.enqueueUnmapMemObject(aBuffer, aPtr, NULL, NULL);
	}

	void waitEvent(cl::Event& aEvent)
	{
		aEvent.wait();
	}

	void waitCommandQueue(cl::CommandQueue& aCommandQueue)
	{
		aCommandQueue.finish();
	}

	void profileEvent(cl::Event& aEvent)
	{
		cl_ulong time_start;
		cl_ulong time_end;

		clGetEventProfilingInfo(aEvent(), CL_PROFILING_COMMAND_SUBMIT, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(aEvent(), CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

		double nanoSeconds = time_end - time_start;
		printf("OpenCl Execution time is: %0.8f milliseconds \n", nanoSeconds / 1000000.0);
	}

	uint32_t getMaxLocalWorkspace(cl::Device aDevice)
	{
		return aDevice.getInfo< CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	}

	//Static Functions//
	static void printAvailableDevices()
	{
		cl::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		//Print all available devices//
		int platform_id = 0;
		std::cout << "Number of Platforms: " << platforms.size() << std::endl << std::endl;
		for (cl::vector<cl::Platform>::iterator it = platforms.begin(); it != platforms.end(); ++it)
		{
			cl::Platform platform(*it);

			std::cout << "Platform ID: " << platform_id++ << std::endl;
			std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
			std::cout << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

			cl::vector<cl::Device> devices;
			platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);

			int device_id = 0;
			for (cl::vector<cl::Device>::iterator it2 = devices.begin(); it2 != devices.end(); ++it2)
			{
				cl::Device device(*it2);

				std::cout << "\tDevice " << device_id++ << ": " << std::endl;
				std::cout << "\t\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
				std::cout << "\t\tDevice Type: " << device.getInfo<CL_DEVICE_TYPE>();
				std::cout << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;
				std::cout << "\t\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
				std::cout << "\t\tDevice Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
				std::cout << "\t\tDevice Global Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
				std::cout << "\t\tDevice Global Memory Cache: " << device.getInfo< CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>() << std::endl;
				std::cout << "\t\tDevice Max Clock Frequency: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
				std::cout << "\t\tDevice Max Allocateable Memory: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
				std::cout << "\t\tDevice Local Memory: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
				std::cout << "\t\tDevice Available: " << device.getInfo< CL_DEVICE_AVAILABLE>() << std::endl;
				std::cout << "\t\tMax workgroup size: " << device.getInfo< CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
				auto workitemsizes = device.getInfo< CL_DEVICE_MAX_WORK_ITEM_SIZES>();
				std::cout << "\t\tMax workitem sizes: " << std::endl;
				for(auto it3 = workitemsizes.begin(); it3 != workitemsizes.end(); ++ it3)
					std::cout << *it3 << std::endl;
					

				//If an AMD platform//
				if (strstr(platform.getInfo<CL_PLATFORM_NAME>().c_str(), "AMD"))
				{
					std::cout << "\tAMD Specific:" << std::endl;
					//If AMD//
					//std::cout << "\t\tAMD Wavefront size: " << device.getInfo<CL_DEVICE_WAVEFRONT_WIDTH_AMD>() << std::endl;
				}
			}
			std::cout << std::endl;
		}
	}

	static std::vector<OpenCL_Device> getOpenclDevices()
	{
		std::vector<OpenCL_Device> retDevices;
		OpenCL_Device clDevice;

		cl::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		//Print all available devices//
		int platform_id = 0;
		for (cl::vector<cl::Platform>::iterator it = platforms.begin(); it != platforms.end(); ++it)
		{
			cl::Platform platform(*it);

			clDevice.platform_id = platform_id++;
			clDevice.platform_name = platform.getInfo<CL_PLATFORM_NAME>();

			cl::vector<cl::Device> devices;
			platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);

			int device_id = 0;
			for (cl::vector<cl::Device>::iterator it2 = devices.begin(); it2 != devices.end(); ++it2)
			{
				cl::Device device(*it2);

				clDevice.device_id = device.getInfo<CL_DEVICE_VENDOR_ID>();
				clDevice.device_name = device.getInfo<CL_DEVICE_NAME>();

				retDevices.push_back(clDevice);
			}
		}

		return retDevices;
	}
};

#endif