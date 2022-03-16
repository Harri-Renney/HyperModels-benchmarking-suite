#ifndef FDTD_ACCELERATED_HPP
#define FDTD_ACCELERATED_HPP

#include <utility>
#include <stdint.h>
#include <iostream>
#include <fstream>

//#define CL_HPP_TARGET_OPENCL_VERSION 210
//#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>
//#include <CL/cl.hpp>
#include <CL/cl_gl.h>

//Parsing parameters as json file//
#include "third_party/json.hpp"
using nlohmann::json;

#include "FDTD_Grid.hpp"
#include "Buffer.hpp"

#include "Visualizer.hpp"

enum DeviceType { INTEGRATED = 32902, DISCRETE = 4098, NVIDIA = 4318 };
enum Implementation { OPENCL, CUDA, VULKAN, DIRECT3D };

struct Neighbour_Structure
{
	//neighbour.x.x = x coord, neighbour.x.y = y coord, neighbour.y = weight//
	std::pair<std::pair<int, int>, int> neighbour;
};

#include <string>

class FDTD_Accelerated
{
private:
	Implementation implementation_;
	uint32_t sampleRate_;
	uint32_t deviceType_;

	//CL//
	cl_int errorStatus_ = 0;
	cl_uint num_platforms, num_devices;
	cl::Platform platform_;
	cl::Context context_;
	cl::Device device_;
	cl::CommandQueue commandQueue_;
	cl::Program kernelProgram_;
	std::string kernelSourcePath_;
	cl::Kernel kernel_;
	cl::NDRange globalws_;
	cl::NDRange localws_;

	//CL Buffers//
	cl::Buffer idGrid_;
	cl::Buffer modelGrid_;
	cl::Buffer boundaryGridBuffer_;
	cl::Buffer outputBuffer_;
	cl::Buffer excitationBuffer_;
	cl::Buffer localBuffer_;

	//Model//
	int listenerPosition_[2];
	int excitationPosition_[2];
	Model* model_;
	int modelWidth_;
	int modelHeight_;
	int gridElements_;
	int gridByteSize_;

	int numConnections_ = 4;
	int* connections_ = new int[numConnections_];
	cl::Buffer connectionsBuffer_;

	//Output and excitations//
	typedef float base_type_;
	unsigned int bufferSize_;
	Buffer<base_type_> output_;
	Buffer<base_type_> excitation_;

	int bufferRotationIndex_ = 1;

	Visualizer* vis;

	float* renderGrid;
	int* idGridInput_;
	float* boundaryGridInput_;
	//Cartisian_Grid<int> idGridInput_;
	//int* two_dimensional_grid_;

	void initOpenCL()
	{
		std::vector <cl::Platform> platforms;
		cl::Platform::get(&platforms);
		for (cl::vector<cl::Platform>::iterator it = platforms.begin(); it != platforms.end(); ++it)
		{
			cl::Platform platform(*it);

			cl_context_properties contextProperties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };
			context_ = cl::Context(CL_DEVICE_TYPE_ALL, contextProperties);

			cl::vector<cl::Device> devices = context_.getInfo<CL_CONTEXT_DEVICES>();

			int device_id = 0;
			for (cl::vector<cl::Device>::iterator it2 = devices.begin(); it2 != devices.end(); ++it2)
			{
				cl::Device device(*it2);
				auto d = device.getInfo<CL_DEVICE_VENDOR_ID>();
				if (d == deviceType_)
				{
					//Create command queue for first device - Profiling enabled//
					commandQueue_ = cl::CommandQueue(context_, device, CL_QUEUE_PROFILING_ENABLE, &errorStatus_);	//Need to specify device 1[0] of platform 3[2] for dedicated graphics - Harri Laptop.
					if (errorStatus_)
						std::cout << "ERROR creating command queue for device. Status code: " << errorStatus_ << std::endl;

					std::cout << "\t\tDevice Name Chosen: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

					// A way of automatically setting local work group sizes.
					auto sizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
					auto max = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE >();
					//localws_.get()[0] = sizes[0];
					//localws_.get()[1] = sizes[1];


					return;
				}
			}
			std::cout << std::endl;
		}
	}
	void initBuffersCL()
	{
		//Create input and output buffer for grid points//
		idGrid_ = cl::Buffer(context_, CL_MEM_READ_WRITE, gridByteSize_);
		modelGrid_ = cl::Buffer(context_, CL_MEM_READ_WRITE, gridByteSize_ * 3);
		boundaryGridBuffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, gridByteSize_);
		outputBuffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, output_.bufferSize_ * sizeof(float));
		excitationBuffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, excitation_.bufferSize_ * sizeof(float));
		connectionsBuffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, numConnections_ * sizeof(int));

		//Connections

		//64
		/*connections_[0] = 604;
		connections_[1] = 1056;
		connections_[2] = 2208;
		connections_[3] = 3111;*/

		//128
		/*connections_[0] = 1032;
		connections_[1] = 2102;
		connections_[2] = 12726;
		connections_[3] = 5079;*/

		//256
		/*connections_[0] = 2890;
		connections_[1] = 3706;
		connections_[2] = 22138;
		connections_[3] = 17077;*/

		//512
		connections_[0] = 6313;
		connections_[1] = 21748;
		connections_[2] = 224500;
		connections_[3] = 66401;

		//1024
		//connections_[0] = 27745;
		//connections_[1] = 36382;
		//connections_[2] = 935454;
		//connections_[3] = 247573;

		//Copy data to newly created device's memory//
		float* temporaryGrid =  new float[gridElements_ * 3];
		memset(temporaryGrid, 0, gridByteSize_ * 3);

		commandQueue_.enqueueWriteBuffer(idGrid_, CL_TRUE, 0, gridByteSize_ , idGridInput_);
		commandQueue_.enqueueWriteBuffer(modelGrid_, CL_TRUE, 0, gridByteSize_*3, temporaryGrid);
		commandQueue_.enqueueWriteBuffer(boundaryGridBuffer_, CL_TRUE, 0, gridByteSize_, boundaryGridInput_);
		commandQueue_.enqueueWriteBuffer(connectionsBuffer_, CL_TRUE, 0, numConnections_ * sizeof(int), connections_);
	}
	void step()
	{
		commandQueue_.enqueueNDRangeKernel(kernel_, cl::NullRange/*globaloffset*/, globalws_, localws_, NULL);
		//commandQueue_.finish();

		output_.bufferIndex_++;
		excitation_.bufferIndex_++;
		bufferRotationIndex_ = (bufferRotationIndex_ + 1) % 3;
	}
protected:
public:
	FDTD_Accelerated(Implementation aImplementation, uint32_t aDevice, uint32_t aSampleRate, float aGridSpacing) :
		implementation_(aImplementation),
		deviceType_(aDevice),
		sampleRate_(aSampleRate),
		modelWidth_(128),
		modelHeight_(128),
		bufferSize_(aSampleRate),	//@ToDo - Make these controllable.
		output_(bufferSize_),
		excitation_(bufferSize_)
	{
		listenerPosition_[0] = 16;
		listenerPosition_[1] = 16;
		excitationPosition_[0] = 32;
		excitationPosition_[1] = 32;

		initOpenCL();
	}
	~FDTD_Accelerated()
	{

	}

	void buildProgram()
	{
		//Build program//
		kernelProgram_.build();

		kernel_ = cl::Kernel(kernelProgram_, "compute", &errorStatus_);	//@ToDo - Hard coded the kernel name. Find way to generate this?
	}

	void fillBuffer(float* input, float* output, uint32_t numSteps)
	{
		//Load excitation samples into GPU//
		commandQueue_.enqueueWriteBuffer(excitationBuffer_, CL_TRUE, 0, numSteps * sizeof(float), input);
		kernel_.setArg(5, sizeof(cl_mem), &excitationBuffer_);

		//Calculate buffer size of synthesizer output samples//
		for (unsigned int i = 0; i != numSteps; ++i)
		{
			input[i] = 0.0;
			//Increments kernel indices//
			kernel_.setArg(4, sizeof(int), &output_.bufferIndex_);
			kernel_.setArg(3, sizeof(int), &bufferRotationIndex_);

			step();
		}

		output_.resetIndex();
		excitation_.resetIndex();

		commandQueue_.enqueueReadBuffer(outputBuffer_, CL_TRUE, 0, numSteps * sizeof(float), output);
		//std::memcpy(output, output_.buffer_, sizeof(float) * (numSteps));
		//for (int k = 0; k != numSteps; ++k)
		//	output[k] = output_[k];

		//float* temporaryGrid = new float[gridElements_ * 3];
		//commandQueue_.enqueueReadBuffer(modelGrid_, CL_TRUE, 0, gridByteSize_ * 3, temporaryGrid);
		//render(temporaryGrid);

		//delete temporaryGrid;
	}
	void renderSimulation()
	{
		commandQueue_.enqueueReadBuffer(modelGrid_, CL_TRUE, 0, gridByteSize_, renderGrid);
		render(renderGrid);
	}


	void createModel(const std::string aPath, float aBoundaryValue, uint32_t aInputPosition[2], uint32_t aOutputPosition[2])
	{
		// JOSN parsing.
		//Read json file into program object//
		std::ifstream ifs(aPath);
		json jsonFile = json::parse(ifs);
		//std::cout << j << std::endl;

		 modelWidth_ = jsonFile["buffer"].size();
		 modelHeight_= jsonFile["buffer"][0].size();

		 
		 boundaryGridInput_ = new float[modelWidth_*modelHeight_];
		 idGridInput_ = new int[modelWidth_*modelHeight_];
		for (uint32_t i = 0; i != modelWidth_; ++i)
		{
			for (uint32_t j = 0; j != modelHeight_; ++j)
			{
				idGridInput_[i*modelWidth_ + j] = jsonFile["buffer"][i][j];
				//idGridInput_[i][j] = jsonFile["buffer"][i][j];
				//std::cout << idGridInput_.valueAt(i, j) << " | ";
			}
			//std::cout << std::endl;
		}

		globalws_ = cl::NDRange(modelWidth_, modelHeight_);
		localws_ = cl::NDRange(8, 8);						//@ToDo - CHANGE TO OPTIMIZED GROUP SIZE.

		//createExplicitEquation("2DWaveEquation2.cl");
		//initRender();

		model_ = new Model(modelWidth_, modelHeight_, aBoundaryValue);
		model_->setInputPosition(aInputPosition[0], aInputPosition[1]);
		model_->setOutputPosition(aOutputPosition[0], aOutputPosition[1]);

		int boundaryCount = 0;
		for (uint32_t i = 0; i != (modelWidth_); ++i)
		{
			for (uint32_t j = 0; j != (modelHeight_); ++j)
			{
				if(i == 0 || j == 0 || i == modelWidth_-1 || i == modelHeight_-1)
					boundaryGridInput_[i*modelWidth_ + j] = 1.0;
				else
					boundaryGridInput_[i*modelWidth_ + j] = 0.0;
			}
		}
		int gridUseCount = 0;
		for (uint32_t i = 1; i != (modelWidth_- 1); ++i)
		{
			for (uint32_t j = 1; j != (modelHeight_ - 1); ++j)
			{
				if (idGridInput_[i*modelWidth_ + j] > 0)
				{
					++gridUseCount;
					//std::cout << i * modelWidth_ + j << "\n";
					//std::cout << j << ", " << i << "\n";
				}
				//@ToDo - Work out calcualting boundary grid correctly.
				//This way trying to manually treat string differently from others.
				//if (idGridInput_[i*modelWidth_ + j] == 3)
				//{
				//	int gridID = idGridInput_[i*modelWidth_ + j];

				//	boundaryCount = 0;
				//	if (idGridInput_[(i - 1)*modelWidth_ + j] == gridID)
				//		++boundaryCount;
				//	if (idGridInput_[(i + 1)*modelWidth_ + j] == gridID)
				//		++boundaryCount;
				//	if (idGridInput_[(i - 1)*modelWidth_ + j + 1] == gridID)
				//		++boundaryCount;
				//	if (idGridInput_[(i - 1)*modelWidth_ + j - 1] == gridID)
				//		++boundaryCount;
				//	if (idGridInput_[(i + 1)*modelWidth_ + j + 1] == gridID)
				//		++boundaryCount;
				//	if (idGridInput_[(i + 1)*modelWidth_ + j - 1] == gridID)
				//		++boundaryCount;
				//	if (idGridInput_[(i)*modelWidth_ + j - 1] == gridID)
				//		++boundaryCount;
				//	if (idGridInput_[(i)*modelWidth_ + j + 1] == gridID)
				//		++boundaryCount;

				//	if (boundaryCount == 1)
				//		boundaryGridInput_[i*modelWidth_ + j] = 1.0;
				//}
				//else
				//{
				//	if (idGridInput_[i*modelWidth_ + j] > 0 && (idGridInput_[(i-1)*modelWidth_ + j] == 0 || idGridInput_[(i+1)*modelWidth_ + j] == 0 || idGridInput_[i*modelWidth_ + j-1] == 0 || idGridInput_[i*modelWidth_ + j+1] == 0))
				//	//if (idGridInput_[i][j] > 0 && (idGridInput_[i-1][j] == 0 || idGridInput_[i+1][j] == 0 || idGridInput_[i][j-1] == 0 || idGridInput_[i][j+1] == 0))
				//	{
				//		boundaryGridInput_[i*modelWidth_ + j] = 1.0;
				//	}
				//}
				/*if (idGridInput_[i*modelWidth_ + j] > 0)
				{
					boundaryCount = 0;
					if (idGridInput_[(i - 1)*modelWidth_ + j] == 0)
						++boundaryCount;
					if (idGridInput_[(i + 1)*modelWidth_ + j] == 0)
						++boundaryCount;
					if (idGridInput_[(i - 1)*modelWidth_ + j+1] == 0)
						++boundaryCount;
					if (idGridInput_[(i - 1)*modelWidth_ + j-1] == 0)
						++boundaryCount;
					if (idGridInput_[(i + 1)*modelWidth_ + j+1] == 0)
						++boundaryCount;
					if (idGridInput_[(i + 1)*modelWidth_ + j-1] == 0)
						++boundaryCount;
					if (idGridInput_[(i)*modelWidth_ + j-1] == 0)
						++boundaryCount;
					if (idGridInput_[(i)*modelWidth_ + j + 1] == 0)
						++boundaryCount;
					if(boundaryCount > 1 && boundaryCount < 5)
						boundaryGridInput_[i*modelWidth_ + j] = 1.0;
				}*/

				//if (idGridInput_[i*modelWidth_ + j] > 0 && (idGridInput_[(i-1)*modelWidth_ + j] == 0 || idGridInput_[(i+1)*modelWidth_ + j] == 0 || idGridInput_[i*modelWidth_ + j-1] == 0 || idGridInput_[i*modelWidth_ + j+1] == 0))
				////if (idGridInput_[i][j] > 0 && (idGridInput_[i-1][j] == 0 || idGridInput_[i+1][j] == 0 || idGridInput_[i][j-1] == 0 || idGridInput_[i][j+1] == 0))
				//{
				//	boundaryGridInput_[i*modelWidth_ + j] = aBoundaryValue;
				//}
				//std::cout << model_->boundaryGrid_.valueAt(i, j) << " | ";
			}
			//std::cout << std::endl;
		}

		std::cout << "GRID COUNT: " << ++gridUseCount << "\n";

		gridElements_ = (modelWidth_ * modelHeight_);
		gridByteSize_ = (gridElements_ * sizeof(float));
		renderGrid = new float[gridElements_];

		if (implementation_ == Implementation::OPENCL)
		{
			initBuffersCL();
		}

		createExplicitEquation(aPath);
	}

	void createExplicitEquation(const std::string aPath)
	{
		//Read json file into program object//
		std::ifstream ifs(aPath);
		json jsonFile = json::parse(ifs);
		//@TODO - Fix which physics equation is collected.
		//std::string sourceFile = jsonFile["controllers"][0]["physics_kernel"];
		std::string sourceFile = jsonFile["controllers"][0]["physics_kernel"];

		std::cout << sourceFile << std::endl;

		//Read in program source//
		//kernelSourcePath_ = aPath;
		//std::ifstream sourceFileName(kernelSourcePath_.c_str());
		//std::string sourceFile(std::istreambuf_iterator<char>(sourceFileName), (std::istreambuf_iterator<char>()));

		//Create program source object from std::string source code//
		std::vector<std::string> programSources;
		programSources.push_back(sourceFile);
		cl::Program::Sources source(programSources);	//Apparently this takes a vector of strings as the program source.

		//Create program from source code//
		kernelProgram_ = cl::Program(context_, source, &errorStatus_);
		if (errorStatus_)
			std::cout << "ERROR creating program from source. Status code: " << errorStatus_ << std::endl;
		
		//Build program//
		char options[1024];
		snprintf(options, sizeof(options),
			""
			//" -cl-fast-relaxed-math"
			//" -cl-single-precision-constant"
			//""
		);
		kernelProgram_.build();	//@Highlight - Keep this in?

		kernel_ = cl::Kernel(kernelProgram_, "fdtdKernel", &errorStatus_);	//@ToDo - Hard coded the kernel name. Find way to generate this?
		//buildProgram();

		if (errorStatus_)
			std::cout << "ERROR building program from source. Status code: " << errorStatus_ << std::endl;

		kernel_.setArg(0, sizeof(cl_mem), &idGrid_);
		kernel_.setArg(1, sizeof(cl_mem), &modelGrid_);
		kernel_.setArg(2, sizeof(cl_mem), &boundaryGridBuffer_);
		kernel_.setArg(6, sizeof(cl_mem), &outputBuffer_);

		int inPos = model_->getInputPosition();
		int outPos = model_->getOutputPosition();
		kernel_.setArg(7, sizeof(int), &inPos);
		kernel_.setArg(8, sizeof(int), &outPos);

		//CONNECTIONS
		kernel_.setArg(9, sizeof(int), &numConnections_);
		kernel_.setArg(10, sizeof(cl_mem), &connectionsBuffer_);
	}
	void createMatrixEquation(const std::string aPath);	//How is the matrix equations defined? Is there just a default matrix equation that can be formed for many equations or need be defined?

	//@ToDo - Do we need this? Coefficients just need to use .setArg(), don't need to create buffer for them...
	void createCoefficient(std::string aCoeff)
	{

	}
	void updateCoefficient(std::string aCoeff, uint32_t aIndex, float aValue)
	{
		kernel_.setArg(aIndex, sizeof(float), &aValue);	//@ToDo - Need dynamicaly find index for setArg (The first param)
	}

	void setInputPosition(int aInputs[])
	{
		model_->setInputPosition(aInputs[0], aInputs[1]);
		int inPos = model_->getInputPosition();
		kernel_.setArg(7, sizeof(int), &inPos);
	}
	void setOutputPosition(int aOutputs[])
	{
		model_->setOutputPosition(aOutputs[0], aOutputs[1]);
		int outPos = model_->getOutputPosition();
		kernel_.setArg(8, sizeof(int), &outPos);
	}
	void setInputPositions(std::vector<uint32_t> aInputs);
	void setOutputPositions(std::vector<uint32_t> aOutputs);

	void setInputs(std::vector<std::vector<float>> aInputs);

	std::vector<std::vector<float>> getInputs();
	std::vector<std::vector<float>> getOutputs();

	void initRender()
	{
		vis = new Visualizer(modelWidth_, modelHeight_);
	}
	void render(float* aData)
	{
		vis->render(aData);
	}
	GLFWwindow* getWindow()
	{
		return vis->getWindow();
	}
};

#endif