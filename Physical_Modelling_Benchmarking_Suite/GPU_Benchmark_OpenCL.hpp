#ifndef GPU_BENCHMARK_OPENCL_HPP
#define GPU_BENCHMARK_OPENCL_HPP
//
//#define CL_HPP_TARGET_OPENCL_VERSION 210
//#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>
//#include <CL/cl.hpp>
#include <CL/cl_gl.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <random>

#include "OpenCL_Wrapper.h"
#include "Benchmarker.hpp"
#include "FDTD_Accelerated.hpp"
#include "AudioFile.h"

class GPU_Benchmark_OpenCL
{
private:
	static const size_t bufferSizesLength = 11;
	uint64_t bufferSizes[bufferSizesLength];

	typedef float datatype;
	uint32_t sampleRate_ = 44100;
	uint64_t bufferSize_ = 1024;
	uint64_t bufferLength_ = bufferSize_ / sizeof(datatype);
	uint32_t minDimensionSize_ = 64;
	uint32_t maxDimensionSize_ = 1024;
	float* soundBuffer_;
	float* inputBuffer_;
	float* outputBuffer_;

	OpenCL_Wrapper openCL;
	std::string deviceName_;
	uint32_t currentPlatformIdx_;
	uint32_t currentDeviceIdx_;
	cl::NDRange globalWorkspace_;
	cl::NDRange localWorkspace_;

	Benchmarker clBenchmarker_;

	FDTD_Accelerated fdtdSynth;

	//OpenCL objects//
	cl::Context context_;
	cl::Device device_;
	cl::CommandQueue commandQueue_;
	cl::Program kernelProgram_;
	cl::Event kernelBenchmark_;

	cl::Event clEvent;

	void setBufferLength(uint64_t aBufferLength)
	{
		bufferLength_ = aBufferLength;
		bufferSize_ = bufferLength_ * sizeof(datatype);

		setLocalWorkspace(bufferLength_);
	}
	void setWorkspaceSize(uint32_t aGlobalSize, uint32_t aLocalSize)
	{
		globalWorkspace_ = cl::NDRange(aGlobalSize, 1, 1);
		localWorkspace_ = cl::NDRange(aLocalSize, 1, 1);
	}
	void setWorkspaceSize(cl::NDRange aGlobalSize, cl::NDRange aLocalSize)
	{
		globalWorkspace_ = aGlobalSize;
		localWorkspace_ = aLocalSize;
	}

	void setLocalWorkspace(uint64_t aGlobalSize)
	{
		uint64_t maxLocalWorkspace = openCL.getMaxLocalWorkspace(device_);
		uint64_t localWorkspace = aGlobalSize > maxLocalWorkspace ? maxLocalWorkspace : aGlobalSize;

		cl::NDRange newGlobalSize = aGlobalSize;
		cl::NDRange newLocalSize = localWorkspace;
		setWorkspaceSize(newGlobalSize, newLocalSize);
	}
	void setBufferSize(uint64_t aBufferSize)
	{
		bufferSize_ = aBufferSize;
		bufferLength_ = bufferSize_ / sizeof(datatype);

		setLocalWorkspace(bufferLength_);
	}

	void impulse(uint32_t aLength, uint32_t aImpulseLength, float* aInput)
	{
		for (uint32_t i = 0; i != aLength; ++i)
		{
			//Create initial impulse as excitation//
			if (i < aImpulseLength)
				aInput[i] = 0.5;
			else
				aInput[i] = 0.0;
		}
	}

	void runSingleModelTest(uint32_t aN, bool isWarmup, std::string aPath)
	{
		//Test preperation//
		uint32_t numSamplesComputed = 0;
		impulse(bufferLength_, 5, inputBuffer_);

		// Benchmark Auto Shader
		std::string modelPath = aPath;
		uint32_t inputPosition[2] = { 125,60 };
		uint32_t outputPosition[2] = { 580, 235 };
		float boundaryValue = 1.0;
		fdtdSynth.createModel(modelPath, boundaryValue, inputPosition, outputPosition);

		float propagationCoefficient = 0.0018;
		float dampingCoefficient = 0.000010;
		fdtdSynth.updateCoefficient("mu", 9, dampingCoefficient);
		fdtdSynth.updateCoefficient("lambda", 10, propagationCoefficient);

		//Execute and average//
		std::cout << "Executing test: singleModelTestAuto" << std::endl;
		if (isWarmup)
		{
			fdtdSynth.fillBuffer(inputBuffer_, outputBuffer_, 1);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("singleModelTestAuto");

			fdtdSynth.fillBuffer(inputBuffer_, outputBuffer_, bufferLength_);

			clBenchmarker_.pauseTimer("singleModelTestAuto");
		}
		clBenchmarker_.elapsedTimer("singleModelTestAuto");

		//Save audio to file for inspection//
		for (int j = 0; j != bufferLength_; ++j)
			soundBuffer_[j] = outputBuffer_[j];
		outputAudioFile("cl_singleModelTestAuto.wav", soundBuffer_, bufferLength_, sampleRate_);
		std::cout << "singleModelTestAuto successful: Inspect audio log \"cl_singleModelTestAuto.wav\"" << std::endl << std::endl;

		// Benchmark Manual Shader
		//modelPath = "resources/kernels/manual/singleModelTestManual.json";
		//fdtdSynth.createModel(modelPath, boundaryValue, inputPosition, outputPosition);
		//std::cout << "Executing test: singleModelTestManual" << std::endl;
		//if (isWarmup)
		//{
		//	fdtdSynth.fillBuffer(inBuf, outBuf, 1);
		//}
		//for (uint32_t i = 0; i != aN; ++i)
		//{
		//	clBenchmarker_.startTimer("singleModelTestManual");

		//	fdtdSynth.fillBuffer(inBuf, outBuf, bufferLength_);

		//	clBenchmarker_.pauseTimer("singleModelTestManual");
		//}
		//clBenchmarker_.elapsedTimer("singleModelTestManual");

		////Save audio to file for inspection//
		//for (int j = 0; j != bufferLength_; ++j)
		//	soundBuffer[j] = outBuf[j];
		//outputAudioFile("cl_singleModelTestManual.wav", soundBuffer, bufferLength_, sampleRate_);
		//std::cout << "singleModelTestManual successful: Inspect audio log \"cl_singleModelTestManual.wav\"" << std::endl << std::endl;
	}
	void runSimpleSingleModelTestRealtime(size_t aFrameRate, bool isWarmup)
	{
		//Run tests with setup//
		for (uint32_t n = minDimensionSize_; n <= maxDimensionSize_; n *= 2)
		{
			//@ ToDo Skip 512 for now.
			//if (n != 512)
			//	continue;

			std::string modelPathAuto = "resources/kernels/auto/simple_single_model/simpleSingleModelTestAuto";
			modelPathAuto.append(std::to_string(n));
			modelPathAuto.append(".json");

			std::string modelPathManual = "resources/kernels/manual/simple_single_model/simpleSingleModelTestManual";
			modelPathManual.append(std::to_string(n));
			modelPathManual.append(".json");

			//Prepare new file for cl_bidirectional_processing//
			std::string strBenchmarkFileNameAuto = "CL_Logs/";
			strBenchmarkFileNameAuto.append(deviceName_);
			std::string strBenchmarkFileNameManual = strBenchmarkFileNameAuto;
			strBenchmarkFileNameAuto.append("_cl_single_model_test_auto");
			strBenchmarkFileNameManual.append("_cl_single_model_test_manual");
			std::string strFrameRate = std::to_string(aFrameRate);
			strBenchmarkFileNameAuto.append(strFrameRate);
			strBenchmarkFileNameManual.append(strFrameRate);
			strBenchmarkFileNameAuto.append("dimensions");
			strBenchmarkFileNameManual.append("dimensions");
			strBenchmarkFileNameAuto.append(std::to_string(n));
			strBenchmarkFileNameManual.append(std::to_string(n));
			strBenchmarkFileNameAuto.append(".csv");
			strBenchmarkFileNameManual.append(".csv");
			clBenchmarker_ = Benchmarker(strBenchmarkFileNameAuto, { "Buffer_Size", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });

			// AUTO
			uint64_t numSamplesComputed = 0;
			for (size_t i = 0; i != bufferSizesLength; ++i)
			{
				uint64_t currentBufferLength = bufferSizes[i];
				uint64_t currentBufferSize = currentBufferLength * sizeof(float);
				setBufferLength(currentBufferLength);
				if (currentBufferLength > aFrameRate)
					break;

				std::string strBenchmarkName = "";
				std::string strBufferSize = std::to_string(currentBufferLength);
				strBenchmarkName.append(strBufferSize);

				uint32_t centre = n / 2;
				uint32_t inputPosition[2] = { centre,centre };
				uint32_t outputPosition[2] = { centre+10, centre+10 };
				float boundaryValue = 1.0;
				fdtdSynth.createModel(modelPathAuto, boundaryValue, inputPosition, outputPosition);

				float propagationCoefficient = 0.0018;
				float dampingCoefficient = 0.000005;
				fdtdSynth.updateCoefficient("lambda", 10, propagationCoefficient);
				fdtdSynth.updateCoefficient("mu", 9, dampingCoefficient);

				uint64_t numSamplesComputed = 0;
				impulse(currentBufferLength, 5, inputBuffer_);

				if (isWarmup)
				{
					fdtdSynth.fillBuffer(inputBuffer_, outputBuffer_, currentBufferLength);
				}
				impulse(currentBufferLength, 5, inputBuffer_);
				while (numSamplesComputed < aFrameRate)
				{
					clBenchmarker_.startTimer(strBenchmarkName);
					fdtdSynth.fillBuffer(inputBuffer_, outputBuffer_, currentBufferLength);
					clBenchmarker_.pauseTimer(strBenchmarkName);

					//Log audio for inspection if necessary//
					for (int j = 0; j != currentBufferLength; ++j)
						soundBuffer_[numSamplesComputed + j] = outputBuffer_[j];

					numSamplesComputed += currentBufferLength;
				}
				clBenchmarker_.elapsedTimer(strBenchmarkName);

				//Save audio to file for inspection//
				std::string strBenchmarkFileNameAutoWav = strBenchmarkFileNameAuto;
				strBenchmarkFileNameAutoWav.append("bufferlength");
				strBenchmarkFileNameAutoWav.append(std::to_string(i));
				strBenchmarkFileNameAutoWav.append(".wav");
				outputAudioFile(strBenchmarkFileNameAutoWav.c_str(), soundBuffer_, aFrameRate, aFrameRate);
				std::cout << "cl_runSingleModelTestRealtime successful: Inspect audio log \"cl_runSingleModelTestRealtime.wav\"" << std::endl << std::endl;

				numSamplesComputed = 0;
			}

			// MANUAL
			clBenchmarker_ = Benchmarker(strBenchmarkFileNameManual, { "Buffer_Size", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });
			for (size_t i = 0; i != bufferSizesLength; ++i)
			{
				uint64_t currentBufferLength = bufferSizes[i];
				uint64_t currentBufferSize = currentBufferLength * sizeof(float);
				setBufferLength(currentBufferLength);
				if (currentBufferLength > aFrameRate)
					break;

				std::string strBenchmarkName = "";
				std::string strBufferSize = std::to_string(currentBufferLength);
				strBenchmarkName.append(strBufferSize);

				// MANUAL
				uint32_t centre = n / 2;
				uint32_t inputPosition[2] = { centre,centre };
				uint32_t outputPosition[2] = { centre + 10, centre + 10 };
				float boundaryValue = 1.0;
				fdtdSynth.createModel(modelPathManual, boundaryValue, inputPosition, outputPosition);

				float propagationCoefficient = 0.0018;
				float dampingCoefficient = 0.000005;
				float muOne = dampingCoefficient - 1.0;
				float muTwo = 1.0 / (dampingCoefficient + 1.0);
				float lambdaOne = propagationCoefficient;
				fdtdSynth.updateCoefficient("muOne", 9, muOne);
				fdtdSynth.updateCoefficient("muTwo", 10, muTwo);
				fdtdSynth.updateCoefficient("lambdaOne", 11, lambdaOne);

				uint64_t numSamplesComputed = 0;
				impulse(currentBufferLength, 5, inputBuffer_);

				if (isWarmup)
				{
					fdtdSynth.fillBuffer(inputBuffer_, outputBuffer_, currentBufferLength);
				}
				while (numSamplesComputed < aFrameRate)
				{
					clBenchmarker_.startTimer(strBenchmarkName);
					fdtdSynth.fillBuffer(inputBuffer_, outputBuffer_, currentBufferLength);
					clBenchmarker_.pauseTimer(strBenchmarkName);

					//Log audio for inspection if necessary//
					for (int j = 0; j != currentBufferLength; ++j)
						soundBuffer_[numSamplesComputed + j] = outputBuffer_[j];

					numSamplesComputed += currentBufferLength;
				}
				clBenchmarker_.elapsedTimer(strBenchmarkName);

				//Save audio to file for inspection//
				std::string strBenchmarkFileNameManualWav = strBenchmarkFileNameManual;
				strBenchmarkFileNameManualWav.append("bufferlength");
				strBenchmarkFileNameManualWav.append(std::to_string(i));
				strBenchmarkFileNameManualWav.append(".wav");
				outputAudioFile(strBenchmarkFileNameManualWav.c_str(), soundBuffer_, aFrameRate, aFrameRate);
				std::cout << "cl_runSingleModelTestRealtime successful: Inspect audio log \"cl_runSingleModelTestRealtime.wav\"" << std::endl << std::endl;

				numSamplesComputed = 0;
			}
		}
	}
	void runSimpleMultiModelTestRealtime(size_t aFrameRate, bool isWarmup)
	{
		//Run tests with setup//
		for (uint32_t n = minDimensionSize_; n <= maxDimensionSize_; n *= 2)
		{
			//@ ToDo Skip 512 for now.
			//if (n != 128)
			//	continue;

			std::string modelPathAuto = "resources/kernels/auto/simple_multi_model/simpleMultiModelTestAuto";
			modelPathAuto.append(std::to_string(n));
			modelPathAuto.append(".json");

			std::string modelPathManual = "resources/kernels/manual/simple_multi_model/simpleMultiModelTestManual";
			modelPathManual.append(std::to_string(n));
			modelPathManual.append(".json");

			//Prepare new file for cl_bidirectional_processing//
			std::string strBenchmarkFileNameAuto = "CL_Logs/";
			strBenchmarkFileNameAuto.append(deviceName_);
			std::string strBenchmarkFileNameManual = strBenchmarkFileNameAuto;
			strBenchmarkFileNameAuto.append("_cl_multi_model_test_auto");
			strBenchmarkFileNameManual.append("_cl_multi_model_test_manual");
			std::string strFrameRate = std::to_string(aFrameRate);
			strBenchmarkFileNameAuto.append(strFrameRate);
			strBenchmarkFileNameManual.append(strFrameRate);
			strBenchmarkFileNameAuto.append("dimensions");
			strBenchmarkFileNameManual.append("dimensions");
			strBenchmarkFileNameAuto.append(std::to_string(n));
			strBenchmarkFileNameManual.append(std::to_string(n));
			strBenchmarkFileNameAuto.append(".csv");
			strBenchmarkFileNameManual.append(".csv");
			clBenchmarker_ = Benchmarker(strBenchmarkFileNameAuto, { "Buffer_Size", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });

			// AUTO
			uint64_t numSamplesComputed = 0;
			for (size_t i = 0; i != bufferSizesLength; ++i)
			{
				uint64_t currentBufferLength = bufferSizes[i];
				uint64_t currentBufferSize = currentBufferLength * sizeof(float);
				setBufferLength(currentBufferLength);
				if (currentBufferLength > aFrameRate)
					break;

				std::string strBenchmarkName = "";
				std::string strBufferSize = std::to_string(currentBufferLength);
				strBenchmarkName.append(strBufferSize);

				uint32_t centre = n / 2;
				/*uint32_t inputPosition[2] = { 18,12 };
				uint32_t outputPosition[2] = { 18, 111 };*/
				uint32_t inputPosition[2] = { 69,10 };
				uint32_t outputPosition[2] = { 69, 450 };
				float boundaryValue = 1.0;
				fdtdSynth.createModel(modelPathAuto, boundaryValue, inputPosition, outputPosition);

				float propagationCoefficient = 0.18;
				float dampingCoefficient = 0.00050;
				fdtdSynth.updateCoefficient("stringLambda", 10, propagationCoefficient);
				fdtdSynth.updateCoefficient("stringMu", 9, dampingCoefficient);

				uint64_t numSamplesComputed = 0;
				impulse(currentBufferLength, 5, inputBuffer_);

				if (isWarmup)
				{
					fdtdSynth.fillBuffer(inputBuffer_, outputBuffer_, currentBufferLength);
				}
				while (numSamplesComputed < aFrameRate)
				{
					clBenchmarker_.startTimer(strBenchmarkName);
					fdtdSynth.fillBuffer(inputBuffer_, outputBuffer_, currentBufferLength);
					clBenchmarker_.pauseTimer(strBenchmarkName);

					//Log audio for inspection if necessary//
					for (int j = 0; j != currentBufferLength; ++j)
						soundBuffer_[numSamplesComputed + j] = outputBuffer_[j];

					numSamplesComputed += currentBufferLength;
				}
				clBenchmarker_.elapsedTimer(strBenchmarkName);

				//Save audio to file for inspection//
				std::string strBenchmarkFileNameAutoWav = strBenchmarkFileNameAuto;
				strBenchmarkFileNameAutoWav.append("bufferlength");
				strBenchmarkFileNameAutoWav.append(std::to_string(i));
				strBenchmarkFileNameAutoWav.append(".wav");
				outputAudioFile(strBenchmarkFileNameAutoWav.c_str(), soundBuffer_, aFrameRate, aFrameRate);
				std::cout << "cl_runMultiModelTestRealtime successful: Inspect audio log \"cl_runMultiModelTestRealtime.wav\"" << std::endl << std::endl;

				numSamplesComputed = 0;
			}

			// MANUAL
			clBenchmarker_ = Benchmarker(strBenchmarkFileNameManual, { "Buffer_Size", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });
			for (size_t i = 0; i != bufferSizesLength; ++i)
			{
				uint64_t currentBufferLength = bufferSizes[i];
				uint64_t currentBufferSize = currentBufferLength * sizeof(float);
				setBufferLength(currentBufferLength);
				if (currentBufferLength > aFrameRate)
					break;

				std::string strBenchmarkName = "";
				std::string strBufferSize = std::to_string(currentBufferLength);
				strBenchmarkName.append(strBufferSize);

				// Manual
				uint32_t centre = n / 2;
				/*uint32_t inputPosition[2] = { 18,12 };
				uint32_t outputPosition[2] = { 18, 111 };*/
				uint32_t inputPosition[2] = { 69,10 };
				uint32_t outputPosition[2] = { 69, 450 };
				float boundaryValue = 1.0;
				fdtdSynth.createModel(modelPathManual, boundaryValue, inputPosition, outputPosition);

				float propagationCoefficient = 0.18;
				float dampingCoefficient = 0.00050;
				float stringLambda = propagationCoefficient * propagationCoefficient;
				float stringMu = (1.0 / (dampingCoefficient + 1.0));
				fdtdSynth.updateCoefficient("stringLambda", 10, stringLambda);
				fdtdSynth.updateCoefficient("stringMu", 9, stringMu);

				uint64_t numSamplesComputed = 0;
				impulse(currentBufferLength, 5, inputBuffer_);

				if (isWarmup)
				{
					fdtdSynth.fillBuffer(inputBuffer_, outputBuffer_, currentBufferLength);
				}
				while (numSamplesComputed < aFrameRate)
				{
					clBenchmarker_.startTimer(strBenchmarkName);
					fdtdSynth.fillBuffer(inputBuffer_, outputBuffer_, currentBufferLength);
					clBenchmarker_.pauseTimer(strBenchmarkName);

					//Log audio for inspection if necessary//
					for (int j = 0; j != currentBufferLength; ++j)
						soundBuffer_[numSamplesComputed + j] = outputBuffer_[j];

					numSamplesComputed += currentBufferLength;
				}
				clBenchmarker_.elapsedTimer(strBenchmarkName);

				//Save audio to file for inspection//
				std::string strBenchmarkFileNameManualWav = strBenchmarkFileNameManual;
				strBenchmarkFileNameManualWav.append("bufferlength");
				strBenchmarkFileNameManualWav.append(std::to_string(i));
				strBenchmarkFileNameManualWav.append(".wav");
				outputAudioFile(strBenchmarkFileNameManualWav.c_str(), soundBuffer_, aFrameRate, aFrameRate);
				std::cout << "cl_runMultiModelTestRealtime successful: Inspect audio log \"cl_runMultiModelTestRealtime.wav\"" << std::endl << std::endl;

				numSamplesComputed = 0;
			}
		}
	}
	void runComplexMultiModelTestRealtime(size_t aFrameRate, bool isWarmup)
	{
		//Run tests with setup//
		for (uint32_t n = minDimensionSize_; n <= maxDimensionSize_; n *= 2)
		{
			//@ ToDo Skip 512 for now.
			if (n != 512)
				continue;

			std::string modelPathAuto = "resources/kernels/auto/complex_multi_model/complexMultiModelTestAuto";
			modelPathAuto.append(std::to_string(n));
			modelPathAuto.append(".json");

			std::string modelPathManual = "resources/kernels/manual/complex_multi_model/complexMultiModelTestManual";
			modelPathManual.append(std::to_string(n));
			modelPathManual.append(".json");

			//Prepare new file for cl_bidirectional_processing//
			std::string strBenchmarkFileNameAuto = "CL_Logs/";
			strBenchmarkFileNameAuto.append(deviceName_);
			std::string strBenchmarkFileNameManual = strBenchmarkFileNameAuto;
			strBenchmarkFileNameAuto.append("_cl_complex_multi_model_test_auto");
			strBenchmarkFileNameManual.append("_cl_complex_multi_model_test_manual");
			std::string strFrameRate = std::to_string(aFrameRate);
			strBenchmarkFileNameAuto.append(strFrameRate);
			strBenchmarkFileNameManual.append(strFrameRate);
			strBenchmarkFileNameAuto.append("dimensions");
			strBenchmarkFileNameManual.append("dimensions");
			strBenchmarkFileNameAuto.append(std::to_string(n));
			strBenchmarkFileNameManual.append(std::to_string(n));
			strBenchmarkFileNameAuto.append(".csv");
			strBenchmarkFileNameManual.append(".csv");
			clBenchmarker_ = Benchmarker(strBenchmarkFileNameAuto, { "Buffer_Size", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });

			// AUTO
			uint64_t numSamplesComputed = 0;
			for (size_t i = 0; i != bufferSizesLength; ++i)
			{
				uint64_t currentBufferLength = bufferSizes[i];
				uint64_t currentBufferSize = currentBufferLength * sizeof(float);
				setBufferLength(currentBufferLength);
				if (currentBufferLength > aFrameRate)
					break;

				std::string strBenchmarkName = "";
				std::string strBufferSize = std::to_string(currentBufferLength);
				strBenchmarkName.append(strBufferSize);

				uint32_t centre = n / 2;
				//64
				/*uint32_t inputPosition[2] = { 9,8 };
				uint32_t outputPosition[2] = { 32, 41 };*/
				//128
				/*uint32_t inputPosition[2] = { 17,6 };
				uint32_t outputPosition[2] = { 66, 41 };*/
				//256
				/*uint32_t inputPosition[2] = { 26,5 };
				uint32_t outputPosition[2] = { 142, 82 };*/
				//512
				uint32_t inputPosition[2] = { 100,50 };
				uint32_t outputPosition[2] = { 369, 147 };
				//1024
				//uint32_t inputPosition[2] = { 192,5 };
				//uint32_t outputPosition[2] = { 763, 269 };
				float boundaryValue = 1.0;
				fdtdSynth.createModel(modelPathAuto, boundaryValue, inputPosition, outputPosition);

				float lambda = 0.018;
				float mu = 0.000005;
				float stringMu = 0.001;
				float stringLambda = 0.1;
				float muTwo = 0.1;
				float sigma = 50.01;
				float deltaT = 1.0 / 44100.0;
				fdtdSynth.updateCoefficient("lambda", 11, lambda);
				fdtdSynth.updateCoefficient("mu", 12, mu);
				fdtdSynth.updateCoefficient("stringMu", 13, stringMu);
				fdtdSynth.updateCoefficient("stringLambda", 14, stringLambda);
				fdtdSynth.updateCoefficient("deltaT", 15, deltaT);
				fdtdSynth.updateCoefficient("muTwo", 16, muTwo);
				fdtdSynth.updateCoefficient("sigma", 17, sigma);

				uint64_t numSamplesComputed = 0;
				impulse(currentBufferLength, 5, inputBuffer_);

				if (isWarmup)
				{
					fdtdSynth.fillBuffer(inputBuffer_, outputBuffer_, currentBufferLength);
				}
				impulse(currentBufferLength, 5, inputBuffer_);
				while (numSamplesComputed < aFrameRate)
				{
					clBenchmarker_.startTimer(strBenchmarkName);
					fdtdSynth.fillBuffer(inputBuffer_, outputBuffer_, currentBufferLength);
					clBenchmarker_.pauseTimer(strBenchmarkName);

					//Log audio for inspection if necessary//
					for (int j = 0; j != currentBufferLength; ++j)
						soundBuffer_[numSamplesComputed + j] = outputBuffer_[j];

					numSamplesComputed += currentBufferLength;
				}
				clBenchmarker_.elapsedTimer(strBenchmarkName);

				//Save audio to file for inspection//
				std::string strBenchmarkFileNameAutoWav = strBenchmarkFileNameAuto;
				strBenchmarkFileNameAutoWav.append("bufferlength");
				strBenchmarkFileNameAutoWav.append(std::to_string(i));
				strBenchmarkFileNameAutoWav.append(".wav");
				outputAudioFile(strBenchmarkFileNameAutoWav.c_str(), soundBuffer_, aFrameRate, aFrameRate);
				std::cout << "cl_runSingleModelTestRealtime successful: Inspect audio log \"cl_runComplexMultiModelTestRealtime.wav\"" << std::endl << std::endl;

				numSamplesComputed = 0;
			}

			// MANUAL
			clBenchmarker_ = Benchmarker(strBenchmarkFileNameManual, { "Buffer_Size", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });
			for (size_t i = 0; i != bufferSizesLength; ++i)
			{
				uint64_t currentBufferLength = bufferSizes[i];
				uint64_t currentBufferSize = currentBufferLength * sizeof(float);
				setBufferLength(currentBufferLength);
				if (currentBufferLength > aFrameRate)
					break;

				std::string strBenchmarkName = "";
				std::string strBufferSize = std::to_string(currentBufferLength);
				strBenchmarkName.append(strBufferSize);

				uint32_t centre = n / 2;
				//64
				/*uint32_t inputPosition[2] = { 9,8 };
				uint32_t outputPosition[2] = { 32, 41 };*/
				//128
				/*uint32_t inputPosition[2] = { 17,6 };
				uint32_t outputPosition[2] = { 66, 41 };*/
				//256
				/*uint32_t inputPosition[2] = { 26,5 };
				uint32_t outputPosition[2] = { 142, 82 };*/
				//512
				uint32_t inputPosition[2] = { 100,50 };
				uint32_t outputPosition[2] = { 369, 147 };
				//1024
				//uint32_t inputPosition[2] = { 192,5 };
				//uint32_t outputPosition[2] = { 763, 269 };
				float boundaryValue = 0.0;
				fdtdSynth.createModel(modelPathManual, boundaryValue, inputPosition, outputPosition);

				float lambda = 0.018;
				float mu = 0.000005;
				float stringMu = 0.001;
				float stringLambda = 0.1;
				float muTwo = 0.1;
				float sigma = 50.01;
				float deltaT = 1.0 / 44100.0;
				fdtdSynth.updateCoefficient("lambda", 11, lambda);
				fdtdSynth.updateCoefficient("mu", 12, mu);
				fdtdSynth.updateCoefficient("stringMu", 13, stringMu);
				fdtdSynth.updateCoefficient("stringLambda", 14, stringLambda);
				fdtdSynth.updateCoefficient("deltaT", 15, deltaT);
				fdtdSynth.updateCoefficient("muTwo", 16, muTwo);
				fdtdSynth.updateCoefficient("sigma", 17, sigma);

				uint64_t numSamplesComputed = 0;
				impulse(currentBufferLength, 5, inputBuffer_);

				if (isWarmup)
				{
					fdtdSynth.fillBuffer(inputBuffer_, outputBuffer_, currentBufferLength);
				}
				impulse(currentBufferLength, 5, inputBuffer_);
				while (numSamplesComputed < aFrameRate)
				{
					clBenchmarker_.startTimer(strBenchmarkName);
					fdtdSynth.fillBuffer(inputBuffer_, outputBuffer_, currentBufferLength);
					clBenchmarker_.pauseTimer(strBenchmarkName);

					//Log audio for inspection if necessary//
					for (int j = 0; j != currentBufferLength; ++j)
						soundBuffer_[numSamplesComputed + j] = outputBuffer_[j];

					numSamplesComputed += currentBufferLength;
				}
				clBenchmarker_.elapsedTimer(strBenchmarkName);

				//Save audio to file for inspection//
				std::string strBenchmarkFileNameManualWav = strBenchmarkFileNameManual;
				strBenchmarkFileNameManualWav.append("bufferlength");
				strBenchmarkFileNameManualWav.append(std::to_string(i));
				strBenchmarkFileNameManualWav.append(".wav");
				outputAudioFile(strBenchmarkFileNameManualWav.c_str(), soundBuffer_, aFrameRate, aFrameRate);
				std::cout << "cl_runSingleModelTestRealtime successful: Inspect audio log \"cl_runComplexMultiModelTestRealtime.wav\"" << std::endl << std::endl;

				numSamplesComputed = 0;
			}
		}
	}
	void runComplexSingleModelTestRealtime(size_t aFrameRate, bool isWarmup)
	{
		//Run tests with setup//
		for (uint32_t n = minDimensionSize_; n <= maxDimensionSize_; n *= 2)
		{
			//@ ToDo Skip 512 for now.
			//if (n != 512)
			//	continue;

			std::string modelPathAuto = "resources/kernels/auto/complex_single_model/complexSingleModelTestAuto";
			modelPathAuto.append(std::to_string(n));
			modelPathAuto.append(".json");

			std::string modelPathManual = "resources/kernels/manual/complex_single_model/complexSingleModelTestManual";
			modelPathManual.append(std::to_string(n));
			modelPathManual.append(".json");

			//Prepare new file for cl_bidirectional_processing//
			std::string strBenchmarkFileNameAuto = "CL_Logs/";
			strBenchmarkFileNameAuto.append(deviceName_);
			std::string strBenchmarkFileNameManual = strBenchmarkFileNameAuto;
			strBenchmarkFileNameAuto.append("_cl_complex_single_model_test_auto");
			strBenchmarkFileNameManual.append("_cl_complex_single_model_test_manual");
			std::string strFrameRate = std::to_string(aFrameRate);
			strBenchmarkFileNameAuto.append(strFrameRate);
			strBenchmarkFileNameManual.append(strFrameRate);
			strBenchmarkFileNameAuto.append("dimensions");
			strBenchmarkFileNameManual.append("dimensions");
			strBenchmarkFileNameAuto.append(std::to_string(n));
			strBenchmarkFileNameManual.append(std::to_string(n));
			strBenchmarkFileNameAuto.append(".csv");
			strBenchmarkFileNameManual.append(".csv");
			clBenchmarker_ = Benchmarker(strBenchmarkFileNameAuto, { "Buffer_Size", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });

			// AUTO
			uint64_t numSamplesComputed = 0;
			for (size_t i = 0; i != bufferSizesLength; ++i)
			{
				uint64_t currentBufferLength = bufferSizes[i];
				uint64_t currentBufferSize = currentBufferLength * sizeof(float);
				setBufferLength(currentBufferLength);
				if (currentBufferLength > aFrameRate)
					break;

				std::string strBenchmarkName = "";
				std::string strBufferSize = std::to_string(currentBufferLength);
				strBenchmarkName.append(strBufferSize);

				uint32_t centre = n / 2;
				uint32_t inputPosition[2] = { centre,centre };
				uint32_t outputPosition[2] = { centre - 10, centre - 10 };
				float boundaryValue = 0.0;
				fdtdSynth.createModel(modelPathAuto, boundaryValue, inputPosition, outputPosition);

				float mu = 0.1;
				float sigma = 50.01;
				float deltaT = 1.0 / 44100.0;
				fdtdSynth.updateCoefficient("mu", 9, mu);
				fdtdSynth.updateCoefficient("sigma", 10, sigma);
				fdtdSynth.updateCoefficient("deltaT", 11, deltaT);

				uint64_t numSamplesComputed = 0;
				impulse(currentBufferLength, 5, inputBuffer_);

				if (isWarmup)
				{
					fdtdSynth.fillBuffer(inputBuffer_, outputBuffer_, currentBufferLength);
				}
				impulse(currentBufferLength, 5, inputBuffer_);
				while (numSamplesComputed < aFrameRate)
				{
					clBenchmarker_.startTimer(strBenchmarkName);
					fdtdSynth.fillBuffer(inputBuffer_, outputBuffer_, currentBufferLength);
					clBenchmarker_.pauseTimer(strBenchmarkName);

					//Log audio for inspection if necessary//
					for (int j = 0; j != currentBufferLength; ++j)
						soundBuffer_[numSamplesComputed + j] = outputBuffer_[j];

					numSamplesComputed += currentBufferLength;
				}
				clBenchmarker_.elapsedTimer(strBenchmarkName);

				//Save audio to file for inspection//
				std::string strBenchmarkFileNameAutoWav = strBenchmarkFileNameAuto;
				strBenchmarkFileNameAutoWav.append("bufferlength");
				strBenchmarkFileNameAutoWav.append(std::to_string(i));
				strBenchmarkFileNameAutoWav.append(".wav");
				outputAudioFile(strBenchmarkFileNameAutoWav.c_str(), soundBuffer_, aFrameRate, aFrameRate);
				std::cout << "cl_runSingleModelTestRealtime successful: Inspect audio log \"cl_runComplexSingleModelTestRealtime.wav\"" << std::endl << std::endl;

				numSamplesComputed = 0;
			}

			// MANUAL
			clBenchmarker_ = Benchmarker(strBenchmarkFileNameManual, { "Buffer_Size", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });
			for (size_t i = 0; i != bufferSizesLength; ++i)
			{
				uint64_t currentBufferLength = bufferSizes[i];
				uint64_t currentBufferSize = currentBufferLength * sizeof(float);
				setBufferLength(currentBufferLength);
				if (currentBufferLength > aFrameRate)
					break;

				std::string strBenchmarkName = "";
				std::string strBufferSize = std::to_string(currentBufferLength);
				strBenchmarkName.append(strBufferSize);

				uint32_t centre = n / 2;
				uint32_t inputPosition[2] = { centre,centre };
				uint32_t outputPosition[2] = { centre - 10, centre - 10 };
				float boundaryValue = 1.0;
				fdtdSynth.createModel(modelPathManual, boundaryValue, inputPosition, outputPosition);

				float mu = 0.1;
				float muSquared = mu * mu;
				float muSquaredTwo = 2 * mu * mu;
				float muSquaredEight = 8 * mu * mu;
				float muSquaredTwenty = 20 * mu * mu;
				float sigma = 50.01;
				float deltaT = 1.0 / 44100.0;
				float sigmaMinus = 1 - sigma * deltaT;
				float sigmaPlus = 1.0 / (float)(1.0 + sigma * deltaT);
				fdtdSynth.updateCoefficient("muSquared", 9, muSquared);
				fdtdSynth.updateCoefficient("muSquaredTwo", 10, muSquaredTwo);
				fdtdSynth.updateCoefficient("muSquaredEight", 11, muSquaredEight);
				fdtdSynth.updateCoefficient("muSquaredTwenty", 12, muSquaredTwenty);
				fdtdSynth.updateCoefficient("sigmaMinus", 13, sigmaMinus);
				fdtdSynth.updateCoefficient("sigmaPlus", 14, sigmaPlus);

				uint64_t numSamplesComputed = 0;
				impulse(currentBufferLength, 5, inputBuffer_);

				if (isWarmup)
				{
					fdtdSynth.fillBuffer(inputBuffer_, outputBuffer_, currentBufferLength);
				}
				impulse(currentBufferLength, 5, inputBuffer_);
				while (numSamplesComputed < aFrameRate)
				{
					clBenchmarker_.startTimer(strBenchmarkName);
					fdtdSynth.fillBuffer(inputBuffer_, outputBuffer_, currentBufferLength);
					clBenchmarker_.pauseTimer(strBenchmarkName);

					//Log audio for inspection if necessary//
					for (int j = 0; j != currentBufferLength; ++j)
						soundBuffer_[numSamplesComputed + j] = outputBuffer_[j];

					numSamplesComputed += currentBufferLength;
				}
				clBenchmarker_.elapsedTimer(strBenchmarkName);

				//Save audio to file for inspection//
				std::string strBenchmarkFileNameManualWav = strBenchmarkFileNameManual;
				strBenchmarkFileNameManualWav.append("bufferlength");
				strBenchmarkFileNameManualWav.append(std::to_string(i));
				strBenchmarkFileNameManualWav.append(".wav");
				outputAudioFile(strBenchmarkFileNameManualWav.c_str(), soundBuffer_, aFrameRate, aFrameRate);
				std::cout << "cl_runSingleModelTestRealtime successful: Inspect audio log \"cl_runComplexSingleModelTestRealtime.wav\"" << std::endl << std::endl;

				numSamplesComputed = 0;
			}
		}
	}
	static void outputAudioFile(const char* aPath, float* aAudioBuffer, uint32_t aAudioLength, uint32_t aSampleRate)
	{
		AudioFile<float> audioFile;
		AudioFile<float>::AudioBuffer buffer;

		buffer.resize(1);
		buffer[0].resize(aAudioLength);
		audioFile.setBitDepth(24);
		audioFile.setSampleRate(aSampleRate);

		for (int k = 0; k != aAudioLength; ++k)
			buffer[0][k] = (float)aAudioBuffer[k];

		audioFile.setAudioBuffer(buffer);
		audioFile.save(aPath);
	}
public:
	GPU_Benchmark_OpenCL(std::string aDeviceName, uint32_t aPlatform, uint32_t aDevice) : fdtdSynth(Implementation::OPENCL, aDevice, 44100, 0.001), deviceName_(aDeviceName), clBenchmarker_("CL_Logs/openclog.csv", { "Test_Name", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" })
	{
		currentPlatformIdx_ = aPlatform;
		currentDeviceIdx_ = aDevice;
		bufferSizes[0] = 1;
		for (size_t i = 1; i != bufferSizesLength; ++i)
		{
			bufferSizes[i] = bufferSizes[i - 1] * 2;
		}

		//Initialise workgroup dimensions//
		globalWorkspace_ = cl::NDRange(1024);
		localWorkspace_ = cl::NDRange(256);

		// Allocate space for input and output buffers.
		soundBuffer_ = new float[sampleRate_ * 5];
		inputBuffer_ = new float[bufferSizes[bufferSizesLength]];
		outputBuffer_ = new float[bufferSizes[bufferSizesLength]];
	}
	~GPU_Benchmark_OpenCL()
	{
		delete soundBuffer_;
		delete inputBuffer_;
		delete outputBuffer_;
	}

	void runGeneralBenchmarks(uint32_t aNumRepetitions, bool isWarmup)
	{
		//Run tests with setup//
		for (uint32_t n = minDimensionSize_; n <= maxDimensionSize_; n *= 2)
		{
			std::string modelPathAuto = "resources/kernels/auto/singleModelTestAuto";
			modelPathAuto.append(std::to_string(n));
			modelPathAuto.append(".json");
			for (uint32_t i = 0; i != bufferSizesLength; ++i)
			{
				uint64_t currentBufferSize = bufferSizes[i];
				std::string benchmarkFileName = "CL_Logs/";
				benchmarkFileName.append(deviceName_);
				benchmarkFileName.append("_cl_");
				std::string strBufferSize = std::to_string(currentBufferSize);
				benchmarkFileName.append("buffersize");
				benchmarkFileName.append(strBufferSize);
				benchmarkFileName.append("dimensions");
				benchmarkFileName.append(std::to_string(n));
				benchmarkFileName.append(".csv");
				clBenchmarker_ = Benchmarker(benchmarkFileName, { "Test_Name", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });
				setBufferLength(currentBufferSize);


				runSingleModelTest(aNumRepetitions, isWarmup, modelPathAuto);
			}
		}
	}
	void runRealTimeBenchmarks(uint32_t aSampleRate, bool isWarmup)
	{
		runSimpleSingleModelTestRealtime(aSampleRate, false);
		//runSimpleMultiModelTestRealtime(aSampleRate, false);
		//runComplexSingleModelTestRealtime(aSampleRate, false);
		//runComplexMultiModelTestRealtime(aSampleRate, false);
	}

	static bool openclCompatible()
	{
		cl::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		if (platforms.size() != 0)
		{
			std::cout << "OpenCL Platform Version: " << platforms[0].getInfo<CL_PLATFORM_VERSION>() << std::endl << std::endl;
			return true;
		}

		return false;
	}

};

#endif