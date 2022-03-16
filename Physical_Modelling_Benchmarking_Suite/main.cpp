#include <vector>

#include "GPU_Benchmark_OpenCL.hpp"

int main(int argc, char** argv)
{
	OpenCL_Wrapper::printAvailableDevices();

	//Check OpenCL support and device availability//
	bool isOpenCl = GPU_Benchmark_OpenCL::openclCompatible();
	if (isOpenCl)
	{
		std::vector<OpenCL_Device> clDevices = OpenCL_Wrapper::getOpenclDevices();

		std::cout << "OpenCL device and support detected." << std::endl;
		std::cout << "Beginning OpenCL benchmarking" << std::endl << std::endl;
		for (uint32_t i = 0; i != clDevices.size(); ++i)
		{
			std::cout << "Runnning tests for platform " << clDevices[i].platform_name << " device " << clDevices[i].device_name << std::endl;
			GPU_Benchmark_OpenCL clBenchmark(clDevices[i].platform_name, clDevices[i].platform_id, clDevices[i].device_id);

			//clBenchmark.setBufferLength(44100);

			//clBenchmark.cl_mappingmemory(100);
			//clBenchmark.runGeneralBenchmarks(86, false);
			clBenchmark.runRealTimeBenchmarks(44100, true);
		}
	}
	else
		std::cout << "OpenCL device or support not present to benchmark OpenCL." << std::endl;

	char haltc;
	std::cin >> haltc;
}