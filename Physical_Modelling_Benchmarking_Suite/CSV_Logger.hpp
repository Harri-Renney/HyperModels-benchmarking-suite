#ifndef CSV_LOGGER_HPP
#define CSV_LOGGER_HPP

#include <string>
#include <vector>
#include <fstream>

class CSV_Logger
{
public:
	size_t recordLength_;
	std::ofstream csvFile;
	CSV_Logger(const std::string aFilePath, std::vector<std::string> aFields)
	{
		recordLength_ = aFields.size();

		csvFile.open(aFilePath);

		for (uint32_t i = 0; i != recordLength_; ++i)
			csvFile << aFields[i] << ",";
		csvFile << "\n";
	}
	bool addRecord(std::vector<std::string> aRecord)
	{
		if (aRecord.size() != recordLength_)
			return false;

		for (uint32_t i = 0; i != aRecord.size(); ++i)
			csvFile << aRecord[i] << ",";
		csvFile << "\n";

		return true;
	}
	bool addField(std::string aRecord)
	{
		//if (aRecord.size() != recordLength_)
		//	return false;

		csvFile << aRecord << ",";

		return true;
	}
	bool endRecord()
	{
		//if (aRecord.size() != recordLength_)
		//	return false;

		csvFile << "\n";

		return true;
	}
};

#endif