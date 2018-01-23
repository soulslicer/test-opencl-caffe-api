#include <clManager.hpp>

namespace op
{
    std::shared_ptr<CLManager> CLManager::getInstance(int deviceId, int deviceType, bool getFromVienna)
    {
        static std::map<int, std::shared_ptr<CLManager>> clManagers;
        if(clManagers.count(deviceId))
            return clManagers[deviceId];
        else
        {
            clManagers[deviceId] = std::shared_ptr<CLManager>(new CLManager(deviceId, deviceType, getFromVienna));
            return clManagers[deviceId];
        }
    }

    CLManager::CLManager(int deviceId, int deviceType, bool getFromVienna)
    {
        if(getFromVienna)
        {

            context = cl::Context(caffe::Caffe::GetOpenCLContext(deviceId, 0), true);
            queue = cl::CommandQueue(caffe::Caffe::GetOpenCLQueue(deviceId, 0), true);
            //context.printContext();
        }
        else
        {
            throw std::runtime_error("Error: Set variable getFromVienna to true");
        }
    }

    CLManager::~CLManager()
    {

    }

    cl::Context& CLManager::getContext()
    {
        return context;
    }

    cl::CommandQueue& CLManager::getQueue()
    {
        return queue;
    }

    cl::Device& CLManager::getDevice()
    {
        return device;
    }

    cl::Program CLManager::buildProgramFromSource(std::string src, bool isFile)
    {
        cl::Program program;
        try{
            if(isFile)
            {
                std::ifstream programFile((char*) src.c_str());
                std::string programString(std::istreambuf_iterator<char>(programFile),
                                                  (std::istreambuf_iterator<char>()));
                src = programString;
                //src = std::regex_replace(src, std::regex(";"), std::string(";\n"));
            }
            program = cl::Program(src, true);
        }
        catch(cl::Error e) {
            cerr << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(getDevice()) << endl;
            cerr << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(getDevice()) << endl;
            cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(getDevice()) << endl;
            exit(-1);
        }
        return program;
    }

    bool CLManager::buildKernelIntoManager(std::string kernelName, std::string src, bool isFile){
        // Program not built
        if (!(clPrograms.find(src) != clPrograms.end()))
        {
            clPrograms[src] = buildProgramFromSource(src, isFile);
        }

        cl::Program& program = clPrograms[src];

        // Kernel not built
        if (!(clKernels.find(kernelName) != clKernels.end()))
        {
            clKernels[kernelName] = cl::Kernel(program, kernelName.c_str());
            log("Kernel " + kernelName + " built successfully");
            return true;
        }
        else
        {
            log("Kernel " + kernelName + " already built");
            return false;
        }
    }

    cl::Kernel& CLManager::getKernelFromManager(std::string kernelName){
        if (!(clKernels.find(kernelName) != clKernels.end()))
        {
            throw std::runtime_error("Error: Kernel not found in Manager: " + kernelName);
        }
        return clKernels[kernelName];
    }
}
