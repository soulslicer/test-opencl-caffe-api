#include <clManager.hpp>

namespace op
{
    std::shared_ptr<CLManager> CLManager::getInstance(int deviceId, int deviceType, bool getFromVienna)
    {
        static std::map<int, std::shared_ptr<CLManager>> clManagers;
        static std::mutex m;
        if(clManagers.count(deviceId))
            return clManagers[deviceId];
        else
        {
            m.lock();
            clManagers[deviceId] = std::shared_ptr<CLManager>(new CLManager(deviceId, deviceType, getFromVienna));
            m.unlock();
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
            std::vector<cl::Platform> platforms;
            std::vector<cl::Device> devices;
            std::string deviceName;
            cl_uint i, type;
            try {
                cl::Platform::get(&platforms);
                switch(deviceType)
                {
                    case CL_DEVICE_TYPE_GPU:
                    {
                        type = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
                        if( type == CL_SUCCESS)
                        {
                            // Get only relavent device
                            cl::Context allContext(devices);
                            std::vector<cl::Device> gpuDevices;
                            gpuDevices = allContext.getInfo<CL_CONTEXT_DEVICES>();
                            bool deviceFound = false;
                            for(int i=0; i<gpuDevices.size(); i++){
                                if(i == deviceId){
                                    device = gpuDevices[i];
                                    context = cl::Context(device);
                                    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                                    deviceFound = true;
                                    cout << "Made new GPU Instance: " << deviceId << endl;
                                    break;
                                }
                            }
                            if(!deviceFound)
                            {
                                throw std::runtime_error("Error: Invalid GPU ID");
                            }
                        }
                        else if(type == CL_INVALID_DEVICE_TYPE || type == CL_DEVICE_NOT_FOUND)
                        {
                            throw std::runtime_error("Error: GPU Invalid Device or Device not found");
                        }
                        break;
                    }

                    case CL_DEVICE_TYPE_CPU:
                    {
                        cl::Platform::get(&platforms);
                        type = platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);
                        if( type == CL_SUCCESS)
                        {
                            // Get only relavent device
                            std::vector<cl::Device> devices;
                            cl::Context allContext(devices);
                            std::vector<cl::Device> cpuDevices;
                            cpuDevices = allContext.getInfo<CL_CONTEXT_DEVICES>();
                            bool deviceFound = false;
                            for(int i=0; i<cpuDevices.size(); i++){
                                if(i == deviceId){
                                    device = cpuDevices[i];
                                    context = cl::Context(device);
                                    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                                    deviceFound = true;
                                    cout << "Made new CPU Instance: " << deviceId << endl;
                                    break;
                                }
                            }
                            if(!deviceFound)
                            {
                                throw std::runtime_error("Error: Invalid CPU ID");
                            }
                        }
                        else if(type == CL_INVALID_DEVICE_TYPE || type == CL_DEVICE_NOT_FOUND)
                        {
                            throw std::runtime_error("Error: CPU Invalid Device or Device not found");
                        }
                        break;
                    }

                    case CL_DEVICE_TYPE_ACCELERATOR:
                    {
                        cl::Platform::get(&platforms);
                        type = platforms[0].getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
                        if( type == CL_SUCCESS)
                        {
                            // Get only relavent device
                            std::vector<cl::Device> devices;
                            cl::Context allContext(devices);
                            std::vector<cl::Device> accDevices;
                            accDevices = allContext.getInfo<CL_CONTEXT_DEVICES>();
                            bool deviceFound = false;
                            for(int i=0; i<accDevices.size(); i++){
                                if(i == deviceId){
                                    device = accDevices[i];
                                    context = cl::Context(device);
                                    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                                    deviceFound = true;
                                    cout << "Made new ACC Instance: " << deviceId << endl;
                                    break;
                                }
                            }
                            if(!deviceFound)
                            {
                                throw std::runtime_error("Error: Invalid ACC ID");
                            }
                        }
                        else if(type == CL_INVALID_DEVICE_TYPE || type == CL_DEVICE_NOT_FOUND)
                        {
                            throw std::runtime_error("Error: ACC Invalid Device or Device not found");
                        }
                        break;
                    }

                    default:
                    {
                        throw std::runtime_error("Error: No such CL Device Type");
                    }
                }
            }
            catch(cl::Error e) {
                log("Error: " + std::string(e.what()));
            }
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
            program = cl::Program(context, src, true);
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
