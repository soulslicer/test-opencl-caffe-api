#ifndef OPENPOSE_CORE_CL_MANAGER_HPP
#define OPENPOSE_CORE_CL_MANAGER_HPP

#include <atomic>
#include <tuple>
#include <map>
#include <fstream>
#include <regex>

#include <iostream>
using namespace std;

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl2.hpp>

// Singleton structure
// https://stackoverflow.com/questions/1008019/c-singleton-design-pattern

#define DELETE_COPY(className) \
    className(const className&) = delete; \
    className& operator=(const className&) = delete

#include <caffe/net.hpp>
#include <caffe/caffe.hpp>
#include <viennacl/backend/opencl.hpp>
#include <mutex>

namespace op
{
    class CLManager
    {
    private:
    public:
        static std::shared_ptr<CLManager> getInstance(int deviceId = 0, int deviceType = CL_DEVICE_TYPE_GPU, bool getFromVienna = false);
        ~CLManager();
        DELETE_COPY(CLManager);

    private:
        CLManager(int deviceId, int deviceType, bool getFromVienna);
        std::map<std::string, cl::Program> clPrograms;
        std::map<std::string, cl::Kernel> clKernels;
        cl::Device device;
        cl::CommandQueue queue;
        cl::Context context;
        cl::Program buildProgramFromSource(std::string src, bool isFile = false);

    public:
        cl::Context& getContext();
        cl::CommandQueue& getQueue();
        cl::Device& getDevice();
        bool buildKernelIntoManager(std::string kernelName, std::string src, bool isFile = false);
        cl::Kernel& getKernelFromManager(std::string kernelName);
        inline void log(std::string x){std::cout << x << std::endl;}
    };
}

#endif // OPENPOSE_CORE_CL_MANAGER_HPP
