#include <iostream>
using namespace std;

#include <caffe/net.hpp>
#include <caffe/caffe.hpp>
#include <viennacl/backend/opencl.hpp>
#include <opencv2/opencv.hpp>
#include "clManager.hpp"
#include <viennacl/backend/opencl.hpp>
#include <thread>
#include <mutex>
void f(){viennacl::ocl::context &ctx = viennacl::ocl::get_context(0);}

#define MULTI_LINE_STRING(ARG) #ARG

namespace op
{
    const char *resizeAndMergeKernel = MULTI_LINE_STRING(
        __kernel void resizeAndMergeKernel(__global float* image)
        {
            int i = get_global_id(0);
            int j = get_global_id(1);
            if(i == 0 && j==0)
                printf("%f\n",image[0]);
        }
    );
}

class Net
{
public:
    std::unique_ptr<caffe::Net<float>> upCaffeNet;

    Net(){

    }

    void initNet(std::string protoPath, std::string modelPath, int gpuID)
    {
        // Load net
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
        caffe::Caffe::SetDevice(gpuID);
        caffe::device* device = caffe::Caffe::GetDevice(gpuID,0);
        upCaffeNet.reset(new caffe::Net<float>{protoPath, caffe::TEST, device});
        upCaffeNet->CopyTrainedLayersFrom(modelPath);
        cout << "Net Loaded " << gpuID << endl;

        // Setup my OpenCL
        op::CLManager::getInstance(gpuID, CL_DEVICE_TYPE_GPU, true);
        op::CLManager::getInstance(gpuID)->buildKernelIntoManager("resizeAndMergeKernel",op::resizeAndMergeKernel);
        cout << "OpenCL Setup" << gpuID << endl;

        // Load 1 image
        cv::Mat image, image2, image3;
        image = cv::imread(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/image.jpg");
        cv::resize(image, image2, cv::Size(224,244));
        image2.convertTo(image3, CV_32FC3);
        caffe::BlobProto blob_proto;
        blob_proto.set_channels(3);
        blob_proto.set_height(224);
        blob_proto.set_width(224);
        blob_proto.clear_data();
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < 224; ++h) {
                for (int w = 0; w < 224; ++w) {
                    blob_proto.add_data(image3.at<cv::Vec3f>(h, w)[c]);
                }
            }
        }
        blob_proto.set_num(1);
        caffe::Blob<float>* input_layer = upCaffeNet->input_blobs()[0];
        input_layer->FromProto(blob_proto);
        cout << "Image Loaded" << gpuID << endl;

        // Forward Pass
        upCaffeNet->Forward();
        cout << "Forward Done" << gpuID << endl;

        // Get my original image as a cl::Buffer
        const float* gpuData = input_layer->gpu_data();
        const float* cpuData = input_layer->cpu_data();
        cout << "CPU: " << cpuData[0] << endl;
        cl_int err;
        cl::Buffer x((cl_mem)gpuData, true);
        cout << "cl::Buffer setup" << gpuID << endl;

        // Run my Kernel
        cl::Kernel& resizeAndMergeKernel = op::CLManager::getInstance(gpuID)->getKernelFromManager("resizeAndMergeKernel");
        try{
            resizeAndMergeKernel.setArg(0,x);
        }catch(cl::Error& e){
            cout << "ERROR: " << e.what() << gpuID << endl;
            cout << "ERROR: " << e.err() << gpuID << endl;
        }
        op::CLManager::getInstance(gpuID)->getQueue().enqueueNDRangeKernel(resizeAndMergeKernel, cl::NDRange(), cl::NDRange(224,224), cl::NDRange(), NULL, NULL);
        cout << "OpenCL Kernel Run" << gpuID << endl;
    }
};

void thread_handler(int gpuID){
    static std::mutex m;
    cout << gpuID << endl;
    std::string protoPath = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/bvlc_googlenet/deploy.prototxt";
    std::string modelPath = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/bvlc_googlenet/bvlc_googlenet.caffemodel";
    Net n;
    //m.lock();
    n.initNet(protoPath, modelPath, gpuID);
    //m.unlock();
}

int main(){
    std::thread t1(&thread_handler,1);
    std::thread t2(&thread_handler,2);
    t1.join();
    t2.join();
    return 0;
}
