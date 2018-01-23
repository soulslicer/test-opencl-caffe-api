#include <iostream>
using namespace std;

#include <caffe/net.hpp>
#include <caffe/caffe.hpp>
#include <viennacl/backend/opencl.hpp>
#include <opencv2/opencv.hpp>
#include "clManager.hpp"
#include <viennacl/backend/opencl.hpp>
void f(){viennacl::ocl::context &ctx = viennacl::ocl::get_context(0);}

#define MULTI_LINE_STRING(ARG) #ARG

namespace op
{
    const char *resizeAndMergeKernel = MULTI_LINE_STRING(
        __kernel void resizeAndMergeKernel(const int size, __global float* image)
        {
        }
    );
}

class Net
{
public:
    std::unique_ptr<caffe::Net<float>> upCaffeNet;
    cl::Buffer x;

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
        cout << "Net Loaded" << endl;

        // Setup my OpenCL
        op::CLManager::getInstance(gpuID, CL_DEVICE_TYPE_GPU, true);
        op::CLManager::getInstance(gpuID)->buildKernelIntoManager("resizeAndMergeKernel",op::resizeAndMergeKernel);
        cout << "OpenCL Setup" << endl;

//        // Load 1 image
//        cv::Mat image, image2, image3;
//        image = cv::imread(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/image.jpg");
//        cv::resize(image, image2, cv::Size(224,244));
//        image2.convertTo(image3, CV_32FC3);
//        caffe::BlobProto blob_proto;
//        blob_proto.set_channels(3);
//        blob_proto.set_height(224);
//        blob_proto.set_width(224);
//        blob_proto.clear_data();
//        for (int c = 0; c < 3; ++c) {
//            for (int h = 0; h < 224; ++h) {
//                for (int w = 0; w < 224; ++w) {
//                    blob_proto.add_data(image3.at<cv::Vec3f>(h, w)[c]);
//                }
//            }
//        }
//        blob_proto.set_num(1);
//        caffe::Blob<float>* input_layer = upCaffeNet->input_blobs()[0];
//        input_layer->FromProto(blob_proto);
//        cout << "Image Loaded" << endl;

//        // Forward Pass
//        upCaffeNet->Forward();
//        cout << "Forward Done" << endl;

        // My own buffer
        cl_int err;
        float* test = new float[3*224*224];
        x = cl::Buffer(op::CLManager::getInstance(gpuID)->getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * 3 * 224 * 224, (void*)test, &err);
        cout << err << endl;

        //cl::Buffer x = cl::Buffer(op::CLManager::getInstance(gpuID)->getContext(), CL_MEM_READ_WRITE, sizeof(float) * 3 * 224 * 224);
        //cl::Buffer x = cl::Buffer(op::CLManager::getInstance(gpuID)->getContext(), CL_MEM_READ_WRITE, sizeof(float) * 3 * 224 * 224);

//        // Get my original image as a cl::Buffer
//        const float* gpuData = input_layer->gpu_data();
//        cl_int err;
//        cl::Buffer x((cl_mem)gpuData, true);
//        //cl::Buffer x(reinterpret_cast<cl_mem>(const_cast<float*>(gpuData)), true);
//        //cl::Buffer x(reinterpret_cast<const cl_mem>(gpuData), true);
//        //cl::Buffer x(op::CLManager::getInstance(gpuID)->getContext(), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * 3 * 224 * 224, const_cast<float*> (gpuData), &err);
//        cout << err << endl;
//        cout << "cl::Buffer setup" << endl;

        // Run my Kernel
//        cl::Kernel& resizeAndMergeKernel = op::CLManager::getInstance(gpuID)->getKernelFromManager("resizeAndMergeKernel");
//        try{
//            resizeAndMergeKernel.setArg(0,(int)(3 * 224 * 224));

//            resizeAndMergeKernel.setArg(1,x);
//            //resizeAndMergeKernel.setArg(0,x);
//            cout << "A" << endl;
//        }catch(cl::Error& e){
//            cout << "ERROR: " << e.what() << endl;
//            cout << "ERROR: " << e.err() << endl;
//        }
//        op::CLManager::getInstance(gpuID)->getQueue().enqueueNDRangeKernel(resizeAndMergeKernel, cl::NDRange(), cl::NDRange(224,224), cl::NDRange(), NULL, NULL);
//        cout << "OpenCL Kernel Run" << endl;
    }
};


int main(){
    // Load net
    std::string protoPath = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/bvlc_googlenet/deploy.prototxt";
    std::string modelPath = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/bvlc_googlenet/bvlc_googlenet.caffemodel";
    Net n;
    std::vector<std::shared_ptr<Net>> nets;
    nets.emplace_back(std::make_shared<Net>(Net()));
    nets.back()->initNet(protoPath, modelPath, 0);
    return 0;
}
