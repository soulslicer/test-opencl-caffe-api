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

#include <random>

namespace op
{
    const char *resizeAndMergeKernel = MULTI_LINE_STRING(
        __kernel void resizeAndMergeKernel(__global float* image)
        {
            int i = get_global_id(0);
            int j = get_global_id(1);
            //if(i == 0 && j==0)
            //    printf("%f\n",image[0]);
        }
    );
}

class Net
{
public:
    std::unique_ptr<caffe::Net<float>> upCaffeNet;
    bool log = true;
    int gpuID = 0;
    std::shared_ptr<caffe::Caffe> caffe;

    std::random_device rd;
    std::mt19937 mt;
    std::uniform_real_distribution<double> dist;

    Net(){
        mt = std::mt19937(rd());
        dist = std::uniform_real_distribution<double>(224, 230);
    }

    void initNet(std::string protoPath, std::string modelPath, int gpuID)
    {

        //caffe::Caffe::set_mode(caffe::Caffe::GPU);
        //caffe::Caffe::SetDevice(gpuID);
        this->gpuID = gpuID;
        cout << "GPU: " << this->gpuID << endl;
        caffe::Caffe::SelectDevice(this->gpuID, false);
        upCaffeNet.reset(new caffe::Net<float>{protoPath, caffe::TEST, caffe::Caffe::GetDefaultDevice()});
        upCaffeNet->CopyTrainedLayersFrom(modelPath);
        if(log) cout << "Net Loaded " << gpuID << endl;

        // Setup my OpenCL
        op::CLManager::getInstance(gpuID, CL_DEVICE_TYPE_GPU, true);
        op::CLManager::getInstance(gpuID)->buildKernelIntoManager("resizeAndMergeKernel",op::resizeAndMergeKernel);
        if(log) cout << "OpenCL Setup" << gpuID << endl;
    }

    void forwardNet(){
        cv::Size newSize(int(dist(mt)),int(dist(mt)));
        //cv::Size newSize(120,150);
        cout << newSize << endl;

        // Load 1 image
        cv::Mat image, image2, image3;
        image = cv::imread(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/image.jpg");
        cv::resize(image, image2, newSize);
        image2.convertTo(image3, CV_32FC3);
        caffe::BlobProto blob_proto;
        blob_proto.set_channels(3);
        blob_proto.set_height(newSize.width);
        blob_proto.set_width(newSize.height);
        blob_proto.clear_data();
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < newSize.height; ++h) {
                for (int w = 0; w < newSize.width; ++w) {
                    blob_proto.add_data(image3.at<cv::Vec3f>(h, w)[c]);
                }
            }
        }
        blob_proto.set_num(1);
        caffe::Blob<float>* input_layer = upCaffeNet->input_blobs()[0];
        if(log) cout << "Image Loaded" << gpuID << endl;

        upCaffeNet->blobs()[0]->Reshape({1,3,newSize.width,newSize.height});
        upCaffeNet->Reshape();
        if(log) cout << "Reshape Done" << gpuID << endl;

        input_layer->FromProto(blob_proto);
        if(log) cout << "Caffe Loaded" << gpuID << endl;

        // Forward Pass
        upCaffeNet->Forward(0);
        if(log) cout << "Forward Done" << gpuID << endl;
        cout << "Forward Done" << gpuID << endl;

        // Get my original image as a cl::Buffer
        const float* gpuData = input_layer->gpu_data();
        const float* cpuData = input_layer->cpu_data();
        if(log) cout << "CPU: " << cpuData[0] << endl;
        cl_int err;
        cl::Buffer x((cl_mem)gpuData, true);
        if(log) cout << "cl::Buffer setup" << gpuID << endl;

        // Run my Kernel
        cl::Kernel& resizeAndMergeKernel = op::CLManager::getInstance(gpuID)->getKernelFromManager("resizeAndMergeKernel");
        try{
            resizeAndMergeKernel.setArg(0,x);
        }catch(cl::Error& e){
            if(log) cout << "*******ERROR: " << e.what() << gpuID << endl;
            if(log) cout << "*******ERROR: " << e.err() << gpuID << endl;
            return;
        }
        op::CLManager::getInstance(gpuID)->getQueue().enqueueNDRangeKernel(resizeAndMergeKernel, cl::NDRange(), cl::NDRange(newSize.width,newSize.height), cl::NDRange(), NULL, NULL);
        if(log) cout << "OpenCL Kernel Run" << gpuID << endl;

        cout << dist(mt) << endl;

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
    for(int i=0; i<1000; i++){
        n.forwardNet();
    }
    cout << "Finish" << endl;
    //m.unlock();
}

int main(){
    caffe::Caffe::set_mode(caffe::Caffe::GPU);  // Global effect, ONLY to be used on main thread
    // Select all absolute/global device IDs you want to use for computation later on
    std::vector<int> device_ids = {0,1};  // You could also use GPUs 2 and 3, but you wouldn't have to change the IDs for below if you use `list_id = true`.
    caffe::Caffe::SetDevices(device_ids);

    std::thread t1(&thread_handler,0);
    //std::thread t2(&thread_handler,1);
    t1.join();
    //t2.join();
    return 0;
}
