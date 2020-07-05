#define BLOCK_SIZE 256
#define DOWNSCALE_MAX_FRAME_HIGHT 400

#include <mutex>
#include <condition_variable>
#include "salient/salient.cuh"
#include "vrbounds.hpp"
#include "vulkanheadless.hpp"
#include "cameraD415.hpp"
#include "saber-salient.hpp"


__global__ void draw_foreground(
    int W, int H, int downsample_ratio, float near, float far,
    uint8_t* out_argb, float* out_depth, const float* probabilities, const uint8_t* in_color, const float* in_depth)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= W || j >= H)
        return;

    int idx = i + j * W;
    int idx_argb = i + (H - 1 - j) * W;
    int idx_depth = i / downsample_ratio + ((H - 1 - j) / downsample_ratio) * (W / downsample_ratio);

    // make foreground semi-transparent if the probability is between 0.45 and 0.55:
    const float a = max(0.0f, min(1.0f, probabilities[idx] * 10.0f - 4.5f));

    // input is in YUYV
    float Y = ((float)in_color[idx * 2]) - 16.0f;
    float Cb = ((float)in_color[(idx - idx % 2) * 2 + 1]) - 128.0f;
    float Cr = ((float)in_color[(idx - idx % 2) * 2 + 3]) - 128.0f;

    // ... transform into ARGB
    float b = max(0.0f, min(255.0f, 1.163999557f * Y + 2.017999649f * Cb));
    float g = max(0.0f, min(255.0f, 1.163999557f * Y - 0.812999725f * Cr - 0.390999794f * Cb));
    float r = max(0.0f, min(255.0f, 1.163999557f * Y + 1.595999718f * Cr));

    // combine with background
    out_argb[idx_argb * 4 + 0] = (uint8_t)__float2int_rd(a * 255.0f);
    out_argb[idx_argb * 4 + 1] = (uint8_t)__float2int_rd(r);
    out_argb[idx_argb * 4 + 2] = (uint8_t)__float2int_rd(g);
    out_argb[idx_argb * 4 + 3] = (uint8_t)__float2int_rd(b);

    out_depth[idx_argb] = a <= 0 ? 2.0f : ((abs(in_depth[idx_depth]) - near) / far);
}


// I screw the camera in on top of the vive controller, thus fixing one of the axes solid.
// The two remaining axes are defined by one angle - the position where the camera is screwed in tight.
// I determine this angle by (1) guessing the approximate value (2) tuning it by drawing controller and camera positions on screen.
salient::CameraExtrinsics cameraAttachedToTracker(const float angle_y = 2.02f)
{
    const float sph = sin(angle_y), cph = cos(angle_y);
    return salient::CameraExtrinsics {
        {cph, sph, 0,
         0, 0, -1,
         -sph, cph, 0},
        {0, 0, 0.02f} };
}

class semaphore
{
private:

    std::mutex mtx;
    std::condition_variable cv;
    int count;

public:

    semaphore(int count_ = 0) : count{ count_ }
    {}

    void notify()
    {
        std::unique_lock<std::mutex> lck(mtx);
        ++count;
        cv.notify_one();
    }

    void wait()
    {
        std::unique_lock<std::mutex> lck(mtx);
        while (count == 0)
        {
            cv.wait(lck);
        }

        --count;
    }
};


class SaberSalient
{
private:
    semaphore initialized;
    int devId = 0;
    cudaStream_t mainStream = nullptr;
    uint8_t deviceUUID[sizeof(cudaUUID_t)];
    VulkanHeadless* vulkanHeadless = nullptr;
    std::thread* workerThread;
    bool stillRunning = true;

    /** Original image from the color camera. */
    uint8_t* yuyvGPU = nullptr;

    /** RGBA image rendered with only foreground being visible. */
    uint8_t* argbGPU = nullptr;
    float* depthGPU = nullptr;

    const salient::CameraExtrinsics color2tracker;

public:

    /** CPU copy of the rendered camera image. */
    uint8_t* argbA, * argbB;
    float* depthA, * depthB;

    int color_W, color_H, color_N, downscaled_W, downscaled_H, downsample_ratio;

    VrBounds<VulkanHeadless::Vertex>* vrBounds = nullptr;
    camera::Camera* camera = nullptr;

    void loop()
    {
        camera = new camera::IntelD415Camera();

        int devCount;
        cudaGetDeviceCount(&devCount);

        // Select the GPU.
        // The idea is to select a secondary, less powerfull GPU for this, so that it does not interfere with
        // the main user activity (such as playing VR).
        int devId = devCount > 1 ? 1 : 0;
        cudaSetDevice(devId);
        cudaStreamCreate(&mainStream);

        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, devId);
        printf("Selected CUDA device: %s\n", deviceProps.name);
        memcpy(deviceUUID, &(deviceProps.uuid), sizeof(cudaUUID_t));

        color_W = camera->getColorIntrinsics().width;
        color_H = camera->getColorIntrinsics().height;
        color_N = color_W * color_H;
        downscaled_W = color_W;
        downscaled_H = color_H;
        downsample_ratio = 1;
        while (downscaled_H > DOWNSCALE_MAX_FRAME_HIGHT)
        {
            downsample_ratio <<= 1;
            downscaled_H >>= 1;
            downscaled_W >>= 1;
        }

        auto toVertex = [](float x, float y, float z) { return VulkanHeadless::Vertex{ {x, y, z} }; };
        vrBounds = new VrBounds<VulkanHeadless::Vertex>(toVertex, color2tracker, camera->getColorIntrinsics());

        // original image from the color camera
        cudaMalloc((void**)&yuyvGPU, sizeof(uint8_t) * color_N * 2);
        cudaErrorCheck(nullptr);

        // transformed RGB image with foreground mask applied
        cudaMalloc((void**)&argbGPU, sizeof(uint8_t) * color_N * 4);
        cudaErrorCheck(nullptr);
        cudaMalloc((void**)&depthGPU, sizeof(float) * color_N);
        cudaErrorCheck(nullptr);

        argbA = new uint8_t[color_N * 4];
        argbB = new uint8_t[color_N * 4];
        depthA = new float[color_N];
        depthB = new float[color_N];


        // how to access color data at every pixel position.
        auto yuyvGPU0 = yuyvGPU;
        auto color_W0 = color_W;
        auto getFeature = [yuyvGPU0, color_W0] __device__(const int i, const int j, float* out_feature) {
            const int base_off = i + j * color_W0;
            out_feature[0] = (float)yuyvGPU0[base_off * 2];
            out_feature[1] = (float)yuyvGPU0[(base_off - i % 2) * 2 + 1];
            out_feature[2] = (float)yuyvGPU0[(base_off - i % 2) * 2 + 3];
        };
        salient::RealSalient<3, 7, decltype(getFeature)> realSalient(
            mainStream,
            camera->getDepthIntrinsics(),
            camera->getColorIntrinsics(),
            camera->getColorToDepthTransform(),
            downsample_ratio,
            camera->getDepthScale(), getFeature);


        // vulkan-to-cuda rendering
        vulkanHeadless = new VulkanHeadless(downscaled_W, downscaled_H, vrBounds->vertices, vrBounds->indices, deviceUUID);

        dim3 work(color_W, color_H, 1);
        dim3 blockSize(16, 16, 1);
        dim3 blocks = salient::distribute(work, blockSize);

        initialized.notify();
        while (stillRunning)
        {
            // update camera model-view-projection matrix and the color-image-space bounding box.
            vrBounds->update();

            if (vrBounds->trackerFound)
                vulkanHeadless->render(vrBounds->mvpMatrix, mainStream);
            auto frames = camera->waitForFrames();

            // copy the color frame, so that getFeature gets the actual color data.
            cudaMemcpyAsync(yuyvGPU, frames->getColor(), sizeof(uint8_t) * color_N * 2, cudaMemcpyHostToDevice, mainStream);
            cudaErrorCheck(mainStream);

            // load frames to gpu and preprocess
            if (vrBounds->validDevCount > 0)
                realSalient.processFrames(
                    (const uint16_t*)frames->getDepth(),
                    vrBounds->trackerBounds,
                    vulkanHeadless->getCudaTexture());

            draw_foreground<<<blocks, blockSize, 0, mainStream>>>(
                color_W, color_H, downsample_ratio, vrBounds->near, vrBounds->far,
                argbGPU, depthGPU, realSalient.probabilities, yuyvGPU, realSalient.aligned_depth);
            cudaErrorCheck(mainStream);

            cudaMemcpyAsync(argbB, argbGPU, sizeof(uint8_t) * color_W * color_H * 4, cudaMemcpyDeviceToHost, mainStream);
            cudaErrorCheck(mainStream);
            cudaMemcpyAsync(depthB, depthGPU, sizeof(float) * color_W * color_H, cudaMemcpyDeviceToHost, mainStream);
            cudaErrorCheck(mainStream);

            // before using the results coming from GPU, we need to wait the GPU stream to finish.
            cudaStreamSynchronize(mainStream);
            std::swap(argbA, argbB);
            std::swap(depthA, depthB);
        }

        delete vulkanHeadless;
        delete vrBounds;

        cudaFree(argbGPU);
        cudaErrorCheck(nullptr);
        cudaFree(depthGPU);
        cudaErrorCheck(nullptr);
        cudaFree(yuyvGPU);
        cudaErrorCheck(nullptr);
        cudaStreamDestroy(mainStream);
        cudaErrorCheck(nullptr);

        delete argbA;
        delete argbB;
        delete depthA;
        delete depthB;
        delete camera;
    }

    SaberSalient(void (*in_loadColor)(uint8_t*), void (*in_loadDepth)(float*)):
        color2tracker(cameraAttachedToTracker()), initialized(0)
    {
        // start the whole thing in another thread!
        workerThread = new std::thread(&SaberSalient::loop, this);
        initialized.wait();
    }


    ~SaberSalient()
    {
        stillRunning = false;
        workerThread->join();
        delete workerThread;
    }
};


__declspec(dllexport) int SaberSalient_init(SaberSalient** out_SaberSalient, void (*in_loadColor)(uint8_t*), void (*in_loadDepth)(float*))
{
    try
    {
        * out_SaberSalient = new SaberSalient(in_loadColor, in_loadDepth);
    }
    catch (...)
    {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

__declspec(dllexport) void SaberSalient_destroy(SaberSalient* in_SaberSalient)
{
    delete in_SaberSalient;
}

__declspec(dllexport) int SaberSalient_cameraIntrinsics(SaberSalient* in_SaberSalient, salient::CameraIntrinsics* out_intrinsics)
{
    * out_intrinsics = in_SaberSalient->camera->getColorIntrinsics();
    return EXIT_SUCCESS;
}

__declspec(dllexport) int SaberSalient_currentTransform(SaberSalient* in_SaberSalient, float* out_mat44)
{
    memcpy(out_mat44, in_SaberSalient->vrBounds->mvpMatrix, sizeof(float) * 16);
    return EXIT_SUCCESS;
}

__declspec(dllexport) int SaberSalient_currentPosition(SaberSalient* in_SaberSalient, float* out_vec3)
{
    memcpy(out_vec3, in_SaberSalient->vrBounds->world2color.translation, sizeof(float) * 3);
    return EXIT_SUCCESS;
}

__declspec(dllexport) int SaberSalient_currentRotation(SaberSalient* in_SaberSalient, float* out_mat33)
{
    memcpy(out_mat33, in_SaberSalient->vrBounds->world2color.rotation, sizeof(float) * 9);
    return EXIT_SUCCESS;
}

__declspec(dllexport) uint8_t* SaberSalient_getColorBuf(SaberSalient* in_SaberSalient)
{
    return in_SaberSalient->argbA;
}

__declspec(dllexport) float* SaberSalient_getDepthBuf(SaberSalient* in_SaberSalient)
{
    return in_SaberSalient->depthA;
}