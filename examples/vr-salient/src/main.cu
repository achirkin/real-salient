#define BLOCK_SIZE 256
#define DOWNSCALE_MAX_FRAME_HIGHT 400

#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp> // Include OpenCV API
#include "salient/salient.cuh"
#include "vrbounds.hpp"
#include "vulkanheadless.hpp"
#include "cameraD415.hpp"

__global__ void draw_foreground(int N, uint8_t *out_rgb, const float *probabilities, const uint8_t *in_color)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    if (probabilities[idx] > 0.5f)
    {
        float Y = ((float)in_color[idx * 2]) - 16.0f;
        float Cb = ((float)in_color[(idx - idx % 2) * 2 + 1]) - 128.0f;
        float Cr = ((float)in_color[(idx - idx % 2) * 2 + 3]) - 128.0f;

        out_rgb[idx * 3 + 0] = (uint8_t)__float2int_rd(max(0.0f, min(255.0f, 1.163999557f * Y + 2.017999649f * Cb)));
        out_rgb[idx * 3 + 1] = (uint8_t)__float2int_rd(max(0.0f, min(255.0f, 1.163999557f * Y - 0.812999725f * Cr - 0.390999794f * Cb)));
        out_rgb[idx * 3 + 2] = (uint8_t)__float2int_rd(max(0.0f, min(255.0f, 1.163999557f * Y + 1.595999718f * Cr)));
    }
    else
    {
        out_rgb[idx * 3 + 0] = 0;
        out_rgb[idx * 3 + 1] = 150;
        out_rgb[idx * 3 + 2] = 0;
    }
}

__global__ void draw_foreground(int downsample_ratio, int W, int H, uint8_t *out_rgb, const float *probabilities, const uint8_t *in_color, cudaTextureObject_t in_extra)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= W || j >= H)
        return;

    int idx = i + j * W;

    if (probabilities[idx] > 0.5f)
    {
        float Y = ((float)in_color[idx * 2]) - 16.0f;
        float Cb = ((float)in_color[(idx - idx % 2) * 2 + 1]) - 128.0f;
        float Cr = ((float)in_color[(idx - idx % 2) * 2 + 3]) - 128.0f;

        out_rgb[idx * 3 + 0] = (uint8_t)__float2int_rd(max(0.0f, min(255.0f, 1.163999557f * Y + 2.017999649f * Cb)));
        out_rgb[idx * 3 + 1] = (uint8_t)__float2int_rd(max(0.0f, min(255.0f, 1.163999557f * Y - 0.812999725f * Cr - 0.390999794f * Cb)));
        out_rgb[idx * 3 + 2] = (uint8_t)__float2int_rd(max(0.0f, min(255.0f, 1.163999557f * Y + 1.595999718f * Cr)));
    }
    else
    {
        uint8_t extra = (uint8_t)max(0.0f, min(255.0f, 255.0f - tex2D<float>(in_extra, (float)i / (float)downsample_ratio, (float)j / (float)downsample_ratio) * 100.0f));
        out_rgb[idx * 3 + 0] = extra;
        out_rgb[idx * 3 + 1] = extra;
        out_rgb[idx * 3 + 2] = extra;
    }
}

int main(int argc, char *argv[])
{
    int devCount;
    cudaGetDeviceCount(&devCount);

    // Select the GPU.
    // The idea is to select a secondary, less powerfull GPU for this, so that it does not interfere with
    // the main user activity (such as playing VR).
    int devId = devCount > 1 ? 1 : 0;
    cudaSetDevice(devId);

    cudaStream_t mainStream;
    cudaStreamCreate(&mainStream);

    uint8_t *deviceUUID;
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, devId);
    printf("Selected CUDA device: %s\n", deviceProps.name);
    deviceUUID = (uint8_t *)&(deviceProps.uuid);

    // I screw the camera in on top of the vive controller, thus fixing one of the axes solid.
    // The two remaining axes are defined by one angle - the position where the camera is screwed in tight.
    // I determine this angle by (1) guessing the approximate value (2) tuning it by drawing controller and camera positions on screen.
    const float phi = 2.02f;
    const float sph = sin(phi), cph = cos(phi);
    const salient::CameraExtrinsics color2tracker = {
        {cph, sph, 0,
         0, 0, -1,
         -sph, cph, 0},
        {0, 0, 0.02f}};

    camera::Camera *camera = new camera::IntelD415Camera();

    auto color_W = camera->getColorIntrinsics().width;
    auto color_H = camera->getColorIntrinsics().height;
    auto color_N(color_W * color_H);
    auto W = color_W;
    auto H = color_H;
    auto downsample_ratio = 1;
    while (H > DOWNSCALE_MAX_FRAME_HIGHT)
    {
        downsample_ratio <<= 1;
        H >>= 1;
        W >>= 1;
    }

    auto toVertex = [](float x, float y, float z) { return VulkanHeadless::Vertex{{x, y, z}}; };
    VrBounds<VulkanHeadless::Vertex> vrBounds(toVertex, color2tracker, camera->getColorIntrinsics());

    // original image from the color camera
    uint8_t *yuyvGPU = nullptr;
    cudaMalloc((void **)&yuyvGPU, sizeof(uint8_t) * color_W * color_H * 2);
    cudaErrorCheck(nullptr);

    // transformed RGB image with foreground mask applied
    uint8_t *rgbGPU = nullptr;
    cudaMalloc((void **)&rgbGPU, sizeof(uint8_t) * color_W * color_H * 3);
    cudaErrorCheck(nullptr);

    // how to access color data at every pixel position.
    auto getFeature = [yuyvGPU, color_W] __device__(const int i, const int j, float *out_feature) {
        const int base_off = i + j * color_W;
        out_feature[0] = (float)yuyvGPU[base_off * 2];
        out_feature[1] = (float)yuyvGPU[(base_off - i % 2) * 2 + 1];
        out_feature[2] = (float)yuyvGPU[(base_off - i % 2) * 2 + 3];
    };
    salient::RealSalient<3, 7, decltype(getFeature)> realSalient(
        mainStream,
        camera->getDepthIntrinsics(),
        camera->getColorIntrinsics(),
        camera->getColorToDepthTransform(),
        downsample_ratio,
        camera->getDepthScale(), getFeature);

    // This initial bound is used to find the salient object when no VR tracking is available.
    salient::SceneBounds foregroundBounds{
        (int)round(0.1f * color_W) /* int left in pixels */,
        (int)round(0.1f * color_H) /* int top in pixels */,
        (int)round(0.9f * color_W) /* int right in pixels */,
        color_H /* int bottom in pixels */,
        0.4f /* float near distance meters */,
        2.0f /* float far distance meters */
    };

    // vulkan-to-cuda rendering
    auto vulkanHeadless = VulkanHeadless(W, H, vrBounds.vertices, vrBounds.indices, deviceUUID);

    const auto window_name = "real-salient";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    double frame_cap = 90, fps = frame_cap;
    auto frame_avg_time = 1.0 / fps;
    auto frame_start_time = std::chrono::high_resolution_clock::now();
    auto frame_stop_time = frame_start_time;
    // auto frame_cap_ms = 1000 / frame_cap;
    auto ema_alpha = 0.1;
    char fpsText[20];

    cv::Mat3b foreground;
    foreground.create(cv::Size(color_W, color_H));

#ifdef SLIDERS
    // Control all analysis parameters via trackbars
    int timeAlpha = (int)round(realSalient.analysisSettings.timeAlpha * 100);
    cv::createTrackbar("Frame α (x100)", window_name, &timeAlpha, 100);
    int gmmIterations = realSalient.analysisSettings.gmmIterations;
    cv::createTrackbar("GMM iterations", window_name, &gmmIterations, 100);
    int gmmAlpha = (int)round(realSalient.analysisSettings.gmmAlpha * 100);
    cv::createTrackbar("GMM EMA α (x100)", window_name, &gmmAlpha, 100);
    int imputedLabelWeight = (int)round(realSalient.analysisSettings.imputedLabelWeight * 100);
    cv::createTrackbar("Imputed label weight (x100)", window_name, &imputedLabelWeight, 100);
    int crfIterations = realSalient.analysisSettings.crfIterations;
    cv::createTrackbar("CRF iterations", window_name, &crfIterations, 20);
    int smoothnessWeight = (int)round(realSalient.analysisSettings.smoothnessWeight * 10);
    cv::createTrackbar("CRF smoothness weight (x10)", window_name, &smoothnessWeight, 200);
    int smoothnessVarPos = (int)round(sqrt(realSalient.analysisSettings.smoothnessVarPos) * 10);
    cv::createTrackbar("CRF smoothness σ (x10)", window_name, &smoothnessVarPos, 1000);
    int appearanceWeight = (int)round(realSalient.analysisSettings.appearanceWeight * 10);
    cv::createTrackbar("CRF appearance weight (x10)", window_name, &appearanceWeight, 200);
    int appearanceVarPos = (int)round(sqrt(realSalient.analysisSettings.appearanceVarPos) * 10);
    cv::createTrackbar("CRF appearance σ-pos (x10)", window_name, &appearanceVarPos, 1000);
    int appearanceVarCol = (int)round(sqrt(realSalient.analysisSettings.appearanceVarCol) * 10);
    cv::createTrackbar("CRF appearance σ-col (x10)", window_name, &appearanceVarCol, 1000);
    int similarityWeight = (int)round(realSalient.analysisSettings.similarityWeight * 10);
    cv::createTrackbar("CRF similarity weight (x10)", window_name, &similarityWeight, 200);
    int similarityVarCol = (int)round(sqrt(realSalient.analysisSettings.similarityVarCol) * 10);
    cv::createTrackbar("CRF similarity σ (x10)", window_name, &similarityVarCol, 1000);
#endif

    for (int frame_number = 0; cv::waitKey(1) < 0 && cv::getWindowProperty(window_name, cv::WND_PROP_AUTOSIZE) >= 0; frame_number++)
    {
        // update camera model-view-projection matrix and the color-image-space bounding box.
        vrBounds.update();

#ifdef SLIDERS
        // update analysis parameters from the trackbar every frame (avoiding 100500 callbacks to createTrackbar fun)
        realSalient.analysisSettings.timeAlpha = (float)timeAlpha * 0.01f;
        realSalient.analysisSettings.gmmIterations = gmmIterations;
        realSalient.analysisSettings.gmmAlpha = (float)gmmAlpha * 0.01f;
        realSalient.analysisSettings.imputedLabelWeight = (float)imputedLabelWeight * 0.01f;
        realSalient.analysisSettings.crfIterations = crfIterations;
        realSalient.analysisSettings.smoothnessWeight = (float)smoothnessWeight * 0.1f;
        realSalient.analysisSettings.smoothnessVarPos = max(0.01f, (float)(smoothnessVarPos * smoothnessVarPos) * 0.01f);
        realSalient.analysisSettings.appearanceWeight = (float)appearanceWeight * 0.1f;
        realSalient.analysisSettings.appearanceVarPos = max(0.01f, (float)(appearanceVarPos * appearanceVarPos) * 0.01f);
        realSalient.analysisSettings.appearanceVarCol = max(0.01f, (float)(appearanceVarCol * appearanceVarCol) * 0.01f);
        realSalient.analysisSettings.similarityWeight = (float)similarityWeight * 0.1f;
        realSalient.analysisSettings.similarityVarCol = max(0.01f, (float)(similarityVarCol * similarityVarCol) * 0.01f);
#endif

        frame_stop_time = std::chrono::high_resolution_clock::now();
        auto frame_time = std::chrono::duration_cast<std::chrono::microseconds>(frame_stop_time - frame_start_time);
        frame_avg_time = frame_avg_time * (1 - ema_alpha) + frame_time.count() * ema_alpha / 1000000;
        if (frame_number % std::max(1, (int)std::round(fps / 4.0)) == 0)
            fps = 1 / frame_avg_time;
        frame_start_time = frame_stop_time;

        vulkanHeadless.render(vrBounds.mvpMatrix, mainStream);
        auto frames = camera->waitForFrames();

        // copy the color frame, so that getFeature gets the actual color data.
        cudaMemcpyAsync(yuyvGPU, frames->getColor(), sizeof(uint8_t) * color_N * 2, cudaMemcpyHostToDevice, mainStream);
        cudaErrorCheck(mainStream);

        // load frames to gpu and preprocess
        realSalient.processFrames(
            (const uint16_t *)frames->getDepth(),
            vrBounds.validDevCount > 0 ? vrBounds.trackerBounds : foregroundBounds,
            vulkanHeadless.getCudaTexture());

        if (vulkanHeadless.isValid)
        {
            dim3 bs(32, 32, 1);
            draw_foreground<<<salient::distribute(dim3(color_W, color_H, 1), bs), bs, 0, mainStream>>>(downsample_ratio, color_W, color_H, rgbGPU, realSalient.probabilities, yuyvGPU, *vulkanHeadless.getCudaTexture());
        }
        else
        {
            draw_foreground<<<salient::distribute(color_N, BLOCK_SIZE), BLOCK_SIZE, 0, mainStream>>>(color_N, rgbGPU, realSalient.probabilities, yuyvGPU);
        }

        cudaErrorCheck(mainStream);

        cudaMemcpyAsync(foreground.data, rgbGPU, sizeof(uint8_t) * color_W * color_H * 3, cudaMemcpyDeviceToHost, mainStream);
        cudaErrorCheck(mainStream);

        // before using the results coming from GPU, we need to wait the GPU stream to finish.
        cudaStreamSynchronize(mainStream);
        // Show FPS
        sprintf(fpsText, "FPS: %.1f", fps);
        cv::putText(foreground, fpsText, cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

        // show VR headset
        auto toPoint = [](int x, int y) { return cv::Point(x, y); };
        if (vrBounds.validDevCount == 1)
        {
            cv::circle(foreground, vrBounds.getTrackedPoint(toPoint, 0), 3, cv::Scalar(0, 0, 255), 2);
        }
        else if (vrBounds.validDevCount > 1)
        {
            auto head = vrBounds.getTrackedPoint(toPoint, 0);
            for (int k = vrBounds.validDevCount - 1; k > 0; k--)
                cv::line(foreground, head, vrBounds.getTrackedPoint(toPoint, k), cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow(window_name, foreground);

        // update bounds given our refined labeling, assuming they don't change too much.
        // NB: in reality, it's much better to infer these bounds from the VR tracker positions.
        if (vrBounds.validDevCount <= 0) // if cannot track through VR.
            foregroundBounds = realSalient.postprocessInferBounds();
    }

    cudaStreamDestroy(mainStream);
    cudaErrorCheck(nullptr);
    cudaFree(rgbGPU);
    cudaErrorCheck(nullptr);
    cudaFree(yuyvGPU);
    cudaErrorCheck(nullptr);
    delete camera;

    return EXIT_SUCCESS;
}