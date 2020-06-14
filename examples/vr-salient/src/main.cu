#define BLOCK_SIZE 256
#define DOWNSCALE_MAX_FRAME_HIGHT 400

#include <fstream>
#include <chrono>
#include <thread>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2/rs_advanced_mode.hpp>
#include <librealsense2/rsutil.h>
#include <opencv2/opencv.hpp> // Include OpenCV API
#include "salient/salient.cuh"
#include "assets.hpp"
#include "vrbounds.hpp"
#include "vulkanheadless.hpp"

__global__ void draw_foreground(int downsample_ratio, int W, int H, uint8_t* out_rgb, const float* probabilities, const uint8_t* in_color, cudaTextureObject_t in_extra)
{
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //if (idx >= N)
    //    return;

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

rs2::device get_rs_device()
{
    rs2::context ctx;
    rs2::device dev;
    const int total_attempts = 1000;
    for (int i = 0; i < total_attempts; i++)
    {
        auto devices_list = ctx.query_devices();
        size_t device_count = devices_list.size();
        if (device_count > 0)
            try
            {
                dev = devices_list[i % device_count];
                std::cout << "Loaded a device on attempt " << (i + 1) << "." << std::endl;
                break;
            }
            catch (const std::exception &e)
            {
                if (i == total_attempts - 1)
                {
                    std::cout << "Could not create device - " << e.what() << "." << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
            catch (...)
            {
                if (i == total_attempts - 1)
                {
                    std::cout << "Failed to created device." << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
        else if (i == total_attempts - 1)
        {
            std::cout << "Could not find any camera devices." << std::endl;
            exit(EXIT_FAILURE);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // load json preset for high accuracy mode
    try
    {
        auto dev_adv = rs400::advanced_mode::advanced_mode(dev);
        dev_adv.load_json(std::string(assets::REALSENSE_CAMERA_SETTINGS));
    }
    catch (...)
    {
        std::cout << "Could not load depth camera settings; yet, continue with defaults." << std::endl;
    }
    return dev;
}

int main(int argc, char *argv[])
try
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    
    // Select the GPU.
    // The idea is to select a secondary, less powerfull GPU for this, so that it does not interfere with
    // the main user activity (such as playing VR).
    int devId = devCount > 1 ?  1 : 0;
    cudaSetDevice(devId);

    cudaStream_t mainStream;
    cudaStreamCreate(&mainStream);

    uint8_t *deviceUUID;
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, devId);
    printf("Selected CUDA device: %s\n", deviceProps.name);
    deviceUUID = (uint8_t *)&(deviceProps.uuid);

    using namespace cv;
    using namespace rs2;


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



    // find the camera
    pipeline pipe;
    rs2::config config;
    auto dev = get_rs_device();

    // Start the camera
    config.enable_device(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
    config.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);
    // config.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_YUYV, 30);
    config.enable_stream(RS2_STREAM_COLOR, 1920, 1080, RS2_FORMAT_YUYV, 30);
    // config.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 90)
    // config.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_YUYV, 60)

    auto selection = pipe.start(config);
    auto sensor = dev.first<rs2::depth_sensor>();

    // get dimension (to be sure)
    auto depth_stream = selection.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto color_stream = selection.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto color_W = color_stream.width();
    auto color_H = color_stream.height();
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

    auto rs2_depth_intr = depth_stream.get_intrinsics();
    auto rs2_color_intr = color_stream.get_intrinsics();
    auto rs2_color_to_depth = color_stream.get_extrinsics_to(depth_stream);

    const salient::CameraIntrinsics depthIntr = {
        rs2_depth_intr.width,
        rs2_depth_intr.height,
        rs2_depth_intr.ppx,
        rs2_depth_intr.ppy,
        rs2_depth_intr.fx,
        rs2_depth_intr.fy};

    const salient::CameraIntrinsics colorIntr = {
        rs2_color_intr.width,
        rs2_color_intr.height,
        rs2_color_intr.ppx,
        rs2_color_intr.ppy,
        rs2_color_intr.fx,
        rs2_color_intr.fy};

    auto toVertex = [](float x, float y, float z) { return VulkanHeadless::Vertex{ { x, y, z } }; };
    VrBounds<VulkanHeadless::Vertex> vrBounds(toVertex, color2tracker, colorIntr);

    // rely on the fact that we have the same representation as librealsense
    const salient::CameraExtrinsics color2depth(*reinterpret_cast<salient::CameraExtrinsics *>(&rs2_color_to_depth));

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
        mainStream, depthIntr, colorIntr, color2depth, downsample_ratio, sensor.get_depth_scale(), getFeature);

    // vulkan-to-cuda rendering
    auto vulkanHeadless = VulkanHeadless(W, H, vrBounds.vertices, vrBounds.indices, deviceUUID);

    const auto window_name = "real-salient";
    namedWindow(window_name, WINDOW_AUTOSIZE);

    double frame_cap = 90, fps = frame_cap;
    auto frame_avg_time = 1.0 / fps;
    auto frame_start_time = std::chrono::high_resolution_clock::now();
    auto frame_stop_time = frame_start_time;
    // auto frame_cap_ms = 1000 / frame_cap;
    auto ema_alpha = 0.1;
    char fpsText[20];

    // Skips some frames to allow for auto-exposure stabilization
    for (int i = 0; i < 10; i++)
        pipe.wait_for_frames();

    Mat3b foreground;
    foreground.create(Size(color_W, color_H));


    // Control all analysis parameters via trackbars
    int gmmIterations = 10; // realSalient.analysisSettings.gmmIterations;
    createTrackbar("GMM iterations", window_name, &gmmIterations, 100);
    int timeAlpha = (int)round(realSalient.analysisSettings.timeAlpha * 100);
    createTrackbar("GMM EMA α (x100)", window_name, &timeAlpha, 100);
    int imputedLabelWeight = (int)round(realSalient.analysisSettings.imputedLabelWeight * 100);
    createTrackbar("Imputed label weight (x100)", window_name, &imputedLabelWeight, 100);
    int crfIterations = 3; // realSalient.analysisSettings.crfIterations;
    createTrackbar("CRF iterations", window_name, &crfIterations, 20);
    int smoothnessWeight = (int)round(realSalient.analysisSettings.smoothnessWeight * 10);
    createTrackbar("CRF smoothness weight (x10)", window_name, &smoothnessWeight, 200);
    int smoothnessVarPos = (int)round(sqrt(realSalient.analysisSettings.smoothnessVarPos) * 10);
    createTrackbar("CRF smoothness σ (x10)", window_name, &smoothnessVarPos, 1000);
    int appearanceWeight = (int)round(realSalient.analysisSettings.appearanceWeight * 10);
    createTrackbar("CRF appearance weight (x10)", window_name, &appearanceWeight, 200);
    int appearanceVarPos = (int)round(sqrt(realSalient.analysisSettings.appearanceVarPos) * 10);
    createTrackbar("CRF appearance σ-pos (x10)", window_name, &appearanceVarPos, 1000);
    int appearanceVarCol = (int)round(sqrt(realSalient.analysisSettings.appearanceVarCol) * 10);
    createTrackbar("CRF appearance σ-col (x10)", window_name, &appearanceVarCol, 1000);
    int similarityWeight = (int)round(realSalient.analysisSettings.similarityWeight * 10);
    createTrackbar("CRF similarity weight (x10)", window_name, &similarityWeight, 200);
    int similarityVarCol = (int)round(sqrt(realSalient.analysisSettings.similarityVarCol) * 10);
    createTrackbar("CRF similarity σ (x10)", window_name, &similarityVarCol, 1000);

    for (int frame_number = 0; waitKey(1) < 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0; frame_number++)
    {
        // update camera model-view-projection matrix and the color-image-space bounding box.
        vrBounds.update();

        // update analysis parameters from the trackbar every frame (avoiding 100500 callbacks to createTrackbar fun)
        realSalient.analysisSettings.gmmIterations = gmmIterations;
        realSalient.analysisSettings.timeAlpha = (float)timeAlpha * 0.01f;
        realSalient.analysisSettings.imputedLabelWeight = (float)imputedLabelWeight * 0.01f;
        realSalient.analysisSettings.crfIterations = crfIterations;
        realSalient.analysisSettings.smoothnessWeight = (float)smoothnessWeight * 0.1f;
        realSalient.analysisSettings.smoothnessVarPos = max(0.01f, (float)(smoothnessVarPos * smoothnessVarPos) * 0.01f);
        realSalient.analysisSettings.appearanceWeight = (float)appearanceWeight * 0.1f;
        realSalient.analysisSettings.appearanceVarPos = max(0.01f, (float)(appearanceVarPos * appearanceVarPos) * 0.01f);
        realSalient.analysisSettings.appearanceVarCol = max(0.01f, (float)(appearanceVarCol * appearanceVarCol) * 0.01f);
        realSalient.analysisSettings.similarityWeight = (float)similarityWeight * 0.1f;
        realSalient.analysisSettings.similarityVarCol = max(0.01f, (float)(similarityVarCol * similarityVarCol) * 0.01f);

        frame_stop_time = std::chrono::high_resolution_clock::now();
        auto frame_time = std::chrono::duration_cast<std::chrono::microseconds>(frame_stop_time - frame_start_time);
        frame_avg_time = frame_avg_time * (1 - ema_alpha) + frame_time.count() * ema_alpha / 1000000;
        if (frame_number % std::max(1, (int)std::round(fps / 4.0)) == 0)
            fps = 1 / frame_avg_time;
        frame_start_time = frame_stop_time;


        vulkanHeadless.render(vrBounds.mvpMatrix, mainStream);
        frameset data = pipe.wait_for_frames();

        // copy the color frame, so that getFeature gets the actual color data.
        cudaMemcpyAsync(yuyvGPU, data.get_color_frame().get_data(), sizeof(uint8_t) * color_N * 2, cudaMemcpyHostToDevice, mainStream);
        cudaErrorCheck(mainStream);

        // load frames to gpu and preprocess
        realSalient.processFrames(
            (const uint16_t *)data.get_depth_frame().get_data(),
            vrBounds.trackerBounds,
            &(vulkanHeadless.cudaTexture));

        // draw_foreground<<<salient::distribute(color_N, BLOCK_SIZE), BLOCK_SIZE, 0, mainStream>>>(color_N, rgbGPU, realSalient.probabilities, yuyvGPU);
        dim3 bs(32, 32, 1);
        draw_foreground<<<salient::distribute(dim3(color_W, color_H, 1), bs), bs, 0, mainStream>>>(downsample_ratio, color_W, color_H, rgbGPU, realSalient.probabilities, yuyvGPU, vulkanHeadless.cudaTexture);
        

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
        cv::line(foreground, vrBounds.getTrackedPoint(toPoint, 0), vrBounds.getTrackedPoint(toPoint, 1), cv::Scalar(0, 0, 255), 2);
        cv::line(foreground, vrBounds.getTrackedPoint(toPoint, 0), vrBounds.getTrackedPoint(toPoint, 2), cv::Scalar(0, 0, 255), 2);

        imshow(window_name, foreground);

        // update bounds given our refined labeling, assuming they don't change too much.
        // NB: in reality, it's much better to infer these bounds from the VR tracker positions.
        // foregroundBounds = realSalient.postprocessInferBounds();
    }

    cudaStreamDestroy(mainStream);
    cudaErrorCheck(nullptr);
    cudaFree(rgbGPU);
    cudaErrorCheck(nullptr);
    cudaFree(yuyvGPU);
    cudaErrorCheck(nullptr);

    return EXIT_SUCCESS;
}
catch (const rs2::error &e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception &e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
