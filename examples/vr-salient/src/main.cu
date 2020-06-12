#define BLOCK_SIZE 256
#define DOWNSCALE_MAX_FRAME_HIGHT 400

#include <fstream>
#include <chrono>
#include <thread>
#include <openvr.h>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2/rs_advanced_mode.hpp>
#include <librealsense2/rsutil.h>
#include <opencv2/opencv.hpp> // Include OpenCV API
#include "salient/salient.cuh"
#include "assets.hpp"
#include "vulkanheadless.hpp"

__global__ void draw_foreground(int N, uint8_t* out_rgb, const float* probabilities, const uint8_t* in_color)
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

rs2::device get_rs_device()
{
    rs2::context ctx;
    rs2::device dev;
    const int total_attempts = 1000;
    for (int i = 0; i < total_attempts; i++)
    {
        auto devices_list = ctx.query_devices();
        size_t device_count = devices_list.size();
        if (device_count > 0) try
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

salient::SceneBounds boundPoints(const int w, const int h, const float d, const int n, const float* depths, const int* xys, const salient::CameraIntrinsics camIntr)
{
    float marginTop = 0.2f;
    float marginSide = 0.5f;
    float marginBottom = 2.0f; // don't remove legs!
    float cur_depth;
    salient::SceneBounds result;
    result.left = w;
    result.top = h;
    result.right = 0;
    result.bottom = 0;
    result.near = d;
    result.far = 0.01f;
    for (int i = 0; i < n; i++)
    {
        cur_depth = depths[i];
        if (cur_depth > 0.01f && cur_depth < d)
        {
            result.near = min(result.near, max(0.01f, cur_depth - marginSide));
            result.far = max(result.far, min(d, cur_depth + marginSide));
            result.left = min(result.left, max(0, (int)round(xys[2 * i] - marginSide * camIntr.fx / cur_depth)));
            result.top = min(result.top, max(0, (int)round(xys[2 * i + 1] - marginTop * camIntr.fy / cur_depth)));
            result.right = max(result.right, min(w, (int)round(xys[2 * i] + marginSide * camIntr.fx / cur_depth)));
            result.bottom = max(result.bottom, min(h, (int)round(xys[2 * i + 1] + marginBottom * camIntr.fy / cur_depth)));
        }
    }
    return result;
}

int main(int argc, char *argv[])
try
{
    // Select the GPU.
    // The idea is to select a secondary, less powerfull GPU for this, so that it does not interfere with
    // the main user activity (such as playing VR).
    cudaSetDevice(0);

    cudaStream_t mainStream;
    cudaStreamCreate(&mainStream);

    using namespace cv;
    using namespace rs2;

    // Loading the SteamVR Runtime
    vr::EVRInitError eError = vr::VRInitError_None;
    vr::IVRSystem * m_pHMD = vr::VR_Init(&eError, vr::VRApplication_Background); // or VRApplication_Scene
    vr::TrackedDevicePose_t devicePositions[vr::k_unMaxTrackedDeviceCount];

    if (eError != vr::VRInitError_None)
    {
        m_pHMD = nullptr;
        std::cout << "Unable to init VR runtime: " << vr::VR_GetVRInitErrorAsEnglishDescription(eError) << std::endl;
        exit(EXIT_FAILURE);
    }

    int hmdIdx = -1;
    int controller1Idx = -1;
    int controller2Idx = -1;
    int trackerIdx = -1;
    int deviceCount = 0;

    for (int deviceIdx = 0; deviceIdx < vr::k_unMaxTrackedDeviceCount; deviceIdx++)
    {
        if (m_pHMD->IsTrackedDeviceConnected(deviceIdx))
        {
            std::cout << deviceIdx << ": ";
            switch (m_pHMD->GetTrackedDeviceClass(deviceIdx))
            {
            case vr::TrackedDeviceClass_Controller:
                if (controller1Idx < 0)
                    controller1Idx = deviceIdx;
                else if (controller2Idx < 0)
                    controller2Idx = deviceIdx;
                else if (trackerIdx < 0)
                    trackerIdx = deviceIdx;
                std::cout << "controller";
                break;
            case vr::TrackedDeviceClass_HMD:
                hmdIdx = deviceIdx;
                std::cout << "HMD";
                break;
            case vr::TrackedDeviceClass_Invalid:
                std::cout << "Invalid";
                break;
            case vr::TrackedDeviceClass_GenericTracker:
                trackerIdx = deviceIdx;
                std::cout << "Generic tracker";
                break;
            case vr::TrackedDeviceClass_TrackingReference:
                std::cout << "Tracking reference";
                break;
            default:
                std::cout << "default";
                break;
            }
            std::cout << std::endl;
            deviceCount = deviceIdx + 1;
            if (hmdIdx >= 0 && controller1Idx >= 0 && controller2Idx >= 0 && trackerIdx >= 0)
                break;
        }
    }
    std::cout << "Device ids " << hmdIdx << " " << controller1Idx << " " << controller2Idx << " " << trackerIdx << " " << deviceCount << std::endl;
    
    float hmdPos[3], co1Pos[3], co2Pos[3];
    auto trackerPos(devicePositions[trackerIdx].mDeviceToAbsoluteTracking.m);


    auto vrsetup = vr::VRChaperoneSetup();
    uint32_t chaperoneQuadsCount;
    vrsetup->GetLiveCollisionBoundsInfo(NULL, &chaperoneQuadsCount);
    vr::HmdQuad_t * chaperoneBounds = new vr::HmdQuad_t[chaperoneQuadsCount];
    vrsetup->GetLiveCollisionBoundsInfo(chaperoneBounds, &chaperoneQuadsCount);

    std::vector<VulkanHeadless::Vertex> vertices;
    std::vector<uint32_t> indices;
    //std::cout << "Chaperone quads: " << chaperoneQuadsCount << std::endl;
    float smins[3], smaxs[3];
    for (int k = 0; k < 3; k++)
    {
        smins[k] = std::numeric_limits<float>::infinity();
        smaxs[k] = -smins[k];
    }
    for (int i = 0; i < chaperoneQuadsCount; i++)
    {
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 3; k++)
            {
                smins[k] = min(smins[k], chaperoneBounds[i].vCorners[j].v[k]);
                smaxs[k] = max(smaxs[k], chaperoneBounds[i].vCorners[j].v[k]);
            }
        vertices.push_back({
                { chaperoneBounds[i].vCorners[0].v[0],
                  chaperoneBounds[i].vCorners[0].v[1],
                  chaperoneBounds[i].vCorners[0].v[2]}});
        vertices.push_back({
                { chaperoneBounds[i].vCorners[1].v[0],
                  chaperoneBounds[i].vCorners[1].v[1],
                  chaperoneBounds[i].vCorners[1].v[2]}});
        vertices.push_back({
                { chaperoneBounds[i].vCorners[2].v[0],
                  chaperoneBounds[i].vCorners[2].v[1],
                  chaperoneBounds[i].vCorners[2].v[2]} });
        vertices.push_back({
                { chaperoneBounds[i].vCorners[3].v[0],
                  chaperoneBounds[i].vCorners[3].v[1],
                  chaperoneBounds[i].vCorners[3].v[2]} });
        indices.push_back(i * 4);
        indices.push_back(i * 4 + 2);
        indices.push_back(i * 4 + 1);
        indices.push_back(i * 4 + 3);
        indices.push_back(i * 4 + 2);
        indices.push_back(i * 4);
    }
    vertices.push_back({ { smins[0], smins[1], smins[2] } });
    vertices.push_back({ { smins[0], smins[1], smaxs[2] } });
    vertices.push_back({ { smaxs[0], smins[1], smaxs[2] } });
    vertices.push_back({ { smaxs[0], smins[1], smins[2] } });
    vertices.push_back({ { smins[0], smaxs[1], smins[2] } });
    vertices.push_back({ { smins[0], smaxs[1], smaxs[2] } });
    vertices.push_back({ { smaxs[0], smaxs[1], smaxs[2] } });
    vertices.push_back({ { smaxs[0], smaxs[1], smins[2] } });
    indices.push_back(chaperoneQuadsCount * 4);
    indices.push_back(chaperoneQuadsCount * 4 + 2);
    indices.push_back(chaperoneQuadsCount * 4 + 1);
    indices.push_back(chaperoneQuadsCount * 4 + 3);
    indices.push_back(chaperoneQuadsCount * 4 + 2);
    indices.push_back(chaperoneQuadsCount * 4);
    indices.push_back(chaperoneQuadsCount * 4 + 4);
    indices.push_back(chaperoneQuadsCount * 4 + 5);
    indices.push_back(chaperoneQuadsCount * 4 + 6);
    indices.push_back(chaperoneQuadsCount * 4 + 7);
    indices.push_back(chaperoneQuadsCount * 4 + 4);
    indices.push_back(chaperoneQuadsCount * 4 + 6);


    // I screw the camera in on top of the vive controller, thus fixing one of the axes solid.
    // The two remaining axes are defined by one angle - the position where the camera is screwed in tight.
    // I determine this angle by (1) guessing the approximate value (2) tuning it by drawing controller and camera positions on screen.
    const float phi = 2.02f;
    const float sph = sin(phi), cph = cos(phi);
    const salient::CameraExtrinsics color2tracker = {
        { cph, sph, 0,
          0, 0, -1,
          -sph, cph, 0 },
        { 0, 0, 0.02f }
    };
    salient::CameraExtrinsics color2world;

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
    auto vulkanHeadless = VulkanHeadless(W, H, vertices, indices);

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

    // initialize the hard bounds where to look for the salient object in the color camera space
    salient::SceneBounds foregroundBounds = {
        0 /* int left in pixels */,
        0 /* int top in pixels */,
        color_W /* int right in pixels */,
        color_H /* int bottom in pixels */,
        0.1f /* float near distance meters */,
        1.5f /* float far distance meters */
    };


    float mvMatrix[16], pMatrix[16], mvpMatrix[16];
    memset(&mvMatrix, 0, sizeof(mvMatrix));
    memset(&mvpMatrix, 0, sizeof(mvpMatrix));
    memset(&pMatrix, 0, sizeof(pMatrix));
    float veryFar = 10.0f;
    float veryNear = 0.1f;
    float veryvery = veryFar + veryNear;
    pMatrix[0] = 2 * colorIntr.fx / colorIntr.width;
    pMatrix[5] = 2 * colorIntr.fy / colorIntr.height;
    pMatrix[10] = (veryFar + veryNear) / (veryFar - veryNear);
    pMatrix[11] = 1;
    pMatrix[14] = 2 * veryFar * veryNear / (veryNear - veryFar);
    for (int i = 0; i < 4; i++)
    {
        printf("\t%.3f %.3f %.3f %.3f\n", pMatrix[i + 0], pMatrix[i + 4], pMatrix[i + 8], pMatrix[i + 12]);
    }

    // Control all analysis parameters via trackbars
    int gmmIterations = 5; // realSalient.analysisSettings.gmmIterations;
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
        // load position matrices.
        m_pHMD->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0.0f, devicePositions, deviceCount);

        hmdPos[0] = devicePositions[hmdIdx].mDeviceToAbsoluteTracking.m[0][3];
        hmdPos[1] = devicePositions[hmdIdx].mDeviceToAbsoluteTracking.m[1][3];
        hmdPos[2] = devicePositions[hmdIdx].mDeviceToAbsoluteTracking.m[2][3];
        co1Pos[0] = devicePositions[controller1Idx].mDeviceToAbsoluteTracking.m[0][3];
        co1Pos[1] = devicePositions[controller1Idx].mDeviceToAbsoluteTracking.m[1][3];
        co1Pos[2] = devicePositions[controller1Idx].mDeviceToAbsoluteTracking.m[2][3];
        co2Pos[0] = devicePositions[controller2Idx].mDeviceToAbsoluteTracking.m[0][3];
        co2Pos[1] = devicePositions[controller2Idx].mDeviceToAbsoluteTracking.m[1][3];
        co2Pos[2] = devicePositions[controller2Idx].mDeviceToAbsoluteTracking.m[2][3];
        
        /*printf("HMD: %.3f %.3f %.3f\n", hmdPos[0], hmdPos[1], hmdPos[2]);
        printf("Controller 1: %.3f %.3f %.3f\n", co1Pos[0], co1Pos[1], co1Pos[2]);
        printf("Controller 2: %.3f %.3f %.3f\n", co2Pos[0], co2Pos[1], co2Pos[2]);
        printf("Tracker:\n  %.3f %.3f %.3f %.3f\n  %.3f %.3f %.3f %.3f\n  %.3f %.3f %.3f %.3f\n",
            trackerPos[0][0], trackerPos[0][1], trackerPos[0][2], trackerPos[0][3],
            trackerPos[1][0], trackerPos[1][1], trackerPos[1][2], trackerPos[1][3],
            trackerPos[2][0], trackerPos[2][1], trackerPos[2][2], trackerPos[2][3]);*/

        for (int i = 0; i < 3; i++)
        {
            color2world.translation[i] = trackerPos[i][3];
            for (int j = 0; j < 3; j++)
            {
                color2world.translation[i] += trackerPos[i][j] * color2tracker.translation[j];
                color2world.rotation[i + j * 3] = 0;
                for (int k = 0; k < 3; k++)
                    color2world.rotation[i + j * 3] += trackerPos[i][k] * color2tracker.rotation[k + j * 3];
            }
        }
        /*printf("Camera:\n  %.3f %.3f %.3f %.3f\n  %.3f %.3f %.3f %.3f\n  %.3f %.3f %.3f %.3f\n",
            color2world.rotation[0], color2world.rotation[3], color2world.rotation[6], color2world.translation[0],
            color2world.rotation[1], color2world.rotation[4], color2world.rotation[7], color2world.translation[1],
            color2world.rotation[2], color2world.rotation[5], color2world.rotation[8], color2world.translation[2]);*/

        float hmdPosC[3], co1PosC[3], co2PosC[3];
        float invt[3];
        for (int i = 0; i < 3; i++)
        {
            hmdPosC[i] = 0;
            co1PosC[i] = 0;
            co2PosC[i] = 0;
            invt[i] = 0;
            for (int j = 0; j < 3; j++)
            {
                hmdPosC[i] += (hmdPos[j] - color2world.translation[j]) * color2world.rotation[i * 3 + j]; // inverse of orthogonal is transpose
                co1PosC[i] += (co1Pos[j] - color2world.translation[j]) * color2world.rotation[i * 3 + j];
                co2PosC[i] += (co2Pos[j] - color2world.translation[j]) * color2world.rotation[i * 3 + j];
                invt[i] += (- color2world.translation[j]) * color2world.rotation[i * 3 + j];
            }
        }

        mvMatrix[0] = color2world.rotation[0];
        mvMatrix[1] = color2world.rotation[3];
        mvMatrix[2] = color2world.rotation[6];
        mvMatrix[3] = 0;
        mvMatrix[4] = color2world.rotation[1];
        mvMatrix[5] = color2world.rotation[4];
        mvMatrix[6] = color2world.rotation[7];
        mvMatrix[7] = 0;
        mvMatrix[8] = color2world.rotation[2];
        mvMatrix[9] = color2world.rotation[5];
        mvMatrix[10] = color2world.rotation[8];
        mvMatrix[11] = 0;
        mvMatrix[12] = invt[0];
        mvMatrix[13] = invt[1];
        mvMatrix[14] = invt[2];
        mvMatrix[15] = 1;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                mvpMatrix[i + j * 4] = 0;
                for (int k = 0; k < 4; k++)
                {
                    mvpMatrix[i + j * 4] += pMatrix[i + k * 4] * mvMatrix[k + j * 4];
                }
            }
        }



        // printf("Camera space - HMD: %.3f %.3f %.3f\n", hmdPosC[0], hmdPosC[1], hmdPosC[2]);
        // printf("Camera space - Controller 1: %.3f %.3f %.3f\n", co1PosC[0], co1PosC[1], co1PosC[2]);
        // printf("Camera space - Controller 2: %.3f %.3f %.3f\n", co2PosC[0], co2PosC[1], co2PosC[2]);


        int hmdPosS[2], co1PosS[2], co2PosS[2];
        hmdPosS[0] = (int)round(hmdPosC[0] / hmdPosC[2] * colorIntr.fx + colorIntr.ppx);
        hmdPosS[1] = (int)round(hmdPosC[1] / hmdPosC[2] * colorIntr.fy + colorIntr.ppy);
        co1PosS[0] = (int)round(co1PosC[0] / co1PosC[2] * colorIntr.fx + colorIntr.ppx);
        co1PosS[1] = (int)round(co1PosC[1] / co1PosC[2] * colorIntr.fy + colorIntr.ppy);
        co2PosS[0] = (int)round(co2PosC[0] / co2PosC[2] * colorIntr.fx + colorIntr.ppx);
        co2PosS[1] = (int)round(co2PosC[1] / co2PosC[2] * colorIntr.fy + colorIntr.ppy);
        // printf("Camera screen space - HMD: %d %d %.3f\n", hmdPosS[0], hmdPosS[1], hmdPosC[2]);
        // printf("Camera screen space - Controller 1: %d %d %.3f\n", co1PosS[0], co1PosS[1], co1PosC[2]);
        // printf("Camera screen space - Controller 2: %d %d %.3f\n", co2PosS[0], co2PosS[1], co2PosC[2]);

        float depths[3];
        int xys[6];
        depths[0] = hmdPosC[2];
        depths[1] = co2PosC[2];
        depths[2] = co2PosC[2];
        xys[0] = hmdPosS[0];
        xys[1] = hmdPosS[1];
        xys[2] = co1PosS[0];
        xys[3] = co2PosS[1];
        xys[4] = co2PosS[0];
        xys[5] = co2PosS[1];

        foregroundBounds = boundPoints(color_W, color_H, 10.0f, 3, depths, xys, colorIntr);


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

        vulkanHeadless.render(mvpMatrix);
        frameset data = pipe.wait_for_frames();

        // copy the color frame, so that getFeature gets the actual color data.
        cudaMemcpyAsync(yuyvGPU, data.get_color_frame().get_data(), sizeof(uint8_t) * color_N * 2, cudaMemcpyHostToDevice, mainStream);
        cudaErrorCheck(mainStream);

        // load frames to gpu and preprocess
        realSalient.processFrames(
            (const uint16_t *)data.get_depth_frame().get_data(),
            foregroundBounds,
            &(vulkanHeadless.cudaTexture));

        draw_foreground<<<salient::distribute(color_N, BLOCK_SIZE), BLOCK_SIZE, 0, mainStream>>>(color_N, rgbGPU, realSalient.probabilities, yuyvGPU);
        
        cudaErrorCheck(mainStream);

        cudaMemcpyAsync(foreground.data, rgbGPU, sizeof(uint8_t) * color_W * color_H * 3, cudaMemcpyDeviceToHost, mainStream);
        cudaErrorCheck(mainStream);

        // before using the results coming from GPU, we need to wait the GPU stream to finish.
        cudaStreamSynchronize(mainStream);
        // Show FPS
        sprintf(fpsText, "FPS: %.1f", fps);
        cv::putText(foreground, fpsText, cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);


        char posText[50];
        sprintf(posText, "          HMD: %.2f %.2f %.2f", hmdPos[0], hmdPos[1], hmdPos[2]);
        cv::putText(foreground, posText, cv::Point(40, 70), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 200, 200), 1, cv::LINE_AA);
        sprintf(posText, "        Con 1: %.2f %.2f %.2f", co1Pos[0], co1Pos[1], co1Pos[2]);
        cv::putText(foreground, posText, cv::Point(40, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 200, 200), 1, cv::LINE_AA);
        sprintf(posText, "        Con 2: %.2f %.2f %.2f", co2Pos[0], co2Pos[1], co2Pos[2]);
        cv::putText(foreground, posText, cv::Point(40, 110), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 200, 200), 1, cv::LINE_AA);
        sprintf(posText, "      Tracker: %.2f %.2f %.2f", trackerPos[0][3], trackerPos[1][3], trackerPos[2][3]);
        cv::putText(foreground, posText, cv::Point(40, 130), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 200, 200), 1, cv::LINE_AA);
        sprintf(posText, "  HMD/tracker: %.2f %.2f %.2f", hmdPosC[0], hmdPosC[1], hmdPosC[2]);
        cv::putText(foreground, posText, cv::Point(40, 150), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 200, 200), 1, cv::LINE_AA);
        sprintf(posText, "Con 1/tracker: %.2f %.2f %.2f", co1PosC[0], co1PosC[1], co1PosC[2]);
        cv::putText(foreground, posText, cv::Point(40, 170), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 200, 200), 1, cv::LINE_AA);
        sprintf(posText, "Con 2/tracker: %.2f %.2f %.2f", co2PosC[0], co2PosC[1], co2PosC[2]);
        cv::putText(foreground, posText, cv::Point(40, 190), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 200, 200), 1, cv::LINE_AA);

        // show VR headset
        cv::line(foreground, cv::Point(hmdPosS[0], hmdPosS[1]), cv::Point(co1PosS[0], co1PosS[1]), cv::Scalar(0, 0, 255), 2);
        cv::line(foreground, cv::Point(hmdPosS[0], hmdPosS[1]), cv::Point(co2PosS[0], co2PosS[1]), cv::Scalar(0, 0, 255), 2);

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
    delete chaperoneBounds;

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
