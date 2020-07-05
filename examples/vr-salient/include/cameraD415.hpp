#pragma once

#include <iostream>
#include <chrono>
#include <thread>
#include <librealsense2/rs.hpp>
#include <librealsense2/rs_advanced_mode.hpp>
#include <librealsense2/rsutil.h>
#include "assets.hpp"
#include "camera.hpp"

namespace camera
{
    class IntelFrameSet : public FrameSet
    {
    private:
        rs2::frameset intelFrameSet;

    public:
        IntelFrameSet() {}
        IntelFrameSet(rs2::frameset frameset) : intelFrameSet(frameset) {}

        const void *getColor() const override
        {
            return intelFrameSet.get_color_frame().get_data();
        }

        const void *getDepth() const override
        {
            return intelFrameSet.get_depth_frame().get_data();
        }
    };

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
                    // load json preset for high accuracy mode
                    auto dev_adv = rs400::advanced_mode::advanced_mode(dev);
                    dev_adv.load_json(std::string(assets::REALSENSE_CAMERA_SETTINGS));
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
        return dev;
    }

    class IntelD415Camera : public Camera
    {
    private:
        rs2::pipeline pipe;
        float depthScale;
        salient::CameraIntrinsics depthIntr;
        salient::CameraIntrinsics colorIntr;
        salient::CameraExtrinsics color2depth;
        IntelFrameSet lastFrameSet;

    public:
        IntelD415Camera()
        try
        {
            // find the camera
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
            depthScale = sensor.get_depth_scale();

            // get dimension (to be sure)
            auto depth_stream = selection.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
            auto color_stream = selection.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

            auto rs2_depth_intr = depth_stream.get_intrinsics();
            auto rs2_color_intr = color_stream.get_intrinsics();
            auto rs2_color_to_depth = color_stream.get_extrinsics_to(depth_stream);

            depthIntr = {
                rs2_depth_intr.width,
                rs2_depth_intr.height,
                rs2_depth_intr.ppx,
                rs2_depth_intr.ppy,
                rs2_depth_intr.fx,
                rs2_depth_intr.fy};

            colorIntr = {
                rs2_color_intr.width,
                rs2_color_intr.height,
                rs2_color_intr.ppx,
                rs2_color_intr.ppy,
                rs2_color_intr.fx,
                rs2_color_intr.fy};

            // rely on the fact that we have the same representation as librealsense
            color2depth = *reinterpret_cast<salient::CameraExtrinsics *>(&rs2_color_to_depth);

            // Skips some frames to allow for auto-exposure stabilization
            for (int i = 0; i < 10; i++)
                pipe.wait_for_frames();
        }
        catch (const rs2::error &e)
        {
            std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }

        const salient::CameraIntrinsics getColorIntrinsics() const override
        {
            return colorIntr;
        }

        const salient::CameraIntrinsics getDepthIntrinsics() const override
        {
            return depthIntr;
        }

        const salient::CameraExtrinsics getColorToDepthTransform() const override
        {
            return color2depth;
        }

        float getDepthScale() const { return depthScale; }

        const IntelFrameSet *waitForFrames() override
        {
            lastFrameSet = IntelFrameSet(pipe.wait_for_frames());
            return &lastFrameSet;
        }
    };
} // namespace camera