#pragma once

#include "salient/salient_structs.hpp"

namespace camera
{
    class FrameSet
    {
    public:
        /** Get a pointer to the color buffer. */
        virtual const void *getColor() const = 0;

        /** Get a pointer to the depth buffer. */
        virtual const void *getDepth() const = 0;
    };

    class Camera
    {
    public:
        virtual const salient::CameraIntrinsics getColorIntrinsics() const = 0;
        virtual const salient::CameraIntrinsics getDepthIntrinsics() const = 0;
        virtual const salient::CameraExtrinsics getColorToDepthTransform() const = 0;
        virtual float getDepthScale() const = 0;
        virtual const FrameSet *waitForFrames() = 0;
    };
} // namespace camera