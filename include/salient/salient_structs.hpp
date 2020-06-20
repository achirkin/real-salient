#pragma once

namespace salient
{

    /** Parameters of a camera that describe the dimensions of the projected space. */
    struct CameraIntrinsics
    {
        /** Width of a frame in pixels. */
        int width;
        /** Height of a frame in pixels. */
        int height;
        /** Horizontal coordinate of the principal point of a frame, as a pixel offset from the left edge. */
        float ppx;
        /** Vertical coordinate of the principal point of a frame, as a pixel offset from the top edge. */
        float ppy;
        /** Focal length of the image plane, as a multiple of pixel width */
        float fx;
        /** Focal length of the image plane, as a multiple of pixel height */
        float fy;
    };

    /** Spatial transform describing a relationship between two cameras. */
    struct CameraExtrinsics
    {
        /** A 3x3 column-major matrix describing a rotation in 3D space. */
        float rotation[9];
        /** Shift vector, describing a translation in 3D space _in meters_. */
        float translation[3];
    };

    /** A volume in the image+depth space. */
    struct SceneBounds
    {
        int left;
        int top;
        int right;
        int bottom;
        float near;
        float far;
    };

} // namespace salient