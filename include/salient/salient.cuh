#pragma once

#include "util.hpp"
#include "crf.cuh"
#include "gmm.cuh"

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

    /** Settings for the components of RealSalient. Can be changed every frame. */
    struct AnalysisSettings
    {
        // GMM settings
        int gmmIterations = 10;
        float timeAlpha = 0.1f;
        float imputedLabelWeight = 0.3f;

        // CRF settings
        int crfIterations = 5;
        float smoothnessWeight = 3.0f;
        float smoothnessVarPos = 4.0f;
        float appearanceWeight = 10.0f;
        float appearanceVarPos = 60.0f;
        float appearanceVarCol = 25.0f;
        float similarityWeight = 0.5f;
        float similarityVarCol = 100.0f;
    };

    class FrameTransform
    {
    private:
        /**
         * These constants are used for the alignment transform between images from two different cameras.
         * They can be calculated based on the cameras intrinsic and extrinsic parameters.
         * 
         * I referred to the following links to derive the transform:
         *   https://intelrealsense.github.io/librealsense/doxygen/structrs2__extrinsics.html
         *   https://intelrealsense.github.io/librealsense/doxygen/structrs2__intrinsics.html
         *
         *   https://github.com/IntelRealSense/librealsense/blob/9f99fa9a509555f85bffc15ce27531aaa6db6f7e/src/proc/align.cpp
         *   https://github.com/IntelRealSense/librealsense/blob/842ee1e1e5c4bb96d63582a7fde061dbc1bebf69/include/librealsense2/rsutil.h
         */
        float M[6], s0[3], s1[3], pp[2];

    public:
        __host__ FrameTransform(
            const CameraIntrinsics destInstrinsics,
            const CameraIntrinsics sourceIntrinsics,
            const CameraExtrinsics extrinsics,
            const float downsampleRatio)
        {
            M[0] = downsampleRatio * extrinsics.rotation[0] * destInstrinsics.fx / sourceIntrinsics.fx;
            M[1] = downsampleRatio * extrinsics.rotation[1] * destInstrinsics.fy / sourceIntrinsics.fx;
            M[2] = downsampleRatio * extrinsics.rotation[2] / sourceIntrinsics.fx;
            M[3] = downsampleRatio * extrinsics.rotation[3] * destInstrinsics.fx / sourceIntrinsics.fy;
            M[4] = downsampleRatio * extrinsics.rotation[4] * destInstrinsics.fy / sourceIntrinsics.fy;
            M[5] = downsampleRatio * extrinsics.rotation[5] / sourceIntrinsics.fy;

            s0[0] = (sourceIntrinsics.ppx * M[0] + sourceIntrinsics.ppy * M[3]) / downsampleRatio - extrinsics.rotation[6] * destInstrinsics.fx;
            s0[1] = (sourceIntrinsics.ppx * M[1] + sourceIntrinsics.ppy * M[4]) / downsampleRatio - extrinsics.rotation[7] * destInstrinsics.fy;
            s0[2] = (sourceIntrinsics.ppx * M[2] + sourceIntrinsics.ppy * M[5]) / downsampleRatio - extrinsics.rotation[8];

            s1[0] = extrinsics.translation[0] * destInstrinsics.fx;
            s1[1] = extrinsics.translation[1] * destInstrinsics.fy;
            s1[2] = extrinsics.translation[2];

            pp[0] = destInstrinsics.ppx;
            pp[1] = destInstrinsics.ppy;
        }

        /** Transform pixel coordinates in the source space to the pixel coordinates in the dest space.  */
        __device__ void transform(const float x_px, const float y_px, const float focus_meters, float &out_x_px, float &out_y_px) const
        {
            float focus_inv = 1.0f / max(0.1f, focus_meters); // assume minimum focus is 10cm
            float dz = 1.0f / (x_px * M[2] + y_px * M[5] + s1[2] * focus_inv - s0[2]);
            out_x_px = dz * (x_px * M[0] + y_px * M[3] + s1[0] * focus_inv - s0[0]) + pp[0];
            out_y_px = dz * (x_px * M[1] + y_px * M[4] + s1[1] * focus_inv - s0[1]) + pp[1];
        }
    };

    /**
     * Detect the salient object in the scene every frame
     * (i.e. 2-class per-pixel classification).
     * 
     * @param C Number of feature channels in the data.
     * @param GaussianK Number of Gaussians in each GMM.
     * @param GetFeature GPU device function to access the color feature at the given pixel position (i, j).
     */
    template <int C, int GaussianK, class GetFeature>
    class RealSalient
    {
    private:
        const cudaStream_t mainStream;

        /** The block size for pre- and post-processing CUDA kernels in this class. */
        const dim3 squareBlockSize = dim3(32, 32, 1);

        cudaArray *depthArray, *depthInterpArray, *lhoodArray;
        cudaTextureObject_t depthTex, depthInterpTex, lhoodTex;
        cudaSurfaceObject_t depthInterpSurf;
        /** A buffer to render a temporary histogram and output SceneBounds in the optional post-processing step. */
        uint32_t *postProcess_temp;
        uint32_t *postProcess_temp_cpu;
        static const int depthHistSize = 256;

        static void initUint16Frame(const int w, const int h, cudaArray *&array, cudaTextureObject_t *texture);
        template <int TexChannels>
        static void initInterpolatedFrame(const int w, const int h, cudaArray *&array, cudaTextureObject_t *texture, cudaSurfaceObject_t *surface);

        /** Model-specific initialization, called in the constructor. */
        void initModels();

        /** Copy the depth buffer into one GPU array and fill another GPU array imputing values via interpolation.  */
        void loadDepthFrame(const uint16_t *host_depth_buffer);

        /** Copy the color buffer into the GPU and populate the feature buffer with the provided GetFeature function. */
        void loadColorFrame();

        /** Use the earlier initialized depth data to initialize labels. */
        void buildLabels(const SceneBounds foreground_bounds, const cudaTextureObject_t* renderedDepth = nullptr);

        /** Run all models on the current frame. */
        void runModels();

    public:
        /** Divide width and height of the original color image by this value to reduce the computational burden. */
        const int downsampleRatio;

        /** A multiplier for getting the real depth in meters given the uint_16t depth buffer. */
        const float depthScale;

        const int color_W;
        const int color_H;
        const int depth_W;
        const int depth_H;
        /** Downsampled color_W. */
        const int W;
        /** Downsampled color_H. */
        const int H;

        const dim3 dimDepth;
        const dim3 dimColor;
        const dim3 dimWork;

        /** How to probe the depth frame at the position known for the color frame.  */
        const FrameTransform color2depth;

        const GetFeature getFeature;

        /** Gaussian mixture model */
        gmm::GaussianMixtures<C, GaussianK, 2> gmmModel;

        /** Conditional random field */
        crf::DenseCRF<2> crfModel;
        crf::IPairwisePotential *smoothnessPairwise, *appearancePairwise, *similarityPairwise;

        /**
         * The buffer with dimensions defined by colorIntrinsics,
         * contains the calculated probabilites of being foreground for every pixel.
         * 
         * Updated after in processFrames function.
         */
        float *probabilities;

        /**
         * Aligned and imputed values of depth in the processing image space (downscaled color).
         * 
         * Imputed values are saved with negative sign.
         */
        float *aligned_depth;

        /** Focal length of the color image plane, as a multiple of pixel size */
        const float colorFx, colorFy;

        AnalysisSettings analysisSettings{};

        RealSalient(
            const cudaStream_t mainStream,
            const CameraIntrinsics depthIntrinsics,
            const CameraIntrinsics colorIntrinsics,
            const CameraExtrinsics colorToDepthExtrinsics,
            const int downsampleRatio,
            const float depthScale,
            const GetFeature getFeature)
            : mainStream(mainStream),
              downsampleRatio(downsampleRatio),
              depthScale(depthScale),
              getFeature(getFeature),
              color2depth(depthIntrinsics, colorIntrinsics, colorToDepthExtrinsics, downsampleRatio),
              color_W(colorIntrinsics.width), color_H(colorIntrinsics.height),
              depth_W(depthIntrinsics.width), depth_H(depthIntrinsics.height),
              W(colorIntrinsics.width / downsampleRatio),
              H(colorIntrinsics.height / downsampleRatio),
              dimDepth(depth_W, depth_H, 1),
              dimColor(color_W, color_H, 1),
              dimWork(W, H, 1),
              depthArray(nullptr), depthInterpArray(nullptr), lhoodArray(nullptr),
              gmmModel(mainStream, W * H),
              crfModel(mainStream, W * H, gmmModel.logLikelihoodPtr()),
              probabilities(nullptr), aligned_depth(nullptr),
              postProcess_temp(nullptr), postProcess_temp_cpu(nullptr),
              colorFx(colorIntrinsics.fx), colorFy(colorIntrinsics.fy)
        {
            cudaMalloc((void **)&probabilities, sizeof(float) * color_W * color_H);
            cudaErrorCheck(nullptr);
            cudaMalloc((void **)&postProcess_temp, sizeof(uint32_t) * (W + H + depthHistSize));
            cudaErrorCheck(nullptr);
            cudaMalloc((void **)&aligned_depth, sizeof(float) * W * H);
            cudaErrorCheck(nullptr);
            postProcess_temp_cpu = new uint32_t[W + H + depthHistSize];

            initUint16Frame(depth_W, depth_H, depthArray, &depthTex);
            initInterpolatedFrame<1>(depth_W, depth_H, depthInterpArray, &depthInterpTex, &depthInterpSurf);
            initInterpolatedFrame<2>(W, H, lhoodArray, &lhoodTex, nullptr);

            initModels();
        }

        ~RealSalient()
        {
            delete[] postProcess_temp_cpu;
            cudaFree(aligned_depth);
            cudaErrorCheck(nullptr);
            cudaFree(postProcess_temp);
            cudaErrorCheck(nullptr);
            cudaFree(probabilities);
            cudaErrorCheck(nullptr);
            cudaDestroyTextureObject(depthInterpTex);
            cudaErrorCheck(nullptr);
            cudaDestroySurfaceObject(depthInterpSurf);
            cudaErrorCheck(nullptr);
            cudaDestroyTextureObject(depthTex);
            cudaErrorCheck(nullptr);
            cudaDestroyTextureObject(lhoodTex);
            cudaErrorCheck(nullptr);
            cudaFreeArray(depthInterpArray);
            cudaErrorCheck(nullptr);
            cudaFreeArray(depthArray);
            cudaErrorCheck(nullptr);
            cudaFreeArray(lhoodArray);
            cudaErrorCheck(nullptr);
        }

        RealSalient(const RealSalient &o) = delete;

        /**
         * Process a pair of frames (depth and color) and store the probabilities (of being foreground, per-pixel).
         * 
         * @param host_depth_buffer an image from the depth camera.
         * @param foreground_bounds cutoff near and far distance in meters and min/max x/y pixel positions in the space of the color camera.
         * @param gmmIterations number of iterations in EM estimation algorithm for GMMs.
         * @param crfIterations number of passes in CRF inference.
         * @param renderedDepth optional texture objects that helps to cut-off near salient object from far background on per-pixel basis; it's size must be equal to the downscaled color frame.
         */
        void processFrames(
            const uint16_t *host_depth_buffer,
            const SceneBounds foreground_bounds,
            const cudaTextureObject_t * renderedDepth = nullptr);

        /**
         * Try to infer bounds on the salient object (foreground).
         * It gets the final labeling for the current frame and aggregates minimum/maximum positions and depths of the foreground pixels.
         * The bounds are in the space of the color camera.
         * 
         * @param percentile is used to ignore a small fraction of outlier foreground pixels.
         * @param maxDepth maximum value of depth to keep in the histogram (in meters).
         * @param margin add this number to the bounds size (in meters).
         * @param minSize minimum size of the bounds (in meters).
         * @param maxSize maximum size of the bounds (in meters).
         */
        SceneBounds postprocessInferBounds(const float percentile = 0.95f, const float maxDepth = 6.0f, const float margin = 0.3f, const float minSize = 0.5f, const float maxSize = 2.5f);
    };

    dim3 enlargeSquare(const dim3 square, const int pow2)
    {
        return dim3(square.x << pow2, square.y << pow2, square.z);
    }

    template <int C, int GaussianK, class GetFeature>
    void RealSalient<C, GaussianK, GetFeature>::initUint16Frame(const int w, const int h, cudaArray *&array, cudaTextureObject_t *texture)
    {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;     // Same type as content (i.e. not a normalized float)
        texDesc.addressMode[0] = cudaAddressModeBorder; // zero outside of the region
        texDesc.addressMode[1] = cudaAddressModeBorder; // -/-
        texDesc.filterMode = cudaFilterModePoint;       // nearest neighbour interpolation
        texDesc.normalizedCoords = false;               // do not normalize to 0..1

        cudaMallocArray(&array, &channelDesc, w, h);
        cudaResourceDesc resourceDesc;
        memset(&resourceDesc, 0, sizeof(resourceDesc));
        resourceDesc.resType = cudaResourceTypeArray;
        resourceDesc.res.array.array = array;
        cudaCreateTextureObject(texture, &resourceDesc, &texDesc, NULL);
        cudaErrorCheck(nullptr);
    }

    template <int C, int GaussianK, class GetFeature>
    template <int TexChannels>
    void RealSalient<C, GaussianK, GetFeature>::initInterpolatedFrame(const int w, const int h, cudaArray *&array, cudaTextureObject_t *texture, cudaSurfaceObject_t *surface)
    {
        bool withSurface = surface != nullptr;

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(
            TexChannels > 0 ? 32 : 0,
            TexChannels > 1 ? 32 : 0,
            TexChannels > 2 ? 32 : 0,
            TexChannels > 3 ? 32 : 0, cudaChannelFormatKindFloat);
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;    // Same type as content (i.e. not a normalized float)
        texDesc.addressMode[0] = cudaAddressModeClamp; // clamp to coordinates to the image region
        texDesc.addressMode[1] = cudaAddressModeClamp; // -/-
        texDesc.filterMode = cudaFilterModeLinear;     // linear interpolation
        texDesc.normalizedCoords = false;              // do not normalize to 0..1

        cudaMallocArray(&array, &channelDesc, w, h, withSurface ? cudaArraySurfaceLoadStore : cudaArrayDefault);
        cudaResourceDesc resourceDesc;
        memset(&resourceDesc, 0, sizeof(resourceDesc));
        resourceDesc.resType = cudaResourceTypeArray;
        resourceDesc.res.array.array = array;
        cudaCreateTextureObject(texture, &resourceDesc, &texDesc, NULL);
        cudaErrorCheck(nullptr);
        if (withSurface)
        {
            cudaCreateSurfaceObject(surface, &resourceDesc);
            cudaErrorCheck(nullptr);
        }
    }

    template <unsigned int X>
    __global__ void depth_interpolate(int W, int H, float depthScale, cudaTextureObject_t depthTex, cudaSurfaceObject_t depthInterpSurf)
    {
        extern __shared__ float sDepthMipmap[];
        uint32_t *sWeightMipmap = (uint32_t *)(sDepthMipmap + blockDim.x * blockDim.y * 2);
        uint32_t w0 = (blockIdx.x * blockDim.x + threadIdx.x) << X;
        uint32_t h0 = (blockIdx.y * blockDim.y + threadIdx.y) << X;
        const uint32_t D(1 << X);

        float totalDepth = 0;
        uint16_t curDepth = 0;
        uint32_t weight = 0;
#pragma unfold
        for (uint32_t w = w0; w < w0 + D; w++)
#pragma unfold
            for (uint32_t h = h0; h < h0 + D; h++)
            {
                curDepth = tex2D<uint16_t>(depthTex, w, h);
                totalDepth += curDepth * depthScale;
                weight += curDepth > 0 ? 1 : 0;
            }

        float interpolatedDepth = weight == 0 ? 0 : (totalDepth / (float)weight);
        uint32_t t = blockDim.x;
        sDepthMipmap[(t + threadIdx.y) * t + threadIdx.x] = interpolatedDepth;
        sWeightMipmap[(t + threadIdx.y) * t + threadIdx.x] = weight;
        __syncthreads();

        uint16_t wa, wb;
        float da, db;
        uint32_t ix = threadIdx.x;
        uint32_t iy = threadIdx.y;
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1)
        {
            if (threadIdx.x < t && threadIdx.y < s)
            {
                wa = sWeightMipmap[(t + threadIdx.y * 2) * t + threadIdx.x];
                wb = sWeightMipmap[(t + threadIdx.y * 2 + 1) * t + threadIdx.x];
                da = sDepthMipmap[(t + threadIdx.y * 2) * t + threadIdx.x];
                db = sDepthMipmap[(t + threadIdx.y * 2 + 1) * t + threadIdx.x];
                weight = wa + wb;
                totalDepth = (float)wa * da + (float)wb * db;
                sWeightMipmap[(s + threadIdx.y) * t + threadIdx.x] = weight;
                sDepthMipmap[(s + threadIdx.y) * t + threadIdx.x] = weight == 0 ? 0 : (totalDepth / (float)weight);
            }
            __syncthreads();

            if (threadIdx.x < s && threadIdx.y < s)
            {
                wa = sWeightMipmap[(s + threadIdx.y) * t + threadIdx.x * 2];
                wb = sWeightMipmap[(s + threadIdx.y) * t + threadIdx.x * 2 + 1];
                da = sDepthMipmap[(s + threadIdx.y) * t + threadIdx.x * 2];
                db = sDepthMipmap[(s + threadIdx.y) * t + threadIdx.x * 2 + 1];
                weight = wa + wb;
                totalDepth = (float)wa * da + (float)wb * db;
                sWeightMipmap[(s + threadIdx.y) * s + threadIdx.x] = weight;
                sDepthMipmap[(s + threadIdx.y) * s + threadIdx.x] = weight == 0 ? 0 : (totalDepth / (float)weight);
            }
            __syncthreads();
            t = s;
            ix >>= 1;
            iy >>= 1;
            if (interpolatedDepth == 0)
                interpolatedDepth = sDepthMipmap[(s + iy) * s + ix];
        }

#pragma unfold
        for (int w = w0; w < w0 + D; w++)
#pragma unfold
            for (int h = h0; h < h0 + D; h++)
            {
                curDepth = tex2D<uint16_t>(depthTex, (float)w, (float)h);
                surf2Dwrite<float>(curDepth > 0 ? (curDepth * depthScale) : interpolatedDepth, depthInterpSurf, w * sizeof(float), h, cudaBoundaryModeZero);
            }
    }

    template <int C, int GaussianK, class GetFeature>
    void RealSalient<C, GaussianK, GetFeature>::loadDepthFrame(const uint16_t *host_depth_buffer)
    {
        const unsigned int X(3); // gets to the power-of-two; the image scan block size multiplier
        dim3 blocks = distribute(dimDepth, enlargeSquare(squareBlockSize, X));

        // Copy some data located at address h_data in host memory into CUDA array
        cudaMemcpyToArrayAsync(depthArray, 0, 0, host_depth_buffer, sizeof(uint16_t) * depth_W * depth_H, cudaMemcpyHostToDevice, mainStream);
        cudaErrorCheck(mainStream);
        depth_interpolate<X><<<blocks, squareBlockSize, squareBlockSize.x * squareBlockSize.y * sizeof(float) * 4, mainStream>>>(depth_W, depth_H, depthScale, depthTex, depthInterpSurf);
        cudaErrorCheck(mainStream);
    }

    template <int C, class GetFeature>
    __global__ void build_features(const int W, const int H, const int downsampleRatio, float *out_feature, GetFeature getFeature)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= W || j >= H)
            return;

        float d2 = 1.0f / (float)(downsampleRatio * downsampleRatio);
        float tmp_pixel[C];
        float *out_pixel = out_feature + (i + j * W) * C;
#pragma unroll
        for (int c = 0; c < C; c++)
            out_pixel[c] = 0;

        for (int ki = i * downsampleRatio; ki < (i + 1) * downsampleRatio; ki++)
            for (int kj = j * downsampleRatio; kj < (j + 1) * downsampleRatio; kj++)
            {
                getFeature(ki, kj, tmp_pixel);
#pragma unroll
                for (int c = 0; c < C; c++)
                    out_pixel[c] += tmp_pixel[c];
            }

#pragma unroll
        for (int c = 0; c < C; c++)
            out_pixel[c] *= d2;
    }

    template <int C, int GaussianK, class GetFeature>
    void RealSalient<C, GaussianK, GetFeature>::loadColorFrame()
    {
        build_features<C, GetFeature><<<distribute(dimWork, squareBlockSize), squareBlockSize, 0, mainStream>>>(W, H, downsampleRatio, gmmModel.featuresPtr(), getFeature);
        cudaErrorCheck(mainStream);
    }

    // Classes:
    // 0 - foreground
    // 1 - background
    __global__ void depth_to_labels(
        const int W, const int H,
        const int downsampleRatio,
        cudaTextureObject_t depthTex,
        cudaTextureObject_t depthInterpTex,
        int8_t *out_labels,
        float *out_depth,
        const float depthScale,
        const SceneBounds foreground_bounds,
        const salient::FrameTransform color2depth,
        const cudaTextureObject_t renderedDepth,
        const bool withDepthThreshold)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= W || j >= H)
            return;

        float pfar = withDepthThreshold ? min(tex2D<float>(renderedDepth, i, j), foreground_bounds.far) : foreground_bounds.far;

        const bool inBounds =
            downsampleRatio * i >= foreground_bounds.left &&
            downsampleRatio * i < foreground_bounds.right &&
            downsampleRatio * j >= foreground_bounds.top &&
            downsampleRatio * j < foreground_bounds.bottom;
        float di = (float)i;
        float dj = (float)j;
        float dx, dy;
        float depth = max(0.3f, 0.5f * (pfar + foreground_bounds.near));
        // get approximate depth first
        color2depth.transform(di, dj, depth, dx, dy);
        depth = tex2D<float>(depthInterpTex, dx, dy);
        // now get the real depth
        if (depth > 0.01f)
            color2depth.transform(di, dj, depth, dx, dy);
        depth = tex2D<uint16_t>(depthTex, dx, dy) * depthScale;
        int8_t label_base;
        if (depth > 0)
        {
            out_depth[i + j * W] = depth;
            label_base = (!inBounds || depth > pfar || depth < foreground_bounds.near) ? 1 : 0;
            if (abs(depth - pfar) < (depth + pfar) * 0.05f)
                label_base -= 0x80; // make the label appear imputed if depth is too close to the threshold
            out_labels[i + j * W] = label_base;
        }
        else
        {
            out_depth[i + j * W] = -tex2D<float>(depthInterpTex, dx, dy);
            out_labels[i + j * W] = inBounds ? -1 : 1;
        }
    }

    __device__ int8_t match_labels(int8_t a, int8_t b)
    {
        if (a == -1)
            return b;
        if (b != a && b != -1)
            return -2;
        return a;
    }

    template <unsigned int X>
    __global__ void labels_impute(int W, int H, int8_t *labels)
    {
        extern __shared__ int8_t sdata[];
        // -1 means not specified and can be any
        // -2 means contradiction
        // other value means all are either not specified or that class.
        int8_t common_label = -1;
        int8_t label_tmp;
        int w0 = (blockIdx.x * blockDim.x + threadIdx.x) << X;
        int h0 = (blockIdx.y * blockDim.y + threadIdx.y) << X;
        const int D(1 << X);
#pragma unfold
        for (int w = w0; w < W && w < w0 + D; w++)
#pragma unfold
            for (int h = h0; h < H && h < h0 + D; h++)
            {
                label_tmp = labels[h * W + w];
                label_tmp = (label_tmp == -1) ? label_tmp : (label_tmp & 0x7F); // use both GT and imputed data.
                common_label = match_labels(common_label, label_tmp);
            }
        unsigned int t = blockDim.x;
        sdata[(t + threadIdx.y) * t + threadIdx.x] = common_label;
        __syncthreads();

        unsigned int ix = threadIdx.x;
        unsigned int iy = threadIdx.y;
        for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
        {
            if (threadIdx.x < t && threadIdx.y < s)
            {
                sdata[(s + threadIdx.y) * t + threadIdx.x] = match_labels(
                    sdata[(t + threadIdx.y * 2) * t + threadIdx.x],
                    sdata[(t + threadIdx.y * 2 + 1) * t + threadIdx.x]);
            }
            __syncthreads();

            if (threadIdx.x < s && threadIdx.y < s)
            {
                sdata[(s + threadIdx.y) * s + threadIdx.x] = match_labels(
                    sdata[(s + threadIdx.y) * t + threadIdx.x * 2],
                    sdata[(s + threadIdx.y) * t + threadIdx.x * 2 + 1]);
            }
            __syncthreads();
            t = s;
            ix >>= 1;
            iy >>= 1;
            if (common_label == -1)
                common_label = sdata[(s + iy) * s + ix];
        }

        if (common_label >= 0)
#pragma unfold
            for (int w = w0; w < W && w < w0 + D; w++)
#pragma unfold
                for (int h = h0; h < H && h < h0 + D; h++)
                    if (labels[h * W + w] < 0)
                        labels[h * W + w] = common_label - 0x80;
    }

    template <int C, int GaussianK, class GetFeature>
    void RealSalient<C, GaussianK, GetFeature>::buildLabels(const SceneBounds foreground_bounds, const cudaTextureObject_t* renderedDepth)
    {
        const unsigned int X(3); // gets to the power-of-two; the image scan block size multiplier
        depth_to_labels<<<distribute(dimWork, squareBlockSize), squareBlockSize, 0, mainStream>>>(W, H, downsampleRatio, depthTex, depthInterpTex, gmmModel.labelsPtr(), aligned_depth, depthScale, foreground_bounds, color2depth,
            *renderedDepth, renderedDepth != nullptr);
        cudaErrorCheck(mainStream);
        labels_impute<X><<<distribute(dimWork, enlargeSquare(squareBlockSize, X)), squareBlockSize, squareBlockSize.x * squareBlockSize.y * sizeof(int8_t) * 2, mainStream>>>(W, H, gmmModel.labelsPtr());
        cudaErrorCheck(mainStream);
    }

    template <int C, int GaussianK, class GetFeature>
    void RealSalient<C, GaussianK, GetFeature>::initModels()
    {
        // add a color independent term (feature = pixel location 0..W-1, 0..H-1)
        smoothnessPairwise = new crf::PairwisePotential<2, 2>(mainStream, W, H);
        crfModel.addPairwiseEnergy(smoothnessPairwise);

        // add a color dependent term (feature = xyrgb)
        appearancePairwise = new crf::PairwisePotential<2, 2 + C>(mainStream, W, H);
        crfModel.addPairwiseEnergy(appearancePairwise);

        // add a position independent term (feature = rgb)
        similarityPairwise = new crf::PairwisePotential<2, C>(mainStream, W, H);
        crfModel.addPairwiseEnergy(similarityPairwise);
    }

    template <int C, int GaussianK, class GetFeature>
    void RealSalient<C, GaussianK, GetFeature>::runModels()
    {
        if (analysisSettings.gmmIterations > 0)
        {
            gmmModel.alpha = analysisSettings.timeAlpha;
            gmmModel.imputed_weight = analysisSettings.imputedLabelWeight;
            gmmModel.iterate(analysisSettings.gmmIterations);
        }

        gmmModel.infer();

        if (analysisSettings.crfIterations > 0)
        {
            smoothnessPairwise->loadImage(
                analysisSettings.smoothnessWeight,
                analysisSettings.smoothnessVarPos);
            appearancePairwise->loadImage(
                analysisSettings.appearanceWeight,
                analysisSettings.appearanceVarPos,
                analysisSettings.appearanceVarCol,
                gmmModel.featuresPtr());
            similarityPairwise->loadImage(
                analysisSettings.similarityWeight,
                0,
                analysisSettings.similarityVarCol,
                gmmModel.featuresPtr());
            crfModel.inference(analysisSettings.crfIterations);
        }
    }

    __global__ void compute_probabilities(
        const int probs_w, const int probs_h,
        const int lhood_w, const int lhood_h,
        float *out_probs, cudaTextureObject_t lhoodTex)
    {
        int indexX = blockIdx.x * blockDim.x + threadIdx.x;
        int strideX = blockDim.x * gridDim.x;
        int indexY = blockIdx.y * blockDim.y + threadIdx.y;
        int strideY = blockDim.y * gridDim.y;
        float dx = (float)lhood_w / (float)probs_w;
        float dy = (float)lhood_h / (float)probs_h;
        float2 lhood;
        for (int i = indexX; i < probs_w; i += strideX)
            for (int j = indexY; j < probs_h; j += strideY)
            {
                lhood = tex2D<float2>(lhoodTex, dx * i, dy * j);
                lhood.x = expf(lhood.x);
                lhood.y = expf(lhood.y);
                out_probs[i + probs_w * j] = min(1.0f, max(0.0f, lhood.x / max(0.000001, lhood.x + lhood.y)));
            }
    }

    template <int C, int GaussianK, class GetFeature>
    void RealSalient<C, GaussianK, GetFeature>::processFrames(const uint16_t *host_depth_buffer, const SceneBounds foreground_bounds, const cudaTextureObject_t *renderedDepth)
    {
        loadDepthFrame(host_depth_buffer);
        loadColorFrame();
        buildLabels(foreground_bounds, renderedDepth);
        runModels();

        auto lhoodPtr = analysisSettings.crfIterations > 0 ? crfModel.logLikelihoodPtr() : gmmModel.logLikelihoodPtr();
        cudaMemcpyToArrayAsync(lhoodArray, 0, 0, lhoodPtr, sizeof(float) * 2 * W * H, cudaMemcpyDeviceToDevice, mainStream);
        cudaErrorCheck(mainStream);
        compute_probabilities<<<distribute(dimColor, squareBlockSize), squareBlockSize, 0, mainStream>>>(color_W, color_H, W, H, probabilities, lhoodTex);
        cudaErrorCheck(mainStream);
    }

    __global__ void postprocess_histograms(
        const int W, const int H, const int D,
        const float max_depth,
        uint32_t *out_hists,
        cudaTextureObject_t lhoodTex /** likelihoods of being in a class. */,
        const float *in_depth /** aligned, float-valued depth in meters. Negative values mean imputed. */)
    {
        extern __shared__ uint32_t shists[];
        const int bd = blockDim.x; // == blockDim.y;
        uint32_t *hist_sx = shists;
        uint32_t *hist_sy = hist_sx + bd * bd;
        uint32_t *hist_x = out_hists;
        uint32_t *hist_y = hist_x + W;
        uint32_t *hist_d = hist_y + H;
        const int si = threadIdx.x;
        const int sj = threadIdx.y;
        const int i = blockIdx.x * bd + si;
        const int j = blockIdx.y * bd + sj;
        const bool outOfBounds = i >= W || j >= H;
        const float2 lhoods = tex2D<float2>(lhoodTex, i, j);
        const bool foreground = !outOfBounds && lhoods.x > lhoods.y;
        const float depth = foreground ? in_depth[i + j * W] : 0;

        if (depth > 0 && depth < max_depth)
            atomicAdd(hist_d + lrintf((D - 1) * depth / max_depth), 1);

        hist_sx[si + sj * bd] = foreground ? 1 : 0;
        hist_sy[si + sj * bd] = foreground ? 1 : 0;

        __syncthreads();

        for (unsigned int t = bd >> 1; t > 0; t >>= 1)
        {
            if (sj < t)
                hist_sx[si + sj * bd] += hist_sx[si + (sj + t) * bd];

            if (si < t)
                hist_sy[si + sj * bd] += hist_sy[si + t + sj * bd];

            __syncthreads();
        }

        if (si == 0 && !outOfBounds)
            atomicAdd(hist_y + j, hist_sy[sj * bd]);

        if (sj == 0 && !outOfBounds)
            atomicAdd(hist_x + i, hist_sx[si]);
    }

    void getArrayPercentile(int N, const uint32_t *array, const float percentile, int &out_min, int &out_max)
    {
        uint32_t sum = 0;
        for (int i = 0; i < N; i++)
            sum += array[i];
        uint32_t thresh = (uint32_t)roundf((1.0f - percentile) * 0.5f * sum);
        out_min = 0;
        out_max = N;
        uint32_t s = 0;
        while (s < thresh)
        {
            s += array[out_min];
            out_min++;
        }
        s = 0;
        while (s < thresh)
        {
            out_max--;
            s += array[out_max];
        }
    }

    void applyLimits(const float minValue, const float maxValue, const float margin, const float minSize, const float maxSize, float &x, float &y)
    {
        x = max(minValue, x - margin);
        y = min(maxValue, y + margin);
        float diff = min(maxSize, max(minSize, y - x)) * 0.5f;
        float avg = max(minValue + diff, min(maxValue - diff, (x + y) * 0.5f));
        x = avg - diff;
        y = avg + diff;
    }

    template <int C, int GaussianK, class GetFeature>
    SceneBounds RealSalient<C, GaussianK, GetFeature>::postprocessInferBounds(const float percentile, const float maxDepth, const float margin, const float minSize, const float maxSize)
    {
        dim3 blocks = distribute(dim3(W, H, 1), squareBlockSize);
        cudaMemsetAsync((void *)postProcess_temp, 0, sizeof(uint32_t) * (W + H + depthHistSize), mainStream);
        cudaErrorCheck(mainStream);
        postprocess_histograms<<<blocks, squareBlockSize, sizeof(uint32_t) * squareBlockSize.x * squareBlockSize.y * 2, mainStream>>>(W, H, depthHistSize, maxDepth, postProcess_temp, lhoodTex, aligned_depth);
        cudaErrorCheck(mainStream);
        cudaMemcpyAsync(postProcess_temp_cpu, postProcess_temp, sizeof(uint32_t) * (W + H + depthHistSize), cudaMemcpyDeviceToHost, mainStream);
        cudaErrorCheck(mainStream);
        cudaStreamSynchronize(mainStream);

        int xmin, xmax, ymin, ymax, zmin, zmax;
        getArrayPercentile(W, postProcess_temp_cpu, percentile, xmin, xmax);
        getArrayPercentile(H, postProcess_temp_cpu + W, percentile, ymin, ymax);
        getArrayPercentile(depthHistSize, postProcess_temp_cpu + W + H, percentile, zmin, zmax);

        const float dz = maxDepth / depthHistSize;
        float dmin = dz * zmin;
        float dmax = dz * zmax;
        applyLimits(0.1f, maxDepth, margin, minSize, maxSize, dmin, dmax);
        float davg = 0.5f * (dmin + dmax);

        float a = downsampleRatio * xmin;
        float b = downsampleRatio * xmax;
        float r = colorFx / davg;
        applyLimits(0, color_W, margin * r, minSize * r, maxSize * r, a, b);
        xmin = (int)round(a);
        xmax = (int)round(b);

        a = downsampleRatio * ymin;
        b = downsampleRatio * ymax;
        r = colorFy / davg;
        applyLimits(0, color_H, margin * r, minSize * r, maxSize * r, a, b);
        ymin = (int)round(a);
        ymax = (int)round(b);

        return SceneBounds{
            xmin /* int left in pixels */,
            ymin /* int top in pixels */,
            xmax /* int right in pixels */,
            ymax /* int bottom in pixels */,
            dmin /* float near distance meters */,
            dmax /* float far distance meters */
        };
    }
} // namespace salient