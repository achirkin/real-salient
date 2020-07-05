#pragma once

#include <iostream>
#include <openvr.h>
#include "salient/salient_structs.hpp"

template <class Vertex>
class VrBounds
{
public:
    int validDevCount;

private:
    const salient::CameraIntrinsics colorIntr;
    const salient::CameraExtrinsics color2tracker;
    vr::IVRSystem *m_pHMD;
    vr::TrackedDevicePose_t devicePositions[vr::k_unMaxTrackedDeviceCount];

    float (*trackerPos)[4];
    float pMatrix[16];
    int trackedDevCount;
    int trackedDevIds[vr::k_unMaxTrackedDeviceCount];
    float trackedDevDists[vr::k_unMaxTrackedDeviceCount];
    int trackedDevXYs[vr::k_unMaxTrackedDeviceCount * 2];
    int deviceCount;

    salient::SceneBounds boundPoints()
    {
        if (validDevCount <= 0)
            return salient::SceneBounds{
                0 // left;
                ,
                0 // top;
                ,
                colorIntr.width // right;
                ,
                colorIntr.height // bottom;
                ,
                near // near;
                ,
                far // far;
            };
        float marginTop = 0.2f;
        float marginSide = 0.5f;
        float marginBottom = 2.0f; // don't remove legs!
        float cur_depth;
        salient::SceneBounds result;
        result.left = colorIntr.width;
        result.top = colorIntr.height;
        result.right = 0;
        result.bottom = 0;
        result.near = far;
        result.far = near;
        for (int i = 0; i < validDevCount; i++)
        {
            cur_depth = trackedDevDists[i];
            if (cur_depth > 0.01f && cur_depth < far)
            {
                result.near = min(result.near, max(near, cur_depth - marginSide));
                result.far = max(result.far, min(far, cur_depth + marginSide));
                result.left = min(result.left, max(0, (int)round(trackedDevXYs[2 * i] - marginSide * colorIntr.fx / cur_depth)));
                result.top = min(result.top, max(0, (int)round(trackedDevXYs[2 * i + 1] - marginTop * colorIntr.fy / cur_depth)));
                result.right = max(result.right, min(colorIntr.width, (int)round(trackedDevXYs[2 * i] + marginSide * colorIntr.fx / cur_depth)));
                result.bottom = max(result.bottom, min(colorIntr.height, (int)round(trackedDevXYs[2 * i + 1] + marginBottom * colorIntr.fy / cur_depth)));
            }
        }
        return result;
    };

public:
    const float near, far;

    salient::CameraExtrinsics world2color;
    float mvpMatrix[16];
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    salient::SceneBounds trackerBounds;
    bool trackerFound = false;
    bool shaperoneValid = false;

    VrBounds(Vertex (*toVertex)(float, float, float), const salient::CameraExtrinsics color2tracker, const salient::CameraIntrinsics colorIntr, const float near = 0.1f, const float far = 10.0f)
        : color2tracker(color2tracker), colorIntr(colorIntr), near(near), far(far)
    {
        int trackerIdx = -1;
        deviceCount = 0;
        trackedDevCount = 0;
        validDevCount = 0;
        trackerBounds = boundPoints();
        memset(&mvpMatrix, 0, sizeof(mvpMatrix));
        memset(&pMatrix, 0, sizeof(pMatrix));
        pMatrix[0] = 2 * colorIntr.fx / colorIntr.width;
        pMatrix[5] = 2 * colorIntr.fy / colorIntr.height;
        pMatrix[10] = (far + near) / (far - near);
        pMatrix[11] = 1;
        pMatrix[14] = 2 * far * near / (near - far);

        // Loading the SteamVR Runtime
        vr::EVRInitError eError = vr::VRInitError_None;
        m_pHMD = vr::VR_Init(&eError, vr::VRApplication_Background); // or VRApplication_Scene

        if (eError != vr::VRInitError_None)
        {
            m_pHMD = nullptr;
            std::cout << "Unable to init VR runtime: " << vr::VR_GetVRInitErrorAsEnglishDescription(eError) << std::endl;
            return;
        }

        for (int deviceIdx = 0; deviceIdx < vr::k_unMaxTrackedDeviceCount; deviceIdx++)
        {
            if (m_pHMD->IsTrackedDeviceConnected(deviceIdx))
            {
                std::cout << deviceIdx << ": ";
                switch (m_pHMD->GetTrackedDeviceClass(deviceIdx))
                {
                case vr::TrackedDeviceClass_Controller:
                    deviceCount = deviceIdx + 1;
                    trackedDevIds[trackedDevCount] = deviceIdx;
                    trackedDevCount++;
                    std::cout << "controller";
                    break;
                case vr::TrackedDeviceClass_HMD:
                    deviceCount = deviceIdx + 1;
                    trackedDevIds[trackedDevCount] = deviceIdx;
                    trackedDevCount++;
                    std::cout << "HMD";
                    break;
                case vr::TrackedDeviceClass_Invalid:
                    std::cout << "Invalid";
                    break;
                case vr::TrackedDeviceClass_GenericTracker:
                    deviceCount = deviceIdx + 1;
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
            }
        }
        if (trackerIdx < 0)
        {
            std::cout << "Could not find the camera tracker (searched for a generic tracker class)" << std::endl;
            return;
        }
        trackerFound = true;
        trackerPos = devicePositions[trackerIdx].mDeviceToAbsoluteTracking.m;
        std::cout << "Found " << trackedDevCount << " devices to track." << std::endl;

        auto vrsetup = vr::VRChaperoneSetup();
        uint32_t chaperoneQuadsCount;
        vrsetup->GetLiveCollisionBoundsInfo(NULL, &chaperoneQuadsCount);
        if (chaperoneQuadsCount > 0)
        {
            shaperoneValid = true;
            vr::HmdQuad_t *chaperoneBounds = new vr::HmdQuad_t[chaperoneQuadsCount];
            vrsetup->GetLiveCollisionBoundsInfo(chaperoneBounds, &chaperoneQuadsCount);

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
                vertices.push_back(toVertex(chaperoneBounds[i].vCorners[0].v[0], chaperoneBounds[i].vCorners[0].v[1], chaperoneBounds[i].vCorners[0].v[2]));
                vertices.push_back(toVertex(chaperoneBounds[i].vCorners[1].v[0], chaperoneBounds[i].vCorners[1].v[1], chaperoneBounds[i].vCorners[1].v[2]));
                vertices.push_back(toVertex(chaperoneBounds[i].vCorners[2].v[0], chaperoneBounds[i].vCorners[2].v[1], chaperoneBounds[i].vCorners[2].v[2]));
                vertices.push_back(toVertex(chaperoneBounds[i].vCorners[3].v[0], chaperoneBounds[i].vCorners[3].v[1], chaperoneBounds[i].vCorners[3].v[2]));
                indices.push_back(i * 4);
                indices.push_back(i * 4 + 2);
                indices.push_back(i * 4 + 1);
                indices.push_back(i * 4 + 3);
                indices.push_back(i * 4 + 2);
                indices.push_back(i * 4);
            }
            vertices.push_back(toVertex(smins[0], smins[1], smins[2]));
            vertices.push_back(toVertex(smins[0], smins[1], smaxs[2]));
            vertices.push_back(toVertex(smaxs[0], smins[1], smaxs[2]));
            vertices.push_back(toVertex(smaxs[0], smins[1], smins[2]));
            vertices.push_back(toVertex(smins[0], smaxs[1], smins[2]));
            vertices.push_back(toVertex(smins[0], smaxs[1], smaxs[2]));
            vertices.push_back(toVertex(smaxs[0], smaxs[1], smaxs[2]));
            vertices.push_back(toVertex(smaxs[0], smaxs[1], smins[2]));
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
            delete[] chaperoneBounds;
        }
    };

    ~VrBounds()
    {
        if (m_pHMD != nullptr)
            vr::VR_Shutdown();
    };

    void update()
    {
        if (deviceCount <= 0)
            return;

        // load position matrices.
        m_pHMD->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0.0f, devicePositions, deviceCount);

        // update modev-view matrix
        for (int i = 0; i < 3; i++)
        {
            world2color.translation[i] = 0;
            for (int j = 0; j < 3; j++)
                world2color.translation[i] -= color2tracker.rotation[j + i * 3] * color2tracker.translation[j];
            for (int j = 0; j < 3; j++)
            {
                world2color.rotation[i + j * 3] = 0;
                for (int k = 0; k < 3; k++)
                    world2color.rotation[i + j * 3] += trackerPos[j][k] * color2tracker.rotation[k + i * 3];
                world2color.translation[i] -= world2color.rotation[i + j * 3] * trackerPos[j][3];
            }
        }

        // make a list of tracked points to derive the bounding box
        validDevCount = 0;
        for (int d = 0; d < trackedDevCount; d++)
        {
            auto devPos(devicePositions[trackedDevIds[d]]);
            if (devPos.bPoseIsValid)
            {
                auto x = world2color.translation[0];
                auto y = world2color.translation[1];
                auto z = world2color.translation[2];
                for (int j = 0; j < 3; j++)
                {
                    auto s = devPos.mDeviceToAbsoluteTracking.m[j][3];
                    x += world2color.rotation[j * 3] * s;
                    y += world2color.rotation[j * 3 + 1] * s;
                    z += world2color.rotation[j * 3 + 2] * s;
                }
                trackedDevDists[validDevCount] = z;
                trackedDevXYs[validDevCount * 2] = (int)round(x / z * colorIntr.fx + colorIntr.ppx);
                trackedDevXYs[validDevCount * 2 + 1] = (int)round(y / z * colorIntr.fy + colorIntr.ppy);
                validDevCount++;
            }
        }
        trackerBounds = boundPoints();

        // make the model-view-projection matrix for the vulkan depth renderer
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
                mvpMatrix[i + j * 4] = world2color.rotation[i + j * 3] * pMatrix[i * 5];
            mvpMatrix[i + 12] = world2color.translation[i] * pMatrix[i * 5];
            mvpMatrix[3 + i * 4] = world2color.rotation[2 + i * 3];
        }
        mvpMatrix[14] += pMatrix[14];
        mvpMatrix[15] = world2color.translation[2];
    };

    template <class ToResult>
    auto getTrackedPoint(ToResult f, int i) -> decltype(f(0, 0))
    {
        return f(trackedDevXYs[i * 2], trackedDevXYs[i * 2 + 1]);
    }
};