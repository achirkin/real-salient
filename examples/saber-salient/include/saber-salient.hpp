#pragma once

#include "salient/salient_structs.hpp"

extern "C"
{
	struct SaberSalient;

	__declspec(dllexport) int SaberSalient_init(SaberSalient** out_SaberSalient, void (*in_loadColor)(uint8_t*), void (*in_loadDepth)(float*));

	__declspec(dllexport) void SaberSalient_destroy(SaberSalient *in_SaberSalient);

	__declspec(dllexport) int SaberSalient_cameraIntrinsics(SaberSalient *in_SaberSalient, salient::CameraIntrinsics *out_intrinsics);

	__declspec(dllexport) int SaberSalient_currentTransform(SaberSalient* in_SaberSalient, float *out_mat44);

	__declspec(dllexport) int SaberSalient_currentPosition(SaberSalient* in_SaberSalient, float* out_vec3);

	__declspec(dllexport) int SaberSalient_currentRotation(SaberSalient* in_SaberSalient, float* out_mat33);

	__declspec(dllexport) uint8_t* SaberSalient_getColorBuf(SaberSalient* in_SaberSalient);

	__declspec(dllexport) float* SaberSalient_getDepthBuf(SaberSalient* in_SaberSalient);

}