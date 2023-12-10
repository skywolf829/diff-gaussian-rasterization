/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float focal_x, const float focal_y,
		const float tan_fovx, const float tan_fovy,
		const float* viewmatrix,
		const float* bg_color,
		const float3* means3D,
		const float* cov3D,
		const float2* means2D,
		const float4* conic_opacity,
		const float3* world_space_wave_direction,
		const float2* screen_space_wave_direction,
		const int* frequency_coefficient_indices,
		const uint8_t* num_periods,
		const bool* into_screen,
		const float* colors,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const uint8_t* n_contrib_periodic,
		const float* dL_dpixels,
		float2* dL_dmean2D,
		float3* dL_dmean3D,
		float* dL_dcov3D,
		float* dL_dopacity,
		float* dL_dcolors);

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		const int* frequencies,
		const float2* dL_dmean2D,
		const float* dL_dcov3D,
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot);
}

#endif