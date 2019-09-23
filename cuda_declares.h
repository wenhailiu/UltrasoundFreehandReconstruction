#ifndef CUDA_FREERECON
#define CUDA_FREERECON

#include "BaseImgParams.h"

extern const int frames_number;

void GPU_Setups(bool GPU_Available, Ultrasound_Parameters& US_Params, Volume_Parameters& Vol_Params, const float* TotalMatrices, float* Recon_Volume, float* Weighting_Volume, uint8_t* US_Frames);

#endif