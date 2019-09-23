#include <iostream>
#include <cstring>
#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_declares.h"
#include "BaseImgParams.h"

__constant__ float ImageToVolume_DeviceConstant[640 * 16];

__constant__ int VOL_DIM_X;
__constant__ int VOL_DIM_Y;
__constant__ int VOL_DIM_Z;

__constant__ int US_DIM_X;
__constant__ int US_DIM_Y;
__constant__ int FRAME_NUMBER;

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_Z 2

#define GROUPING_SIZE 1

#define INTERP_KERNEL_RADIUS 1
#define HOLEFILLING_KERNEL_RADIUS 1

__device__ int GetPxlIdx(int col_idx, int row_idx, int pag_idx){
    return col_idx + row_idx * blockDim.x * gridDim.x + pag_idx * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
}

__global__ void HoleFilling_GPU(float *Volume_d, float *Volume_cpy_d, float *Weighting_d){
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int pag_idx = threadIdx.z + blockIdx.z * blockDim.z;

    int Vol_idx = col_idx + row_idx * VOL_DIM_X + pag_idx * VOL_DIM_X * VOL_DIM_Y; //GetPxlIdx(col_idx, row_idx, pag_idx);

    if(col_idx < VOL_DIM_X && row_idx < VOL_DIM_Y && pag_idx < VOL_DIM_Z){
        if(Weighting_d[Vol_idx] == 0.0f){
            int Acc_NotZeroVoxel = 0;
            float Acc_Sum = 0.0;
            {//Accumulate the not zero voxels:
                for(int k_z = 0 - HOLEFILLING_KERNEL_RADIUS; k_z < 1 + HOLEFILLING_KERNEL_RADIUS; ++k_z){
                    for(int k_y = 0 - HOLEFILLING_KERNEL_RADIUS; k_y < 1 + HOLEFILLING_KERNEL_RADIUS; ++k_y){
                        for(int k_x = 0 - HOLEFILLING_KERNEL_RADIUS; k_x < 1 + HOLEFILLING_KERNEL_RADIUS; ++k_x){
                            //check if the current selected pixel [x, y, z] is out of bound:
                            int Selected_Pxl[3] = {col_idx + k_x, row_idx + k_y, pag_idx + k_z};
                            if( (Selected_Pxl[0] >= 0 && Selected_Pxl[0] < VOL_DIM_X) && 
                                (Selected_Pxl[1] >= 0 && Selected_Pxl[1] < VOL_DIM_Y) && 
                                (Selected_Pxl[2] >= 0 && Selected_Pxl[2] < VOL_DIM_Z) ){
                                //is the current selected pxl ZERO?
                                if( Volume_cpy_d[ Selected_Pxl[0] + Selected_Pxl[1] * VOL_DIM_X + Selected_Pxl[2] * VOL_DIM_X * VOL_DIM_Y ] != 0.0f){ 
                                    //Not Zero, to be selected for hole filling:
                                    Acc_Sum += Volume_cpy_d[ Selected_Pxl[0] + Selected_Pxl[1] * VOL_DIM_X + Selected_Pxl[2] * VOL_DIM_X * VOL_DIM_Y ];
                                    ++Acc_NotZeroVoxel;
                                }
                            }
                        }
                    }
                }
            }
            if(Acc_NotZeroVoxel > 0){
                Volume_d[Vol_idx] = Acc_Sum / Acc_NotZeroVoxel;
            }
        }
    }
}

__device__ void Matrix4x4MultiplyPoint(const float* point_in, float* point_out, int frame_idx){
    float sum_tmp = 0.0f;
    for(int it_row = 0; it_row < 4; ++ it_row){
        for(int it_col = 0; it_col < 4; ++it_col){
            sum_tmp += ImageToVolume_DeviceConstant[frame_idx * 16 + it_row * 4 + it_col] * point_in[it_col];
        }
        point_out[it_row] = sum_tmp;
        sum_tmp = 0.0f;
    }
}

__global__ void US_Distribution_GPU(float* Volume_d, float* Weighting_d, uint8_t* US_Frame_d, 
    //Parameters for Volume:
    // int Vol_Dim_x, int Vol_Dim_y, int Vol_Dim_z, 
    float Vxl_size_x, float Vxl_size_y, float Vxl_size_z, 
    float Vol_Ori_x, float Vol_Ori_y, float Vol_Ori_z 

    //Parameters for Ultrasound frames:
    // int US_Dim_x, int US_Dim_y, int US_FrameNumber
    ){

    int frm_idx = (threadIdx.z + blockIdx.z * blockDim.z) * GROUPING_SIZE;
    int col_idx = (threadIdx.x + blockIdx.x * blockDim.x) * GROUPING_SIZE;
    int row_idx = (threadIdx.y + blockIdx.y * blockDim.y) * GROUPING_SIZE;

    for(int it_frame = 0; it_frame < GROUPING_SIZE; ++it_frame){
        
        for(int it_row = 0; it_row < GROUPING_SIZE; ++it_row){
            
            for(int it_col = 0; it_col < GROUPING_SIZE; ++it_col){
                
                //Iteration starts:
                if( frm_idx < FRAME_NUMBER && row_idx < US_DIM_Y && col_idx < US_DIM_X){
                    //Get US pxl under volume, from pxl to [mm]:
                    float US_pxl[4] = {float(col_idx), float(row_idx), 0.0, 1.0};
                    float US_pxl_Under_Vol[4] = {0.0f};
                    Matrix4x4MultiplyPoint(US_pxl, US_pxl_Under_Vol, frm_idx);

                    // printf("%f, %f, %f \n", US_pxl_Under_Vol[0], US_pxl_Under_Vol[1], US_pxl_Under_Vol[2]);

                    //Derive pxl under volume in [mm] to [VOXEL]:
                    float Vxl_From_US_Pxl[3] = { roundf((US_pxl_Under_Vol[0] - Vol_Ori_x) / Vxl_size_x), roundf((US_pxl_Under_Vol[1] - Vol_Ori_y) / Vxl_size_y), roundf((US_pxl_Under_Vol[2] - Vol_Ori_z) / Vxl_size_z) };
                    
                    //Iterate 6 voxels around:
                    {
                        for(int it_i = 0 - INTERP_KERNEL_RADIUS; it_i < 0 + 1 + INTERP_KERNEL_RADIUS; ++it_i){
                            for(int it_j = 0 - INTERP_KERNEL_RADIUS; it_j < 0 + 1 + INTERP_KERNEL_RADIUS; ++it_j){
                                for(int it_k = 0 - INTERP_KERNEL_RADIUS; it_k < 0 + 1 + INTERP_KERNEL_RADIUS; ++it_k){
                                    int select_x = Vxl_From_US_Pxl[0] + it_i;
                                    int select_y = Vxl_From_US_Pxl[1] + it_j;
                                    int select_z = Vxl_From_US_Pxl[2] + it_k;
                                    if( (select_x < VOL_DIM_X) && (select_x >= 0) &&
                                        (select_y < VOL_DIM_Y) && (select_y >= 0) &&
                                        (select_z < VOL_DIM_Z) && (select_z >= 0)){
                                        //Calculate inverse distance: [voxel] to US [pixel]:
                                        float inv_distance = exp( -sqrt(
                                            ( (select_x * Vxl_size_x + Vol_Ori_x) - US_pxl_Under_Vol[0] ) * 
                                            ( (select_x * Vxl_size_x + Vol_Ori_x) - US_pxl_Under_Vol[0] ) + 
                                            ( (select_y * Vxl_size_y + Vol_Ori_y) - US_pxl_Under_Vol[1] ) * 
                                            ( (select_y * Vxl_size_y + Vol_Ori_y) - US_pxl_Under_Vol[1] ) + 
                                            ( (select_z * Vxl_size_z + Vol_Ori_z) - US_pxl_Under_Vol[2] ) * 
                                            ( (select_z * Vxl_size_z + Vol_Ori_z - US_pxl_Under_Vol[2] ))));
                                        
                                        // float Volume_tmp = 
                                        // printf("%f, %f, %f \n", select_x, select_y, select_z);
                                        
                                        float sum = Volume_d[select_x + select_y * VOL_DIM_X + select_z * VOL_DIM_X * VOL_DIM_Y] * Weighting_d[select_x + select_y * VOL_DIM_X + select_z * VOL_DIM_X * VOL_DIM_Y] + 
                                                    US_Frame_d[col_idx + row_idx * US_DIM_X + frm_idx * US_DIM_X * US_DIM_Y] * inv_distance;
                                        Weighting_d[select_x + select_y * VOL_DIM_X + select_z * VOL_DIM_X * VOL_DIM_Y] += inv_distance;
                                        Volume_d[select_x + select_y * VOL_DIM_X + select_z * VOL_DIM_X * VOL_DIM_Y] = sum / Weighting_d[select_x + select_y * VOL_DIM_X + select_z * VOL_DIM_X * VOL_DIM_Y];
                                    }
                                }
                            }
                        }                
                    }
                }
                ++col_idx;
            }
            ++row_idx;
        }
        ++frm_idx;
    }
    
}

__global__ void GetXY_plane(float *Volume_d, float *plane_d, int Location){
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;

    int Vol_x_idx = col_idx;
    int Vol_y_idx = row_idx;
    int Vol_z_idx = Location;

    if(col_idx < VOL_DIM_X && row_idx < VOL_DIM_Y){
        plane_d[col_idx + row_idx * VOL_DIM_X] = Volume_d[Vol_x_idx + Vol_y_idx * VOL_DIM_X + Vol_z_idx * VOL_DIM_X * VOL_DIM_Y];
    }
}

__global__ void GetXZ_plane(float *Volume_d, float *plane_d, int Location){
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;

    int Vol_x_idx = col_idx;
    int Vol_y_idx = Location;
    int Vol_z_idx = row_idx;

    if(col_idx < VOL_DIM_X && row_idx < VOL_DIM_Z){
        plane_d[col_idx + row_idx * VOL_DIM_X] = Volume_d[Vol_x_idx + Vol_y_idx * VOL_DIM_X + Vol_z_idx * VOL_DIM_X * VOL_DIM_Y];
    }
}

__global__ void GetYZ_plane(float *Volume_d, float *plane_d, int Location){
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;

    int Vol_x_idx = Location;
    int Vol_y_idx = row_idx;
    int Vol_z_idx = col_idx;

    if(col_idx < VOL_DIM_Z && row_idx < VOL_DIM_Y){
        plane_d[col_idx + row_idx * VOL_DIM_Z] = Volume_d[Vol_x_idx + Vol_y_idx * VOL_DIM_X + Vol_z_idx * VOL_DIM_X * VOL_DIM_Y];
    }
}

void DisplayPlane(float *Volume_d, float *plane_d, Ultrasound_Parameters US_Params, Volume_Parameters Vol_Params, int Location, char Axis){
    
    dim3 BlockDim_Plane( 
        BLOCK_SIZE_X, 
        BLOCK_SIZE_Y, 
        1
    );

    switch (Axis)
    {
    case 'Z':
        {
            dim3 GridDim_Plane( 
                int(ceil( float(Vol_Params.dim.x) / BLOCK_SIZE_X )), 
                int(ceil( float(Vol_Params.dim.y) / BLOCK_SIZE_Y )), 
                1
            );
            GetXY_plane<<<GridDim_Plane, BlockDim_Plane>>>(Volume_d, plane_d, Location);
            break;
        }
        
    case 'Y': 
        {   
            dim3 GridDim_Plane( 
                int(ceil( float(Vol_Params.dim.x) / BLOCK_SIZE_X )), 
                int(ceil( float(Vol_Params.dim.z) / BLOCK_SIZE_Y )), 
                1
            );
            GetXZ_plane<<<GridDim_Plane, BlockDim_Plane>>>(Volume_d, plane_d, Location);
            break;
        }
    case 'X': 
        {
            dim3 GridDim_Plane( 
                int(ceil( float(Vol_Params.dim.y) / BLOCK_SIZE_X )), 
                int(ceil( float(Vol_Params.dim.z) / BLOCK_SIZE_Y )), 
                1
            );
            GetYZ_plane<<<GridDim_Plane, BlockDim_Plane>>>(Volume_d, plane_d, Location);
            break;
        }
    default:
        std::cout << "Please enter correct Plane indicator: Z (xy plane), Y (xz plane), X (yz plane). " << std::endl;
        break;
    }
}


void GPU_Setups(bool GPU_Available, Ultrasound_Parameters& US_Params, Volume_Parameters& Vol_Params, const float* TotalMatrices, float* Recon_Volume, float* Weighting_Volume, uint8_t* US_Frames){

    //Asignment for Device constant memory:
    cudaMemcpyToSymbol(ImageToVolume_DeviceConstant, TotalMatrices, US_Params.FrameNumber * 16 * sizeof(float));

    cudaMemcpyToSymbol(VOL_DIM_X, &Vol_Params.dim.x, sizeof(int));
    cudaMemcpyToSymbol(VOL_DIM_Y, &Vol_Params.dim.y, sizeof(int));
    cudaMemcpyToSymbol(VOL_DIM_Z, &Vol_Params.dim.z, sizeof(int));

    cudaMemcpyToSymbol(US_DIM_X, &US_Params.dim.x, sizeof(int));
    cudaMemcpyToSymbol(US_DIM_Y, &US_Params.dim.y, sizeof(int));

    cudaMemcpyToSymbol(FRAME_NUMBER, &US_Params.FrameNumber, sizeof(int));

    //Allocate GPU memory:
    float   *Volume_d       = NULL;
    float   *Volume_cpy_d   = NULL;
    float   *Weighting_d    = NULL;
    uint8_t *US_Frame_d     = NULL;

    cudaMalloc((void **)&Volume_d, Vol_Params.dim.x * Vol_Params.dim.y * Vol_Params.dim.z * sizeof(float));
    cudaMalloc((void **)&Volume_cpy_d, Vol_Params.dim.x * Vol_Params.dim.y * Vol_Params.dim.z * sizeof(float));
    cudaMalloc((void **)&Weighting_d, Vol_Params.dim.x * Vol_Params.dim.y * Vol_Params.dim.z * sizeof(float));
    cudaMalloc((void **)&US_Frame_d, US_Params.dim.x * US_Params.dim.y * US_Params.FrameNumber * sizeof(uint8_t));

    //Copy mem from host RAM to device RAM:
    cudaMemcpy(US_Frame_d, US_Frames, US_Params.dim.x * US_Params.dim.y * US_Params.FrameNumber * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Weighting_d, Weighting_Volume, Vol_Params.dim.x * Vol_Params.dim.y * Vol_Params.dim.z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Volume_d, Recon_Volume, Vol_Params.dim.x * Vol_Params.dim.y * Vol_Params.dim.z * sizeof(float), cudaMemcpyHostToDevice);

    /*------------------------------------------- Perform Distribution ------------------------------------------ */
    //Define threads distribution:
    dim3 BlockDim_Distribution( 
        BLOCK_SIZE_X, 
        BLOCK_SIZE_Y, 
        BLOCK_SIZE_Z 
    );
    dim3 GridDim_Distribution( 
        int(ceil(float(US_Params.dim.x) / BLOCK_SIZE_X / float(GROUPING_SIZE))), 
        int(ceil(float(US_Params.dim.y) / BLOCK_SIZE_Y / float(GROUPING_SIZE))), 
        int(ceil(float(US_Params.FrameNumber) / BLOCK_SIZE_Z / float(GROUPING_SIZE))) 
    );
    US_Distribution_GPU<<<GridDim_Distribution, BlockDim_Distribution>>>(Volume_d, Weighting_d, US_Frame_d, Vol_Params.pxlSize.x, Vol_Params.pxlSize.y, Vol_Params.pxlSize.z, Vol_Params.orig.x, Vol_Params.orig.y, Vol_Params.orig.z);
    cudaMemcpy(Volume_cpy_d, Volume_d, Vol_Params.dim.x * Vol_Params.dim.y * Vol_Params.dim.z * sizeof(float), cudaMemcpyDeviceToDevice);
    
    /*----------------------------------------------- Hole Filling ---------------------------------------------- */
    dim3 BlockDim_HoleFilling( 
        BLOCK_SIZE_X, 
        BLOCK_SIZE_Y, 
        BLOCK_SIZE_Z 
    );
    dim3 GridDim_HoleFilling( 
        int(ceil( float(Vol_Params.dim.x) / BLOCK_SIZE_X )), 
        int(ceil( float(Vol_Params.dim.y) / BLOCK_SIZE_Y )), 
        int(ceil( float(Vol_Params.dim.z) / BLOCK_SIZE_Z )) 
    );
    HoleFilling_GPU<<<GridDim_HoleFilling, BlockDim_HoleFilling>>>(Volume_d, Volume_cpy_d, Weighting_d);

    //Complete the remaining memory transfer:
    cudaMemcpy(Recon_Volume, Volume_d, Vol_Params.dim.x * Vol_Params.dim.y * Vol_Params.dim.z * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Weighting_Volume, Weighting_d, Vol_Params.dim.x * Vol_Params.dim.y * Vol_Params.dim.z * sizeof(float), cudaMemcpyDeviceToHost);

    //For display Image:
    float *plane_XY_d, *plane_XZ_d, *plane_YZ_d;
    cudaMalloc((void **)&plane_XY_d, Vol_Params.dim.x * Vol_Params.dim.y * sizeof(float));
    cudaMalloc((void **)&plane_XZ_d, Vol_Params.dim.x * Vol_Params.dim.z * sizeof(float));
    cudaMalloc((void **)&plane_YZ_d, Vol_Params.dim.y * Vol_Params.dim.z * sizeof(float));

    // DisplayPlane(Volume_d, plane_XY_d, US_Params, Vol_Params, 100, 'Z');

    cudaFree(Volume_d);
    cudaFree(Volume_cpy_d);
    cudaFree(Weighting_d);
    cudaFree(US_Frame_d);

    cudaFree(plane_XY_d);
    cudaFree(plane_XZ_d);
    cudaFree(plane_YZ_d);
}