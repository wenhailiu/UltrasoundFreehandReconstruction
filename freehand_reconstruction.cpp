#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cmath>

#include "opencv2/opencv.hpp"
#include "vtkObjectFactory.h"
#include "vtkMatrix4x4.h"
#include "vtkMath.h"
#include "vtkQuaternion.h"

#include "mylibs.h"

#include "BaseImgParams.h"
#include "cms_3d_volume.h"
#include "freehand_reconstruction.h"

#include "cuda_declares.h"

void ImportAllMatrices(MyMatrix& ImageToVolumeMatrix, const MyMatrix& ImageToProbeMatrix_ext, std::ifstream& INPUT_TRANSMTX){
    //Read Probe tracking data:
    std::vector<std::string> LineCSV = getNextLineAndSplitIntoTokens(INPUT_TRANSMTX, ' ');
    MyMatrix ProbeToWordMatrix = MyMatrix::New();
    for(int j = 0; j < 16; ++j){
        ProbeToWordMatrix->SetElement((j) / 4, ((j) % 4), std::stof(LineCSV[j]));
    }

    // float ProbeRotm[3][3] = {0.0f};
    // float ProbeEul[3] = {0.0f};
    // {//test:
        
    //     for(int i = 0; i < 3; ++i){
    //         for(int j = 0; j < 3; ++j){
    //             ProbeRotm[i][j] = ProbeToWordMatrix->GetElement(i, j);
    //         }
    //     }
    //     vtkQuaternion<float> ProbeQuat;
    //     ProbeQuat.FromMatrix3x3(ProbeRotm);
    //     Quat2EulXYZ(ProbeQuat.GetW(), ProbeQuat.GetX(), ProbeQuat.GetY(), ProbeQuat.GetZ(), ProbeEul);
    // }

    //Read Phantom tracking data:
    LineCSV = getNextLineAndSplitIntoTokens(INPUT_TRANSMTX, ' ');
    MyMatrix VolumeToWordMatrix = MyMatrix::New();
    for(int j = 0; j < 16; ++j){
        VolumeToWordMatrix->SetElement((j) / 4, ((j) % 4), std::stof(LineCSV[j]));
    }

    {   
        //Calculate to world transform matrix:
        MyMatrix ImageToWorld = MyMatrix::New();
        vtkMatrix4x4::Multiply4x4(ProbeToWordMatrix, ImageToProbeMatrix_ext, ImageToWorld);

        //World to volume:
        MyMatrix WorldToVolumeMatrix = MyMatrix::New();
        vtkMatrix4x4::Invert(VolumeToWordMatrix, WorldToVolumeMatrix);

        //Image To Volume:
        vtkMatrix4x4::Multiply4x4(WorldToVolumeMatrix, ImageToWorld, ImageToVolumeMatrix);
    }
}

void Volume_Param_Correction(const Ultrasound_Parameters US_params, Volume_Parameters& V_params, std::vector<float>& ImageToVolumeMatrices, const MyMatrix& ImageToProbeMatrix_ext){
    float MAX_Vol_X = 0.0f, MAX_Vol_Y = 0.0f, MAX_Vol_Z = 0.0f, 
          MIN_Vol_X = 0.0f, MIN_Vol_Y = 0.0f, MIN_Vol_Z = 0.0f;

    {   
        float test_Pnt_1[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        float test_Pnt_2[4] = {float(US_params.dim.x), float(US_params.dim.y), 0.0f, 1.0f};
        std::vector<float> Container_X, Container_Y, Container_Z;
        std::ifstream INPUT_TRANSMTX;
        INPUT_TRANSMTX.open(US_params.TrackerPath, std::ifstream::in);
        for(int i = 0; i < US_params.FrameNumber; ++i){

            //Import transform matrix for ith frame:
            MyMatrix ImageToVolume_VTK = MyMatrix::New();
            ImportAllMatrices(ImageToVolume_VTK, ImageToProbeMatrix_ext, INPUT_TRANSMTX);

            //Calculate two test points under volume, in mm:
            float test_Pnt_1_VOL[4] = {0.0};
            ImageToVolume_VTK->MultiplyPoint(test_Pnt_1, test_Pnt_1_VOL);

            float test_Pnt_2_VOL[4] = {0.0};
            ImageToVolume_VTK->MultiplyPoint(test_Pnt_2, test_Pnt_2_VOL);
            
            //Store X, Y, Z, in mm, in container:
            Container_X.push_back(test_Pnt_1_VOL[0]);
            Container_X.push_back(test_Pnt_2_VOL[0]);

            Container_Y.push_back(test_Pnt_1_VOL[1]);
            Container_Y.push_back(test_Pnt_2_VOL[1]);

            Container_Z.push_back(test_Pnt_1_VOL[2]);
            Container_Z.push_back(test_Pnt_2_VOL[2]);

            //copy from External container to internal container:
            for(int element_it = 0; element_it < 16; ++element_it){
                ImageToVolumeMatrices[element_it + i * 16] = float(ImageToVolume_VTK->GetData()[element_it]);
            }
        }

        MAX_Vol_X = *std::max_element(Container_X.begin(), Container_X.end());
        MIN_Vol_X = *std::min_element(Container_X.begin(), Container_X.end());

        MAX_Vol_Y = *std::max_element(Container_Y.begin(), Container_Y.end());
        MIN_Vol_Y = *std::min_element(Container_Y.begin(), Container_Y.end());

        MAX_Vol_Z = *std::max_element(Container_Z.begin(), Container_Z.end());
        MIN_Vol_Z = *std::min_element(Container_Z.begin(), Container_Z.end());
        
        INPUT_TRANSMTX.close();
    }

    std::cout << "Volume parameters correction: X: " << MIN_Vol_X << ", Y: " << MIN_Vol_Y << ", Z: " << MIN_Vol_Z << std::endl;
    V_params.orig.x = MIN_Vol_X;
    V_params.orig.y = MIN_Vol_Y;
    V_params.orig.z = MIN_Vol_Z;

    std::cout << "Dimension: X: " << int(std::ceil( (MAX_Vol_X - MIN_Vol_X) / V_params.pxlSize.x )) + 1 << 
     ", Y: " << int(std::ceil( (MAX_Vol_Y - MIN_Vol_Y) / V_params.pxlSize.y )) + 1 << 
     ", Z: " << int(std::ceil( (MAX_Vol_Z - MIN_Vol_Z) / V_params.pxlSize.z )) + 1 << std::endl;

    V_params.dim.x = int(std::ceil( (MAX_Vol_X - MIN_Vol_X) / V_params.pxlSize.x )) + 1; 
    V_params.dim.y = int(std::ceil( (MAX_Vol_Y - MIN_Vol_Y) / V_params.pxlSize.y )) + 1; 
    V_params.dim.z = int(std::ceil( (MAX_Vol_Z - MIN_Vol_Z) / V_params.pxlSize.z )) + 1; 
}

//Initializers:
//without GPU
freehand_reconstruction::freehand_reconstruction(const Ultrasound_Parameters US_params, const Volume_Parameters V_params) : 
Parameters(US_params), 
AllMatricesReady(false), 
weightingedVolume(V_params.dim.x * V_params.dim.y * V_params.dim.z, 0.0f), 
ImageToVolumeMatrices(US_params.FrameNumber * 16, 0.0f), 
cms_3d_volume(V_params), 
Enable_CUDA(false), 
NumSelectedFrames(1)
{   
    double ImageToProbe[16];
    for(int i = 0; i < 16; ++i){
        ImageToProbe[i] = std::stof(US_params.ImageToProbeMatrix[2 + i]);
    }
    
    ProbeToWordMatrix = MyMatrix::New();
    VolumeToWordMatrix = MyMatrix::New();

    ImageToProbeMatrix = MyMatrix::New();
    vtkMatrix4x4::DeepCopy(ImageToProbeMatrix->GetData(), ImageToProbe);
    
    float norm_x = sqrt(ImageToProbeMatrix->GetElement(0, 0) * ImageToProbeMatrix->GetElement(0, 0) + ImageToProbeMatrix->GetElement(0, 1) * ImageToProbeMatrix->GetElement(0, 1) + ImageToProbeMatrix->GetElement(0, 2) * ImageToProbeMatrix->GetElement(0, 2));
    float norm_y = sqrt(ImageToProbeMatrix->GetElement(1, 0) * ImageToProbeMatrix->GetElement(1, 0) + ImageToProbeMatrix->GetElement(1, 1) * ImageToProbeMatrix->GetElement(1, 1) + ImageToProbeMatrix->GetElement(1, 2) * ImageToProbeMatrix->GetElement(1, 2));
    float norm_z = sqrt(ImageToProbeMatrix->GetElement(2, 0) * ImageToProbeMatrix->GetElement(2, 0) + ImageToProbeMatrix->GetElement(2, 1) * ImageToProbeMatrix->GetElement(2, 1) + ImageToProbeMatrix->GetElement(2, 2) * ImageToProbeMatrix->GetElement(2, 2));

    ImageToProbeMatrix->SetElement(0, 0, ImageToProbeMatrix->GetElement(0, 0) * US_params.pxlSize.x / norm_x);
    ImageToProbeMatrix->SetElement(0, 1, ImageToProbeMatrix->GetElement(0, 1) * US_params.pxlSize.x / norm_x);
    ImageToProbeMatrix->SetElement(0, 2, ImageToProbeMatrix->GetElement(0, 2) * US_params.pxlSize.x / norm_x);
    ImageToProbeMatrix->SetElement(0, 3, ImageToProbeMatrix->GetElement(0, 3) * US_params.pxlSize.x / norm_x);

    ImageToProbeMatrix->SetElement(1, 0, ImageToProbeMatrix->GetElement(1, 0) * US_params.pxlSize.y / norm_y);
    ImageToProbeMatrix->SetElement(1, 1, ImageToProbeMatrix->GetElement(1, 1) * US_params.pxlSize.y / norm_y);
    ImageToProbeMatrix->SetElement(1, 2, ImageToProbeMatrix->GetElement(1, 2) * US_params.pxlSize.y / norm_y);

    ImageToProbeMatrix->SetElement(2, 0, ImageToProbeMatrix->GetElement(2, 0) * US_params.pxlSize.z / norm_z);
    ImageToProbeMatrix->SetElement(2, 1, ImageToProbeMatrix->GetElement(2, 1) * US_params.pxlSize.z / norm_z);
    ImageToProbeMatrix->SetElement(2, 2, ImageToProbeMatrix->GetElement(2, 2) * US_params.pxlSize.z / norm_z);

    std::cout << "ImageToProbe Matrix" << " Norm_x = " << norm_x << ", " << "Norm_y = " << norm_y << ", " << "Norm_z = " << norm_z << std::endl;
    std::cout << "Normalized with pixel size: " << "x - " << US_params.pxlSize.x << ", y - " << US_params.pxlSize.y << ", y - " << US_params.pxlSize.z << std::endl;
    ImageToProbeMatrix->Print(std::cout);

    Volume_Param_Correction(this->Parameters, this->cms_3d_volume::Parameters, this->ImageToVolumeMatrices, ImageToProbeMatrix);
    
    weightingedVolume.resize(cms_3d_volume::Parameters.dim.x * cms_3d_volume::Parameters.dim.y * cms_3d_volume::Parameters.dim.z, 0.0f);
    cms_3d_volume::data.resize(cms_3d_volume::Parameters.dim.x * cms_3d_volume::Parameters.dim.y * cms_3d_volume::Parameters.dim.z, 0.0f);
    cms_3d_volume::XY_Plane.resize(cms_3d_volume::Parameters.dim.x * cms_3d_volume::Parameters.dim.y, 0.0f);
    cms_3d_volume::YZ_Plane.resize(cms_3d_volume::Parameters.dim.y * cms_3d_volume::Parameters.dim.z, 0.0f);
    cms_3d_volume::XZ_Plane.resize(cms_3d_volume::Parameters.dim.x * cms_3d_volume::Parameters.dim.z, 0.0f);
}

freehand_reconstruction::freehand_reconstruction(const Ultrasound_Parameters US_params, const Volume_Parameters V_params, bool GPU_Enable, int Num_Frames) : 
Parameters(US_params), 
AllMatricesReady(false), 
weightingedVolume(V_params.dim.x * V_params.dim.y * V_params.dim.z, 0.0f), 
ImageToVolumeMatrices(US_params.FrameNumber * 16, 0.0f), 
cms_3d_volume(V_params), 
Enable_CUDA(GPU_Enable), 
NumSelectedFrames(Num_Frames)
{   
    double ImageToProbe[16];
    for(int i = 0; i < 16; ++i){
        ImageToProbe[i] = std::stof(US_params.ImageToProbeMatrix[2 + i]);
    }
    
    ProbeToWordMatrix = MyMatrix::New();
    VolumeToWordMatrix = MyMatrix::New();

    ImageToProbeMatrix = MyMatrix::New();
    vtkMatrix4x4::DeepCopy(ImageToProbeMatrix->GetData(), ImageToProbe);

    float norm_x = sqrt(ImageToProbeMatrix->GetElement(0, 0) * ImageToProbeMatrix->GetElement(0, 0) + ImageToProbeMatrix->GetElement(0, 1) * ImageToProbeMatrix->GetElement(0, 1) + ImageToProbeMatrix->GetElement(0, 2) * ImageToProbeMatrix->GetElement(0, 2));
    float norm_y = sqrt(ImageToProbeMatrix->GetElement(1, 0) * ImageToProbeMatrix->GetElement(1, 0) + ImageToProbeMatrix->GetElement(1, 1) * ImageToProbeMatrix->GetElement(1, 1) + ImageToProbeMatrix->GetElement(1, 2) * ImageToProbeMatrix->GetElement(1, 2));
    float norm_z = sqrt(ImageToProbeMatrix->GetElement(2, 0) * ImageToProbeMatrix->GetElement(2, 0) + ImageToProbeMatrix->GetElement(2, 1) * ImageToProbeMatrix->GetElement(2, 1) + ImageToProbeMatrix->GetElement(2, 2) * ImageToProbeMatrix->GetElement(2, 2));

    ImageToProbeMatrix->SetElement(0, 0, ImageToProbeMatrix->GetElement(0, 0) * US_params.pxlSize.x / norm_x);
    ImageToProbeMatrix->SetElement(0, 1, ImageToProbeMatrix->GetElement(0, 1) * US_params.pxlSize.x / norm_x);
    ImageToProbeMatrix->SetElement(0, 2, ImageToProbeMatrix->GetElement(0, 2) * US_params.pxlSize.x / norm_x);
    ImageToProbeMatrix->SetElement(0, 3, ImageToProbeMatrix->GetElement(0, 3) * US_params.pxlSize.x / norm_x);

    ImageToProbeMatrix->SetElement(1, 0, ImageToProbeMatrix->GetElement(1, 0) * US_params.pxlSize.y / norm_y);
    ImageToProbeMatrix->SetElement(1, 1, ImageToProbeMatrix->GetElement(1, 1) * US_params.pxlSize.y / norm_y);
    ImageToProbeMatrix->SetElement(1, 2, ImageToProbeMatrix->GetElement(1, 2) * US_params.pxlSize.y / norm_y);

    ImageToProbeMatrix->SetElement(2, 0, ImageToProbeMatrix->GetElement(2, 0) * US_params.pxlSize.z / norm_z);
    ImageToProbeMatrix->SetElement(2, 1, ImageToProbeMatrix->GetElement(2, 1) * US_params.pxlSize.z / norm_z);
    ImageToProbeMatrix->SetElement(2, 2, ImageToProbeMatrix->GetElement(2, 2) * US_params.pxlSize.z / norm_z);

    std::cout << "ImageToProbe Matrix" << " Norm_x = " << norm_x << ", " << "Norm_y = " << norm_y << ", " << "Norm_z = " << norm_z << std::endl;
    std::cout << "Normalized with pixel size: " << "x - " << US_params.pxlSize.x << ", y - " << US_params.pxlSize.y << ", y - " << US_params.pxlSize.z << std::endl;
    ImageToProbeMatrix->Print(std::cout);

    Volume_Param_Correction(this->Parameters, this->cms_3d_volume::Parameters, this->ImageToVolumeMatrices, ImageToProbeMatrix);
    
    weightingedVolume.resize(cms_3d_volume::Parameters.dim.x * cms_3d_volume::Parameters.dim.y * cms_3d_volume::Parameters.dim.z, 0.0f);
    cms_3d_volume::data.resize(cms_3d_volume::Parameters.dim.x * cms_3d_volume::Parameters.dim.y * cms_3d_volume::Parameters.dim.z, 0.0f);
    cms_3d_volume::XY_Plane.resize(cms_3d_volume::Parameters.dim.x * cms_3d_volume::Parameters.dim.y, 0.0f);
    cms_3d_volume::YZ_Plane.resize(cms_3d_volume::Parameters.dim.y * cms_3d_volume::Parameters.dim.z, 0.0f);
    cms_3d_volume::XZ_Plane.resize(cms_3d_volume::Parameters.dim.x * cms_3d_volume::Parameters.dim.z, 0.0f);
}

void freehand_reconstruction::Parameter_Importer(std::string ParamPath, Ultrasound_Parameters& Extn_US_params, Volume_Parameters& Extn_V_params){
    std::ifstream PARAMETERS;
    PARAMETERS.open(ParamPath, std::ifstream::in);

    std::vector<std::string> LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_US_params.dim.x = std::stoi(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_US_params.dim.y = std::stoi(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_US_params.dim.z = std::stoi(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_US_params.pxlSize.x = std::stof(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_US_params.pxlSize.y = std::stof(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_US_params.pxlSize.z = std::stof(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_US_params.orig.x = std::stof(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_US_params.orig.y = std::stof(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_US_params.orig.z = std::stof(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_US_params.FramesPath = LineCSV[2];

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_US_params.FrameNumber = std::stoi(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_V_params.dim.x = std::stoi(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_V_params.dim.y = std::stoi(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_V_params.dim.z = std::stoi(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_V_params.pxlSize.x = std::stof(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_V_params.pxlSize.y = std::stof(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_V_params.pxlSize.z = std::stof(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_V_params.orig.x = std::stof(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_V_params.orig.y = std::stof(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_V_params.orig.z = std::stof(LineCSV[2]);

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_V_params.VolumePath = LineCSV[2];

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_US_params.TrackerPath = LineCSV[2];

    LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    Extn_US_params.ImageToProbeMatrix = LineCSV;

    PARAMETERS.close();



}

void freehand_reconstruction::ImportTrackingData(){
    //Import all data, number of frames:
    std::ifstream INPUT_TRANSMTX;
    INPUT_TRANSMTX.open(Parameters.TrackerPath, std::ios::in);
    for(int i = 0; i < NumSelectedFrames; ++i){
        std::vector<std::string> LineCSV = getNextLineAndSplitIntoTokens(INPUT_TRANSMTX, ' ');
        MyMatrix ProbeToWordMatrix_local = MyMatrix::New();
        for(int j = 0; j < 16; ++j){
            ProbeToWordMatrix_local->SetElement((j) / 4, ((j) % 4), std::stof(LineCSV[j]));
        }
        ProbeToWordMatrixSet.push_back(ProbeToWordMatrix_local);

        LineCSV = getNextLineAndSplitIntoTokens(INPUT_TRANSMTX, ' ');
        MyMatrix VolumeToWordMatrix_local = MyMatrix::New();
        for(int j = 0; j < 16; ++j){
            VolumeToWordMatrix_local->SetElement((j) / 4, ((j) % 4), std::stof(LineCSV[j]));
        }
        VolumeToWordMatrixSet.push_back(VolumeToWordMatrix_local);
    }
    INPUT_TRANSMTX.close();
}

void freehand_reconstruction::TrackingDataSmoothing(int kernelSize){

    if(kernelSize % 2 == 0){
        std::cout << "Smoothing kernel size should be odd number!" << std::endl;
        exit(0);
    }

    //Initial containers, to store Euler angles for probe. 
    std::vector<float> Eul_X_probe, Eul_Y_probe, Eul_Z_probe;
    {//Process all the matrix to extract euler angles:
        for(int i = 0; i < NumSelectedFrames; ++i){
            //Matrix4x4 to Matrix3x3:
            float Rotm[3][3] = {0.0f}; 
            for(int row = 0; row < 3; ++row){
                for(int col = 0; col < 3; ++col){
                    Rotm[row][col] = ProbeToWordMatrixSet[i]->GetElement(row, col);
                }
            }

            //Rotation matrix to quaternion:
            float Quat[4] = {0.0f};
            vtkMath::Matrix3x3ToQuaternion(Rotm, Quat);

            //Convert quaternions to euler angles:
            float Eul[3] = {0.0f};
            Quat2EulXYZ(Quat, Eul);
            
            //pushback to containers:
            Eul_X_probe.push_back(Eul[0]);
            Eul_Y_probe.push_back(Eul[1]);
            Eul_Z_probe.push_back(Eul[2]);
        }
    }

    //Perform smoothing on euler angles: with the kernelSize(default: 3)
    std::vector<float> Eul_X_probe_smoothed, Eul_Y_probe_smoothed, Eul_Z_probe_smoothed;

    MovingMean(Eul_X_probe, Eul_X_probe_smoothed, kernelSize);
    MovingMean(Eul_Y_probe, Eul_Y_probe_smoothed, kernelSize);
    MovingMean(Eul_Z_probe, Eul_Z_probe_smoothed, kernelSize);
    // {
    //     int sidetailSize = (kernelSize - 1) / 2;

    //     Eul_X_probe_smoothed.resize(Eul_X_probe.size());
    //     Eul_Y_probe_smoothed.resize(Eul_Y_probe.size());
    //     Eul_Z_probe_smoothed.resize(Eul_Z_probe.size());

    //     //Euler angle loop: iterate for every angle; 
    //     //copy the beginning and end tail, smoothen the middle data with the kernel. 
    //     for(int it_eul = 0; it_eul < Eul_X_probe.size(); ++it_eul){
    //         //within the middle range kernel can smooth:
    //         if(it_eul >= sidetailSize && it_eul < Eul_X_probe.size() - sidetailSize){
    //             //kernel loop:
    //             float acc_x = 0.0f, acc_y = 0.0f, acc_z = 0.0f;
    //             for(int it_kernel = -sidetailSize; it_kernel <= sidetailSize; ++it_kernel){
    //                 int selected_eul_idx = it_kernel + it_eul;
    //                 acc_x += Eul_X_probe[selected_eul_idx];
    //                 acc_y += Eul_Y_probe[selected_eul_idx];
    //                 acc_z += Eul_Z_probe[selected_eul_idx];
    //             }
    //             //update smoothed Euler angle:
    //             Eul_X_probe_smoothed[it_eul] = acc_x / kernelSize;
    //             Eul_Y_probe_smoothed[it_eul] = acc_y / kernelSize;
    //             Eul_Z_probe_smoothed[it_eul] = acc_z / kernelSize;
    //         }
    //         //Out of the middle range, copy the original data:
    //         else{
    //             Eul_X_probe_smoothed[it_eul] = Eul_X_probe[it_eul];
    //             Eul_Y_probe_smoothed[it_eul] = Eul_Y_probe[it_eul];
    //             Eul_Z_probe_smoothed[it_eul] = Eul_Z_probe[it_eul];
    //         }
    //     }
    // }
    
    //Convert smoothed Euler angles back to rotation matrix:
    {
        for(int it_probe2wMatrix = 0; it_probe2wMatrix < ProbeToWordMatrixSet.size(); ++it_probe2wMatrix){
            //Eul to quaternion:
            float eul[3] = {Eul_X_probe_smoothed[it_probe2wMatrix], Eul_Y_probe_smoothed[it_probe2wMatrix], Eul_Z_probe_smoothed[it_probe2wMatrix]};
            float quat[4] = {0.0f};
            EulXYZ2Quat(eul, quat);

            //quaternion to rotation matrix:
            float rotm[3][3];
            vtkMath::QuaternionToMatrix3x3(quat, rotm);

            //Over-write the smoothed results to the MatrixSet members:
            for(int it_row = 0; it_row < 3; ++it_row){
                for(int it_col = 0; it_col < 3; ++it_col){
                    ProbeToWordMatrixSet[it_probe2wMatrix]->SetElement(it_row, it_col, rotm[it_row][it_col]);
                }
            }
        }
        std::cout << "Smoothing with kernelSize " << kernelSize << " has been finished" << std::endl;
    }
}

void freehand_reconstruction::USFrame_Distribution(const MyMatrix& TrackedProbe, const MyMatrix& TrackedVolume, const uint8_t* US_Frame){
    vtkMatrix4x4::DeepCopy(ProbeToWordMatrix->GetData(), TrackedProbe->GetData());
    vtkMatrix4x4::DeepCopy(VolumeToWordMatrix->GetData(), TrackedVolume->GetData());
    AllMatricesReady = true;

    if(AllMatricesReady != true){
        std::cout << "Please set matrices first!" << std::endl;
        return;
    }

    MyMatrix ImageToVolumeMatrix = MyMatrix::New();
    {   
        //Calculate to world transform matrix:
        MyMatrix ImageToWorld = MyMatrix::New();
        vtkMatrix4x4::Multiply4x4(ProbeToWordMatrix, ImageToProbeMatrix, ImageToWorld);

        //World to volume:
        MyMatrix WorldToVolumeMatrix = MyMatrix::New();
        vtkMatrix4x4::Invert(VolumeToWordMatrix, WorldToVolumeMatrix);

        //Image To Volume:
        vtkMatrix4x4::Multiply4x4(WorldToVolumeMatrix, ImageToWorld, ImageToVolumeMatrix);
    }

    //US pixels loop:
    for(int it_row = 0; it_row < Parameters.dim.y; ++it_row){
        for(int it_col = 0; it_col < Parameters.dim.x; ++it_col){
            //column -> x, row -> y, z = 0, form a volumetric point (in mm):
            float Point_US[4] = {float(it_col), float(it_row), 0.0, 1.0};
            float Point_US_Volume[4] = {0.0}; 
            ImageToVolumeMatrix->MultiplyPoint(Point_US, Point_US_Volume);

            //Convert volume in [mm] to [vxl], iterate surroundings and update:
            float Voxel_US_Volume[4] = { (Point_US_Volume[0] - GetOrigX()) / GetPxlX(), (Point_US_Volume[1] - GetOrigY()) / GetPxlY(), (Point_US_Volume[2] - GetOrigZ()) / GetPxlZ(), 1.0};
            {//update voxels:
                //iterate eight surrounding voxels, around the target pixel(in mm, not rounded pixel)

                //Select voxel: with different methods, 1. nearest, TODO. alternatives
                float Selected_Voxel[3] = {round(Voxel_US_Volume[0]), round(Voxel_US_Volume[1]), round(Voxel_US_Volume[2])};
                // std::cout << Selected_Voxel[0] << ", " << Selected_Voxel[1] << ", " << Selected_Voxel[2] << std::endl;
                //Iteration start:
                {
                    for(int it_i = -1; it_i < 2; ++it_i){
                        for(int it_j = -1; it_j < 2; ++it_j){
                            for(int it_k = -1; it_k < 2; ++it_k){
                                int select_x = Selected_Voxel[0] + it_i;
                                int select_y = Selected_Voxel[1] + it_j;
                                int select_z = Selected_Voxel[2] + it_k;
                                if( (select_x < GetDimX()) && (select_x >= 0) &&
                                    (select_y < GetDimY()) && (select_y >= 0) &&
                                    (select_z < GetDimZ()) && (select_z >= 0)){
                                    //Calculate inverse distance: [voxel] to US [pixel]:
                                    float inv_distance = expf( -sqrt(( (select_x * GetPxlX() + GetOrigX()) - Point_US_Volume[0] ) * 
                                                        ( (select_x * GetPxlX() + GetOrigX()) - Point_US_Volume[0] ) + 
                                                        ( (select_y * GetPxlY() + GetOrigY()) - Point_US_Volume[1] ) * 
                                                        ( (select_y * GetPxlY() + GetOrigY()) - Point_US_Volume[1] ) + 
                                                        ( (select_z * GetPxlZ() + GetOrigZ()) - Point_US_Volume[2] ) * 
                                                        ( (select_z * GetPxlZ() + GetOrigZ()) - Point_US_Volume[2] )));
                                    // float inv_distance = 1.0 / sqrt(( (select_x * GetPxlX() + GetOrigX()) - Point_US_Volume[0] ) * 
                                    //                     ( (select_x * GetPxlX() + GetOrigX()) - Point_US_Volume[0] ) + 
                                    //                     ( (select_y * GetPxlY() + GetOrigY()) - Point_US_Volume[1] ) * 
                                    //                     ( (select_y * GetPxlY() + GetOrigY()) - Point_US_Volume[1] ) + 
                                    //                     ( (select_z * GetPxlZ() + GetOrigZ()) - Point_US_Volume[2] ) * 
                                    //                     ( (select_z * GetPxlZ() + GetOrigZ()) - Point_US_Volume[2] ));
                                    // if( inv_distance > INF){
                                    //     weightingedVolume[select_x + select_y * GetDimX() + select_z * GetDimX() * GetDimY()] = INF;
                                    //     cms_3d_volume::SetVoxel(select_x, select_y, select_z, US_Frame[it_col + it_row * Parameters.dim.x]);
                                    // }
                                    // else{
                                        float sum = cms_3d_volume::GetVoxel(select_x, select_y, select_z) * weightingedVolume[select_x + select_y * GetDimX() + select_z * GetDimX() * GetDimY()] + 
                                                    US_Frame[it_col + it_row * Parameters.dim.x] * inv_distance;
                                        weightingedVolume[select_x + select_y * GetDimX() + select_z * GetDimX() * GetDimY()] += inv_distance;
                                        cms_3d_volume::SetVoxel(select_x, select_y, select_z, sum / weightingedVolume[select_x + select_y * GetDimX() + select_z * GetDimX() * GetDimY()]);
                                    // }
                                }
                            }
                        }
                    }
                }

                // { //Only the nearest voxel being selected:
                //     int select_x = Selected_Voxel[0];
                //     int select_y = Selected_Voxel[1];
                //     int select_z = Selected_Voxel[2];
                //     if( (select_x < GetDimX()) && (select_x >= 0) &&
                //         (select_y < GetDimY()) && (select_y >= 0) &&
                //         (select_z < GetDimZ()) && (select_z >= 0)){
                //         //Calculate inverse distance: [voxel] to US [pixel]:
                //         float inv_distance = 1.0 / sqrt(( (select_x * GetPxlX() + GetOrigX()) - Point_US_Volume[0] ) * 
                //                             ( (select_x * GetPxlX() + GetOrigX()) - Point_US_Volume[0] ) + 
                //                             ( (select_y * GetPxlY() + GetOrigY()) - Point_US_Volume[1] ) * 
                //                             ( (select_y * GetPxlY() + GetOrigY()) - Point_US_Volume[1] ) + 
                //                             ( (select_z * GetPxlZ() + GetOrigZ()) - Point_US_Volume[2] ) * 
                //                             ( (select_z * GetPxlZ() + GetOrigZ()) - Point_US_Volume[2] ));
                //         if( inv_distance > INF){
                //             weightingedVolume[select_x + select_y * GetDimX() + select_z * GetDimX() * GetDimY()] = INF;
                //             cms_3d_volume::SetVoxel(select_x, select_y, select_z, US_Frame[it_col + it_row * Parameters.dim.x]);
                //         }
                //         else{
                //             float sum = cms_3d_volume::GetVoxel(select_x, select_y, select_z) * weightingedVolume[select_x + select_y * GetDimX() + select_z * GetDimX() * GetDimY()] + 
                //                         US_Frame[it_col + it_row * Parameters.dim.x] * inv_distance;
                //             weightingedVolume[select_x + select_y * GetDimX() + select_z * GetDimX() * GetDimY()] += inv_distance;
                //             cms_3d_volume::SetVoxel(select_x, select_y, select_z, sum / weightingedVolume[select_x + select_y * GetDimX() + select_z * GetDimX() * GetDimY()]);
                //         }
                //     }
                // }

            }
        }
    }
    AllMatricesReady = false;
}

void freehand_reconstruction::HoleFilling(){ 
    std::vector<float> Volume_Cpy(this->cms_3d_volume::data);

    std::cout << "Start hole filling......" << std::endl;
    ProgressBar progressBar(GetDimX(), 70);
    //Iterations for every voxel in order:
    for(int it_vxl_x = 0; it_vxl_x < GetDimX(); ++it_vxl_x){
        for(int it_vxl_y = 0; it_vxl_y < GetDimY(); ++it_vxl_y){
            for(int it_vxl_z = 0; it_vxl_z < GetDimZ(); ++it_vxl_z){
                //If current voxel NOT zero, then perform Hole Filling:
                if(weightingedVolume[it_vxl_x + it_vxl_y * GetDimX() + it_vxl_z * GetDimX() * GetDimY()] == 0.0f){
                    //filling with the average kernel of 3, around target voxel:
                    int Acc_NotZeroVoxel = 0;
                    float Acc_Sum = 0.0;
                    {//Accumulate the not zero voxels:
                        for(int k_x = -1; k_x < 2; ++k_x){
                            for(int k_y = -1; k_y < 2; ++k_y){
                                for(int k_z = -1; k_z < 2; ++k_z){
                                    //check if the current selected pixel [x, y, z] is out of bound:
                                    int Selected_Pxl[3] = {it_vxl_x + k_x, it_vxl_y + k_y, it_vxl_z + k_z};
                                    if( 
                                        (Selected_Pxl[0] >=0 && Selected_Pxl[0] < GetDimX()) && 
                                        (Selected_Pxl[1] >=0 && Selected_Pxl[1] < GetDimY()) && 
                                        (Selected_Pxl[2] >=0 && Selected_Pxl[2] < GetDimZ()) ){
                                        //is the current selected pxl ZERO?
                                        if( Volume_Cpy[Selected_Pxl[0] + Selected_Pxl[1] * GetDimX() + Selected_Pxl[2] * GetDimX() * GetDimY()] != 0.0f){ 
                                            //Not Zero, to be selected for hole filling:
                                            Acc_Sum += Volume_Cpy[Selected_Pxl[0] + Selected_Pxl[1] * GetDimX() + Selected_Pxl[2] * GetDimX() * GetDimY()];
                                            ++Acc_NotZeroVoxel;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    //Filling:
                    if(Acc_NotZeroVoxel > 0)
                        SetVoxel(it_vxl_x, it_vxl_y, it_vxl_z, Acc_Sum / Acc_NotZeroVoxel);
                }
                else{
                    continue;
                }
            }
        }
        ++progressBar;
        progressBar.display();
    }
    progressBar.done();
}

void freehand_reconstruction::SetMatrix(std::ifstream& INPUT_TRANSMTX, MyMatrix& ImageToVolumeMatrix, int idxSelectedFrames){
    //Read Probe tracking data:
    // std::vector<std::string> LineCSV = getNextLineAndSplitIntoTokens(INPUT_TRANSMTX, ' ');
    // for(int j = 0; j < 16; ++j){
    //     ProbeToWordMatrix->SetElement((j) / 4, ((j) % 4), std::stof(LineCSV[j]));
    // }
    ProbeToWordMatrix->DeepCopy(ProbeToWordMatrixSet[idxSelectedFrames]);

    //Read Phantom tracking data:
    // LineCSV = getNextLineAndSplitIntoTokens(INPUT_TRANSMTX, ' ');
    // for(int j = 0; j < 16; ++j){
    //     VolumeToWordMatrix->SetElement((j) / 4, ((j) % 4), std::stof(LineCSV[j]));
    // }
    VolumeToWordMatrix->DeepCopy(VolumeToWordMatrixSet[idxSelectedFrames]);

    {   
        //Calculate to world transform matrix:
        MyMatrix ImageToWorld = MyMatrix::New();
        vtkMatrix4x4::Multiply4x4(ProbeToWordMatrix, ImageToProbeMatrix, ImageToWorld);

        //World to volume:
        MyMatrix WorldToVolumeMatrix = MyMatrix::New();
        vtkMatrix4x4::Invert(VolumeToWordMatrix, WorldToVolumeMatrix);

        //Image To Volume:
        vtkMatrix4x4::Multiply4x4(WorldToVolumeMatrix, ImageToWorld, ImageToVolumeMatrix);
    }

}

void freehand_reconstruction::PerformFrameDistributionOnGPU(){
    if(Enable_CUDA){
        //Setup all matrices:
        std::ifstream INPUT_TRANSMTX;
        INPUT_TRANSMTX.open(Parameters.TrackerPath, std::ifstream::in);
        float TotalMatrices[NumSelectedFrames * 16];

        MyMatrix ImageToVolumeMatrix = MyMatrix::New();
        for(int i = 0; i < NumSelectedFrames; ++i){
            ImageToVolumeMatrix->Identity();
            SetMatrix(INPUT_TRANSMTX, ImageToVolumeMatrix, i);
            for(int element_it = 0; element_it < 16; ++element_it){
                TotalMatrices[element_it + i * 16] = float(ImageToVolumeMatrix->GetData()[element_it]);
            }
        }
        INPUT_TRANSMTX.close();

        //Import all US frames:
        uint8_t* US_Frame = new uint8_t[Parameters.dim.x * Parameters.dim.y * NumSelectedFrames];
        readFromBin(US_Frame, Parameters.dim.x * Parameters.dim.y * NumSelectedFrames, Parameters.FramesPath);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        GPU_Setups(Enable_CUDA, Parameters, cms_3d_volume::Parameters, TotalMatrices, &cms_3d_volume::data[0], &weightingedVolume[0], US_Frame);
        std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
        std::cout << "GPU calculation time (sec) = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 <<std::endl;
        //Free memory:
        delete[] US_Frame;
        
    }
    else{
        std::cout << "ERROR: CUDA is not enabled in your platform." << std::endl;
    }
}
