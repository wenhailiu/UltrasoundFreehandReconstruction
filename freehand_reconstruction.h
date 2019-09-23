#ifndef FREEHAND3DRECON
#define FREEHAND3DRECON

#include "BaseImgParams.h"
#include "cms_3d_volume.h"

#include "vtkObjectFactory.h"
#include "vtkMatrix4x4.h"
#include "vtkMath.h"

#define KERNEL_SIZE 8
#define INF 99999

typedef vtkSmartPointer<vtkMatrix4x4> MyMatrix;

// Filling in Matrix:
// double ImageToProbe[16] = { 
//     -0.3054, 0.0094, -0.0382, 67.3653, 
// 0.0073, 0.3128, 0.0166, 72.3705, 
// 0.0383, 0.0157, -0.3078, -45.2306, 
// 0, 0, 0, 1}; 

// double ImageToProbe[16] = { 
//     -0.243132, 0.145637, -0.22401, 108.204, 
//     0.210999, -0.0791473, -0.282343, -119.189, 
//     -0.160964, -0.323731, -0.031747, -17.2477, 
//     0, 0, 0, 1}; 


class freehand_reconstruction: public cms_3d_volume{
public: 
    //intialize without GPU
    freehand_reconstruction(const Ultrasound_Parameters, const Volume_Parameters);
    //with GPU enabled:
    freehand_reconstruction(const Ultrasound_Parameters, const Volume_Parameters, bool, int);

    static void Parameter_Importer(std::string ParamPath, Ultrasound_Parameters&, Volume_Parameters&);

    void ImportTrackingData();

    void TrackingDataSmoothing(int kernelSize = 1);

    int GetUSDimX(){ return Parameters.dim.x; }
    int GetUSDimY(){ return Parameters.dim.y; }

    int GetUSPxlX(){ return Parameters.pxlSize.x; }
    int GetUSPxlY(){ return Parameters.pxlSize.y; }
    int GetUSPxlZ(){ return Parameters.pxlSize.z; }

    void SetMatrices(const MyMatrix& TrackedProbe, const MyMatrix& TrackedVolume){
        vtkMatrix4x4::DeepCopy(ProbeToWordMatrix->GetData(), TrackedProbe->GetData());
        vtkMatrix4x4::DeepCopy(VolumeToWordMatrix->GetData(), TrackedVolume->GetData());
        AllMatricesReady = true;
    }

    void SetMatrix(std::ifstream&, MyMatrix&, int);

    void PerformFrameDistributionOnGPU();

// #ifdef __CUDACC__
    void USFrame_Distribution_GPU();

    void HoleFilling_GPU(bool);

// #else
    void USFrame_Distribution(const MyMatrix&, const MyMatrix&, const uint8_t*);

    void HoleFilling();
// #endif

private:
    freehand_reconstruction(const freehand_reconstruction&) = delete;
    void operator= (const freehand_reconstruction&) = delete;

private:
    Ultrasound_Parameters Parameters;

    bool AllMatricesReady;
    bool Enable_CUDA;

    MyMatrix ImageToProbeMatrix;
    MyMatrix ProbeToWordMatrix;
    MyMatrix VolumeToWordMatrix;

    std::vector<float> weightingedVolume;
    std::vector<float> ImageToVolumeMatrices;

    std::vector<MyMatrix> ProbeToWordMatrixSet; 
    std::vector<MyMatrix> VolumeToWordMatrixSet;

    const int NumSelectedFrames;

};
#endif