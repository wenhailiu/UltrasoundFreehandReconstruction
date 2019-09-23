#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "vtkObjectFactory.h"
#include "vtkMatrix4x4.h"
#include "vtkMath.h"

#include "mylibs.h"
#include "cxxopts.hpp"

#include "BaseImgParams.h"
#include "cms_3d_volume.h"
#include "freehand_reconstruction.h"

int smoothKernel;

void Recon_On_CPU(std::string ParameterPath){

    /* ----------------------------------- Parameter Initialization: ---------------------------- */
    Ultrasound_Parameters US_Parameters;
    Volume_Parameters Vol_Parameters;
    freehand_reconstruction::Parameter_Importer(ParameterPath, US_Parameters, Vol_Parameters);

    //Perform Reconstruction:
    freehand_reconstruction FreehandRecon(US_Parameters, Vol_Parameters);
    MyMatrix ProbeToWorldMatrix = MyMatrix::New();
    MyMatrix PhantomToWorld = MyMatrix::New();

    uint8_t* US_Frame = new uint8_t[US_Parameters.dim.x * US_Parameters.dim.y * US_Parameters.FrameNumber];
    readFromBin(US_Frame, US_Parameters.dim.x * US_Parameters.dim.y * US_Parameters.FrameNumber, US_Parameters.FramesPath);

    std::ifstream INPUT_TRANSMTX;
    INPUT_TRANSMTX.open(US_Parameters.TrackerPath, std::ifstream::in);

    //Initialize bar:
    std::cout << "Ultrasound frames are distributed......" << std::endl;
    ProgressBar progressBar(US_Parameters.FrameNumber, 70);

    for(int i = 0; i < US_Parameters.FrameNumber; ++i){
        //Read Probe tracking data:
        std::vector<std::string> LineCSV = getNextLineAndSplitIntoTokens(INPUT_TRANSMTX, ' ');
        for(int j = 0; j < 16; ++j){
            ProbeToWorldMatrix->SetElement((j) / 4, ((j) % 4), std::stof(LineCSV[j]));
        }
        // ProbeToWorldMatrix->Print(std::cout);

        //Read Phantom tracking data:
        LineCSV = getNextLineAndSplitIntoTokens(INPUT_TRANSMTX, ' ');
        for(int j = 0; j < 16; ++j){
            PhantomToWorld->SetElement((j) / 4, ((j) % 4), std::stof(LineCSV[j]));
        }
        // PhantomToWorld->Print(std::cout);
        
        // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        FreehandRecon.USFrame_Distribution(ProbeToWorldMatrix, PhantomToWorld, US_Frame + (US_Parameters.dim.x * US_Parameters.dim.y) * i);
        // std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
        // std::cout << "Time difference (sec) = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 <<std::endl;
        ++progressBar;
        progressBar.display();
    }
    progressBar.done();

    FreehandRecon.HoleFilling();
    
    INPUT_TRANSMTX.close();

    FreehandRecon.SaveVolume();

    delete[] US_Frame;
    
}

void Recon_On_GPU(std::string ParameterPath){
    Ultrasound_Parameters US_Parameters;
    Volume_Parameters Vol_Parameters;
    freehand_reconstruction::Parameter_Importer(ParameterPath, US_Parameters, Vol_Parameters);
    /* ----------------------------------- Parameter Initialization End: ---------------------------- */
    {//GPU program testing:
        freehand_reconstruction FreehandRecon(US_Parameters, Vol_Parameters, true, US_Parameters.FrameNumber);
        FreehandRecon.ImportTrackingData();
        FreehandRecon.TrackingDataSmoothing(smoothKernel);
        FreehandRecon.PerformFrameDistributionOnGPU();
        FreehandRecon.SaveVolume();
        return; 
    }
}

int main(int argc, char* argv[]){

    cxxopts::Options options("UltraRecon", "CMS freehand ultrasound images reconstructor");

    options.add_options() 
    ("p,parameters", "Parameter File Name", cxxopts::value<std::string>()) 
    ("G,EnableGPU", "Enable GPU or not.")
    ("S,smooth", "smoothing kernel size", cxxopts::value<int>());

    auto result = options.parse(argc, argv);

    std::string parameterPath = result["parameters"].as<std::string>();
    bool Platform_Spec = result["EnableGPU"].as<bool>();
    smoothKernel = result["smooth"].as<int>();
    // std::string parameterPath = "/home/wenhai/vsc_workspace/Freehand3DUSReconstruction/Parameters.txt";

    if(Platform_Spec){
        Recon_On_GPU(parameterPath);
    }
    else{
        Recon_On_CPU(parameterPath);
    }

    return 0;
}