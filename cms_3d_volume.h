#ifndef CMS_3D_VOLUME
#define CMS_3D_VOLUME

#include <fstream>
#include <string>
#include "BaseImgParams.h"

class Img_Plane{

public:
    void show(){
        cv::Mat Image_obj(dim_i * dim_j, 1, CV_32F, plane_data.data());
        cv::normalize(Image_obj, Image_obj, 0, 1, cv::NORM_MINMAX);
        cv::imshow("ImagePlane Show", Image_obj);
        cv::waitKey(0);
    }

    void SetDim(const int i, const int j){
        plane_data.resize(i*j, 0.0f);
        dim_i = i;
        dim_j = j;
    }

    void SetValue(const int i, const int j, const float Value){
        plane_data[i + j * dim_i] = Value;
    }

private:
    std::vector<float> plane_data;
    int dim_i, dim_j;
};

class cms_3d_volume{

public:
    cms_3d_volume() : Data_Valid(false){}

    cms_3d_volume(const Volume_Parameters params) : 
        Parameters(params), 
        Data_Valid(true), 
        Data_Reconstructed(false), 
        data(params.dim.x * params.dim.y * params.dim.z, 0.0f), 
        XY_Plane(params.dim.x * params.dim.y, 0.0f), 
        XZ_Plane(params.dim.x * params.dim.z, 0.0f), 
        YZ_Plane(params.dim.y * params.dim.z, 0.0f){

        // XY_Plane.SetDim(params.dim.x, params.dim.y);
        std::cout << "Volume constructed!" << " with size of " << data.size() << std::endl;
    }

    cms_3d_volume(std::string InputFilePath, const Volume_Parameters params) : 
        Data_Reconstructed(true), 
        Parameters(params), 
        Data_Valid(true), 
        data(params.dim.x * params.dim.y * params.dim.z, 0.0f), 
        XY_Plane(params.dim.x * params.dim.y, 0.0f), 
        XZ_Plane(params.dim.x * params.dim.z, 0.0f), 
        YZ_Plane(params.dim.y * params.dim.z, 0.0f){ 
        
        readFromBin(&data[0], params.dim.x * params.dim.y * params.dim.z, InputFilePath);
        std::cout << "External reconstructed volume imported!" << " from path: " << InputFilePath << std::endl;
    }

    ~cms_3d_volume(){
        Data_Valid = false;
    }

    int GetDimX(){ return Parameters.dim.x; }
    int GetDimY(){ return Parameters.dim.y; }
    int GetDimZ(){ return Parameters.dim.z; }

    float GetPxlX(){ return Parameters.pxlSize.x; }
    float GetPxlY(){ return Parameters.pxlSize.y; }
    float GetPxlZ(){ return Parameters.pxlSize.z; }

    float GetOrigX(){ return Parameters.orig.x; }
    float GetOrigY(){ return Parameters.orig.y; }
    float GetOrigZ(){ return Parameters.orig.z; }

    void SetVoxel(int i, int j, int k, float V_value){
        if(Data_Valid){
            data[i + j * Parameters.dim.x + k * Parameters.dim.x * Parameters.dim.y] = V_value;
        }
        else{
            std::cout << "Please check volume data valibilidy!" << std::endl;
        }
    }

    float GetVoxel(int i, int j, int k){
        if(Data_Valid){
            return data[i + j * Parameters.dim.x + k * Parameters.dim.x * Parameters.dim.y];
        }
        else{
            std::cout << "Please check volume data valibilidy!" << std::endl;
        }
    }

    std::vector<float> Get_XY_Plane(const int z){
        if(Data_Valid){
            std::copy(data.begin() + Parameters.dim.x * Parameters.dim.y * z, data.begin() + Parameters.dim.x * Parameters.dim.y * (z + 1), XY_Plane.data());
            return XY_Plane;
        }
        else{
            std::cout << "Please check volume data valibilidy!" << std::endl;
        }
    }

    std::vector<float> Get_YZ_Plane(const int x){
        if(Data_Valid){
            {
                for(int it_z = 0; it_z < Parameters.dim.z; ++it_z){
                    for(int it_y = 0; it_y < Parameters.dim.y; ++it_y){
                        YZ_Plane[it_y + it_z * Parameters.dim.y] = data[x + it_y * Parameters.dim.x + it_z * Parameters.dim.x * Parameters.dim.y];
                    }
                }
            }
            return YZ_Plane;
        }
        else{
            std::cout << "Please check volume data valibilidy!" << std::endl;
        }
    }

    std::vector<float> Get_XZ_Plane(const int y){
        if(Data_Valid){
            {
                for(int it_z = 0; it_z < Parameters.dim.z; ++it_z){
                    for(int it_x = 0; it_x < Parameters.dim.x; ++it_x){
                        XZ_Plane[it_x + it_z * Parameters.dim.x] = data[it_x + y * Parameters.dim.x + it_z * Parameters.dim.x * Parameters.dim.y];
                    }
                }
            }
            return XZ_Plane;
        }
        else{
            std::cout << "Please check volume data valibilidy!" << std::endl;
        }
    }

    void Show_XY(const int z){
        cv::Mat Image_obj(Parameters.dim.x, Parameters.dim.y, CV_32F, Get_XY_Plane(z).data());
        cv::normalize(Image_obj, Image_obj, 0, 1, cv::NORM_MINMAX);
        cv::imshow("XY - Plane Show", Image_obj);
        cv::waitKey(0);
    }

    void Show_YZ(const int x){
        cv::Mat Image_obj(Parameters.dim.y, Parameters.dim.z, CV_32F, Get_YZ_Plane(x).data());
        cv::normalize(Image_obj, Image_obj, 0, 1, cv::NORM_MINMAX);
        cv::imshow("YZ - Plane Show", Image_obj);
        cv::waitKey(0);
    }

    void Show_XZ(const int y){
        cv::Mat Image_obj(Parameters.dim.x, Parameters.dim.z, CV_32F, Get_XZ_Plane(y).data());
        cv::normalize(Image_obj, Image_obj, 0, 1, cv::NORM_MINMAX);
        cv::imshow("XZ - Plane Show", Image_obj);
        cv::waitKey(0);
    }

    void SaveVolume(){
        //Export header:
        std::ofstream Export_NRRD;
        Export_NRRD.open(Parameters.VolumePath, std::ofstream::out);

        Export_NRRD << "NRRD0004" << std::endl;
        Export_NRRD << "type: float" << std::endl;
        Export_NRRD << "dimension: 3" << std::endl;
        Export_NRRD << "space: left-posterior-superior" << std::endl;
        Export_NRRD << "sizes: " << Parameters.dim.x << " " << Parameters.dim.y << " " << Parameters.dim.z << std::endl;
        Export_NRRD << "space directions: (1,0,0) (0,1,0) (0,0,1)" << std::endl;
        Export_NRRD << "kinds: domain domain domain" << std::endl;
        Export_NRRD << "endian: little" << std::endl;
        Export_NRRD << "encoding: raw" << std::endl;
        Export_NRRD << "space origin: (0,0,0)" << std::endl << std::endl;

        Export_NRRD.close();

        //Export volume data:
        writeToBin(&data[0], Parameters.dim.x * Parameters.dim.y * Parameters.dim.z, Parameters.VolumePath);
        std::cout << "Reconstructed volume was saved at:  " << Parameters.VolumePath << std::endl;

        //Export report: 
        std::string ReportPath;
        {
            auto found = Parameters.VolumePath.find_last_of("/\\");
            ReportPath = Parameters.VolumePath.substr(0, found) + "/recon_report.txt";
        }
        std::ofstream Report_output;
        Report_output.open(ReportPath, std::ofstream::out | std::ofstream::trunc);
        Report_output << "ReconVolumeOrigin_X = " << Parameters.orig.x << std::endl;
        Report_output << "ReconVolumeOrigin_Y = " << Parameters.orig.y << std::endl;
        Report_output << "ReconVolumeOrigin_Z = " << Parameters.orig.z << std::endl;
        Report_output << "ReconVolumePixelSize_X = " << Parameters.pxlSize.x << std::endl;
        Report_output << "ReconVolumePixelSize_Y = " << Parameters.pxlSize.y << std::endl;
        Report_output << "ReconVolumePixelSize_Z = " << Parameters.pxlSize.z << std::endl;
        Report_output << "ReconVolumeDimension_X = " << Parameters.dim.x << std::endl;
        Report_output << "ReconVolumeDimension_Y = " << Parameters.dim.y << std::endl;
        Report_output << "ReconVolumeDimension_Z = " << Parameters.dim.z << std::endl;
        Report_output.close();
        std::cout << "Reconstruction report was saved at: " << ReportPath << std::endl;
    }

    float* GetRawPtr(){ return &data[0]; }

    bool IsVolumeValid(){ return Data_Valid; }

private:
    cms_3d_volume(const cms_3d_volume&) = delete;
    void operator=(const cms_3d_volume&) = delete;
    

protected:
    //Parameters:
    Volume_Parameters Parameters;

    //Data:
    std::vector<float> data;


    // Img_Plane XY_Plane
    std::vector<float> XY_Plane;
    std::vector<float> XZ_Plane;
    std::vector<float> YZ_Plane;

    //Flags:
    bool Data_Valid;
    bool Data_Reconstructed;

    //volume PATH:
    std::string Volume_PATH;

};

#endif