#ifndef BASEPARAM
#define BASEPARAM

#include <string>
#include <vector>

struct Dim{
    int x;
    int y;
    int z;
};

struct PixelSize{
    float x;
    float y;
    float z;
};

struct Origin{
    float x;
    float y;
    float z;
};

struct Volume_Parameters{
    Dim dim;
    PixelSize pxlSize;
    Origin orig;
    std::string VolumePath;
};

struct Ultrasound_Parameters{
    Dim dim;
    PixelSize pxlSize;
    Origin orig;
    std::string FramesPath;
    std::string TrackerPath;
    int FrameNumber;
    std::vector<std::string> ImageToProbeMatrix;
};

#endif