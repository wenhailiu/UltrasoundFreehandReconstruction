#ifndef MYLIBS
#define MYLIBS

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <chrono>

#define MYSECOND      1
#define MYMILLISECOND 2
#define MYMICROSECOND 3
#define MYNANOSECOND  4

/*
Parse csv file, split by SPLITTER, which could be ' ', or ','

Example usage:
    std::ifstream PARAMETERS;
    PARAMETERS.open(ParameterPath, std::ifstream::in);
    std::vector<std::string> LineCSV = getNextLineAndSplitIntoTokens(PARAMETERS, ' ');
    PARAMETERS.close();
 */
inline std::vector<std::string> getNextLineAndSplitIntoTokens(std::istream& str, const char SPLITTER)
{
    std::vector<std::string>   result;
    std::string                line;
    std::getline(str,line);

    std::stringstream          lineStream(line);
    std::string                cell;

    while(std::getline(lineStream,cell, SPLITTER))
    {
        result.push_back(cell);
    }
    // This checks for a trailing comma with no data after it.
    if (!lineStream && cell.empty())
    {
        // If there was a trailing comma then add an empty element.
        result.push_back("");
    }
    return result;
}

template<typename T> 
extern void readFromBin(T *Output, int Num_Elements, const std::string FILENAME) {
	std::ifstream InputStream;
	InputStream.open(FILENAME, std::ios::in | std::ios::binary);

	if (!InputStream.good()) {
		std::cout << "Failed to open " << FILENAME << std::endl;
		exit(0);
	}

	InputStream.read(reinterpret_cast<char*>(Output), sizeof(T) * Num_Elements);
	Output = reinterpret_cast<T*>(Output);

	InputStream.close();
}

template<typename T>
extern void writeToBin(T *Output, int Num_Elements, const std::string FILENAME) {
	std::ofstream OutputStream;
	OutputStream.open(FILENAME, std::ios::app | std::ios::binary);

	if (!OutputStream.good()) {
		std::cout << "Failed to open " << FILENAME << std::endl;
		exit(0);
	}

	OutputStream.write(reinterpret_cast<char*>(Output), sizeof(T) * Num_Elements);

	OutputStream.close();
}


//Progress Bar:
/*
Usage:
The bar takes the following options at initialization:

    -Limit: the total number of ticks that need to be completed
    -Width: width of the bar
    +Complete Char: the character to indicate completion (defaults to =)
    +Incomplete Char: the character to indicate pending. (defaults to ' ')

Example Program:

#include "ProgressBar.hpp"
int main() {

    const int limit = 10000;

    // initialize the bar
    ProgressBar progressBar(limit, 70);

    for (int i = 0; i < limit; i++) {
        // record the tick
        ++progressBar;

        // display the bar
        progressBar.display();
    }

    // tell the bar to finish
    progressBar.done();
}
*/
class ProgressBar {
private:
    unsigned int ticks = 0;

    const unsigned int total_ticks;
    const unsigned int bar_width;
    const char complete_char = '=';
    const char incomplete_char = ' ';
    const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

public:
    ProgressBar(unsigned int total, unsigned int width, char complete, char incomplete) :
            total_ticks {total}, bar_width {width}, complete_char {complete}, incomplete_char {incomplete} {}

    ProgressBar(unsigned int total, unsigned int width) : total_ticks {total}, bar_width {width} {}

    unsigned int operator++() { return ++ticks; }

    void display() const
    {
        float progress = (float) ticks / total_ticks;
        int pos = (int) (bar_width * progress);

        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now-start_time).count();

        std::cout << "[";

        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << complete_char;
            else if (i == pos) std::cout << ">";
            else std::cout << incomplete_char;
        }
        std::cout << "] " << int(progress * 100.0) << "% "
                  << float(time_elapsed) / 1000.0 << "s\r";
        std::cout.flush();
    }

    void done() const
    {
        display();
        std::cout << std::endl;
    }
};

template<typename TimeT = std::chrono::milliseconds>
struct measure
{
    template<typename F, typename ...Args>
    static typename TimeT::rep execution(F func, Args&&... args)
    {
        auto start = std::chrono::system_clock::now();
 
        // Now call the function with all the parameters you need.
        func(std::forward<Args>(args)...);
 
        auto duration = std::chrono::duration_cast< TimeT> 
                            (std::chrono::system_clock::now() - start);
        return duration.count();
    }

    template<typename F, typename ...Args>
    static auto duration(F&& func, Args&&... args)
    {
        auto start = std::chrono::steady_clock::now();
        std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        return std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now()-start);
    } 
};

// class Mytimer{

// public:

// void Start(){
//     begin = std::chrono::steady_clock::now();
// }

// void Stop(){
//     end= std::chrono::steady_clock::now();
// }

// double Duration(){
//     return static_cast<double>()
// }

// private:

// std::chrono::steady_clock::time_point begin; 
// std::chrono::steady_clock::time_point end; 

// };

template<typename T>
void Quat2EulXYZ(const T quat[4], T Eul_XYZ[3]){
    Eul_XYZ[0] = std::atan2(-2.0 * (quat[2] * quat[3] - quat[1] * quat[0]), quat[0] * quat[0] - quat[1] * quat[1] - quat[2] * quat[2] + quat[3] * quat[3]);
    Eul_XYZ[1] = std::asin(2.0 * (quat[1] * quat[3] + quat[2] * quat[0]));
    Eul_XYZ[2] = std::atan2(-2.0 * (quat[1] * quat[2] - quat[3] * quat[0]), quat[0] * quat[0] + quat[1] * quat[1] - quat[2] * quat[2] - quat[3] * quat[3]);
}

template<typename T>
int MovingMean(const std::vector<T>& In, std::vector<T>& Out, const int KernelSize){
    if(In.size() != Out.size()){
        std::cout << "WARNING: In and Out should have same size. " << std::endl;
        std::cout << "Perform forced resize to Out with In. " << std::endl;

        Out.resize(In.size());

    }

    int elementSize = Out.size();
    //Peform moving mean: with KernelSize
    {
        int sidetailSize = (KernelSize - 1) / 2;

        //Element loop: iterate for every elements to be smoothed; 
        //copy the beginning and end tail, smoothen the middle data with the kernel. 
        for(int it_Object = 0; it_Object < elementSize; ++it_Object){
            //within the middle range kernel can smooth:
            if(it_Object >= sidetailSize && it_Object < elementSize - sidetailSize){
                //kernel loop:
                float acc = 0.0f;
                for(int it_kernel = -sidetailSize; it_kernel <= sidetailSize; ++it_kernel){
                    int selected_idx = it_kernel + it_Object;
                    acc += In[selected_idx];
                }
                //update smoothed Euler angle:
                Out[it_Object] = acc / KernelSize;
            }
            //Out of the middle range, copy the original data:
            else{
                Out[it_Object] = In[it_Object];
            }
        }
    }

}

template<typename T>
void EulXYZ2Quat(const T Eul_XYZ[3], T quat[4]){
    //calculate cos for half angle:
    T cos_half_x = std::cos(Eul_XYZ[0] / 2.0);
    T cos_half_y = std::cos(Eul_XYZ[1] / 2.0);
    T cos_half_z = std::cos(Eul_XYZ[2] / 2.0);

    //calculate sin for half angle:
    T sin_half_x = std::sin(Eul_XYZ[0] / 2.0);
    T sin_half_y = std::sin(Eul_XYZ[1] / 2.0);
    T sin_half_z = std::sin(Eul_XYZ[2] / 2.0);

    //quat: w
    quat[0] = cos_half_x * cos_half_y * cos_half_z - sin_half_x * sin_half_y * sin_half_z;

    //quat: x
    quat[1] = sin_half_x * cos_half_y * cos_half_z + cos_half_x * sin_half_y * sin_half_z;

    //quat: y
    quat[2] = - sin_half_x * cos_half_y * sin_half_z + cos_half_x * sin_half_y * cos_half_z;

    //quat: z
    quat[3] = cos_half_x * cos_half_y * sin_half_z + sin_half_x * sin_half_y * cos_half_z;
}

#endif