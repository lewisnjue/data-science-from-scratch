#include "ds/statistics.hpp"
#include <vector> 
#include <iostream> 
#include <iomanip> 
using namespace ds; 
int main(){
    std::vector<double> myVector = {3.0,4.0,5.0};

    double my_mean = mean(myVector);
    std::cout << "MY CALCUATED MEAN IS : " << my_mean << " the expected one was " << 4 << std::endl; 


    return 0;
}