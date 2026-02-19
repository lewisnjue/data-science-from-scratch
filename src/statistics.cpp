#include "ds/statistics.hpp"
#include <unordered_map>
#include <ds/linear_algebra.hpp>
#include <vector> 
#include <cassert>
#include <math.h>
#include <algorithm> // for std::sort
namespace ds {
   double mean(const Vector& v) {
       double sum = 0.0;
       for (double x : v)
           sum += x;
       return sum / v.size();
   } 
   double median(const Vector& v){
         std::vector<double> sorted_v = v; 
         std::sort(sorted_v.begin(), sorted_v.end());
         size_t n = sorted_v.size();
         if (n % 2 == 1) { // odd number of elements 
            return sorted_v[n / 2];
         } else {
              return (sorted_v[n / 2 - 1] + sorted_v[n / 2]) / 2.0;
         }
   }

    double quantile(const Vector& v,double p){
        int p_index = (int)(p * v.size());
        Vector sorted_v = v;  // doing this because i dont want to modify the original v 
        std::sort(sorted_v.begin(),sorted_v.end());
        return sorted_v[p_index];

    }
    
   Vector mode(const Vector& x) {
        if (x.empty()) return {};
        std::unordered_map<double, int> counts; // key(double),value(int)
        for (double val : x) {
            counts[val]++;
        }
        int max_count = 0;
        for (const auto& pair : counts) { // pair {key,value}
            if (pair.second > max_count) {
                max_count = pair.second;
            }
        }
        Vector modes;
        for (const auto& pair : counts) {
            if (pair.second == max_count) {
                modes.push_back(pair.first);
            }
    }

    return modes;
    
}
    double data_range(const Vector& v){
            auto [min_val,max_val] = std::minmax_element(v.begin(),v.end()); 
            double i_min = *min_val; 
            double i_max = *max_val; 
            return i_max - i_min;


        }

    Vector de_mean(const Vector& xs){
        double x_bar = mean(xs);
        Vector result(xs.size(),0.0); 
       for (size_t i = 0; i < xs.size(); i++){
        result[i] = xs[i] - x_bar;

       }
       return result;
        
    } 

    double variance(const Vector& xs){
        assert(xs.size() >= 2); 
        auto n = xs.size();
        Vector deviations = de_mean(xs);
        double result =  sum_of_squares(deviations); 
        return result / (n -1); 

    }
    double standard_deviation(const Vector& xs){
       return std::sqrt(variance(xs));

    }
    double interquartile_range(const Vector& xs){
        return quantile(xs,0.75) - quantile(xs,0.25);
    }
    
    double covariance(const Vector& xs, const Vector& ys){
        assert(xs.size() == ys.size());
        return dot(de_mean(xs),de_mean(ys)) / (xs.size() -1);

    }

    double correlation(const Vector& xs,const Vector& ys){
       auto stdev_x = standard_deviation(xs);
       auto stdev_y = standard_deviation(ys); 
       if(stdev_x > 0 && stdev_y > 0){
        return covariance(xs,ys) / stdev_x / stdev_y; 
       } else{
        return 0;
       }

    }


}