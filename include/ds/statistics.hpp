#if !defined(__STATISTICS__)
#define __STATISTICS__
#include <vector>
#include "ds/linear_algebra.hpp"
namespace ds {
    // a function called mean  
    double mean(const Vector& ); 
    double median(const Vector& ); 
    double quantile(const Vector& ,double );
    Vector mode(const Vector& );
    double data_range(const Vector& );
    Vector de_mean(const Vector&); 
    double variance(const Vector&);
    double standard_deviation(const Vector&);
    double interquartile_range(const Vector&);
    double covariance(const Vector&, const Vector&);
    double correlation(const Vector&,const Vector&);
}

#endif // __STATISTICS__
