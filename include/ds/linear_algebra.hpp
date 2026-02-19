#if !defined(__LINEAR_ALGEBRA__)
#define __LINEAR_ALGEBRA__
#include <vector>
#include <utility>
#include <functional>


namespace ds
{

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

// Vector operations
Vector add(const Vector& v, const Vector& w);
Vector subtract(const Vector& v, const Vector& w);
Vector scalar_multiply(double c, const Vector& v);
Vector vector_sum(const std::vector<Vector>& vectors);
Vector vector_mean(const std::vector<Vector>& vectors);
double dot(const Vector& v, const Vector& w);
double sum_of_squares(const Vector& v);
double magnitude(const Vector& v);
double squared_distance(const Vector& v, const Vector& w);
double distance(const Vector& v, const Vector& w);

// Matrix operations
std::pair<int, int> shape(const Matrix& A); // a std::par is like a tuple in python 
Vector get_row(const Matrix& A, int i);
Vector get_column(const Matrix& A, int j);

Matrix make_matrix(
    int num_rows,
    int num_cols,
    std::function<double(int, int)> entry_fn);

Matrix identity_matrix(int n);

 

}


#endif // __LINEAR_ALGEBRA__
