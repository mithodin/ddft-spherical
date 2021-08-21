__global__ void central_diff(size_t num, const double **f, double dx, double **out);
__global__ void laplace(size_t num, const double **f, double dx, double **out);
__global__ void divergence(size_t num, const double **f, double dx, double **out);
__global__ void average(size_t num, const double **f, double **out);
__global__ void min_val(size_t num, double **f, double min);
