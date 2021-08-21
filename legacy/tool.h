__global__ void dfts_extrapolate(size_t num, size_t extrapolate, size_t fit, double *y, double *buffer);
__host__ void dfts_log_array(size_t num, double *arr, const char filename[]);
