#ifdef __CUDACC__
__device__ __host__ double weight_4pir2(long lb, long ub, long k, double dr);
__device__ __host__ double weight_wd(dfts_wd alpha, long lb, long ub, long r, long rp, long rs, double dr);
__device__ __host__ double weight_psi(dfts_wd alpha, long lb, long ub, long r, long rp, long rs, double dr);
__global__ void dfts_get_weights(long lb, long ub, double dr, double *w);
#endif
