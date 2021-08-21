#ifdef __CUDACC__
__global__ void dfts_free_energy(DFTS_t conf, bool grad, double *f);
__global__ void dfts_f_ideal(DFTS_t conf, bool grad, double *f);
__global__ void dfts_f_exc(DFTS_t conf, bool grad, double *f);
#endif
