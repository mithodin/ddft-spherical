#ifdef __CUDACC__
extern "C" {
#endif
DFTS_t dfts_init(size_t num_bins, double dr, size_t bins_sphere);
void dfts_destroy(DFTS_t conf);
void dfts_sync(DFTS_t conf);
#ifdef __CUDACC__
}
#endif
