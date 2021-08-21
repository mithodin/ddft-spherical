#ifdef __CUDACC__
__global__ void pfts_rt(PFTS_t conf, double *result, bool grad);
__global__ void pfts_fdot(PFTS_t conf, double *result, bool grad);
__global__ void pfts_p_exc(PFTS_t conf, double *result, bool grad);
__global__ void pfts_pexc_update(PFTS_t conf, double **density_update);
__global__ void pfts_update_memory(PFTS_t conf);
extern "C" {
#endif
PFTS_t pfts_create(size_t num_bins, double dr, size_t bins_sphere, double dt);
void pfts_pexc_settings(PFTS_t conf, double a, double zeta, double tau_mem, double gauss_f);
void pfts_destroy(PFTS_t conf);
void pfts_set_current(PFTS_t conf, const double **current);
void pfts_get_current(PFTS_t conf, double **current);
void pfts_set_density(PFTS_t conf, const double **density);
double pfts_rt_c(PFTS_t conf, double **gradient);
double pfts_p_ideal_c(PFTS_t conf, double **gradient);
double pfts_fdot_c(PFTS_t conf, double **gradient);
double pfts_rt_nlopt(unsigned dims, const double *j, double *grad, void *conf_);
void pfts_advance_time(PFTS_t conf);
void pfts_init_rdf(PFTS_t conf);
void pfts_minimize(PFTS_t conf);
void pfts_renormalize(PFTS_t conf);
#ifdef __CUDACC__
}
#endif
