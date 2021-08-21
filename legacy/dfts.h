#ifdef __CUDACC__
__global__ void dfts_omega_(DFTS_t conf, bool grad, double *f);
__global__ void dfts_potential(DFTS_t conf, bool grad, double *f);
extern "C" {
#endif
double dfts_omega(DFTS_t conf, double **grad);
void dfts_set_density(DFTS_t conf, const double **density);
void dfts_set_density_component(DFTS_t conf, const double **density,int mask);
void dfts_get_density(DFTS_t conf, double **density);
void dfts_set_potential(DFTS_t conf, double *potential);
void dfts_set_chemical_potential(DFTS_t conf, double mu);
void dfts_set_selfinteraction(DFTS_t conf, DFTS_selfinteraction self);
void dfts_minimize(DFTS_t conf);
void dfts_minimize_component(DFTS_t conf, size_t component);
void dfts_minimize_component_mask(DFTS_t conf, size_t component, bool *mask);
void dfts_log_wd(DFTS_t conf, const char filename[]);
void dfts_log_psi(DFTS_t conf, const char filename[]);
double dfts_fexc(DFTS_t conf, double **rho, double **grad);
double dfts_fid(DFTS_t conf, double **rho, double **grad);
double dfts_f(DFTS_t conf, double **rho, double **grad);
double dfts_get_mean_density(DFTS_t conf, double *total_volume, double *mean_volume);
double dfts_get_chemical_potential_mean_density(DFTS_t conf, double mean_density, double mean_volume_fraction);
#ifdef __CUDACC__
}
#endif
