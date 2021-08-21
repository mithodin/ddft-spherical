#define DEBUG
#define THREADS 1024
#define NUM_BUFFERS 8
#define NUM_RESULTS 10
#define D_SELF 0
#define D_DISTINCT 1

#ifndef size_t
#include <stdlib.h>
#endif

#ifndef bool
#include <stdbool.h>
#endif

#ifndef dim3
#include <vector_types.h>
#endif

typedef enum {
	semi_linearized,
	quenched,
	full
} DFTS_selfinteraction;

typedef enum { dfts_n2 = 0, dfts_n3 = 1, dfts_n2v = 2, dfts_n11 = 3, dfts_num_wd = 4 } dfts_wd;

typedef struct _kconf {
	size_t blocks_1;
	size_t threads;
	dim3 blocks_2;
	dim3 blocks_wd;
} DFTS_kernelconfig;

typedef struct _c {
        size_t num_bins;
	double dr;
        double radius_sphere;
        size_t bins_sphere;
        double chemical_potential;
	DFTS_kernelconfig kc;
	double *results;
	double *density_sum[2];
        double *potential;
	double *grad_potential;
	double *buffer[NUM_BUFFERS];
	double *weighted_density[2][dfts_num_wd];
	double *psi[2][dfts_num_wd];
        double *density[2];
        double *gradient[2];
	bool *min_mask;
	struct _c *_self;
	DFTS_selfinteraction selfinteraction;
} DFTS_conf;

typedef DFTS_conf * DFTS_t;
#ifdef __CUDACC__
__global__ void dfts_free_energy(DFTS_t conf, bool grad, double *f);
__global__ void dfts_f_ideal(DFTS_t conf, bool grad, double *f);
__global__ void dfts_f_exc(DFTS_t conf, bool grad, double *f);
#endif
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
#ifdef __CUDACC__
__device__ __host__ double weight_4pir2(long lb, long ub, long k, double dr);
__device__ __host__ double weight_wd(dfts_wd alpha, long lb, long ub, long r, long rp, long rs, double dr);
__device__ __host__ double weight_psi(dfts_wd alpha, long lb, long ub, long r, long rp, long rs, double dr);
__global__ void dfts_get_weights(long lb, long ub, double dr, double *w);
#endif
#ifdef __CUDACC__
extern "C" {
#endif
DFTS_t dfts_init(size_t num_bins, double dr, size_t bins_sphere);
void dfts_destroy(DFTS_t conf);
void dfts_sync(DFTS_t conf);
#ifdef __CUDACC__
}
#endif
