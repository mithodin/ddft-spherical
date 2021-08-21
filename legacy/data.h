
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
