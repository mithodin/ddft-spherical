#include "pc-config.h"
typedef struct _picc {
	double *alpha;
	double *alpha_max;
	double *direction[COMPONENTS];
	double *x[COMPONENTS];
	double *x_old[COMPONENTS];
	double *gradient[COMPONENTS];
	double *result;
	bool *mask;
	size_t dim;
	size_t num_optimize_components;
	size_t optimize_component;
	dim3 blocks;
	size_t threads;
	TYPE_OBJCONF conf;
} Picard;

typedef Picard * Picard_t;
__host__ Picard_t picard_init(TYPE_OBJCONF conf, size_t num_bins, size_t components, size_t component, bool *mask);
__host__ void picard_destroy(Picard_t p);
__host__ void picard_minimize(Picard_t p, double **x0);
