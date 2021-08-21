typedef struct _p {
	double *buffer[4];
	double *alpha;
	double *alpha_max;
	double *direction[2];
	size_t optimize_component;
	size_t bins;
	dim3 blocks;
	size_t threads;
	DFTS_t dfts;
	struct _p *_self;
} Picard;

typedef Picard * Picard_t;

__host__ Picard_t dfts_picard_init(DFTS_t conf, size_t components, size_t component);
__host__ void dfts_picard_destroy(Picard_t p);
__host__ void dfts_picard_minimize(Picard_t p);
