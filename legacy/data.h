#define PFTS_NUM_BUFFERS 5
#define PFTS_SIZE_RESULTS 5

//Datatypes
typedef struct _pexc {
	double prefactor;
	double timestep;
	double gauss_f; //irrelevant for local p_exc
	int iterations;
} PFTS_Pexc_conf;

typedef PFTS_Pexc_conf * PFTS_Pexc_t;

typedef struct _pc {
	DFTS_t dfts;
	DFTS_t dfts_gpu;
	double *results;
	double *buffer[PFTS_NUM_BUFFERS];
	double *gradient[2];
	double *current[2];
	double mean_density[2];
	double dt;
	unsigned long timestep;
	unsigned long time_fdot;
	double *fdot[2];
	double *density_update[2];
	double *current_sum;
	double *memory;
	PFTS_Pexc_t pexc_conf;
} PFTS_conf;

typedef PFTS_conf * PFTS_t;
