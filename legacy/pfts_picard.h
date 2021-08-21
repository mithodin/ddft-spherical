#include <dfts.h>
#include "data.h"
extern "C" {
	void pfts_rt_picard(PFTS_t conf, double *j_now, double *grad);
	void pfts_picard_direction(PFTS_t conf, double *j_now, double *j_next);
	void pfts_alpha_max(PFTS_t conf, double *j_next, double *alpha_max, double *alpha);
}
