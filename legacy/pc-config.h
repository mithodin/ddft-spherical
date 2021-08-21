#define THREADS 1024
#define FUNCTION_GRADIENT pfts_rt_picard
#define FUNCTION_DIRECTION pfts_picard_direction
#define FUNCTION_MAX_ALPHA pfts_alpha_max
#define GRAD_EPSILON 1e-13
#define TYPE_OBJCONF PFTS_t
#define OBJ_HEADER "pfts_picard.h"
#define EPSILON 1e-20
#define COMPONENTS 2
#define X0_DEVICE
//#define DEBUG
#define SILENT
