#include <cuhelp.h>
#include <stdio.h>
#include <stdbool.h>
#include <dfts.h>
#include "debug.h"
#include "data.h"
#include "pft_math.h"

#define NUM_BINS (conf->dfts_gpu->num_bins)

//#include "pexc_local.cu"
#include "pexc_gauss.cu"

__global__ void pfts_fdot_k(size_t num, double dr, double **j, double **fdot, double **out, double **gradient){
        int i = blockIdx.x*blockDim.x+threadIdx.x;
        int block = blockIdx.y;
	if( i >= num ) return;
	double w = weight_4pir2(0,num-1,i,dr);
	out[block][i] = j[block][i]*fdot[block][i]*w;
	if( gradient ){
		gradient[block][i] = fdot[block][i];
	}
}

__global__ void pfts_fdot(PFTS_t conf, double *result, bool grad){
	size_t t = conf->dfts_gpu->kc.threads;
	if( conf->time_fdot != conf->timestep ){
		dfts_free_energy<<<1,1>>>(conf->dfts_gpu, true, result);
		central_diff<<<conf->dfts_gpu->kc.blocks_2,t>>>(conf->dfts_gpu->num_bins,
				(const double **)conf->dfts_gpu->gradient,
				conf->dfts_gpu->dr,
				conf->fdot);
		conf->time_fdot = conf->timestep;
	}
	double **gradient = NULL;
	if( grad ){
		gradient = conf->gradient;
	}
	pfts_fdot_k<<<conf->dfts_gpu->kc.blocks_2,t>>>(conf->dfts_gpu->num_bins,conf->dfts_gpu->dr,conf->current,conf->fdot,conf->buffer,gradient);
	reduce_add<<<1,1>>>(conf->dfts_gpu->num_bins, conf->buffer[0], conf->buffer[1], result, OP_ADD);
}

__global__ void pfts_p_ideal_k(size_t num, double dr, double **j, double **rho, double **out, double **gradient){
        int i = blockIdx.x*blockDim.x+threadIdx.x;
        int block = blockIdx.y;
	if( i >= num ){
		return;
	}
	double w = weight_4pir2(0,num-1,i,dr);
	out[block][i] = j[block][i]/rho[block][i];
	if( gradient ){
		gradient[block][i] += out[block][i];
	}
	out[block][i] *= j[block][i]/2.0*w;
}

__global__ void pfts_p_ideal(PFTS_t conf, double *result, bool grad){
	int t = conf->dfts_gpu->kc.threads;
	dim3 b = conf->dfts_gpu->kc.blocks_2;
	double **g = NULL;
	if( grad ){
		g = conf->gradient;
	}
	pfts_p_ideal_k<<<b,t>>>(conf->dfts_gpu->num_bins, conf->dfts_gpu->dr, conf->current, conf->dfts_gpu->density, conf->buffer, g);
	reduce_add<<<1,1>>>(conf->dfts_gpu->num_bins, conf->buffer[0], conf->buffer[1], result, OP_ADD);
}

__global__ void pfts_rt(PFTS_t conf, double *res, bool grad){
	pfts_fdot<<<1,1>>>(conf, res, grad);
	pfts_p_ideal<<<1,1>>>(conf, res+1, grad);
#ifdef DDFT
	*(res+2) = 0.0;
#else
	pfts_p_exc<<<1,1>>>(conf, res+2, grad);
#endif
	op_add_kernel<<<1,1>>>(1, res+1, res+2, res, OP_ADD);
}

extern "C"{
__host__ void pfts_set_current(PFTS_t conf, const double **current){
	do_check( cudaMemcpy(conf->current[0],current[0],conf->dfts->num_bins*sizeof(double),cudaMemcpyHostToDevice) == cudaSuccess);
	do_check( cudaMemcpy(conf->current[1],current[1],conf->dfts->num_bins*sizeof(double),cudaMemcpyHostToDevice) == cudaSuccess);
}

__host__ void pfts_get_current(PFTS_t conf, double **current){
	do_check( cudaMemcpy(current[0],conf->current[0],conf->dfts->num_bins*sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess);
	do_check( cudaMemcpy(current[1],conf->current[1],conf->dfts->num_bins*sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess);
}

__host__ double pfts_rt_nlopt(unsigned dims, const double *j, double *grad, void *conf_){
	PFTS_t conf = (PFTS_t)conf_;
	const double *current[2] = {j,j+dims/2};
	pfts_set_current(conf,current);
	pfts_rt<<<1,1>>>(conf, conf->results, grad);
	double rt = 0.0;
	cudaDeviceSynchronize();
	do_check( cudaMemcpy(&rt, conf->results, sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess );
	if( grad ){
		do_check( cudaMemcpy(grad, conf->gradient[0], conf->dfts->num_bins*sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess );
		do_check( cudaMemcpy(grad+conf->dfts->num_bins, conf->gradient[1], conf->dfts->num_bins*sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess );
	}
	return rt;
}

__host__ double pfts_rt_c(PFTS_t conf, double **gradient){
	double f = 0.0;
	pfts_rt<<<1,1>>>(conf, conf->results, gradient != NULL);
	cudaDeviceSynchronize();
	do_check( cudaMemcpy(&f, conf->results, sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess );
	if( gradient ){
		do_check( cudaMemcpy(gradient[0],conf->gradient[0],conf->dfts->num_bins*sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess );
		do_check( cudaMemcpy(gradient[1],conf->gradient[1],conf->dfts->num_bins*sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess );
	}
	return f;
}

__host__ double pfts_p_ideal_c(PFTS_t conf, double **gradient){
	double f = 0.0;
	pfts_p_ideal<<<1,1>>>(conf, conf->results, gradient != NULL);
	cudaDeviceSynchronize();
	do_check( cudaMemcpy(&f, conf->results, sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess );
	if( gradient ){
		do_check( cudaMemcpy(gradient[0],conf->gradient[0],conf->dfts->num_bins*sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess );
		do_check( cudaMemcpy(gradient[1],conf->gradient[1],conf->dfts->num_bins*sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess );
	}
	return f;
}

__host__ double pfts_fdot_c(PFTS_t conf, double **gradient){
	double f = 0.0;
	init_vec<<<conf->dfts->kc.blocks_1,conf->dfts->kc.threads>>>(conf->dfts->num_bins,0.0,conf->gradient[0]);
	cudaDeviceSynchronize();
	init_vec<<<conf->dfts->kc.blocks_1,conf->dfts->kc.threads>>>(conf->dfts->num_bins,0.0,conf->gradient[1]);
	cudaDeviceSynchronize();
	pfts_fdot<<<1,1>>>(conf, conf->results, gradient != NULL);
	cudaDeviceSynchronize();
	do_check( cudaMemcpy(&f, conf->results, sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess );
	if( gradient ){
		do_check( cudaMemcpy(gradient[0],conf->gradient[0],conf->dfts->num_bins*sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess );
		do_check( cudaMemcpy(gradient[1],conf->gradient[1],conf->dfts->num_bins*sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess );
	}
	return f;
}

}//end extern "C"
