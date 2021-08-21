#include <stdio.h>
#include <dfts.h>
#include "debug.h"
#include "data.h"

__global__ void pfts_pexc_settings_(PFTS_Pexc_t conf, double a, double zeta, double tau_mem, double gauss_f, double dt, double dr){
	conf->prefactor = a/2.*zeta/tau_mem*dt;
	conf->timestep = exp(-dt/tau_mem);
	//the following is irrelevant for local P_exc
	conf->gauss_f = gauss_f*dt/dr*dr;
	conf->iterations = (int)round(conf->gauss_f/.2);
	if( conf->iterations <= 0 ){
		conf->iterations = 1;
	}
	conf->gauss_f /= conf->iterations;
}

extern "C"{
void pfts_pexc_settings(PFTS_t conf, double a, double zeta, double tau_mem, double gauss_f){
	cudaDeviceSynchronize();
	pfts_pexc_settings_<<<1,1>>>(conf->pexc_conf, a, zeta, tau_mem, gauss_f, conf->dt, conf->dfts->dr);
	cudaDeviceSynchronize();
}

PFTS_t pfts_create(size_t num_bins, double dr, size_t bins_sphere, double dt){
	PFTS_t conf;
	do_check( cudaMallocManaged(&conf, sizeof(PFTS_conf)) == cudaSuccess );
	conf->dfts = dfts_init(num_bins, dr, bins_sphere);
	conf->dfts_gpu = conf->dfts->_self;
	do_check( cudaMalloc(&(conf->current_sum), num_bins*sizeof(double)) == cudaSuccess );
	do_check( cudaMalloc(&(conf->memory), num_bins*sizeof(double)) == cudaSuccess );
	do_check( cudaMalloc(&(conf->pexc_conf), sizeof(PFTS_Pexc_conf)) == cudaSuccess );
	pfts_pexc_settings_<<<1,1>>>(conf->pexc_conf, 0.0, 1., 1., .2*conf->dfts->dr*conf->dfts->dr*dt, dt, conf->dfts->dr); //default values
	cudaDeviceSynchronize();
	cudaMemset( conf->memory, 0, num_bins*sizeof(double) );
	do_check( cudaMalloc(&(conf->results), PFTS_SIZE_RESULTS*sizeof(double)) == cudaSuccess );
	do_check( cudaMalloc(conf->buffer, PFTS_NUM_BUFFERS*num_bins*sizeof(double)) == cudaSuccess );
	for(int i=1;i<PFTS_NUM_BUFFERS;++i){
		conf->buffer[i] = conf->buffer[0]+i*num_bins;
	}
	cudaDeviceSynchronize();
	do_check( cudaMalloc(conf->gradient, 2*num_bins*sizeof(double)) == cudaSuccess );
	conf->gradient[1] = conf->gradient[0]+num_bins;
	do_check( cudaMalloc(conf->current, 2*num_bins*sizeof(double)) == cudaSuccess );
	conf->current[1] = conf->current[0]+num_bins;
	do_check( cudaMalloc(conf->fdot, 2*num_bins*sizeof(double)) == cudaSuccess );
	conf->fdot[1] = conf->fdot[0]+num_bins;
	do_check( cudaMalloc(conf->density_update, 2*num_bins*sizeof(double)) == cudaSuccess );
	conf->density_update[1] = conf->density_update[0]+num_bins;
	conf->time_fdot = 10000000;
	conf->timestep = 0;
	conf->dt = dt;
	return conf;
}

void pfts_destroy(PFTS_t conf){
	dfts_destroy(conf->dfts);
	do_check( cudaFree(conf->fdot[0]) == cudaSuccess );
	do_check( cudaFree(conf->current[0]) == cudaSuccess );
	do_check( cudaFree(conf->gradient[0]) == cudaSuccess );
	do_check( cudaFree(conf->buffer[0]) == cudaSuccess );
	do_check( cudaFree(conf->results) == cudaSuccess );
	do_check( cudaFree(conf->memory) == cudaSuccess );
	do_check( cudaFree(conf->current_sum) == cudaSuccess );
	do_check( cudaFree(conf->density_update[0]) == cudaSuccess );
	do_check( cudaFree(conf->pexc_conf) == cudaSuccess );
	do_check( cudaFree(conf) == cudaSuccess );
}
}
