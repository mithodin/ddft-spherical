#include <stdio.h>
#include "config.h"
#include "data.h"
#include "debug.h"
#include "dfts.h"

__host__ void dfts_init_kernelconfig(DFTS_kernelconfig *kc, size_t num_bins){
        kc->blocks_1 = (num_bins+THREADS-1)/THREADS;
        kc->threads = THREADS;
	kc->blocks_2.x = kc->blocks_1;
	kc->blocks_2.y = 2;
	kc->blocks_2.z = 1;
	kc->blocks_wd.x = kc->blocks_1;
	kc->blocks_wd.y = 2;
	kc->blocks_wd.z = dfts_num_wd;
}

extern "C" {
__host__ void dfts_sync(DFTS_t conf){
	do_check( cudaMemcpy(conf->_self,conf,sizeof(DFTS_conf),cudaMemcpyHostToDevice) == cudaSuccess );
	cudaDeviceSynchronize();
}

__host__ DFTS_t dfts_init(size_t num_bins, double dr, size_t bins_sphere){
	DFTS_t conf = (DFTS_t)malloc(sizeof(DFTS_conf));
	conf->num_bins = num_bins;
	conf->bins_sphere = bins_sphere;
	conf->dr = dr;
	conf->radius_sphere = dr*bins_sphere;
	conf->chemical_potential = 0.0;
	conf->selfinteraction = full;
	dfts_init_kernelconfig(&(conf->kc),num_bins);
	do_check( cudaMalloc(&(conf->results),NUM_RESULTS*sizeof(double)) == cudaSuccess );
	do_check( cudaMalloc(conf->density_sum,2*num_bins*sizeof(double)) == cudaSuccess );
	conf->density_sum[1] = conf->density_sum[0]+num_bins;
	do_check( cudaMalloc(&(conf->min_mask),num_bins*sizeof(double)) == cudaSuccess );
	do_check( cudaMalloc(&(conf->potential),num_bins*sizeof(double)) == cudaSuccess );
	do_check( cudaMalloc(&(conf->grad_potential),num_bins*sizeof(double)) == cudaSuccess );
	do_check( cudaMalloc(conf->buffer,NUM_BUFFERS*num_bins*sizeof(double)) == cudaSuccess );
	for(int i=1;i<NUM_BUFFERS;++i){
		conf->buffer[i] = conf->buffer[0]+i*num_bins;
	}
	do_check( cudaMalloc(conf->weighted_density[0],2*dfts_num_wd*num_bins*sizeof(double)) == cudaSuccess );
	conf->weighted_density[1][0] = conf->weighted_density[0][0]+dfts_num_wd*num_bins;
	for(int i=1;i<dfts_num_wd;++i){
		conf->weighted_density[0][i] = conf->weighted_density[0][0]+i*num_bins;
		conf->weighted_density[1][i] = conf->weighted_density[1][0]+i*num_bins;
	}

	do_check( cudaMalloc(conf->psi[0],2*dfts_num_wd*num_bins*sizeof(double)) == cudaSuccess );
	conf->psi[1][0] = conf->psi[0][0]+dfts_num_wd*num_bins;
	for(int i=1;i<dfts_num_wd;++i){
		conf->psi[0][i] = conf->psi[0][0]+i*num_bins;
		conf->psi[1][i] = conf->psi[1][0]+i*num_bins;
	}
	do_check( cudaMalloc(conf->density,2*num_bins*sizeof(double)) == cudaSuccess );
	conf->density[1] = conf->density[0]+num_bins;
	do_check( cudaMalloc(conf->gradient,2*num_bins*sizeof(double)) == cudaSuccess );
	conf->gradient[1] = conf->gradient[0]+num_bins;
	do_check( cudaMalloc(&(conf->_self),sizeof(DFTS_conf)) == cudaSuccess );
	do_check( cudaMemcpy(conf->_self,conf,sizeof(DFTS_conf),cudaMemcpyHostToDevice) == cudaSuccess );
	return conf;
}

__host__ void dfts_destroy(DFTS_t conf){
	cudaFree(conf->_self);
	cudaFree(conf->gradient[0]);
	cudaFree(conf->density[0]);
	cudaFree(conf->weighted_density[0][0]);
	cudaFree(conf->psi[0][0]);
	cudaFree(conf->buffer[0]);
	cudaFree(conf->grad_potential);
	cudaFree(conf->min_mask);
	cudaFree(conf->potential);
	cudaFree(conf->density_sum[0]);
	cudaFree(conf->results);
	free(conf);
}

}//end extern "C"
