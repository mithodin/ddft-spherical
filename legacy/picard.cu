#include <cuhelp.h>
#include <stdio.h>
#include "config.h"
#include "debug.h"
#include "data.h"
#include "picard.h"
#include "free_energy.h"
#include "dfts.h"
#include "tool.h"

#define EPSILON 1e-12

__global__ void dfts_picard_expm(size_t num, double *in, double *out){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int block = blockIdx.y;
	int offset = block*num;
	if( i < num ){
		out[i+offset] = exp(-in[i+offset]);
	}
}

__global__ void dfts_max_alpha(size_t num, double *new_density, double *old_density, double *alpha, double *alpha_max){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int offset = blockIdx.y*num;
	if( i < num ){
		if( new_density[i] <= 0 ){
			alpha[i+offset] = 1.0/(1.0-new_density[i+offset]/old_density[i+offset])-EPSILON;
			if( alpha[i+offset] > *alpha_max ){
				alpha[i+offset] = *alpha_max;
			}
		}else{
			alpha[i+offset] = *alpha_max;
		}
	}
}

__global__ void dfts_mix_densities(size_t num, double *alpha, double *new_density, double *old_density, double *out, bool *mask){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int offset = blockIdx.y*num;
	if( i < num ){
		if( mask[i] ){
			out[i+offset] = (1-*alpha)*old_density[i+offset]+*alpha*new_density[i+offset];
		}
	}
}


__global__ void dfts_get_picard_direction(Picard_t p){
	memcpy(p->dfts->gradient[p->optimize_component],p->buffer[p->optimize_component],p->bins*sizeof(double));
	init_vec<<<p->dfts->kc.blocks_1,p->dfts->kc.threads>>>(p->bins,0.0,p->dfts->gradient[p->optimize_component]);
	dfts_f_exc<<<1,1>>>(p->dfts, true, p->dfts->results+1);
	op_kernel<<<p->blocks,p->threads>>>(p->dfts->num_bins,p->buffer[p->optimize_component],p->dfts->gradient[p->optimize_component],p->dfts->gradient[p->optimize_component],OP_ADD);
	dfts_picard_expm<<<p->blocks,p->threads>>>(p->dfts->num_bins, p->dfts->gradient[p->optimize_component], p->direction[p->optimize_component]);
}

__global__ void dfts_find_alpha_max(Picard_t p, double *alpha, bool *mask){
	memcpy(p->buffer[2],p->dfts->density[p->optimize_component],p->bins*sizeof(double));
	dfts_mix_densities<<<p->blocks,p->threads>>>(p->dfts->num_bins, alpha, p->direction[p->optimize_component], p->buffer[2], p->dfts->density[p->optimize_component], mask);
	dfts_f_exc<<<1,1>>>(p->dfts, NULL, p->dfts->results);
	cudaDeviceSynchronize();
	while( isnan(p->dfts->results[0]) ){
		*alpha = *alpha/2;
		dfts_mix_densities<<<p->blocks,p->threads>>>(p->dfts->num_bins, alpha, p->direction[p->optimize_component], p->buffer[2], p->dfts->density[p->optimize_component], mask);
		dfts_f_exc<<<1,1>>>(p->dfts, NULL, p->dfts->results);
		cudaDeviceSynchronize();
	}
}

__global__ void dfts_picard_calc_grad(size_t num, size_t num_mask, double *grad, bool *mask, double *result){
	int num_real = 0;
	*result = 0.0;
	for(int i=0;i<num;++i){
		if(mask[i%num_mask]){
			*result += grad[i]*grad[i];
			++num_real;
		}
	}
	*result = sqrt(*result)/num_real;
}

__global__ void dfts_picard_halve_alpha(double *alpha){
	if( *alpha > 1e-20 ){
		*alpha /= 2;
	}
}

__host__ void dfts_picard_minimize(Picard_t p){
	DFTS_t conf = p->dfts;
	double *alpha = p->alpha;
	double grad=0;
	double last_grad=0;
	init_vec<<<1,1>>>(1,0.5,p->alpha_max);
	init_vec<<<conf->kc.blocks_1,conf->kc.threads>>>(p->bins,0.0,conf->gradient[p->optimize_component]);
	dfts_potential<<<1,1>>>(conf->_self, true, conf->results);
	do_check( cudaMemcpy(p->buffer[p->optimize_component],conf->gradient[p->optimize_component],p->bins*sizeof(double),cudaMemcpyDeviceToDevice) == cudaSuccess );
	dfts_omega_<<<1,1>>>(conf->_self, true, conf->results);
	dfts_picard_calc_grad<<<1,1>>>(p->bins,conf->num_bins,conf->gradient[p->optimize_component],conf->min_mask,conf->results);
	cudaMemcpy(&grad,conf->results,sizeof(double),cudaMemcpyDeviceToHost);
	last_grad = grad;
	printf("|grad|/N = %e",grad);
	fflush(stdout);
	int no_improvement = 0;
	while( grad > 1e-16 ){
		dfts_get_picard_direction<<<1,1>>>(p->_self);
		dfts_max_alpha<<<p->blocks,p->threads>>>(conf->num_bins,p->direction[p->optimize_component],conf->density[p->optimize_component],conf->buffer[0],p->alpha_max);
		reduce_min<<<1,1>>>(p->bins,conf->buffer[0],alpha);
		dfts_find_alpha_max<<<1,1>>>(p->_self, alpha,p->dfts->min_mask);
		cudaMemcpy(&grad,alpha,sizeof(double),cudaMemcpyDeviceToHost);
		printf("\33[2K\ralpha = %e ",grad);
		cudaDeviceSynchronize();
		dfts_omega_<<<1,1>>>(conf->_self, true, conf->results);
		dfts_picard_calc_grad<<<1,1>>>(p->bins,conf->num_bins,conf->gradient[p->optimize_component],conf->min_mask,conf->results);
		cudaMemcpy(&grad,conf->results,sizeof(double),cudaMemcpyDeviceToHost);
		printf("|grad|/N = %e",grad);
		fflush(stdout);
		if( grad > last_grad || last_grad/grad-1 < 1e-10 ){
			dfts_picard_halve_alpha<<<1,1>>>(p->alpha_max);
			cudaDeviceSynchronize();
			no_improvement++;
			printf(" [progress stalled %03d]",no_improvement);
			fflush(stdout);
		}else{
			no_improvement = 0;
		}
		if( no_improvement >= 500 ){
			printf("\nno improvement in 500 cycles, exiting.\n");
			return;
		}
		last_grad = grad;
	}
	printf("\npicard iteration successful.\n");
}

__host__ Picard_t dfts_picard_init(DFTS_t conf, size_t components, size_t component){
	Picard_t p = (Picard_t) malloc(sizeof(Picard));
	do_check( cudaMalloc(&(p->_self),sizeof(Picard)) == cudaSuccess );
	do_check( cudaMalloc(p->buffer,4*conf->num_bins*sizeof(double)) == cudaSuccess );
	for(int i=1;i<4;++i){
		p->buffer[i] = p->buffer[0]+i*conf->num_bins;
	}
	do_check( cudaMalloc(p->direction,2*conf->num_bins*sizeof(double)) == cudaSuccess );
	p->direction[1] = p->direction[0]+conf->num_bins;
	do_check( cudaMalloc(&(p->alpha),sizeof(double)) == cudaSuccess );
	do_check( cudaMalloc(&(p->alpha_max),sizeof(double)) == cudaSuccess );
	p->dfts = conf->_self;
	p->blocks.x = conf->kc.blocks_1;
	p->blocks.y = components;
	p->blocks.z = 1;
	p->threads = conf->kc.threads;
	p->optimize_component = component;
	p->bins = conf->num_bins*components;
	do_check( cudaMemcpy( p->_self, p, sizeof(Picard), cudaMemcpyHostToDevice ) == cudaSuccess );
	p->dfts = conf;
	return p;
}

__host__ void dfts_picard_destroy(Picard_t p){
	cudaFree( p->alpha );
	cudaFree( p->direction[0] );
	cudaFree( p->buffer[0] );
	cudaFree( p->_self );
	free(p);
}
