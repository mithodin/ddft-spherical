#include <cuhelp.h>
#include <stdio.h>
#include "pc-config.h"
#include OBJ_HEADER
#include "picard.h"

#define str2(x) #x
#define str(x) str2(x)

#ifdef DEBUG
	#define do_check(x) if( !(x) ){ fprintf(stderr,"dfts failed check %s at line %d in %s\n",str2(x),__LINE__,__FILE__);exit(-1); }
#else
	#define do_check(x) if( !(x) ){ fprintf(stderr,"dfts has encountered an problem and is exiting.\n");exit(-1); }
#endif

__global__ void picard_mix(size_t num, double *alpha, double *direction, double *old_x, double *new_x, bool *mask){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int offset = blockIdx.y*num;
	if( i < num ){
		if( ( mask && mask[i] ) || !mask ){
			new_x[i+offset] = (1-*alpha)*old_x[i+offset]+(*alpha)*direction[i+offset];
		}
	}
}

__global__ void picard_calc_grad(size_t dim, size_t num_c, double *grad, bool *mask, double *result){
	int num_real = 0;
	*result = 0.0;
	for(int i=0;i<dim;++i){
		if( (mask && mask[i]) || !mask ){
			for(int c=0;c<num_c;++c){
				*result += grad[i+c*dim]*grad[i+c*dim];
				++num_real;
			}
		}
	}
	*result = sqrt(*result)/num_real;
}

__global__ void picard_halve_alpha(double *alpha){
	if( *alpha > 1e-20 ){
		*alpha /= 2;
	}
}

__host__ void picard_minimize(Picard_t p, double **x0){
	double grad=0;
	double last_grad=0;
	init_vec<<<1,1>>>(1,0.5,p->alpha_max);
	init_vec<<<p->blocks,p->threads>>>(p->dim,0.0,p->gradient[p->optimize_component]);
	for(int i=0;i<p->num_optimize_components;++i){
#ifdef X0_DEVICE
		cudaMemcpy(p->x[i],x0[i],p->dim*sizeof(double),cudaMemcpyDeviceToDevice);
#else
		cudaMemcpy(p->x[i],x0[i],p->dim*sizeof(double),cudaMemcpyHostToDevice);
#endif
	}
	FUNCTION_GRADIENT(p->conf,p->x[0],p->gradient[0]);
	picard_calc_grad<<<1,1>>>(p->dim,p->num_optimize_components,p->gradient[p->optimize_component],p->mask,p->result);
	cudaMemcpy(&grad,p->result,sizeof(double),cudaMemcpyDeviceToHost);
	last_grad = grad;
#ifndef SILENT
	printf("|grad|/N = %e",grad);
	fflush(stdout);
#endif
	int no_improvement = 0;
	while( grad > GRAD_EPSILON ){
		do_check( cudaMemcpy(p->x_old[p->optimize_component],p->x[p->optimize_component],p->num_optimize_components*p->dim*sizeof(double),cudaMemcpyDeviceToDevice) == cudaSuccess );
		FUNCTION_DIRECTION(p->conf,p->x_old[0],p->direction[0]);
		FUNCTION_MAX_ALPHA(p->conf,p->direction[0],p->alpha_max,p->alpha);
		cudaMemcpy(&grad,p->alpha,sizeof(double),cudaMemcpyDeviceToHost);
#ifndef SILENT
		printf("\33[2K\ralpha = %e ",grad);
#endif
		//blocks and threads?
		picard_mix<<<p->blocks,p->threads>>>(p->dim, p->alpha, p->direction[p->optimize_component], p->x_old[p->optimize_component], p->x[p->optimize_component], p->mask);
		cudaDeviceSynchronize();
		FUNCTION_GRADIENT(p->conf,p->x[0],p->gradient[0]);
		picard_calc_grad<<<1,1>>>(p->dim,p->num_optimize_components,p->gradient[p->optimize_component],p->mask,p->result);
		cudaMemcpy(&grad,p->result,sizeof(double),cudaMemcpyDeviceToHost);
#ifndef SILENT
		printf("|grad|/N = %e",grad);
		fflush(stdout);
#endif
		if( grad > last_grad || last_grad/grad-1 < 1e-10 ){
			picard_halve_alpha<<<1,1>>>(p->alpha_max);
			cudaDeviceSynchronize();
			no_improvement++;
#ifndef SILENT
			printf(" [progress stalled %03d]",no_improvement);
#endif
			fflush(stdout);
		}else{
			no_improvement = 0;
		}
		if( no_improvement >= 500 ){
#ifndef SILENT
			printf("\nno improvement in 500 cycles, exiting.\n");
#endif
			break;
		}
		last_grad = grad;
	}
#ifndef SILENT
	printf("\npicard iteration successful.\n");
#endif
	for(int i=0;i<p->num_optimize_components;++i){
#ifdef X0_DEVICE
		cudaMemcpy(x0[i],p->x[i],p->dim*sizeof(double),cudaMemcpyDeviceToDevice);
#else
		cudaMemcpy(x0[i],p->x[i],p->dim*sizeof(double),cudaMemcpyDeviceToHost);
#endif
	}
}

__host__ Picard_t picard_init(TYPE_OBJCONF conf, size_t num_bins, size_t components, size_t component, bool *mask){
	Picard_t p = (Picard_t) malloc(sizeof(Picard));
	do_check( cudaMalloc(p->gradient,COMPONENTS*num_bins*sizeof(double)) == cudaSuccess );
	do_check( cudaMalloc(p->direction,COMPONENTS*num_bins*sizeof(double)) == cudaSuccess );
	do_check( cudaMalloc(p->x_old,COMPONENTS*num_bins*sizeof(double)) == cudaSuccess );
	do_check( cudaMalloc(p->x,COMPONENTS*num_bins*sizeof(double)) == cudaSuccess );
	for(int i=1;i<COMPONENTS;++i){
		p->gradient[i] = p->gradient[0]+i*num_bins;
		p->direction[i] = p->direction[0]+i*num_bins;
		p->x_old[i] = p->x_old[0]+i*num_bins;
		p->x[i] = p->x[0]+i*num_bins;
	}
	do_check( cudaMalloc(&(p->alpha),sizeof(double)) == cudaSuccess );
	do_check( cudaMalloc(&(p->alpha_max),sizeof(double)) == cudaSuccess );
	do_check( cudaMalloc(&(p->result),sizeof(double)) == cudaSuccess );
	p->mask = NULL;
	if( mask ){
		do_check( cudaMalloc(&(p->mask),num_bins*sizeof(double)) == cudaSuccess );
		do_check( cudaMemcpy(p->mask,mask,num_bins*sizeof(double),cudaMemcpyHostToDevice) == cudaSuccess );
	}
	p->dim = num_bins;
	p->conf = conf;
	p->blocks.x = (num_bins+THREADS-1)/THREADS;
	p->blocks.y = components;
	p->blocks.z = 1;
	p->threads = THREADS;
	p->num_optimize_components = components;
	p->optimize_component = component;
	return p;
}

__host__ void picard_destroy(Picard_t p){
	cudaFree( p->alpha );
	cudaFree( p->alpha_max );
	cudaFree( p->direction[0] );
	cudaFree( p->gradient[0] );
	cudaFree( p->x[0] );
	cudaFree( p->x_old[0] );
	cudaFree( p->result );
	free(p);
}
