#include <cuhelp.h>
#include <stdio.h>
#include <dfts.h>
#include "data.h"
#include "pfts.h"
#include "pft_math.h"
#include "debug.h"
#include "picard.h"

#define ZERO 1e-70

__global__ void pfts_update_rho(PFTS_t conf){
	size_t t = conf->dfts_gpu->kc.threads;
	dim3 b = conf->dfts_gpu->kc.blocks_2;

	//laplace rho
	laplace<<<b,t>>>(conf->dfts_gpu->num_bins, (const double **)conf->dfts_gpu->density, conf->dfts_gpu->dr, conf->density_update);

	// grad rho * grad d F_exc / d rho
	memset(conf->dfts_gpu->gradient[0],0,2*conf->dfts_gpu->num_bins*sizeof(double));
	cudaDeviceSynchronize();

	dfts_f_exc<<<1,1>>>(conf->dfts_gpu, true, conf->buffer[0]);
	central_diff<<<b,t>>>(conf->dfts_gpu->num_bins, (const double **)conf->dfts_gpu->gradient, conf->dfts_gpu->dr, conf->buffer);
	central_diff<<<b,t>>>(conf->dfts_gpu->num_bins, (const double **)conf->dfts_gpu->density, conf->dfts_gpu->dr, conf->buffer+2);
	op_add_kernel<<<b,t>>>(conf->dfts_gpu->num_bins, conf->buffer[0], conf->buffer[2], conf->density_update[0], OP_MUL);

	// rho * laplace d F_exc / d rho
	laplace<<<b,t>>>(conf->dfts_gpu->num_bins, (const double **)conf->dfts_gpu->gradient, conf->dfts_gpu->dr, conf->buffer);
	op_add_kernel<<<b,t>>>(conf->dfts_gpu->num_bins, conf->buffer[0], conf->dfts_gpu->density[0], conf->density_update[0], OP_MUL);

#ifndef DDFT
	// div rho * d P_exc / d j
	pfts_pexc_update<<<1,1>>>(conf, conf->density_update);
	//update memory
	pfts_update_memory<<<1,1>>>(conf);
#endif

	mul<<<b,t>>>(conf->dfts_gpu->num_bins, conf->density_update[0], conf->dt, conf->density_update[0]);
	op_add_kernel<<<b,t>>>(conf->dfts_gpu->num_bins, conf->density_update[0], NULL, conf->dfts_gpu->density[0], OP_NOP);
	//min_val<<<b,t>>>(conf->dfts_gpu->num_bins, conf->dfts_gpu->density, ZERO);
}

extern "C" {
__host__ void pfts_set_density(PFTS_t conf, const double **density){
	dfts_set_density(conf->dfts, density);
}

__host__ void pfts_print_array(PFTS_t conf, const double **arr){
	double *data = (double *)malloc(conf->dfts->num_bins*sizeof(double)*2);
	do_check( cudaMemcpy(data,arr[0],conf->dfts->num_bins*2*sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess );
	FILE *out = fopen("debug.dat","w");
	for(int i=0;i<conf->dfts->num_bins;++i){
		fprintf(out,"%d\t%e\t%e\n",i,data[i],data[i+conf->dfts->num_bins]);
	}
	fclose(out);
}

__host__ void pfts_advance_time(PFTS_t conf){
	pfts_update_rho<<<1,1>>>(conf);
	cudaDeviceSynchronize();
	conf->timestep += 1;
	cudaDeviceSynchronize();
}

#define ALPHA 1e3
__host__ void pfts_init_rdf(PFTS_t conf){
	double a0 = pow(ALPHA/M_PI,3.0/2.0);
	double *rho[2] = {(double *)malloc(2*(conf->dfts->num_bins)*sizeof(double)),NULL};
	rho[1] = rho[0]+conf->dfts->num_bins;
	for(int i=0;i<conf->dfts->num_bins;++i){
		rho[0][i] = ZERO;
		rho[1][i] = ZERO;
	}
	rho[0][0] = 1.0/(4.0*M_PI*(1.0/3.0-1.0/4.0)*conf->dfts->dr*conf->dfts->dr*conf->dfts->dr)-1e-8;
	dfts_set_density(conf->dfts,(const double**)rho);
	
	bool *mask = (bool *)malloc(conf->dfts->num_bins*sizeof(bool));
	{
		int i = 0;
		for(;i<2*conf->dfts->bins_sphere;++i){
			mask[i] = false;
		}
		for(;i<conf->dfts->num_bins;++i){
			mask[i] = true;
		}
	}

	//calculate distinct part of the density with actual delta in self part
	dfts_minimize_component_mask(conf->dfts,1,mask);
	
	//replace delta with narrow gaußian
	dfts_get_density(conf->dfts,rho);
	for(int i=0;i<conf->dfts->bins_sphere;++i){
		double r=i*conf->dfts->dr;
		rho[0][i] = a0*exp(-ALPHA*r*r);
		if( rho[0][i] < ZERO ){
			rho[0][i] = ZERO;
		}
	}
	dfts_set_density(conf->dfts,(const double**)rho);
	double td[2];
	dfts_get_mean_density(conf->dfts,td,NULL);
	for(int i=0;i<conf->dfts->bins_sphere;++i){
		rho[0][i] /= td[0];
	}
	dfts_set_density(conf->dfts,(const double**)rho);
	dfts_get_mean_density(conf->dfts,conf->mean_density,NULL);
	free(rho[0]);
	free(mask);
}

__host__ void pfts_renormalize(PFTS_t conf){
	double td[2];
	dfts_get_mean_density(conf->dfts,td,NULL);
	mul<<<conf->dfts->kc.blocks_1,conf->dfts->kc.threads>>>(conf->dfts->num_bins,conf->dfts->density[0],conf->mean_density[0]/td[0],conf->dfts->density[0]);
	cudaDeviceSynchronize();
	//mul<<<conf->dfts->kc.blocks_1,conf->dfts->kc.threads>>>(conf->dfts->num_bins,conf->dfts->density[1],conf->mean_density[1]/td[1],conf->dfts->density[1]);
	//cudaDeviceSynchronize();
}

__host__ void pfts_rt_picard(PFTS_t conf, double *j, double *grad){
	cudaDeviceSynchronize();
	const double *_j[2] = {j,j+conf->dfts->num_bins};
	pfts_set_current(conf,_j);
	cudaDeviceSynchronize();
	pfts_rt<<<1,1>>>(conf,conf->results,true);
	cudaDeviceSynchronize();
	do_check( cudaMemcpy(grad,conf->gradient[0],2*conf->dfts->num_bins*sizeof(double),cudaMemcpyDeviceToDevice) == cudaSuccess);
}

__host__ void pfts_picard_initial_guess(PFTS_t conf, double *j){
	pfts_fdot<<<1,1>>>(conf,conf->results,true);
	cudaDeviceSynchronize();
#ifndef DDFT
	pfts_p_exc<<<1,1>>>(conf,conf->results+1,true);
	cudaDeviceSynchronize();
#endif
	op_kernel<<<conf->dfts->kc.blocks_2,conf->dfts->kc.threads>>>(conf->dfts->num_bins,conf->gradient[0],conf->dfts->density[0],j,OP_MUL);
	cudaDeviceSynchronize();
	mul<<<conf->dfts->kc.blocks_2,conf->dfts->kc.threads>>>(conf->dfts->num_bins,j,-1,j);
}

__host__ void pfts_picard_direction(PFTS_t conf, double *j_now, double *j_next){
	const double *_j[2] = {j_now,j_now+conf->dfts->num_bins};
	pfts_set_current(conf,_j);
	cudaDeviceSynchronize();
	pfts_fdot<<<1,1>>>(conf,conf->results,true);
	cudaDeviceSynchronize();
	pfts_p_exc<<<1,1>>>(conf,conf->results,true);
	cudaDeviceSynchronize();
	op_kernel<<<conf->dfts->kc.blocks_2,conf->dfts->kc.threads>>>(conf->dfts->num_bins,conf->gradient[0],conf->dfts->density[0],j_next,OP_MUL);
	cudaDeviceSynchronize();
	mul<<<conf->dfts->kc.blocks_2,conf->dfts->kc.threads>>>(conf->dfts->num_bins,j_next,-1,j_next);
}

__host__ void pfts_alpha_max(PFTS_t conf, double *j_next, double *alpha_max, double *alpha){
	cudaMemcpy(alpha,alpha_max,sizeof(double),cudaMemcpyDeviceToDevice);
}

__host__ void pfts_minimize(PFTS_t conf){
	/*Picard_t p = picard_init(conf,conf->dfts->num_bins,2,0,NULL);
	double *j_init[2];
	cudaMalloc(j_init,2*conf->dfts->num_bins*sizeof(double));
	j_init[1] = j_init[0]+conf->dfts->num_bins;
	pfts_picard_initial_guess(conf, j_init[0]);
	picard_minimize(p,j_init);
	pfts_set_current(conf,(const double **)j_init);
	picard_destroy(p);
	cudaFree(j_init[0]);*/
	pfts_picard_initial_guess(conf,conf->current[0]);
	cudaDeviceSynchronize();
}

}
