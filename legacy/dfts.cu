#include <stdio.h>
#include <cuhelp.h>
#include "config.h"
#include "debug.h"
#include "data.h"
#include "free_energy.h"
#include "memory.h"
#include "picard.h"
#include "weights.h"
#include "tool.h"

#define cat(a,b) a##b
#define con(a,b) cat(a,b)

__global__ void dfts_pot_k(size_t num, double dr, double **density, double *potential, double *mu, double **out, double **grad){
	size_t i = blockIdx.x*blockDim.x+threadIdx.x;
	size_t c = blockIdx.y;
	if( i < num ){
		out[c][i] = potential[i]-(*mu);
		if( grad ){
			grad[c][i] += out[c][i];
		}
		out[c][i] *= density[c][i]*weight_4pir2(0,num-1,i,dr);
	}
}

__global__ void dfts_potential(DFTS_t conf, bool grad, double *f){
	double **g = NULL;
	if( grad ) g = conf->gradient;
	dfts_pot_k<<<conf->kc.blocks_2,conf->kc.threads>>>
		(conf->num_bins,
		 conf->dr,
		 conf->density,
		 conf->potential,
		 &(conf->chemical_potential),
		 conf->buffer,
		 g);
	reduce_add<<<1,1>>>(2*conf->num_bins,conf->buffer[0],NULL,f,OP_NOP);
}

__global__ void dfts_omega_(DFTS_t conf, bool grad, double *f){
	dfts_free_energy<<<1,1>>>(conf,grad,f+1);
	dfts_potential<<<1,1>>>(conf, grad,f+2);
	op_kernel<<<1,1>>>(1,f+2,f+1,f,OP_ADD);
}

__global__ void dfts_clear_mask(DFTS_t conf){
	int i=0;
	for(i=0;i<conf->num_bins;++i){
		conf->min_mask[i] = true;
	}
}

__global__ void dfts_calculate_dphi_dn(size_t num, double radius_sphere, double *wd[3], double *psi[3]);
__global__ void dfts_calculate_grad_fexc(size_t num, double dr, int rs, double *dphi[3], double *psi[3]);

extern "C" {
__host__ double dfts_omega(DFTS_t conf, double **grad){
	double f = 0;
	double **g = NULL;
	if( grad ){
		g = conf->gradient;
	}
	dfts_omega_<<<1,1>>>(conf->_self, g, conf->results);
	cudaMemcpy(&f, conf->results, sizeof(double), cudaMemcpyDeviceToHost);
	if( grad ){
		cudaMemcpy(grad[0],conf->gradient[0],conf->num_bins*sizeof(double),cudaMemcpyDeviceToHost);
		cudaMemcpy(grad[1],conf->gradient[1],conf->num_bins*sizeof(double),cudaMemcpyDeviceToHost);
	}
	return f;
}

__host__ void dfts_minimize_component_mask(DFTS_t conf, size_t component, bool *mask){
	do_check( cudaMemcpy(conf->min_mask,mask,conf->num_bins*sizeof(bool),cudaMemcpyHostToDevice) == cudaSuccess );
	dfts_sync(conf);
	Picard_t p = dfts_picard_init(conf,1,component);
	dfts_picard_minimize(p);
	dfts_picard_destroy(p);
}

__host__ void dfts_minimize_component(DFTS_t conf, size_t component){
	dfts_clear_mask<<<1,1>>>(conf->_self);
	dfts_sync(conf);
	Picard_t p = dfts_picard_init(conf,1,component);
	dfts_picard_minimize(p);
	dfts_picard_destroy(p);
}

__host__ void dfts_minimize(DFTS_t conf){
	dfts_clear_mask<<<1,1>>>(conf->_self);
	dfts_sync(conf);
	Picard_t p = dfts_picard_init(conf,2,0);
	dfts_picard_minimize(p);
	dfts_picard_destroy(p);
}

__host__ void dfts_get_density(DFTS_t conf, double **density){
	for(int i=0;i<2;++i){
		do_check(cudaMemcpy(density[i],conf->density[i],conf->num_bins*sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess);
	}
}

__host__ void dfts_set_density_component(DFTS_t conf, const double **density, int mask){
	cudaDeviceSynchronize();
	for(int i=0;i<2;++i){
		if( mask & 1 ){
			do_check(cudaMemcpy(conf->density[i],density[i],conf->num_bins*sizeof(double),cudaMemcpyHostToDevice) == cudaSuccess);
		}
		mask = mask >> 1;
	}
}

__host__ void dfts_set_density(DFTS_t conf, const double **density){
	dfts_set_density_component(conf,density,3);
}

__host__ void dfts_set_potential(DFTS_t conf, double *potential){
	do_check( cudaMemcpy(conf->potential,potential,conf->num_bins*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess );
}

__host__ void dfts_set_chemical_potential(DFTS_t conf, double mu){
	conf->chemical_potential = mu;
	dfts_sync(conf);
}

__host__ void dfts_set_selfinteraction(DFTS_t conf, DFTS_selfinteraction self){
	conf->selfinteraction = self;
	dfts_sync(conf);
}

__host__ void dfts_log_wd(DFTS_t conf, const char filename[]){
	double *wd = (double*)malloc(2*conf->num_bins*dfts_num_wd*sizeof(double));
	cudaMemcpy(wd,conf->weighted_density[0][0],2*dfts_num_wd*conf->num_bins*sizeof(double),cudaMemcpyDeviceToHost);
	FILE *out = fopen(filename,"w");
	for(int i=0;i<conf->num_bins;++i){
		fprintf(out,"%f\t",i*conf->dr);
		for(int alpha=0;alpha<2*dfts_num_wd;++alpha){
			fprintf(out,"%.20e\t",wd[alpha*conf->num_bins+i]);
		}
		fprintf(out,"\n");
	}
	fclose(out);
}

__global__ void dfts_calc_dphidn(DFTS_t conf){
	printf("This function is unavailable due to restructuring.\n");
	//dfts_calculate_dphi_dn<<<conf->kc.blocks_wd,conf->kc.threads>>>(conf->num_bins, conf->radius_sphere, conf->weighted_density, conf->buffer+3);
	/*dfts_calculate_grad_fexc<<<conf->kc.blocks_3,conf->kc.threads>>>(conf->num_bins, conf->dr, conf->bins_sphere, conf->buffer, conf->buffer+3);
	op_add_kernel<<<conf->kc.blocks_1,conf->kc.threads>>>(conf->num_bins,conf->buffer[3],conf->buffer[4],conf->buffer[5],OP_ADD);
	dfts_extrapolate<<<1,1>>>(conf->num_bins-conf->bins_sphere,conf->bins_sphere,2*conf->bins_sphere+1,conf->buffer[5],conf->buffer[0]);*/
}

__host__ void dfts_log_psi(DFTS_t conf, const char filename[]){
	dfts_calc_dphidn<<<1,1>>>(conf->_self);
	double *wd = (double*)malloc(conf->num_bins*dfts_num_wd*sizeof(double));
	cudaMemcpy(wd,conf->buffer[3],dfts_num_wd*conf->num_bins*sizeof(double),cudaMemcpyDeviceToHost);
	FILE *out = fopen(filename,"w");
	for(int i=0;i<conf->num_bins;++i){
		fprintf(out,"%f\t%.20e\t%.20e\t%.20e\t%.20e\n",i*conf->dr,wd[i],wd[conf->num_bins+i],wd[2*conf->num_bins+i],wd[3*conf->num_bins+i]);
	}
	fclose(out);
}

__host__ double dfts_get_mean_density(DFTS_t conf, double *total_density, double *mean_density){
	double sum;
	double *td;
	if( total_density == NULL ){
		td = (double *)calloc(2,sizeof(double));
	}else{
		td = total_density;
	}
	dfts_get_weights<<<conf->kc.blocks_1,conf->kc.threads>>>(0,conf->num_bins-1,conf->dr,conf->buffer[0]);
	reduce_add<<<1,1>>>(conf->num_bins,conf->buffer[0],conf->density[0],conf->results,OP_MUL);
	reduce_add<<<1,1>>>(conf->num_bins,conf->buffer[0],conf->density[1],conf->results+1,OP_MUL);
	reduce_add<<<1,1>>>(conf->num_bins,conf->buffer[0],NULL,conf->results+2,OP_NOP);
	double volume = 0.0;
	do_check( cudaMemcpy(&volume,conf->results+2,sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess );
	do_check( cudaMemcpy(td,conf->results,2*sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess );
	if( mean_density != NULL ){
		mean_density[0] = total_density[0]/volume;
		mean_density[1] = total_density[1]/volume;
	}
	sum = td[0]+td[1];
	if( total_density == NULL ){
		free(td);
	}
	return sum/volume;
}

__host__ double dfts_get_chemical_potential_mean_density(DFTS_t conf, double mean_density, double mean_volume_fraction){
	double radius = conf->radius_sphere;
	if( mean_density < 0 ){
		if( mean_volume_fraction < 0 ){
			printf("neither mean_density nor mean_volume_fraction make sense. Setting mu = 0.0\n");
			return 0.0;
		}
		mean_density = mean_volume_fraction/(4./3.*M_PI*radius*radius*radius);
	}
	double n2 = mean_density*4*M_PI*radius*radius;
	double n3 = n2*radius/3.;
	return (1.0/216.0)*(2*(n2*n2*n2)*n3*((n3*n3*n3) - 3*(n3*n3) + n3 + ((n3 - 1)*(n3 - 1))*log(1 - n3)) + (n2*n2*n2)*(n3 - 1)*(2*(n3*n3*n3) - 3*(n3*n3) - n3*(3*(n3*n3) - 2*n3 + 2*(n3 - 1)*log(1 - n3)) + 2*n3 + 2*((n3 - 1)*(n3 - 1))*log(1 - n3)) - 6*(n2*n2)*(n3*n3)*(n3 - 1)*((n3*n3) - 5*n3 + 2*(n3 - 1)*log(1 - n3)) + 6*(n2*n2)*n3*(n3 - 1)*(2*n3*(n3 + (n3 - 1)*(n3 + log(1 - n3) - 1) - 1) - (n3 - 1)*((n3*n3) - 2*n3 + 2*(n3 - 1)*log(1 - n3))) - 36*n2*(n3*n3*n3)*((n3 - 1)*(n3 - 1)) - 18*n3*(n3 - 1)*((n2*n2)*((n3*n3*n3) - 3*(n3*n3) + n3 + ((n3 - 1)*(n3 - 1))*log(1 - n3)) + 4*n2*n3*(n3 - 1)*(-(n3*n3) + 5*n3 - 2*(n3 - 1)*log(1 - n3)) + 12*(n3*n3)*((n3 - 1)*(n3 - 1))*log(1 - n3)))/((n3*n3*n3)*((n3 - 1)*(n3 - 1)*(n3 - 1)))+log(mean_density);
}

}//end extern "C"
