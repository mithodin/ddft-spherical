#include <stdio.h>
#include <cuhelp.h>
#include "config.h"
#include "debug.h"
#include "data.h"
#include "weights.h"
#include "tool.h"
#include "dfts.h"

__device__ double dfts_wb2_dphi2(double n3){
	if( n3 == 0 ){
		return 0.0;
	}
	return -2.*(1+log(1-n3)/n3)-(2/n3-1+2*(1-n3)*log(1-n3)/(n3*n3));
}

__device__ double dfts_wb2_dphi3(double n3){
	if( n3 == 0 ){
		return 4./3.;
	}
	return (-4/(n3*n3)+6/n3-4-4*(1-n3)*(1-n3)*log(1-n3)/(n3*n3*n3))
		+(-4/n3+6-4*(1-n3)*log(1-n3)/(n3*n3));
}

__device__ double dfts_wb2_phi2(double n3){
	if( n3 == 0.0 ){
		return 0.0;
	}
	if( n3 == 1.0 ){
		return 1.0;
	}
	return (2-n3+2*(1.-n3)*log(1.-n3)/n3);
}

__device__ double dfts_wb2_phi3(double n3){
	if( n3 == 0.0 ){
		return 0.0;
	}
	if( n3 == 1.0 ){
		return 1.0;
	}
	return (2/n3-3+2*n3+2*(1.-n3)*(1.-n3)*log(1.-n3)/(n3*n3));
}

__device__ double dfts_dphi_dn(dfts_wd alpha, double n2, double n3, double n2v, double n11, double radius_sphere){
	double res = 0.0;
	switch(alpha){
		case dfts_n2:
			res = -log(1.-n3)/(4*M_PI*radius_sphere*radius_sphere)
				+(1+dfts_wb2_phi2(n3)/3)/(1-n3)*2*n2/(4*M_PI*radius_sphere)
				+(3*n2*n2-3*n2v*n2v)*(1-dfts_wb2_phi3(n3)/3)/(24*M_PI*(1-n3)*(1-n3));
			break;
		case dfts_n3:
			res = n2/(4*M_PI*radius_sphere*radius_sphere*(1-n3))
				+(n2*n2-n2v*n2v)/(4*M_PI*radius_sphere)*( (1+dfts_wb2_phi2(n3)/3)/((1-n3)*(1-n3))
						     +dfts_wb2_dphi2(n3)/(3*(1-n3)) )
				+(n2*n2*n2-3*n2*n2v*n2v+9*(3*n11*n11*n11-n11*n2v*n2v))/(24*M_PI*(1-n3)*(1-n3))
						*( 2*(1-dfts_wb2_phi3(n3)/3)/(1-n3)
						   -dfts_wb2_dphi3(n3)/3 );
			break;
		case dfts_n2v:
			res = -n2v/(2*M_PI*radius_sphere)*(1+dfts_wb2_phi2(n3)/3)/(1-n3)
				-(6*n2*n2v+18*n11*n2v)*(1-dfts_wb2_phi3(n3)/3)/(24*M_PI*(1-n3)*(1-n3));
			break;
		case dfts_n11:
			res = 9*(9*n11*n11-n2v*n2v)*(1-dfts_wb2_phi3(n3)/3)/(24*M_PI*(1-n3)*(1-n3));
			break;
	}
	return res;
}

__global__ void dfts_calculate_dphi_dn(size_t num, double radius_sphere, double *wd[2][dfts_num_wd], double *psi[2][dfts_num_wd]){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int component = blockIdx.y;
	int alpha = blockIdx.z;
	if( i >= num ) return;
	psi[component][alpha][i] = dfts_dphi_dn((dfts_wd)alpha,wd[component][dfts_n2][i],wd[component][dfts_n3][i],wd[component][dfts_n2v][i], wd[component][dfts_n11][i], radius_sphere);
}

__global__ void dfts_calculate_phi(size_t num, double dr, double radius_sphere, double *wd[2][dfts_num_wd], double *phi[2]){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int component = blockIdx.y;
	if( i >= num ) return;
	double n3 = wd[component][dfts_n3][i];
	double n2 = wd[component][dfts_n2][i];
	double n2v = wd[component][dfts_n2v][i];
	double n11 = wd[component][dfts_n11][i];
	double n0 = n2/(4*M_PI*radius_sphere*radius_sphere);
	double n1 = n2/(4*M_PI*radius_sphere);
	double n1v = n2v/(4*M_PI*radius_sphere);

	phi[component][i] = weight_4pir2(0,num-1,i,dr)*( -n0*log(1.-n3)
				+(n1*n2-n1v*n2v)*(1+dfts_wb2_phi2(n3)/3)/(1-n3)
				+(n2*n2*n2-3*n2*n2v*n2v+9*(3*n11*n11*n11-n11*n2v*n2v))*(1-dfts_wb2_phi3(n3)/3)/(24.*M_PI*(1-n3)*(1-n3)) );
}

__global__ void dfts_calculate_grad_fexc(size_t num, double dr, int rs, double *dphi[2][dfts_num_wd], double *psi[2*dfts_num_wd]){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int component = blockIdx.y;
	int alpha = blockIdx.z;
	int offset = component*dfts_num_wd;
	if( i >= num-rs ) return;
	psi[alpha+offset][i] = 0.0;
	if( i == 0 && alpha != dfts_n3 ){ //limit r --> 0 must be handled specially for n2 and n2v
		psi[alpha+offset][i] = 4*M_PI*rs*rs*dr*dr*dphi[component][alpha][rs];
		return;
	}
	int lb = i-rs;
	if( lb < 0 ){
		if( alpha == dfts_n3 ){
			lb = 0;
		}else{
			lb = abs(lb);
		}
	}
	int ub = i+rs;
	for(int rp=lb;rp<=ub;++rp){
		double w = weight_psi((dfts_wd)alpha,lb,ub,i,rp,rs,dr);
		psi[alpha+offset][i] += dphi[component][alpha][rp]*w;
	}
}

__global__ void dfts_calculate_weighted_densities(size_t num, double dr, int rs, double *density[2], double *wd[2][dfts_num_wd]){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int component = blockIdx.y;
	int alpha = blockIdx.z;
	if( i >= num-rs ) return;
	wd[component][alpha][i] = 0.0;
	if( i == 0 ){ //limit r --> 0 must be handled specially for n2, n2v and n11
		if( alpha == dfts_n2 ){
			wd[component][alpha][i] = 4*M_PI*rs*rs*dr*dr*density[component][rs];
			return;
		}else if( alpha == dfts_n2v || alpha == dfts_n11 ){
			return;
		}
	}
	int lb = i-rs;
	if( lb < 0 ){
		if( alpha == dfts_n3 ){
			lb = 0;
		}else{
			lb = abs(lb);
		}
	}
	int ub = i+rs;
	for(int rp=lb;rp<=ub;++rp){
		double w = weight_wd((dfts_wd)alpha,lb,ub,i,rp,rs,dr);
		wd[component][alpha][i] += density[component][rp]*w;
	}
}

__global__ void dfts_fid_k(size_t num, double dr, double **density, double **out, double **grad){
	size_t i = blockIdx.x*blockDim.x+threadIdx.x;
	size_t c = blockIdx.y;
	if( i < num ){
		//out[c][i] = log(density[c][i]);
		if( grad ){
			//grad[c][i] = out[c][i];
			grad[c][i] = log(density[c][i]);
		}
		//out[c][i] -= 1.0;
		//out[c][i] *= density[c][i]*weight_4pir2(0,num-1,i,dr);
	}
}

__global__ void dfts_f_ideal(DFTS_t conf, bool grad, double *f){
	double **g = NULL;
	if( grad) g = conf->gradient;
	dfts_fid_k<<<conf->kc.blocks_2,conf->kc.threads>>>(conf->num_bins,conf->dr, conf->density, conf->buffer,g);
	//reduce_add<<<1,1>>>(2*conf->num_bins,conf->buffer[0],NULL,f,OP_NOP);
	*f = 0.0;
}

__global__ void dfts_f_exc(DFTS_t conf, bool grad, double *f){
	op_kernel<<<conf->kc.blocks_1,conf->kc.threads>>>(conf->num_bins,conf->density[0],conf->density[1],conf->density_sum[D_DISTINCT],OP_ADD);
	switch( conf->selfinteraction ){
		case semi_linearized:
			memcpy(conf->density_sum[D_SELF],conf->density[1],conf->num_bins*sizeof(double));
			break;
		case quenched:
			memcpy(conf->density_sum[D_SELF],conf->density[0],conf->num_bins*sizeof(double));
			break;
		case full:
			memcpy(conf->density_sum[D_SELF],conf->density_sum[D_DISTINCT],conf->num_bins*sizeof(double));
			break;
	}
	dfts_calculate_weighted_densities<<<conf->kc.blocks_wd,conf->kc.threads>>>(conf->num_bins, conf->dr, conf->bins_sphere, conf->density_sum, conf->weighted_density);
	for(int alpha=0;alpha<dfts_num_wd;++alpha){
		dfts_extrapolate<<<1,1>>>(conf->num_bins-conf->bins_sphere,conf->bins_sphere,2*conf->bins_sphere+1,conf->weighted_density[D_SELF][alpha],conf->buffer[0]);
		dfts_extrapolate<<<1,1>>>(conf->num_bins-conf->bins_sphere,conf->bins_sphere,2*conf->bins_sphere+1,conf->weighted_density[D_DISTINCT][alpha],conf->buffer[0]);
	}
	dfts_calculate_phi<<<conf->kc.blocks_2,conf->kc.threads>>>(conf->num_bins,conf->dr,conf->radius_sphere,conf->weighted_density,conf->buffer); //using buffer[0] and buffer[1]
	if( conf->selfinteraction == quenched ){
		reduce_add<<<1,1>>>(2*conf->num_bins,conf->buffer[1],conf->buffer[0],f,OP_SUB);
	}else{
		reduce_add<<<1,1>>>(2*conf->num_bins,conf->buffer[1],NULL,f,OP_NOP);
	}
	if( grad ){
		dfts_calculate_dphi_dn<<<conf->kc.blocks_wd,conf->kc.threads>>>(conf->num_bins, conf->radius_sphere, conf->weighted_density, conf->psi);
		dfts_calculate_grad_fexc<<<conf->kc.blocks_wd,conf->kc.threads>>>(conf->num_bins, conf->dr, conf->bins_sphere, conf->psi, conf->buffer);
		if( conf->selfinteraction != full ){ //only need special self component if we don't just use the full functional
			op_add_kernel<<<conf->kc.blocks_1,conf->kc.threads>>>(conf->num_bins,conf->buffer[0],conf->buffer[1],conf->buffer[2],OP_ADD);
			op_add_kernel<<<conf->kc.blocks_1,conf->kc.threads>>>(conf->num_bins,conf->buffer[3],NULL,conf->buffer[2],OP_NOP);
			dfts_extrapolate<<<1,1>>>(conf->num_bins-conf->bins_sphere,conf->bins_sphere,2*conf->bins_sphere+1,conf->buffer[2],conf->buffer[0]);
		}
		op_add_kernel<<<conf->kc.blocks_1,conf->kc.threads>>>(conf->num_bins,conf->buffer[4],conf->buffer[5],conf->buffer[6],OP_ADD);
		op_add_kernel<<<conf->kc.blocks_1,conf->kc.threads>>>(conf->num_bins,conf->buffer[7],NULL,conf->buffer[6],OP_NOP);
		dfts_extrapolate<<<1,1>>>(conf->num_bins-conf->bins_sphere,conf->bins_sphere,2*conf->bins_sphere+1,conf->buffer[6],conf->buffer[0]);

		switch( conf->selfinteraction ){
			case semi_linearized:
				op_add_kernel<<<conf->kc.blocks_1,conf->kc.threads>>>(conf->num_bins,conf->buffer[2],NULL,conf->gradient[D_SELF],OP_NOP);
				op_add_kernel<<<conf->kc.blocks_1,conf->kc.threads>>>(conf->num_bins,conf->buffer[6],NULL,conf->gradient[D_DISTINCT],OP_NOP);
				break;
			case quenched:
				op_add_kernel<<<conf->kc.blocks_1,conf->kc.threads>>>(conf->num_bins,conf->buffer[6],conf->buffer[2],conf->gradient[D_SELF],OP_SUB);
				op_add_kernel<<<conf->kc.blocks_1,conf->kc.threads>>>(conf->num_bins,conf->buffer[6],NULL,conf->gradient[D_DISTINCT],OP_NOP);
				break;
			case full:
				op_add_kernel<<<conf->kc.blocks_1,conf->kc.threads>>>(conf->num_bins,conf->buffer[6],NULL,conf->gradient[D_SELF],OP_NOP);
				op_add_kernel<<<conf->kc.blocks_1,conf->kc.threads>>>(conf->num_bins,conf->buffer[6],NULL,conf->gradient[D_DISTINCT],OP_NOP);
				break;
		}
	}
}

__global__ void dfts_free_energy(DFTS_t conf, bool grad, double *f){
	dfts_f_ideal<<<1,1>>>(conf, grad, f);
	dfts_f_exc<<<1,1>>>(conf, grad, f+1);
	op_kernel<<<1,1>>>(1,f,f+1,f,OP_ADD);
}

extern "C"{
__host__ double dfts_fexc(DFTS_t conf, double **rho, double **grad){
	double f = 0.0;
	dfts_set_density(conf,(const double **)rho);
	init_vec<<<conf->kc.blocks_2,conf->kc.threads>>>(conf->num_bins,0.0,conf->gradient[0]);
	dfts_f_exc<<<1,1>>>(conf->_self, (grad != NULL), conf->results);
	cudaMemcpy(&f,conf->results,sizeof(double),cudaMemcpyDeviceToHost);
	if( grad ){
		cudaMemcpy(grad[0],conf->gradient[0],conf->num_bins*sizeof(double),cudaMemcpyDeviceToHost);
		cudaMemcpy(grad[1],conf->gradient[1],conf->num_bins*sizeof(double),cudaMemcpyDeviceToHost);
	}
	return f;
}

__host__ double dfts_fid(DFTS_t conf, double **rho, double **grad){
	double f = 0.0;
	dfts_set_density(conf,(const double **)rho);
	dfts_f_ideal<<<1,1>>>(conf->_self, (grad != NULL), conf->results);
	cudaMemcpy(&f,conf->results,sizeof(double),cudaMemcpyDeviceToHost);
	if( grad ){
		cudaMemcpy(grad[0],conf->gradient[0],conf->num_bins*sizeof(double),cudaMemcpyDeviceToHost);
		cudaMemcpy(grad[1],conf->gradient[1],conf->num_bins*sizeof(double),cudaMemcpyDeviceToHost);
	}
	return f;
}

__host__ double dfts_f(DFTS_t conf, double **rho, double **grad){
	double f = 0.0;
	dfts_set_density(conf,(const double **)rho);
	dfts_free_energy<<<1,1>>>(conf->_self, (grad != NULL), conf->results);
	do_check( cudaMemcpy(&f,conf->results,sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess );
	if( grad ){
		do_check( cudaMemcpy(grad[0],conf->gradient[0],conf->num_bins*sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess );
		do_check( cudaMemcpy(grad[1],conf->gradient[1],conf->num_bins*sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess );
	}
	return f;
}

}
