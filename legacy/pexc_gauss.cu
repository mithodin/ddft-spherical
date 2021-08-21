#define DELAY 30000
__device__ double my_prefactor = 1.0;
__constant__ double my_memtime = 6e-4;
__constant__ double my_cutoff = 5e-2;

//helpers
__global__ void pfts_mul_r(size_t num, int exp, double *buffer, double dr, double *out){
	const int i = blockIdx.x*blockDim.x+threadIdx.x;
	if( i >= num ){
		return;
	}else if( i == 0 ){//if buffer[0] != 0 in case of division, we're fucked anyways
		out[i] = 0.0;
	}else{
		out[i] = buffer[i];
		double r = i*dr;
		for(;exp > 0;--exp){
			out[i] *= r; 
		}
		for(;exp < 0;++exp){
			out[i] /= r;
		}
	}
}

__constant__ double pexc_coefficients[3] = {-5.0/2.0, 4.0/3.0, -1.0/12.0};
__constant__ size_t pexc_num_coeff = 3;

__global__ void diffuse_sphere(size_t num, double *in, double f, double *out){
	const int i = blockIdx.x*blockDim.x+threadIdx.x;
	if( i >= num ){
		return;
	}
	if( i == 0 ){
		out[i] = 0.0;
		return;
	}
	out[i] = (1.+f*pexc_coefficients[0])*in[i];
	for(int os=1;os<pexc_num_coeff;++os){
		int j = i+os;
		if( j < num ){ //ignore everything beyond the last bin
			out[i] += f*in[j]*pexc_coefficients[os];
		}
		
		j = i-os;
		if( j < 0 ){
			out[i] -= f*in[-j]*pexc_coefficients[os]; //mirror negative for indices < 0
		}else if( j != 0 ){
			out[i] += f*in[j]*pexc_coefficients[os];
		}
	}
}

__global__ void pfts_memory_propagate(PFTS_t conf){
	size_t thr = conf->dfts_gpu->kc.threads;
	dim3 bl1 = conf->dfts_gpu->kc.blocks_1;
	PFTS_Pexc_t pcnf = conf->pexc_conf;
	int iter = 0;
	diffuse_sphere<<<bl1,thr>>>(conf->dfts_gpu->num_bins,conf->memory,pcnf->gauss_f,conf->buffer[iter]);
	for(++iter;iter<pcnf->iterations;++iter){
		diffuse_sphere<<<bl1,thr>>>(conf->dfts_gpu->num_bins,conf->buffer[1-(iter%2)],pcnf->gauss_f,conf->buffer[iter%2]);
	}
	mul<<<bl1,thr>>>(conf->dfts_gpu->num_bins,conf->buffer[1-(iter%2)],pcnf->timestep,conf->memory);
}


//calculate the density update due to p_exc
__global__ void pfts_pexc_update(PFTS_t conf, double **density_update){
	if( conf->timestep < DELAY ) return;
	size_t thr = conf->dfts_gpu->kc.threads;
	dim3 bl1 = conf->dfts_gpu->kc.blocks_1;
	dim3 bl2 = conf->dfts_gpu->kc.blocks_2;
	op_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->current[0], conf->current[1], conf->current_sum, OP_ADD); //calculate total current
	op_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->dfts_gpu->density[0], conf->dfts_gpu->density[1], conf->buffer[1], OP_ADD); //calculate total density in buffer+1
	op_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->buffer[1], conf->memory, conf->buffer[0], OP_MUL); // rho * memory in buffer+0
	central_diff<<<bl1,thr>>>(conf->dfts_gpu->num_bins, (const double **)conf->buffer, conf->dfts_gpu->dr, conf->buffer+2); // grad rho * mem in buffer+2
	pfts_mul_r<<<bl1,thr>>>(conf->dfts_gpu->num_bins, -1, conf->buffer[2], conf->dfts_gpu->dr, conf->buffer[2]); // ( grad rho * mem ) / r in buffer+2
	op_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->buffer[2], conf->buffer[1], conf->buffer[2], OP_DIV); // ( grad rho * mem ) / (r * rho) in buffer+2
	pfts_mul_r<<<bl1,thr>>>(conf->dfts_gpu->num_bins, -2, conf->memory, conf->dfts_gpu->dr, conf->buffer[3]); // mem / r^2 in buffer+3
	op_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->buffer[2], conf->buffer[3], conf->buffer[2], OP_SUB); // [ grad rho * mem / (r * rho) - mem / r^2 ] in buffer+2, buffer+3 free
	mul<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->buffer[2], -conf->pexc_conf->prefactor*my_prefactor, conf->buffer[0]); //d P_exc / d_J in buffer+0
	divergence<<<bl1,thr>>>(conf->dfts_gpu->num_bins, (const double **)conf->buffer, conf->dfts_gpu->dr, conf->buffer+1); //div d P_exc / d_J in buffer+1
	central_diff<<<bl2,thr>>>(conf->dfts_gpu->num_bins, (const double **)conf->dfts_gpu->density, conf->dfts_gpu->dr, conf->buffer+2); //grad rho_i in buffer+(2,3)
	op_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->buffer[2], conf->buffer[0], conf->buffer[2], OP_MUL);
	op_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->buffer[3], conf->buffer[0], conf->buffer[3], OP_MUL);
	op_add_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->dfts_gpu->density[0], conf->buffer[1], conf->buffer[2], OP_MUL);
	op_add_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->dfts_gpu->density[1], conf->buffer[1], conf->buffer[3], OP_MUL);
	//mul<<<bl2,thr>>>(conf->dfts_gpu->num_bins, conf->buffer[2], -1, conf->buffer[2]);
	op_add_kernel<<<bl2,thr>>>(conf->dfts_gpu->num_bins, conf->buffer[2], NULL, density_update[0], OP_NOP);
}

//update the memory terms
__global__ void pfts_update_memory(PFTS_t conf){
	if( conf->timestep < DELAY ) return;
	//double time = conf->timestep*conf->dt;
	//my_prefactor = time < my_cutoff ? (exp(time/my_memtime)-1.0)/(exp(my_cutoff/my_memtime)-1.0) : 1.0;
	//printf("pf = %e\n",my_prefactor);
	size_t thr = conf->dfts_gpu->kc.threads;
	dim3 bl1 = conf->dfts_gpu->kc.blocks_1;
	op_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->current[0], conf->current[1], conf->current_sum, OP_ADD);
	op_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->dfts_gpu->density[0], conf->dfts_gpu->density[1], conf->buffer[1], OP_ADD);
	op_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->current_sum, conf->buffer[1], conf->current_sum, OP_DIV); //calculate v
	divergence<<<bl1,thr>>>(conf->dfts_gpu->num_bins, (const double **)&(conf->current_sum), conf->dfts_gpu->dr, conf->buffer);
	pfts_mul_r<<<bl1,thr>>>(conf->dfts_gpu->num_bins, 1, conf->buffer[0], conf->dfts_gpu->dr, conf->buffer[0]);
	op_add_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->buffer[0], conf->buffer[1], conf->memory, OP_MUL);
	pfts_memory_propagate<<<1,1>>>(conf); //propagate memory with diffusion equation
}

__global__ void pfts_p_exc_k(size_t num, double *div_v, double *mem, double dr, double *out){
        int i = blockIdx.x*blockDim.x+threadIdx.x;
	if( i >= num ) return;
	double w = weight_4pir2(0,num-1,i,dr);
	out[i] = w*div_v[i]*mem[i];
}

//calculate p_exc and grad p_exc
__global__ void pfts_p_exc(PFTS_t conf, double *result, bool grad){
	if( conf->timestep < DELAY ){
		*result = 0.0;
		return;
	}
	size_t thr = conf->dfts_gpu->kc.threads;
	dim3 bl1 = conf->dfts_gpu->kc.blocks_1;
	op_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->current[0], conf->current[1], conf->current_sum, OP_ADD); //calculate total current
	op_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->dfts_gpu->density[0], conf->dfts_gpu->density[1], conf->buffer[1], OP_ADD); //calculate total density in buffer+1
	op_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->buffer[1], conf->memory, conf->buffer[0], OP_MUL); // rho * memory in buffer+0
	if( grad ){
		//calculate gradient
		central_diff<<<bl1,thr>>>(conf->dfts_gpu->num_bins, (const double **)conf->buffer, conf->dfts_gpu->dr, conf->buffer+2); // grad rho * mem in buffer+2
		pfts_mul_r<<<bl1,thr>>>(conf->dfts_gpu->num_bins, -1, conf->buffer[2], conf->dfts_gpu->dr, conf->buffer[2]); // ( grad rho * mem ) / r in buffer+2
		op_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->buffer[2], conf->buffer[1], conf->buffer[2], OP_DIV); // ( grad rho * mem ) / (r * rho) in buffer+2
		pfts_mul_r<<<bl1,thr>>>(conf->dfts_gpu->num_bins, -2, conf->memory, conf->dfts_gpu->dr, conf->buffer[3]); // mem / r^2 in buffer+3
		op_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->buffer[2], conf->buffer[3], conf->buffer[2], OP_SUB); // [ grad rho * mem / (r * rho) - mem / r^2 ] in buffer+2, buffer+3 free
		mul<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->buffer[2], -conf->pexc_conf->prefactor*my_prefactor, conf->buffer[2]);
		//mul<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->buffer[2], conf->pexc_conf->prefactor*my_prefactor, conf->buffer[2]);
		op_add_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->buffer[2], NULL, conf->gradient[0], OP_NOP);
		op_add_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->buffer[2], NULL, conf->gradient[1], OP_NOP);

	}
	pfts_mul_r<<<bl1,thr>>>(conf->dfts_gpu->num_bins,-1, conf->buffer[0], conf->dfts_gpu->dr, conf->buffer[0]); // rho / r * memory in buffer+0;
	op_kernel<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->current_sum, conf->buffer[1], conf->buffer[2], OP_DIV); //buffer+2 has velocity profile
	divergence<<<bl1,thr>>>(conf->dfts_gpu->num_bins, (const double **)conf->buffer+2, conf->dfts_gpu->dr, conf->buffer+3); //div v in buffer+3
	pfts_p_exc_k<<<bl1,thr>>>(conf->dfts_gpu->num_bins, conf->buffer[3], conf->buffer[0], conf->dfts_gpu->dr, conf->buffer[2]);
	reduce_add<<<1,1>>>(conf->dfts_gpu->num_bins, conf->buffer[2], NULL, result, OP_NOP);
	mul<<<1,1>>>(1,result,-conf->pexc_conf->prefactor*my_prefactor,result);
}
