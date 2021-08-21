#define DELAY 50000
__device__ int delay_time = 0;

__global__ void pfts_pexc_update(PFTS_t conf, double **density_update){
	//return;
if( conf->timestep < DELAY ) return;
	size_t t = conf->dfts_gpu->kc.threads;
	dim3 b = conf->dfts_gpu->kc.blocks_1;
	dim3 b2 = conf->dfts_gpu->kc.blocks_2;
	double **density_sum = conf->buffer;
	double **integral = &(conf->memory);
	//calculate total current and total density
	op_kernel<<<b,t>>>(NUM_BINS, conf->dfts_gpu->density[0], conf->dfts_gpu->density[1], *density_sum, OP_ADD);

	//calculate grad rho * grad integral
	central_diff<<<b,t>>>(NUM_BINS, (const double **)integral, conf->dfts_gpu->dr, conf->buffer+2);
	map_log<<<b,t>>>(NUM_BINS, *density_sum, *density_sum); //density_sum now holds log rho
	central_diff<<<b,t>>>(NUM_BINS, (const double **)density_sum, conf->dfts_gpu->dr, conf->buffer+3);
	op_add_kernel<<<b,t>>>(NUM_BINS, conf->buffer[3], *integral, conf->buffer[2], OP_MUL); //buffer3 is now free

	central_diff<<<b2,t>>>(NUM_BINS, (const double **)conf->dfts_gpu->density, conf->dfts_gpu->dr, conf->buffer+3);
	op_kernel<<<b,t>>>(NUM_BINS, conf->buffer[2], conf->buffer[3], conf->buffer[3], OP_MUL);
	op_kernel<<<b,t>>>(NUM_BINS, conf->buffer[2], conf->buffer[4], conf->buffer[4], OP_MUL); //buffer2 is now free
	mul<<<b2,t>>>(NUM_BINS, conf->buffer[3], -conf->pexc_conf->prefactor, conf->buffer[3]);
	op_add_kernel<<<b2,t>>>(NUM_BINS, conf->buffer[3], NULL, density_update[0], OP_NOP); //buffer3 and buffer4 are now free
	//grad rho * grad integral is now in the density update

	//calculate rho * laplace integral
	laplace<<<b,t>>>(NUM_BINS, (const double **)density_sum, conf->dfts_gpu->dr, conf->buffer+3); //buffer3 now has laplace log rho
	laplace<<<b,t>>>(NUM_BINS, (const double **)integral, conf->dfts_gpu->dr, conf->buffer+2);
	op_add_kernel<<<b,t>>>(NUM_BINS, *integral, conf->buffer[3], conf->buffer[2], OP_MUL);
	mul<<<b,t>>>(NUM_BINS, conf->buffer[2], -conf->pexc_conf->prefactor, conf->buffer[2]);
	op_add_kernel<<<b,t>>>(NUM_BINS, conf->buffer[2], conf->dfts_gpu->density[0], density_update[0], OP_MUL);
	op_add_kernel<<<b,t>>>(NUM_BINS, conf->buffer[2], conf->dfts_gpu->density[1], density_update[1], OP_MUL);
	//done
}

__global__ void pfts_update_memory(PFTS_t conf){
if( conf->timestep < DELAY ) return;
	size_t t = conf->dfts_gpu->kc.threads;
	dim3 b = conf->dfts_gpu->kc.blocks_1;
	double **density_sum = conf->buffer;
	double **rho_div_v = conf->buffer+1;
	op_kernel<<<b,t>>>(NUM_BINS, conf->current[0], conf->current[1], conf->current_sum, OP_ADD);
	op_kernel<<<b,t>>>(NUM_BINS, conf->dfts_gpu->density[0], conf->dfts_gpu->density[1], *density_sum, OP_ADD);
	op_kernel<<<b,t>>>(NUM_BINS, conf->current_sum, *density_sum, conf->current_sum, OP_DIV);
	divergence<<<b,t>>>(NUM_BINS, (const double **)&(conf->current_sum), conf->dfts_gpu->dr, rho_div_v);
	op_kernel<<<b,t>>>(NUM_BINS, *rho_div_v, *density_sum, *rho_div_v, OP_MUL);
	op_kernel<<<b,t>>>(NUM_BINS, conf->memory, *rho_div_v, conf->memory, OP_ADD);
	mul<<<b,t>>>(NUM_BINS, conf->memory, conf->pexc_conf->timestep, conf->memory);
}

__global__ void pfts_p_exc_k(size_t num, double dr, double timestep, double *rho_div_v, double *memory, double *out){
        int i = blockIdx.x*blockDim.x+threadIdx.x;
	if( i >= num ) return;
	double w = weight_4pir2(0,num-1,i,dr);
	out[i] = w*rho_div_v[i]*memory[i];
}

__global__ void pfts_p_exc(PFTS_t conf, double *result, bool grad){
if( conf->timestep < DELAY ){
	*result = 0.0;
	return;
}
	size_t t = conf->dfts_gpu->kc.threads;
	dim3 b = conf->dfts_gpu->kc.blocks_1;
	double **density_sum = conf->buffer;
	double **rho_div_v = conf->buffer+1;
	op_kernel<<<b,t>>>(NUM_BINS, conf->current[0], conf->current[1], conf->current_sum, OP_ADD);
	op_kernel<<<b,t>>>(NUM_BINS, conf->dfts_gpu->density[0], conf->dfts_gpu->density[1], *density_sum, OP_ADD);
	op_kernel<<<b,t>>>(NUM_BINS, conf->current_sum, *density_sum, conf->current_sum, OP_DIV);
	divergence<<<b,t>>>(NUM_BINS, (const double **)&(conf->current_sum), conf->dfts_gpu->dr, rho_div_v);
	op_kernel<<<b,t>>>(NUM_BINS, *rho_div_v, *density_sum, *rho_div_v, OP_MUL);
	pfts_p_exc_k<<<b,t>>>(NUM_BINS, conf->dfts_gpu->dr, conf->pexc_conf->timestep, *rho_div_v, conf->memory, conf->buffer[2]);
	reduce_add<<<1,1>>>(NUM_BINS, conf->buffer[2], NULL, result, OP_NOP);
	mul<<<1,1>>>(1,result, conf->pexc_conf->prefactor, result);
	if( grad ){
		op_kernel<<<b,t>>>(NUM_BINS, conf->memory, *density_sum, conf->buffer[2], OP_MUL);
		central_diff<<<b,t>>>(NUM_BINS, (const double **)conf->buffer+2, conf->dfts_gpu->dr, conf->buffer+3);
		op_kernel<<<b,t>>>(NUM_BINS, conf->buffer[3], *density_sum, conf->buffer[3], OP_DIV);
		mul<<<b,t>>>(NUM_BINS, conf->buffer[3], -conf->pexc_conf->prefactor, conf->buffer[3]);
		op_add_kernel<<<b,t>>>(NUM_BINS, conf->buffer[3], NULL, conf->gradient[0], OP_NOP);
		op_add_kernel<<<b,t>>>(NUM_BINS, conf->buffer[3], NULL, conf->gradient[1], OP_NOP);
	}
}
