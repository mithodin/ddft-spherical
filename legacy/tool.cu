#include <stdio.h>
#include <cuhelp.h>
#include <assert.h>
#include "debug.h"

/*
__global__ void dfts_extrapolate_k(size_t num, double *a, double *b, double *y){
        int i = blockIdx.x*blockDim.x+threadIdx.x;
        if( i >= num ){
                return;
        }
        y[i] = *b+*a*i;
}

__global__ void dfts_extrapolate_a_k(int fit, double *y, double *ym, double *out){
        int i = blockIdx.x*blockDim.x+threadIdx.x;
        if( i >= fit ){
                return;
        }
        out[i] = (i-fit)*(y[i]-(*ym/fit));
}

__global__ void dfts_extrapolate_calc_ab(size_t fit, double *ym, double *a, double *b){
        *a = *a*12.0/(fit*(fit*fit-1.0));
        *b = *ym/fit+(fit+1)/2*(*a);
}

__global__ void dfts_extrapolate(size_t num, size_t extrapolate, size_t fit, double *y, double *buffer){
        int blocks,threads,blocks_e,threads_e;
        distribute(fit,&blocks,&threads);
        distribute(extrapolate,&blocks_e,&threads_e);
        double *ym = buffer;
        double *a = buffer+1;
        double *b = buffer+2;
        double *my_buf = buffer+3;
        int i0 = num-fit;
        assert( fit+3 <= num );
        assert( fit%2 == 1 );
        reduce_add<<<1,1>>>(fit,y+i0,NULL,ym,OP_NOP);
        dfts_extrapolate_a_k<<<blocks,threads>>>(fit,y+i0,ym,my_buf);
        reduce_add<<<1,1>>>(fit,my_buf,NULL,a,OP_NOP);
        dfts_extrapolate_calc_ab<<<1,1>>>(fit,ym,a,b);
        dfts_extrapolate_k<<<blocks_e,threads_e>>>(extrapolate,a,b,y+num);
}
*/

__global__ void dfts_extrapolate_k(size_t num, double *a, double *b, double *y0, double *y){
        int i = blockIdx.x*blockDim.x+threadIdx.x;
	if( i >= num ){
		return;
	}
	int x = i+1;
	y[i] = *y0+((*a)*x+(*b))*x;
}

__global__ void dfts_extrapolate_calc_ab(double *yx, double *yx2, long x2, long x3, long x4, double *a, double *b){
	double denom = x2*x4-x3*x3;
	*a = (*yx2*x2+*yx*x3)/denom;
	*b = (*yx*x4+*yx2*x3)/denom;
}

__global__ void dfts_extrapolate_calc_yixi(size_t num, double *y, double *yx, double *yx2){
        int i = blockIdx.x*blockDim.x+threadIdx.x;
	if( i >= num ){
		return;
	}
	int x = -(num-1-i);
	yx[i] = (y[i]-y[num-1])*x;
	yx2[i] = yx[i]*x;
}

__global__ void dfts_extrapolate(size_t num, size_t extrapolate, size_t fit, double *y, double *buffer){
        int blocks,threads,blocks_e,threads_e;
        distribute(fit,&blocks,&threads);
        distribute(extrapolate,&blocks_e,&threads_e);
	double *yx_a = buffer;
	double *yx2_a = buffer+fit;
	double *yx = yx2_a+fit;
	double *yx2 = yx+1;
	double *a = yx2+1;
	double *b = a+1;
	long x2 = ((fit-1)*fit*(2*fit-1))/6;
	long x3 = ((fit-1)*(fit-1)*fit*fit)/4;
	long x4 = ((fit-1)*fit*(2*fit-1)*(3*fit*fit-3*fit-1))/30;
	dfts_extrapolate_calc_yixi<<<blocks,threads>>>(fit,y-fit+num,yx_a,yx2_a);
	reduce_add<<<1,1>>>(fit,yx_a,NULL,yx,OP_NOP);
	reduce_add<<<1,1>>>(fit,yx2_a,NULL,yx2,OP_NOP);
	dfts_extrapolate_calc_ab<<<1,1>>>(yx,yx2,x2,x3,x4,a,b);
	dfts_extrapolate_k<<<blocks_e,threads_e>>>(extrapolate,a,b,y+num-1,y+num);
}

__host__ void dfts_log_array(size_t num, double *arr, const char fname[]){
	FILE *out = fopen(fname,"w");
	double *data = (double*)malloc(num*sizeof(double));
	do_check( cudaMemcpy(data,arr,num*sizeof(double),cudaMemcpyDeviceToHost) == cudaSuccess );
	for(int i=0;i<num;++i){
		fprintf(out,"%d\t%.20e\n",i,data[i]);
	}
	fclose(out);
	free(data);
}
