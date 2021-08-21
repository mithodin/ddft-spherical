#include <dfts.h>

__global__ void central_diff(size_t num, const double **f, double dx, double **out){
	const int i = blockIdx.x*blockDim.x+threadIdx.x;
	const int block = blockIdx.y;
	if( i >= num ){
		return;
	}
	if( i == 0 ){
		//out[block][i] = f[block][1]-f[block][0];
		out[block][i] = 0.0;
	}else if( i == num-1 ){
		out[block][i] = (f[block][i-2]-4.*f[block][i-1]+3.*f[block][i])/2.;
	}else{
		out[block][i] = (f[block][i+1]-f[block][i-1])/2.;
	}
	out[block][i] /= dx;
}

__global__ void laplace(size_t num, const double **f, double dx, double **out){
	const int i = blockIdx.x*blockDim.x+threadIdx.x;
	const int block = blockIdx.y;
	if( i >= num ){
		return;
	}
	if( i == 0 ){
		out[block][i] = 6*(f[block][1] - f[block][0]);
	}else if( i == num-1 ){
		out[block][i] = (f[block][i-2]-2*f[block][i-1]+f[block][i])+(f[block][i-2]-4.*f[block][i-1]+3.*f[block][i])/i;
	}else{
		out[block][i] = ((f[block][i + 1] - 2.*f[block][i] + f[block][i - 1]) + (f[block][i + 1] - f[block][i - 1])/i);
	}
	out[block][i] /= (dx*dx);
}

__global__ void divergence(size_t num, const double **f, double dr, double **out){
	const int i = blockIdx.x*blockDim.x+threadIdx.x;
	const int block = blockIdx.y;
	if( i >= num ){
		return;
	}
	if( i == 0 ){
		//out[block][i] = 3*f[block][1];
		out[block][i] = 0.0;
	}else if( i == num-1 ){
		//out[block][i] = 2*f[block][i]/i+f[block][i]-f[block][i-1];
		out[block][i] = 2*f[block][i]/i+(3*f[block][i]-4*f[block][i-1]+f[block][i-2])/2.;
	}else{
		out[block][i] = 2*f[block][i]/i+(f[block][i+1]-f[block][i-1])/2.0;
	}
	out[block][i] /= dr;
}

__global__ void min_val(size_t num, double **f, double min){
	const int i = blockIdx.x*blockDim.x+threadIdx.x;
	const int block = blockIdx.y;
	if( i >= num ){
		return;
	}
	if( f[block][i] < min ){
		f[block][i] = min;
	}
}
