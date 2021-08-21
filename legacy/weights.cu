#include <math.h>
#include "config.h"
#include "data.h"

__device__ __host__ double weight_4pir2(long lb, long ub, long k, double dr){
	double w = 6*k*k+1;
	if( k == lb ){
		w += 4*k;
	}else if( k == ub ){
		w -= 4*k;
	}else{
		w *= 2;
	}
	return w*M_PI/3*dr*dr*dr;
}

__global__ void dfts_get_weights(long lb, long ub, double dr, double *w){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if( i <= ub-lb ){
		w[i-lb] = weight_4pir2(lb,ub,i+lb,dr);
	}
}

__device__ __host__ double weight_n3(long lb, long ub, long r, long rp, long rs, double dr){
	double w = 0.0;
	if( r < rs ){ //n3 is split into two parts if we're close to the origin
		if( rp <= rs-r ){ //first part of the longegral
			w = weight_4pir2(lb, rs-r, rp, dr);
		}
		if( rp >= rs-r && r > 0 ){ //second part of the longegral
			lb = rs-r;
		}else{ //we're done
			return w;
		}
	}
	if( rp == lb ){
		w += -M_PI*dr*dr*dr*((30*rp+10)*(r*r-rs*rs)+(30*rp*rp*(rp+1)+15*rp+3)-r*(60*rp*rp+40*rp+10))/(60*r);
	}else if( rp == ub ){
		w += M_PI*dr*dr*dr*((30*rp-10)*(rs*rs-r*r)+(30*rp*rp*(1-rp)-15*rp+3)+r*(60*rp*rp-40*rp+10))/(60*r);
	}else{
		w += M_PI*dr*dr*dr*(6*rp*rs*rs-3*rp*(2*rp*rp+1)+r*(12*rp*rp+2)-6*rp*r*r)/(6*r);
	}
	return w;
}

__device__ __host__ double weight_n2v(long lb, long ub, long r, long rp, long rs, double dr){
	double w = 0.0;
	if( rp == lb ){
		w = M_PI/60*dr*dr/(r*r)*((rs*rs+r*r)*(30*rp+10)-(30*rp*rp*(rp+1)+15*rp+3));
	}else if( rp == ub ){
		w = M_PI/60*dr*dr/(r*r)*((rs*rs+r*r)*(30*rp-10)+(30*rp*rp*(1-rp)-15*rp+3));
	}else{
		w = M_PI/2*dr*dr/(r*r)*rp*(2*(rs*rs+r*r)-(2*rp*rp+1));
	}
	return w;
}

__device__ __host__ double weight_n2(long lb, long ub, long r, long rp, long rs, double dr){
	double w = 0.0;
	if( rp == lb ){
		w = M_PI/3*rs*dr*dr/r*(3*rp+1);
	}else if( rp == ub ){
		w = M_PI/3*rs*dr*dr/r*(3*rp-1);
	}else{
		w = 2*M_PI*rs*dr*dr/r*rp;
	}
	return w;
}

__device__ __host__ double weight_n11(long lb, long ub, long r, long rp, long rs, double dr){
	double w = 0.0;
	if( rp == lb ){
		w = -(105*rs*rs*rs*rs*rp + 35*rs*rs*rs*rs - 210*(rs*rs)*(rp*rp*rp) - 210*(rs*rs)*(rp*rp) - 210*(rs*rs)*rp*(r*r) - 105*(rs*rs)*rp - 70*(rs*rs)*(r*r) - 21*(rs*rs) + 105*rp*rp*rp*rp*rp + 175*rp*rp*rp*rp - 210*(rp*rp*rp)*(r*r) + 175*(rp*rp*rp) - 210*(rp*rp)*(r*r) + 105*(rp*rp) + 105*rp*r*r*r*r - 105*rp*(r*r) + 35*rp + 35*r*r*r*r - 21*(r*r) + 5)*M_PI*dr*dr/(840.*rs*r*r*r);
	}else if( rp == ub ){
		w = -(rs*rs*rs*rs*(105*rp-35) + 210*(rs*rs)*rp*rp*(1-rp) - 210*(rs*rs)*rp*(r*r) - 105*(rs*rs)*rp + 70*(rs*rs)*(r*r) + 21*(rs*rs) + 105*rp*rp*rp*rp*rp - 175*rp*rp*rp*rp - 210*(rp*rp*rp)*(r*r) + 175*(rp*rp*rp) + 210*(rp*rp)*(r*r) - 105*(rp*rp) + 105*rp*r*r*r*r - 105*rp*(r*r) + 35*rp - 35*r*r*r*r + 21*(r*r) - 5)*M_PI*dr*dr/(840.*rs*r*r*r);
	}else{
		w = -(3*rs*rs*rs*rs*rp - 6*(rs*rs)*(rp*rp*rp) - 6*(rs*rs)*rp*(r*r) - 3*(rs*rs)*rp + 3*rp*rp*rp*rp*rp - 6*(rp*rp*rp)*(r*r) + 5*(rp*rp*rp) + 3*rp*r*r*r*r - 3*rp*(r*r) + rp)*M_PI*dr*dr/(12.*rs*r*r*r);
	}
	return w - weight_n2(lb,ub,r,rp,rs,dr)/3.;
}

__device__ __host__ double weight_wd(dfts_wd alpha, long lb, long ub, long r, long rp, long rs, double dr){
	switch(alpha){
		case dfts_n2:
			return weight_n2(lb,ub,r,rp,rs,dr);
		case dfts_n2v:
			return weight_n2v(lb,ub,r,rp,rs,dr);
		case dfts_n3:
			return weight_n3(lb,ub,r,rp,rs,dr);
		case dfts_n11:
			return weight_n11(lb,ub,r,rp,rs,dr);
	}
	return 0.0;
}

__device__ __host__ double weight_psi3(long lb, long ub, long r, long rp, long rs, double dr){
	return weight_n3(lb,ub,r,rp,rs,dr);
}

__device__ __host__ double weight_psi2v(long lb, long ub, long r, long rp, long rs, double dr){
	double w = 0.0;
	if( rp == lb ){
		w = M_PI*dr*dr/(12*r)*(6*(rs*rs + rp*rp - r*r) + 4*rp + 1);
	}else if( rp == ub ){
		w = M_PI*dr*dr/(12*r)*(6*(rs*rs + rp*rp - r*r) - 4*rp + 1);
	}else{
		w = M_PI*dr*dr/(6*r)*(6*(rs*rs + rp*rp - r*r) + 1);
	}
	return w;
}

__device__ __host__ double weight_psi2(long lb, long ub, long r, long rp, long rs, double dr){
	return weight_n2(lb,ub,r,rp,rs,dr);
}

__device__ __host__ double weight_psi11(long lb, long ub, long r, long rp, long rs, double dr){
	double w = 0.0;
	if( r == rs ){
		if( rp == lb ){
			w = (rs*rs*(rs*dr*(90*rp + 30) - (240*rp + 80)) - 90*rp*rp*(rp+1) - 45*rp - 9)*M_PI*dr*dr/(720*rs*rs);
		}else if( rp == ub ){
			w = (rs*rs*(rs*dr*(90*rp - 30) - (240*rp - 80)) - 90*rp*rp*(rp-1) - 45*rp + 9)*M_PI*dr*dr/(720.*rs*rs);
		}else{
			w = rp*(rs*rs*(6*rs*dr - 16) - 3*rp*rp - 3)*M_PI*dr*dr/(24.*rs*rs);
		}
	}else{
		if( rp == lb ){
			w = M_PI*dr*dr*( 180*log(rp/(rp+1.))*(rp+1)*(rs*rs-r*r)*(rs*rs-r*r) + rs*rs*(-360*r*r + 180*rs*rs - 60*rp - 20) + r*r*r*dr*(90*rp + 30) - 90*rp*rp*(rp+1) - r*r*(180*rp + 60 - 180*r*r) - 45*rp - 9)/(720.*rs*r);
		}else if( rp == ub ){
			w = M_PI*dr*dr*( 180*log(rp/(rp-1.))*(rp-1)*(rs*rs-r*r)*(rs*rs-r*r) + rs*rs*( 360*r*r - 180*rs*rs - 60*rp + 20) + r*r*r*dr*(90*rp - 30) - 90*rp*rp*(rp-1) - r*r*(180*rp - 60 + 180*r*r) - 45*rp + 9)/(720.*rs*r);
		}else{
			w = M_PI*dr*dr*( 6*(rp*log(rp*rp/(rp*rp-1.)) + log((rp-1.)/(rp+1.)))*(rs*rs-r*r)*(rs*rs-r*r) + rp*( -4*(rs*rs) + 6*dr*(r*r*r) - 6*(rp*rp) - 12*(r*r) + 3))/(24.*rs*r);
		}
	}
	return w;
}

__device__ __host__ double weight_psi(dfts_wd alpha, long lb, long ub, long r, long rp, long rs, double dr){
	switch(alpha){
		case dfts_n2:
			return weight_psi2(lb,ub,r,rp,rs,dr);
		case dfts_n2v:
			return weight_psi2v(lb,ub,r,rp,rs,dr);
		case dfts_n3:
			return weight_psi3(lb,ub,r,rp,rs,dr);
		case dfts_n11:
			return weight_psi11(lb,ub,r,rp,rs,dr);
	}
	return 0.0;
}

/*

Integral n11:
Coefficient for first point:
-1.0/840.0*(105*ipow(R, 4)*((dr)*(dr))*i + 35*ipow(R, 4)*((dr)*(dr)) - 210*((R)*(R))*((dr)*(dr))*((i)*(i)*(i)) - 210*((R)*(R))*((dr)*(dr))*((i)*(i)) - 210*((R)*(R))*((dr)*(dr))*i*((r)*(r)) - 105*((R)*(R))*((dr)*(dr))*i - 70*((R)*(R))*((dr)*(dr))*((r)*(r)) - 21*((R)*(R))*((dr)*(dr)) + 105*((dr)*(dr))*ipow(i, 5) + 175*((dr)*(dr))*ipow(i, 4) - 210*((dr)*(dr))*((i)*(i)*(i))*((r)*(r)) + 175*((dr)*(dr))*((i)*(i)*(i)) - 210*((dr)*(dr))*((i)*(i))*((r)*(r)) + 105*((dr)*(dr))*((i)*(i)) + 105*((dr)*(dr))*i*ipow(r, 4) - 105*((dr)*(dr))*i*((r)*(r)) + 35*((dr)*(dr))*i + 35*((dr)*(dr))*ipow(r, 4) - 21*((dr)*(dr))*((r)*(r)) + 5*((dr)*(dr)))/(R*((r)*(r)*(r)))
Coefficient for last point:
-1.0/840.0*(105*ipow(R, 4)*((dr)*(dr))*i - 35*ipow(R, 4)*((dr)*(dr)) - 210*((R)*(R))*((dr)*(dr))*((i)*(i)*(i)) + 210*((R)*(R))*((dr)*(dr))*((i)*(i)) - 210*((R)*(R))*((dr)*(dr))*i*((r)*(r)) - 105*((R)*(R))*((dr)*(dr))*i + 70*((R)*(R))*((dr)*(dr))*((r)*(r)) + 21*((R)*(R))*((dr)*(dr)) + 105*((dr)*(dr))*ipow(i, 5) - 175*((dr)*(dr))*ipow(i, 4) - 210*((dr)*(dr))*((i)*(i)*(i))*((r)*(r)) + 175*((dr)*(dr))*((i)*(i)*(i)) + 210*((dr)*(dr))*((i)*(i))*((r)*(r)) - 105*((dr)*(dr))*((i)*(i)) + 105*((dr)*(dr))*i*ipow(r, 4) - 105*((dr)*(dr))*i*((r)*(r)) + 35*((dr)*(dr))*i - 35*((dr)*(dr))*ipow(r, 4) + 21*((dr)*(dr))*((r)*(r)) - 5*((dr)*(dr)))/(R*((r)*(r)*(r)))
Coefficient for inbetween points:
-1.0/12.0*(3*ipow(R, 4)*((dr)*(dr))*i - 6*((R)*(R))*((dr)*(dr))*((i)*(i)*(i)) - 6*((R)*(R))*((dr)*(dr))*i*((r)*(r)) - 3*((R)*(R))*((dr)*(dr))*i + 3*((dr)*(dr))*ipow(i, 5) - 6*((dr)*(dr))*((i)*(i)*(i))*((r)*(r)) + 5*((dr)*(dr))*((i)*(i)*(i)) + 3*((dr)*(dr))*i*ipow(r, 4) - 3*((dr)*(dr))*i*((r)*(r)) + ((dr)*(dr))*i)/(R*((r)*(r)*(r)))

*/
