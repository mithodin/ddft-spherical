#include <math.h>
#include <stdio.h>
#include "config.h"
#include "data.h"
#include "memory.h"
#include "dfts.h"

#define N 4096
#define DR 1.0/128.0
#define BINS_SPHERE 64

__host__ void gradient_numeric(DFTS_t conf, double **rho, double **grad){
	double r0 = 0.0;
	double h = 1e-6;
	dfts_fexc(conf, rho, grad);
	for(int i=0;i<N;++i){
		r0 = rho[0][i];
		rho[0][i] += h;
		double fp = dfts_fexc(conf, rho, NULL);
		double fm;
		rho[0][i] = r0-h;
		fm = dfts_fexc(conf, rho, NULL);
		rho[0][i] = r0;
		dfts_set_density(conf,(const double**)rho);
		grad[1][i] = (fp-fm)/(2*h*(1+i)*(1+i)*DR*DR*DR);
		
	}
}

void test_mu_rho(DFTS_t conf){
	double *rho[2] = {malloc(2*N*sizeof(double)),NULL};
	rho[1] = rho[0]+N;
	for(int i=0;i<N;++i){
		rho[0][i] = 1;
		rho[1][i] = 1e-200;
	}
	dfts_set_density(conf,(const double**)rho);
	double bd[100];
	double mu = 0.0;
	for(int k = 0;k<100;++k){
		mu = k*0.02;
		dfts_set_chemical_potential(conf,mu);
		dfts_minimize_component(conf,0);
		bd[k] = dfts_get_mean_density(conf,NULL,NULL);
	}
	FILE *out = fopen("rho-mu.dat","w");
	for(int k=0;k<100;++k){
		fprintf(out,"%d\t%.2f\t%.10e\n",k,k*0.02,bd[k]);
	}
	fclose(out);
	free(rho[0]);
}

#define ZERO 1e-60

int main(void){
	FILE *out = NULL;
	DFTS_t conf = dfts_init(N,DR,BINS_SPHERE);
	double *grad[2] = {malloc(2*N*sizeof(double)),NULL};
	grad[1] = grad[0]+N;
	double *rho[2] = {malloc(2*N*sizeof(double)),NULL};
	rho[1] = rho[0]+N;
	for(int i=0;i<N;++i){
		rho[0][i] = ZERO;
		rho[1][i] = ZERO;
	}
	rho[0][0] = 1.0/(4.0*M_PI*(1.0/3.0-1.0/4.0)*DR*DR*DR)-1e-8;

	dfts_set_density(conf,(const double**)rho);

	/*double *pot = malloc(N*sizeof(double));
	for(int i=0;i<N;++i){
		double r = i*DR;
		//pot[i] = -exp(-r*r/30)*cos(r*M_PI/2);
		pot[i] = 0.0*r;
	}
	dfts_set_potential(conf,pot);*/

	double mu = dfts_get_chemical_potential_mean_density(conf,.7266666666666666666666666,-1);
	printf("mu = %.15e\n",mu);
	//return 0;
	dfts_set_chemical_potential(conf,mu);

	bool *mask = malloc(N*sizeof(bool));
	{
		int i = 0;
		for(;i<2*BINS_SPHERE;++i){
			mask[i] = false;
		}
		for(;i<N;++i){
			mask[i] = true;
		}
	}

	double f = dfts_omega(conf,NULL);
	printf("omega[rho] = %.20e\n",f);

	//dfts_log_wd(conf,"wd.dat");
	//dfts_log_psi(conf,"psi.dat");

	/*gradient_numeric(conf, rho, grad);

	out = fopen("gradient-num.dat","w");
	for(int i=0;i<N;++i){
		fprintf(out,"%.10f\t%e\t%e\n",i*DR,grad[0][i],grad[1][i]);
	}

	return 0; */

	dfts_minimize_component_mask(conf,1,mask);
	//dfts_minimize_component(conf,0);
	//dfts_minimize(conf);

	dfts_get_density(conf,rho);

	out = fopen("density.dat","w");
	for(int i=0;i<N;++i){
		fprintf(out,"%f\t%.20e\t%.20e\n",(i*DR),rho[0][i],rho[1][i]);
	}
	fclose(out);

	double md = dfts_get_mean_density(conf,NULL,NULL);
	printf("Mean density: %.20e\n",md);

	dfts_destroy(conf);

	free(grad[0]);
	free(rho[0]);
	free(mask);
	return 0;
}
