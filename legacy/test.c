#include <nlopt.h>
#include <stdio.h>
#include <math.h>
#include <progressbar.h>
#include "pfts_public.h"
#include "debug.h"

#define N 4096
#define DR (1.0/128)
#define BINS_SPHERE 64
#define INNER_TS 2000
#define TIME 1000
#define PEXC_ALPHA 1e-5
#define ZETA 1.
#define TAU_MEM 1.

void print_mv(DFTS_t conf){
	double mv[5];
	mv[4] = dfts_get_mean_density(conf,mv,mv+2);
	printf("    total self density: %.10e\n     mean self density: %.10e\n",mv[0],mv[2]);
	printf("total distinct density: %.10e\n mean distinct density: %.10e\n",mv[1],mv[3]);
	printf("mean density: %.10e\n",mv[4]);
}

void numerical_gradient(PFTS_t conf, double **current, double **grad){
	double c0 = 0.0;
	double h = 1e-7;
	for(int b = 0;b<2;++b){
		for(int i=0;i<N;++i){
			c0 = current[b][i];
			current[b][i] += h;
			pfts_set_current(conf,(const double**)current);
			double fp = pfts_rt_c(conf,NULL);
			current[b][i] = c0-h;
			pfts_set_current(conf,(const double**)current);
			double fm = pfts_rt_c(conf,NULL);
			current[b][i] = c0;
			pfts_set_current(conf,(const double**)current);
			grad[b][i] = (fp-fm)/(2*h);
		}
	}
}

int main(void){
	PFTS_t conf = pfts_create(N, DR, BINS_SPHERE, 1e-6);
	pfts_pexc_settings(conf,PEXC_ALPHA,ZETA,TAU_MEM);
	double *current[2] = {malloc(2*N*sizeof(double)),NULL};
	current[1] = current[0]+N;
	for(int i=0;i<N;++i){
		current[0][i] = 0.0;
		current[1][i] = 0.0;
	}
	pfts_set_current(conf, (const double**)current);

	double *grad[4] = {malloc(4*N*sizeof(double)),NULL,NULL,NULL};
	for(int i=1;i<4;++i){
		grad[i] = grad[0]+i*N;
	}

	double *density[2] = {malloc(2*N*sizeof(double)),NULL};
	density[1] = density[0]+N;
	/*double alpha = 0.1;
	double a0 = pow(alpha/M_PI,3.0/2.0);
	for(int i=0;i<N;++i){
		//density[0][i] = a0*exp(-alpha*i*i*DR*DR)/1.2;
		density[0][i] = 0.1+exp(-(i*DR-10)*(i*DR-10))*0.01;
		density[1][i] = 0.1+exp(-(i*DR-5)*(i*DR-5))*0.01;
	}
	pfts_set_density(conf,(const double**)density);*/

	//dfts_minimize_component(conf->dfts,1);

	//dfts_set_chemical_potential(conf->dfts,7.98251061110204);
	dfts_set_chemical_potential(conf->dfts,1.0);
	pfts_init_rdf(conf);

	FILE *out = fopen("current.dat","w");fclose(out); //clear file

	pfts_minimize(conf);
	for(int t=0;t<TIME;++t){
		print_mv(conf->dfts);
		pfts_get_current(conf,current);
		dfts_get_density(conf->dfts,density);
		bool ex=false;
		out = fopen("current.dat","a");
		for(int i=0;i<N;++i){
			if( isnan(density[0][i]) || isnan(density[1][i]) ){
				ex=true;
			}
			fprintf(out,"%f\t%.30e\t%.30e\t%.30e\t%.30e\n",i*DR,density[0][i],density[1][i],current[0][i],current[1][i]);
		}
		fprintf(out,"\n\n");
		fclose(out);
		if(ex){
			printf("NaN density, exiting.\n");
			return 1;
		}
	
		progressbar *pr = progressbar_new_with_format("",INNER_TS/20,"[=]");
		for(int tt=0;tt<INNER_TS;++tt){
			pfts_advance_time(conf);
			pfts_minimize(conf);
			if( tt%20 == 0 ){
				progressbar_inc(pr);
			}
		}
		progressbar_finish(pr);

		printf("%04d/%04d\n",t+1,TIME);
		//pfts_renormalize(conf);
	}
	pfts_get_current(conf,current);
	pfts_advance_time(conf);
	dfts_get_density(conf->dfts,density);
	out = fopen("current.dat","a");
	for(int i=0;i<N;++i){
		fprintf(out,"%f\t%.30e\t%.30e\t%.30e\t%.30e\n",i*DR,density[0][i],density[1][i],current[0][i],current[1][i]);
	}
	fprintf(out,"\n\n");
	fclose(out);
	print_mv(conf->dfts);
	pfts_destroy(conf);
	free(current[0]);
	free(grad[0]);
	free(density[0]);
	return 0;
}
