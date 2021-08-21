#include <stdio.h>
#include <cuhelp.h>
#include "pfts_public.h"
extern "C" {
#include "h5_interface_vh_log.h"
}

#define PEXC_ALPHA 1e-10
#define ZETA 1.
#define TAU_MEM 0.000001
#define GAUSS_F 200.0

__global__ void get_update(PFTS_t conf){
	memset(conf->density_update[0], 0, 2*conf->dfts_gpu->num_bins*sizeof(double)); //clear density update
	pfts_pexc_update<<<1,1>>>(conf, conf->density_update);
}

__global__ void print_settings(PFTS_Pexc_t conf){
	printf("prefactor = %.10e\ntimestep = %.10e\n",conf->prefactor,conf->timestep);
}

int main(int argc, char *argv[]){
	if( argc < 3 ){
		printf("usage: test_memory <h5_logfile> <logname>\n");
		return 1;
	}

	//loading log
	vh_log_file_t vanhove_log = vh_log_open(argv[1],argv[2],false);
	if( vanhove_log == NULL ){
		printf("failed to open logfile, exiting.\n");
		return 2;
	}

	vh_log_group_simulation_t log_group = vh_log_open_group_simulation(vanhove_log, ".");
	if( log_group == NULL ){
		printf("failed to open group.\n");
		return 3;
	}

	printf("opened group %s in file %s.\n",argv[2],argv[1]);
	double bulk_density = log_group->attributes.bulk_density;
	unsigned int num_bins = log_group->attributes.num_bins;
	unsigned int bins_sphere = log_group->attributes.bins_sphere;
	double dr = log_group->attributes.dr;
	double dt = log_group->attributes.dt;
	printf("num_bins = %u\nbulk density = %e\n",num_bins,bulk_density);

	vh_log_table_vanhove_t log_table = vh_log_open_table_vanhove(log_group,"vanhove");
	if( log_table == NULL ){
		printf("failed to open table.\n");
		return 4;
	}

	size_t num_records = 0;
	vh_log_table_vanhove_recordset_t log_records = NULL;
	if( !vh_log_get_records_vanhove(log_table, 0, &log_records, &num_records) ){
		printf("failed to load records.\n");
		return 5;
	}

        PFTS_t conf = pfts_create(num_bins, dr, bins_sphere, dt);
        pfts_pexc_settings(conf,PEXC_ALPHA,ZETA,TAU_MEM,GAUSS_F*dr*dr/dt);
	print_settings<<<1,1>>>(conf->pexc_conf);
	cudaDeviceSynchronize();

	double *current[2] = {(double *)malloc(2*num_bins*sizeof(double)),NULL};
	double *density[2] = {(double *)malloc(2*num_bins*sizeof(double)),NULL};
	current[1] = current[0]+num_bins;
	density[1] = density[0]+num_bins;

	for(int r=0;r<num_records;++r){
		vh_log_table_vanhove_record_t rec = &(log_records->set[r]);

		for(int i=0;i<num_bins;++i){
			current[0][i] = rec->current_self[i];
			current[1][i] = rec->current_distinct[i];
		}
		pfts_set_current(conf, (const double**)current);

		for(int i=0;i<num_bins;++i){
			density[0][i] = bulk_density*rec->vanhove_self[i];
			density[1][i] = bulk_density*rec->vanhove_distinct[i];
		}
		pfts_set_density(conf, (const double **)density);

		pfts_update_memory<<<1,1>>>(conf);
		cudaDeviceSynchronize();

		get_update<<<1,1>>>(conf);
		cudaDeviceSynchronize();

		if( r == 0 ){
			print_device_array("memtest-integral.dat","w",num_bins,conf->memory);
			print_device_array("memtest-update.dat","w",num_bins,conf->density_update[0]);
		}else{
			print_device_array("memtest-integral.dat","a",num_bins,conf->memory);
			print_device_array("memtest-update.dat","a",num_bins,conf->density_update[0]);
		}
	}

	pfts_destroy(conf);
	free(density[0]);
	free(current[0]);

	vh_log_close_table_vanhove_recordset(log_records);
	vh_log_close_table_vanhove(log_table);
	vh_log_close_group_simulation(log_group);
	vh_log_close(vanhove_log);
 
	return 0;
}
