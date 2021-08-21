#include <cuda_runtime.h>
#include <nlopt.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <progressbar.h>
#include <stdbool.h>
#include <aslog.h>
#include <gsl/gsl_interp.h>
#include "h5_interface_vh_log.h"
#include "h5_interface_bd.h"
#include "pfts_public.h"
#include "debug.h"
#include "vh-config.h"

struct my_logframe{
	vh_log_table_vanhove_t table;
	vh_log_table_vanhove_recordset_t record;
};

bool vanhove_append_logframe(void *data){
	struct my_logframe *fr = (struct my_logframe *)data;
	vh_log_add_recordset_vanhove(fr->table,fr->record);
	vh_log_flush(fr->table->parent->parent);
	free(data);
	return true;
}

bool vanhove_enqueue_logframe(PFTS_t conf, vh_log_table_vanhove_t table, bool final){
	bool nan = false;
	struct my_logframe *fr = (struct my_logframe *)malloc(sizeof(struct my_logframe));
	fr->table = table;
	vh_log_table_vanhove_recordset_t rs = vh_log_create_table_vanhove_recordset(table,1);
	fr->record = rs;
	//load data from GPU
	cudaMemcpy(rs->set->current_self,conf->current[0],NUM_BINS*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(rs->set->current_distinct,conf->current[1],NUM_BINS*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(rs->set->vanhove_self,conf->dfts->density[0],NUM_BINS*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(rs->set->vanhove_distinct,conf->dfts->density[1],NUM_BINS*sizeof(double),cudaMemcpyDeviceToHost);
	*(rs->set->frame) = conf->timestep;
	*(rs->set->time) = conf->timestep*conf->dt;
	for(int i=0;i<NUM_BINS;++i){
		rs->set->vanhove_self[i] /= BULK_DENSITY;
		rs->set->vanhove_distinct[i] /= BULK_DENSITY;
		if( isnan(rs->set->vanhove_self[i]) || isnan(rs->set->vanhove_distinct[i]) ){
			final = true;
			nan = true;
		}
	}
	aslog_log_enqueue((void*)fr, final);
	return nan;
}

#ifdef BD_CORE_FILE
bd_table_vanhove_t open_bd_logfile(void){
	bd_file_t bdf = bd_open(BD_CORE_FILE,"/",false);
	if( !bdf ){
		return NULL;
	}
	bd_group_log_t g0 = bd_open_group_log(bdf,BD_CORE_GROUP);
	if( !g0 ){
		bd_close(bdf);
		return NULL;
	}

	bd_table_vanhove_t tb = bd_open_table_vanhove(g0,"vanhove");
	if( !tb ){
		bd_close_group_log(g0);
		bd_close(bdf);
		return NULL;
	}
	return tb;
}
#endif

void close_logfile(vh_log_table_vanhove_t tb){
	aslog_shutdown();
	vh_log_group_simulation_t g = tb->parent;
	vh_log_file_t f = g->parent;
	vh_log_close_table_vanhove(tb);
	vh_log_close_group_simulation(g);
	vh_log_close(f);
}

vh_log_table_vanhove_t init_logging(void){
	//create file
	vh_log_file_t lf = NULL;
	if( access( LOGFILE, F_OK ) == 0 ){
		lf = vh_log_open(LOGFILE, NULL, true);
	}else{
		lf = vh_log_create(LOGFILE);
	}
	if( !lf ){
		printf("Failed to create or open logfile.\n");
		exit(1);
	}

	//create group
	char **groups_found = NULL;
	size_t num_groups;
	bool found = vh_log_get_groups(lf, SIMULATION_SHORTNAME, &groups_found, &num_groups);
	for(int i=0;i<num_groups;++i){
		free(groups_found[i]);
	}
	free(groups_found);
	char *gname;
	if( found ){
		size_t len_newname = strlen(SIMULATION_SHORTNAME)+3+(int)floor(log(num_groups)/log(10));
		gname = (char*)calloc(len_newname,sizeof(char));
		snprintf(gname, len_newname, "%s_%ld",SIMULATION_SHORTNAME,num_groups);
	}else{
		gname = strdup(SIMULATION_SHORTNAME);
	}

	//create attributes
	vh_log_group_simulation_attributes attrs = {NUM_BINS,BINS_SPHERE,DR,TIMESTEP,TIMESTEP_LOG,TIME,BULK_DENSITY,strdup(SIMULATION_DESCRIPTION)};
	vh_log_group_simulation_t group = vh_log_create_group_simulation(lf, attrs, gname);
	free(gname);
	if( !group ){
		printf("Failed to create group.\n");
		exit(2);
	}

	//create table
	vh_log_table_vanhove_t table = vh_log_create_table_vanhove(group, "vanhove");
	if( !table ){
		printf("Failed to create table.\n");
		exit(3);
	}

	//start up logging system
	aslog_init(10,vanhove_append_logframe);
	return table;
}

#define ZERO 1e-70
double *points_r = NULL;

void load_vanhove_bd(double *density_bd, double bd_dr, unsigned int bd_n, double *result){
	if( points_r == NULL ){
		points_r = calloc(bd_n+1,sizeof(double));
		for(int i=0;i<bd_n;++i){
			points_r[i] = bd_dr*i;
		}
		points_r[bd_n] = NUM_BINS*DR; //a far out point
	}
	gsl_interp *interpolation = gsl_interp_alloc(gsl_interp_steffen, bd_n+1);
	gsl_interp_accel *accel = gsl_interp_accel_alloc();
	density_bd[bd_n] = ZERO; //zero at far out point
	gsl_interp_init(interpolation, points_r, density_bd, bd_n+1);
	for(int i=0;i<NUM_BINS;++i){
		result[i] = BULK_DENSITY*gsl_interp_eval(interpolation, points_r, density_bd, i*DR ,accel);
		if( result[i] < ZERO ){
			result[i] = ZERO;
		}
	}
	gsl_interp_free(interpolation);
	gsl_interp_accel_free(accel);
}

double time_log[2] = {-1,-1};
double *logframe[2] = {NULL,NULL};
int log_offset = 0;
void set_self_density_bd(bd_table_vanhove_t log, double bd_dr, double bd_dt, unsigned int bd_n, double *density, double time){
	if( logframe[0] == NULL ){
		logframe[0] = calloc(2*NUM_BINS,sizeof(double));
		logframe[1] = logframe[0]+NUM_BINS;
		bd_table_vanhove_recordset_t rec = NULL;
		size_t num_recs = 2;
		bd_get_records_vanhove(log, log_offset, &rec, &num_recs);
		load_vanhove_bd(rec->set[0].vanhove, bd_dr, bd_n, logframe[0]);
		load_vanhove_bd(rec->set[1].vanhove, bd_dr, bd_n, logframe[1]);
		time_log[0] = rec->set[0].time[0];
		time_log[1] = rec->set[1].time[0];
		log_offset += 2;
		bd_close_table_vanhove_recordset(rec);
	}
	while( time > time_log[1] ){
		time_log[0] = time_log[1];
		bd_table_vanhove_recordset_t rec = NULL;
		size_t num_recs = 1;
		if( !bd_get_records_vanhove(log, log_offset, &rec, &num_recs) ){
			printf("Error, could not load next record from the log!\n");
			for(int i=0;i<NUM_BINS;++i){
				density[i] = NAN;
			}
			return;
		}else{
			load_vanhove_bd(rec->set[0].vanhove, bd_dr, bd_n, logframe[log_offset%2]);
			time_log[1] = rec->set[0].time[0];
			bd_close_table_vanhove_recordset(rec);
		}
		log_offset += 1;
	}

	double delta = (time-time_log[0])/bd_dt;
	for(int i=0;i<NUM_BINS;++i){
		density[i] = (1.-delta)*logframe[log_offset%2][i]+delta*logframe[1-(log_offset%2)][i];
	}
}

int main(void){
	vh_log_table_vanhove_t h5_log = init_logging();

	PFTS_t conf = pfts_create(NUM_BINS,DR,BINS_SPHERE,TIMESTEP);
	pfts_pexc_settings(conf,PEXC_ALPHA,ZETA,TAU_MEM,GAUSS_F); 

#ifdef BD_CORE_FILE
	bd_table_vanhove_t bd_log = open_bd_logfile();
	if( !bd_log ){
		printf("unable to open bd logfile.\n");
		return 1;
	}

	double bd_dr = bd_log->parent->attributes.vh_dr;
	double bd_dt = bd_log->parent->attributes.vh_dt;
	unsigned int bd_num_bins = bd_log->parent->attributes.vh_n;
	double *density_self = calloc(NUM_BINS,sizeof(double));
#endif

#ifdef SELFINTERACTION
	dfts_set_selfinteraction(conf->dfts,SELFINTERACTION);
#endif

	double mu = dfts_get_chemical_potential_mean_density(conf->dfts,BULK_DENSITY,-1);
	dfts_set_chemical_potential(conf->dfts,mu);
	pfts_init_rdf(conf);

	const int timesteps = (int)(TIME/TIMESTEP_LOG);
	const int inner_ts = (int)(TIMESTEP_LOG/TIMESTEP);
	
	bool nan = false;

	pfts_minimize(conf);
	for(int t=0;t<timesteps;++t){
		nan = vanhove_enqueue_logframe(conf, h5_log, false);
		if( nan ){
			printf("\nError: NaN detected in density. Exiting.\n");
			break;
		}

		progressbar *pr = progressbar_new_with_format("",inner_ts/5,"[=]");
		for(int tt=0;tt<inner_ts;++tt){
			pfts_advance_time(conf);
#ifdef BD_CORE_FILE
			set_self_density_bd(bd_log, bd_dr, bd_dt, bd_num_bins, density_self, (t*inner_ts+tt+1)*TIMESTEP);
			dfts_set_density_component(conf->dfts,(const double **)&density_self,1);
#endif
			pfts_minimize(conf);
			if( tt%5 == 0 ){
				progressbar_inc(pr);
			}
		}
		progressbar_finish(pr);
		printf("%04d/%04d\n",t+1,timesteps);
	}
	if( !nan ){
		vanhove_enqueue_logframe(conf, h5_log, true);
	}

	close_logfile(h5_log);
	pfts_destroy(conf);
	return 0;
}
