#include <hdf5_hl.h>
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "h5_interface_vh_log.h"

bool vh_log_open_file(const char *filename, hid_t *filehandle, bool rw){
    if( access( filename, F_OK ) == 0 ){
        if( rw ){
            *filehandle = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
        }else{
            *filehandle = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
        }
        if(*filehandle < 0){
            printf("Error opening file.\n");
            return false;
        }
    }else{
        printf("File does not exist.\n");
        return false;
    }
    return true;
}

vh_log_file_t vh_log_open(const char *filename, const char *root, bool rw){
    H5Eset_auto(H5E_DEFAULT, NULL, NULL);
    vh_log_file_t lf = malloc(sizeof(vh_log_file));
    if( !vh_log_open_file(filename, &(lf->h5_file), rw) ){
        return NULL;
    }
    lf->rw = rw;
    if( !root ){
        root = "/";
    }
    lf->root = H5Gopen(lf->h5_file,root,H5P_DEFAULT);
    if( lf->root < 0 ){
        printf("root dir \"%s\" not found in file \"%s\"\n",root,filename);
        free(lf);
        return NULL;
    }
    return lf;
}

vh_log_file_t vh_log_create(const char *filename){
    vh_log_file_t lf = malloc(sizeof(vh_log_file));
    lf->h5_file = H5Fcreate(filename,H5F_ACC_EXCL,H5P_DEFAULT,H5P_DEFAULT);
    if( lf->h5_file < 0 ){
        printf("file %s exists or cannot be created.\n",filename);
        free(lf);
        return NULL;
    }else{
        lf->root = H5Gopen(lf->h5_file,"/",H5P_DEFAULT);
        lf->rw = true;
        return lf;
    }
}

void vh_log_close(vh_log_file_t lf){
    H5Gclose(lf->root);
    H5Fclose(lf->h5_file);
    free(lf);
}

bool vh_log_flush(vh_log_file_t lf){
    if( lf->rw ){
        herr_t retval = H5Fflush(lf->h5_file,H5F_SCOPE_GLOBAL);
        return retval >= 0;
    }
    return false;
}

struct _vh_log_iterate {
    char **groups;
    size_t num_groups;
    const char *wildcard;
};

herr_t vh_log_list_groups(hid_t parent, const char *groupname, const H5L_info_t *info, void *data){
    struct _vh_log_iterate *iter = (struct _vh_log_iterate *)data;
    if( ( iter->wildcard && strstr(groupname, iter->wildcard) ) || !(iter->wildcard) ){
        iter->groups[iter->num_groups] = strdup(groupname);
        iter->num_groups++;
    }
    return 0;
}

bool vh_log_get_groups(vh_log_file_t lf, const char *group, char ***result, size_t *num_results){
    if( *result == NULL ){
        hsize_t num_groups = 0;
        H5Gget_num_objs(lf->root,&num_groups);
        *result = calloc(num_groups,sizeof(char *));
    }
    struct _vh_log_iterate res = {*result,0,group};
    hsize_t idx=0;
    H5Literate( lf->root, H5_INDEX_NAME, H5_ITER_NATIVE, &idx, &vh_log_list_groups, &res );
    *num_results = res.num_groups;
    return res.num_groups > 0;
}

vh_log_group_simulation_t vh_log_open_group_simulation(vh_log_file_t lf, const char gname[]){
    vh_log_group_simulation_t group = malloc(sizeof(vh_log_group_simulation));
    group->name = strdup(gname);
    group->parent = lf;
    group->h5_group = H5Gopen(group->parent->root, group->name, H5P_DEFAULT);
    if( group->h5_group < 0 ){
        printf("group %s not found!\n",gname);
        free(group->name);
        free(group);
        return NULL;
    }
    //read attributes
    herr_t ret;
    ret = H5LTget_attribute(group->h5_group, ".", "num_bins", H5T_NATIVE_UINT, &(group->attributes.num_bins) );
    ret = H5LTget_attribute(group->h5_group, ".", "bins_sphere", H5T_NATIVE_UINT, &(group->attributes.bins_sphere) );
    ret = H5LTget_attribute(group->h5_group, ".", "dr", H5T_NATIVE_DOUBLE, &(group->attributes.dr) );
    ret = H5LTget_attribute(group->h5_group, ".", "dt", H5T_NATIVE_DOUBLE, &(group->attributes.dt) );
    ret = H5LTget_attribute(group->h5_group, ".", "timestep_log", H5T_NATIVE_DOUBLE, &(group->attributes.timestep_log) );
    ret = H5LTget_attribute(group->h5_group, ".", "time_simulation", H5T_NATIVE_DOUBLE, &(group->attributes.time_simulation) );
    ret = H5LTget_attribute(group->h5_group, ".", "bulk_density", H5T_NATIVE_DOUBLE, &(group->attributes.bulk_density) );
    group->attributes.simulation_description = calloc(500,sizeof(char));
    ret = H5LTget_attribute(group->h5_group, ".", "simulation_description", H5T_NATIVE_CHAR, group->attributes.simulation_description );
    if( ret < 0 ){
        free(group->attributes.simulation_description);
        group->attributes.simulation_description = strdup("");
    }

    return group;
}

void vh_log_group_simulation_attribute_sync(vh_log_group_simulation_t group){
    herr_t ret;
    ret = H5LTset_attribute_uint(group->h5_group, ".", "num_bins", &(group->attributes.num_bins), 1);
    if( ret < 0 ){
        printf("failed to set attribute num_bins");
    }
    ret = H5LTset_attribute_uint(group->h5_group, ".", "bins_sphere", &(group->attributes.bins_sphere), 1);
    if( ret < 0 ){
        printf("failed to set attribute bins_sphere");
    }
    ret = H5LTset_attribute_double(group->h5_group, ".", "dr", &(group->attributes.dr), 1);
    if( ret < 0 ){
        printf("failed to set attribute dr");
    }
    ret = H5LTset_attribute_double(group->h5_group, ".", "dt", &(group->attributes.dt), 1);
    if( ret < 0 ){
        printf("failed to set attribute dt");
    }
    ret = H5LTset_attribute_double(group->h5_group, ".", "timestep_log", &(group->attributes.timestep_log), 1);
    if( ret < 0 ){
        printf("failed to set attribute timestep_log");
    }
    ret = H5LTset_attribute_double(group->h5_group, ".", "time_simulation", &(group->attributes.time_simulation), 1);
    if( ret < 0 ){
        printf("failed to set attribute time_simulation");
    }
    ret = H5LTset_attribute_double(group->h5_group, ".", "bulk_density", &(group->attributes.bulk_density), 1);
    if( ret < 0 ){
        printf("failed to set attribute bulk_density");
    }
    if( group->attributes.simulation_description == NULL ){
        group->attributes.simulation_description = strdup("");
    }
    {
        ret = H5LTset_attribute_string(group->h5_group, ".", "simulation_description", group->attributes.simulation_description);
        if( ret < 0 ){
            printf("failed to set attribute simulation_description");
        }
    }

}

vh_log_group_simulation_t vh_log_create_group_simulation(vh_log_file_t lf, vh_log_group_simulation_attributes attrs, const char gname[]){
    hid_t gid = H5Gcreate(lf->root, gname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if( gid < 0 ){
       printf("failed to create group %s.\n",gname);
       return NULL;
    }
    vh_log_group_simulation_t group = malloc(sizeof(vh_log_group_simulation));
    group->name = strdup(gname);
    group->h5_group = gid;
    group->parent = lf;
    group->attributes = attrs;
    vh_log_group_simulation_attribute_sync(group);
    return group;
}

void vh_log_close_group_simulation(vh_log_group_simulation_t group){
    if( group->parent->rw ){
        vh_log_group_simulation_attribute_sync(group);
    }
    //free attributes
    free(group->attributes.simulation_description);

    H5Gclose(group->h5_group);
    free(group->name);
    free(group);
}

bool vh_log_flush_group_simulation(vh_log_group_simulation_t group){
    if( group->parent->rw ){
        herr_t retval = H5Fflush(group->h5_group,H5F_SCOPE_GLOBAL);
        return retval >= 0;
    }
    return false;
}

vh_log_table_vanhove_t vh_log_open_table_vanhove(vh_log_group_simulation_t group, const char *tname){
    vh_log_table_vanhove_t tb = malloc(sizeof(vh_log_table_vanhove));
    if( H5TBget_table_info( group->h5_group, tname, &(tb->num_columns), &(tb->num_records) ) < 0 ){
        printf("table %s not found.\n",tname);
        free(tb);
        return NULL;
    }
    tb->column_h5types = NULL;
    tb->name = strdup(tname);
    tb->parent = group;
    tb->column_offsets = calloc(tb->num_columns,sizeof(size_t));
    tb->column_sizes = calloc(tb->num_columns,sizeof(size_t));
    tb->column_names = calloc(tb->num_columns,sizeof(char *));
    for(int i=0;i<tb->num_columns;++i){
        tb->column_names[i] = calloc(50,sizeof(char));
    }
    if( H5TBget_field_info( tb->parent->h5_group, tname, tb->column_names, tb->column_sizes, tb->column_offsets, &(tb->record_size) ) < 0 ){
        printf("error getting field info.\n");
        for(int i=0;i<tb->num_columns;++i){
            free(tb->column_names[i]);
        }
        free(tb->column_names);
        free(tb->column_sizes);
        free(tb->column_offsets);
        free(tb->name);
        free(tb);
        return NULL;
    }
    return tb;
}

void vh_log_clear_table_vanhove(vh_log_table_vanhove_t tb){
    if( H5TBget_table_info( tb->parent->h5_group, tb->name, &(tb->num_columns), &(tb->num_records) ) < 0 ){
        printf("table %s not found.\n",tb->name);
        return;
    }
    herr_t status = H5TBdelete_record(tb->parent->h5_group, tb->name, 0, tb->num_records);
    if( status < 0 ){
        printf("failed to delete records.\n");
    }
}

void vh_log_close_table_vanhove(vh_log_table_vanhove_t tb){
    for(int i=0;i<tb->num_columns;++i){
        free(tb->column_names[i]);
    }
    free(tb->column_names);
    free(tb->column_sizes);
    free(tb->column_offsets);
    if( tb->column_h5types ){
        free(tb->column_h5types);
    }
    free(tb->name);
    free(tb);
}

bool vh_log_get_records_vanhove(vh_log_table_vanhove_t table, size_t start, vh_log_table_vanhove_recordset_t *records, size_t *num_records){
    bool need_free = false;
    if( *num_records == 0 ){
        *num_records = table->num_records-start;
    }
    if( *records == NULL ){
        need_free = true;
        *records = malloc(sizeof(vh_log_table_vanhove_recordset));
    }
    (*records)->set = calloc(*num_records,sizeof(vh_log_table_vanhove_record));
    void *data = calloc(*num_records,table->record_size);
    (*records)->data_raw = data;

    if( H5TBread_records( table->parent->h5_group, table->name, start, *num_records, table->record_size, table->column_offsets, table->column_sizes, data ) < 0 ){
        if( need_free ){
            free( (*records)->set );
            free( *records );
            *records = NULL;
        }
        printf("error reading records.\n");
        return false;
    }
    for(int i=0;i<*num_records;++i){
        vh_log_table_vanhove_record_t rec = (*records)->set+i;
        void *mydata = data+i*(table->record_size);
        for(int j=0;j<table->num_columns;++j){
            switch(table->column_names[j][0]){
               case 'c':
                  if( strcmp(table->column_names[j],"current_distinct") == 0 ){
                     rec->current_distinct = (double *)(mydata+table->column_offsets[j]);
                  }
                  if( strcmp(table->column_names[j],"current_self") == 0 ){
                     rec->current_self = (double *)(mydata+table->column_offsets[j]);
                  }
                  break;
               case 'f':
                  if( strcmp(table->column_names[j],"frame") == 0 ){
                     rec->frame = (unsigned long *)(mydata+table->column_offsets[j]);
                  }
                  break;
               case 't':
                  if( strcmp(table->column_names[j],"time") == 0 ){
                     rec->time = (double *)(mydata+table->column_offsets[j]);
                  }
                  break;
               case 'v':
                  if( strcmp(table->column_names[j],"vanhove_distinct") == 0 ){
                     rec->vanhove_distinct = (double *)(mydata+table->column_offsets[j]);
                  }
                  if( strcmp(table->column_names[j],"vanhove_self") == 0 ){
                     rec->vanhove_self = (double *)(mydata+table->column_offsets[j]);
                  }
                  break;
            }
        }
    }
    return true;
}

void vh_log_close_table_vanhove_recordset(vh_log_table_vanhove_recordset_t rec){
    if( rec->set ){
        free(rec->set);
    }
    free(rec->data_raw);
    free(rec);
}

vh_log_table_vanhove_t vh_log_create_table_vanhove(vh_log_group_simulation_t group, const char *tname){
    herr_t ret;
    vh_log_table_vanhove_t table = malloc(sizeof(vh_log_table_vanhove));
    table->name = strdup(tname);
    table->num_columns = 6;
    table->num_records = 0;
    table->column_offsets = calloc(6,sizeof(size_t));
    table->column_sizes = calloc(6,sizeof(size_t));
    table->column_h5types = calloc(6,sizeof(hid_t));
    table->column_names = calloc(6,sizeof(char *));
    table->parent = group;
    {
        hsize_t array_dims[1] = { (group->attributes.num_bins) };
        table->column_h5types[0] = H5Tarray_create(H5T_NATIVE_DOUBLE,1,array_dims);
    }
    table->column_sizes[0] = sizeof(double)*(group->attributes.num_bins);
    table->column_names[0] = strdup("current_distinct");
    {
        hsize_t array_dims[1] = { (group->attributes.num_bins) };
        table->column_h5types[1] = H5Tarray_create(H5T_NATIVE_DOUBLE,1,array_dims);
    }
    table->column_sizes[1] = sizeof(double)*(group->attributes.num_bins);
    table->column_names[1] = strdup("current_self");
    table->column_h5types[2] = H5T_NATIVE_ULONG;
    table->column_sizes[2] = sizeof(unsigned long)*1;
    table->column_names[2] = strdup("frame");
    table->column_h5types[3] = H5T_NATIVE_DOUBLE;
    table->column_sizes[3] = sizeof(double)*1;
    table->column_names[3] = strdup("time");
    {
        hsize_t array_dims[1] = { (group->attributes.num_bins) };
        table->column_h5types[4] = H5Tarray_create(H5T_NATIVE_DOUBLE,1,array_dims);
    }
    table->column_sizes[4] = sizeof(double)*(group->attributes.num_bins);
    table->column_names[4] = strdup("vanhove_distinct");
    {
        hsize_t array_dims[1] = { (group->attributes.num_bins) };
        table->column_h5types[5] = H5Tarray_create(H5T_NATIVE_DOUBLE,1,array_dims);
    }
    table->column_sizes[5] = sizeof(double)*(group->attributes.num_bins);
    table->column_names[5] = strdup("vanhove_self");

    table->record_size = 0;
    for(int i=0;i<6;++i){
        table->column_offsets[i] = table->record_size;
        table->record_size += table->column_sizes[i];
    }
    ret = H5TBmake_table(tname, group->h5_group, tname, table->num_columns, 0, table->record_size, (const char **)table->column_names, table->column_offsets, table->column_h5types, 20, NULL, 5, NULL);
    if( ret < 0 ){
        vh_log_close_table_vanhove(table);
        return NULL;
    }
    return table;
}

void vh_log_add_recordset_vanhove(vh_log_table_vanhove_t table,  vh_log_table_vanhove_recordset_t recs){
    herr_t ret = H5TBappend_records(table->parent->h5_group,table->name,recs->num_records,table->record_size,table->column_offsets,table->column_sizes,recs->data_raw);
    if( ret < 0 ){
        printf("failed to append records to table %s\n",table->name);
    }
    vh_log_close_table_vanhove_recordset(recs);
}

vh_log_table_vanhove_recordset_t vh_log_create_table_vanhove_recordset(vh_log_table_vanhove_t table, size_t num_records){
    vh_log_table_vanhove_recordset_t rs = malloc(sizeof(vh_log_table_vanhove_recordset));
    rs->set = malloc(num_records*sizeof(vh_log_table_vanhove_record));
    rs->data_raw = malloc(num_records*(table->record_size));
    rs->num_records = num_records;
    for(int i=0;i<num_records;++i){
        void *data = rs->data_raw+(i*table->record_size);
        vh_log_table_vanhove_record_t record = &(rs->set[i]);
        record->current_distinct = data+table->column_offsets[0];
        record->current_self = data+table->column_offsets[1];
        record->frame = data+table->column_offsets[2];
        record->time = data+table->column_offsets[3];
        record->vanhove_distinct = data+table->column_offsets[4];
        record->vanhove_self = data+table->column_offsets[5];

    }
    return rs;
}

void vh_log_add_records_vanhove(vh_log_table_vanhove_t table, size_t num_records, vh_log_table_vanhove_record_t records){
    void *data_chunk = calloc(num_records,table->record_size);
    for(int i=0;i<num_records;++i){
        void *data = data_chunk+(i*table->record_size);
        memcpy(data+table->column_offsets[0],records[i].current_distinct,table->column_sizes[0]);
        memcpy(data+table->column_offsets[1],records[i].current_self,table->column_sizes[1]);
        memcpy(data+table->column_offsets[2],records[i].frame,table->column_sizes[2]);
        memcpy(data+table->column_offsets[3],records[i].time,table->column_sizes[3]);
        memcpy(data+table->column_offsets[4],records[i].vanhove_distinct,table->column_sizes[4]);
        memcpy(data+table->column_offsets[5],records[i].vanhove_self,table->column_sizes[5]);

    }
    herr_t ret = H5TBappend_records(table->parent->h5_group,table->name,num_records,table->record_size,table->column_offsets,table->column_sizes,data_chunk);
    if( ret < 0 ){
        printf("failed to append records to table %s\n",table->name);
    }
    free(data_chunk);
}

