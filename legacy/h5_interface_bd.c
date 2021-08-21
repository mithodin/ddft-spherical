#include <hdf5_hl.h>
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "h5_interface_bd.h"

bool bd_open_file(const char *filename, hid_t *filehandle, bool rw){
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

bd_file_t bd_open(const char *filename, const char *root, bool rw){
    H5Eset_auto(H5E_DEFAULT, NULL, NULL);
    bd_file_t lf = malloc(sizeof(bd_file));
    if( !bd_open_file(filename, &(lf->h5_file), rw) ){
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

bd_file_t bd_create(const char *filename){
    bd_file_t lf = malloc(sizeof(bd_file));
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

void bd_close(bd_file_t lf){
    H5Gclose(lf->root);
    H5Fclose(lf->h5_file);
    free(lf);
}

bool bd_flush(bd_file_t lf){
    if( lf->rw ){
        herr_t retval = H5Fflush(lf->h5_file,H5F_SCOPE_GLOBAL);
        return retval >= 0;
    }
    return false;
}

struct _bd_iterate {
    char **groups;
    size_t num_groups;
    const char *wildcard;
};

herr_t bd_list_groups(hid_t parent, const char *groupname, const H5L_info_t *info, void *data){
    struct _bd_iterate *iter = (struct _bd_iterate *)data;
    if( ( iter->wildcard && strstr(groupname, iter->wildcard) ) || !(iter->wildcard) ){
        iter->groups[iter->num_groups] = strdup(groupname);
        iter->num_groups++;
    }
    return 0;
}

bool bd_get_groups(bd_file_t lf, const char *group, char ***result, size_t *num_results){
    if( *result == NULL ){
        hsize_t num_groups = 0;
        H5Gget_num_objs(lf->root,&num_groups);
        *result = calloc(num_groups,sizeof(char *));
    }
    struct _bd_iterate res = {*result,0,group};
    hsize_t idx=0;
    H5Literate( lf->root, H5_INDEX_NAME, H5_ITER_NATIVE, &idx, &bd_list_groups, &res );
    *num_results = res.num_groups;
    return res.num_groups > 0;
}

bd_group_log_t bd_open_group_log(bd_file_t lf, const char gname[]){
    bd_group_log_t group = malloc(sizeof(bd_group_log));
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
    ret = H5LTget_attribute(group->h5_group, ".", "dimension", H5T_NATIVE_UINT, &(group->attributes.dimension) );
    ret = H5LTget_attribute(group->h5_group, ".", "num_particles", H5T_NATIVE_UINT, &(group->attributes.num_particles) );
    ret = H5LTget_attribute(group->h5_group, ".", "sigma", H5T_NATIVE_DOUBLE, &(group->attributes.sigma) );
    ret = H5LTget_attribute(group->h5_group, ".", "zeta0", H5T_NATIVE_DOUBLE, &(group->attributes.zeta0) );
    ret = H5LTget_attribute(group->h5_group, ".", "dt", H5T_NATIVE_DOUBLE, &(group->attributes.dt) );
    ret = H5LTget_attribute(group->h5_group, ".", "frame_timestep", H5T_NATIVE_DOUBLE, &(group->attributes.frame_timestep) );
    ret = H5LTget_attribute(group->h5_group, ".", "time", H5T_NATIVE_DOUBLE, &(group->attributes.time) );
    ret = H5LTget_attribute(group->h5_group, ".", "time_init", H5T_NATIVE_DOUBLE, &(group->attributes.time_init) );
    if( ret < 0 ){
        group->attributes.time_init = 0;
    }
    group->attributes.sizes = calloc((group->attributes.dimension),sizeof(double));
    ret = H5LTget_attribute(group->h5_group, ".", "sizes", H5T_NATIVE_DOUBLE, group->attributes.sizes );
    group->attributes.behaviour = calloc((group->attributes.dimension),sizeof(int));
    ret = H5LTget_attribute(group->h5_group, ".", "behaviour", H5T_NATIVE_INT, group->attributes.behaviour );
    group->attributes.force_init_x = calloc(200,sizeof(char));
    ret = H5LTget_attribute(group->h5_group, ".", "force-init-x", H5T_NATIVE_CHAR, group->attributes.force_init_x );
    if( ret < 0 ){
        free(group->attributes.force_init_x);
        group->attributes.force_init_x = strdup("0.0");
    }
    group->attributes.force_init_y = calloc(200,sizeof(char));
    ret = H5LTget_attribute(group->h5_group, ".", "force-init-y", H5T_NATIVE_CHAR, group->attributes.force_init_y );
    if( ret < 0 ){
        free(group->attributes.force_init_y);
        group->attributes.force_init_y = strdup("0.0");
    }
    group->attributes.force_init_z = calloc(200,sizeof(char));
    ret = H5LTget_attribute(group->h5_group, ".", "force-init-z", H5T_NATIVE_CHAR, group->attributes.force_init_z );
    if( ret < 0 ){
        free(group->attributes.force_init_z);
        group->attributes.force_init_z = strdup("0.0");
    }
    group->attributes.force_x = calloc(200,sizeof(char));
    ret = H5LTget_attribute(group->h5_group, ".", "force-x", H5T_NATIVE_CHAR, group->attributes.force_x );
    if( ret < 0 ){
        free(group->attributes.force_x);
        group->attributes.force_x = strdup("0.0");
    }
    group->attributes.force_y = calloc(200,sizeof(char));
    ret = H5LTget_attribute(group->h5_group, ".", "force-y", H5T_NATIVE_CHAR, group->attributes.force_y );
    if( ret < 0 ){
        free(group->attributes.force_y);
        group->attributes.force_y = strdup("0.0");
    }
    group->attributes.force_z = calloc(200,sizeof(char));
    ret = H5LTget_attribute(group->h5_group, ".", "force-z", H5T_NATIVE_CHAR, group->attributes.force_z );
    if( ret < 0 ){
        free(group->attributes.force_z);
        group->attributes.force_z = strdup("0.0");
    }
    ret = H5LTget_attribute(group->h5_group, ".", "vh_dr", H5T_NATIVE_DOUBLE, &(group->attributes.vh_dr) );
    if( ret < 0 ){
        group->attributes.vh_dr = 0;
    }
    ret = H5LTget_attribute(group->h5_group, ".", "vh_dt", H5T_NATIVE_DOUBLE, &(group->attributes.vh_dt) );
    if( ret < 0 ){
        group->attributes.vh_dt = 0;
    }
    ret = H5LTget_attribute(group->h5_group, ".", "vh_n", H5T_NATIVE_UINT, &(group->attributes.vh_n) );
    if( ret < 0 ){
        group->attributes.vh_n = 0;
    }

    return group;
}

void bd_group_log_attribute_sync(bd_group_log_t group){
    herr_t ret;
    ret = H5LTset_attribute_uint(group->h5_group, ".", "dimension", &(group->attributes.dimension), 1);
    if( ret < 0 ){
        printf("failed to set attribute dimension");
    }
    ret = H5LTset_attribute_uint(group->h5_group, ".", "num_particles", &(group->attributes.num_particles), 1);
    if( ret < 0 ){
        printf("failed to set attribute num_particles");
    }
    ret = H5LTset_attribute_double(group->h5_group, ".", "sigma", &(group->attributes.sigma), 1);
    if( ret < 0 ){
        printf("failed to set attribute sigma");
    }
    ret = H5LTset_attribute_double(group->h5_group, ".", "zeta0", &(group->attributes.zeta0), 1);
    if( ret < 0 ){
        printf("failed to set attribute zeta0");
    }
    ret = H5LTset_attribute_double(group->h5_group, ".", "dt", &(group->attributes.dt), 1);
    if( ret < 0 ){
        printf("failed to set attribute dt");
    }
    ret = H5LTset_attribute_double(group->h5_group, ".", "frame_timestep", &(group->attributes.frame_timestep), 1);
    if( ret < 0 ){
        printf("failed to set attribute frame_timestep");
    }
    ret = H5LTset_attribute_double(group->h5_group, ".", "time", &(group->attributes.time), 1);
    if( ret < 0 ){
        printf("failed to set attribute time");
    }
    ret = H5LTset_attribute_double(group->h5_group, ".", "time_init", &(group->attributes.time_init), 1);
    if( ret < 0 ){
        printf("failed to set attribute time_init");
    }
    if( group->attributes.sizes == NULL ){
    }else{
        ret = H5LTset_attribute_double(group->h5_group, ".", "sizes", group->attributes.sizes, (group->attributes.dimension));
        if( ret < 0 ){
            printf("failed to set attribute sizes");
        }
    }
    if( group->attributes.behaviour == NULL ){
    }else{
        ret = H5LTset_attribute_int(group->h5_group, ".", "behaviour", group->attributes.behaviour, (group->attributes.dimension));
        if( ret < 0 ){
            printf("failed to set attribute behaviour");
        }
    }
    if( group->attributes.force_init_x == NULL ){
        group->attributes.force_init_x = strdup("0.0");
    }
    {
        ret = H5LTset_attribute_string(group->h5_group, ".", "force-init-x", group->attributes.force_init_x);
        if( ret < 0 ){
            printf("failed to set attribute force_init_x");
        }
    }
    if( group->attributes.force_init_y == NULL ){
        group->attributes.force_init_y = strdup("0.0");
    }
    {
        ret = H5LTset_attribute_string(group->h5_group, ".", "force-init-y", group->attributes.force_init_y);
        if( ret < 0 ){
            printf("failed to set attribute force_init_y");
        }
    }
    if( group->attributes.force_init_z == NULL ){
        group->attributes.force_init_z = strdup("0.0");
    }
    {
        ret = H5LTset_attribute_string(group->h5_group, ".", "force-init-z", group->attributes.force_init_z);
        if( ret < 0 ){
            printf("failed to set attribute force_init_z");
        }
    }
    if( group->attributes.force_x == NULL ){
        group->attributes.force_x = strdup("0.0");
    }
    {
        ret = H5LTset_attribute_string(group->h5_group, ".", "force-x", group->attributes.force_x);
        if( ret < 0 ){
            printf("failed to set attribute force_x");
        }
    }
    if( group->attributes.force_y == NULL ){
        group->attributes.force_y = strdup("0.0");
    }
    {
        ret = H5LTset_attribute_string(group->h5_group, ".", "force-y", group->attributes.force_y);
        if( ret < 0 ){
            printf("failed to set attribute force_y");
        }
    }
    if( group->attributes.force_z == NULL ){
        group->attributes.force_z = strdup("0.0");
    }
    {
        ret = H5LTset_attribute_string(group->h5_group, ".", "force-z", group->attributes.force_z);
        if( ret < 0 ){
            printf("failed to set attribute force_z");
        }
    }
    ret = H5LTset_attribute_double(group->h5_group, ".", "vh_dr", &(group->attributes.vh_dr), 1);
    if( ret < 0 ){
        printf("failed to set attribute vh_dr");
    }
    ret = H5LTset_attribute_double(group->h5_group, ".", "vh_dt", &(group->attributes.vh_dt), 1);
    if( ret < 0 ){
        printf("failed to set attribute vh_dt");
    }
    ret = H5LTset_attribute_uint(group->h5_group, ".", "vh_n", &(group->attributes.vh_n), 1);
    if( ret < 0 ){
        printf("failed to set attribute vh_n");
    }

}

bd_group_log_t bd_create_group_log(bd_file_t lf, bd_group_log_attributes attrs, const char gname[]){
    hid_t gid = H5Gcreate(lf->root, gname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if( gid < 0 ){
       printf("failed to create group %s.\n",gname);
       return NULL;
    }
    bd_group_log_t group = malloc(sizeof(bd_group_log));
    group->name = strdup(gname);
    group->h5_group = gid;
    group->parent = lf;
    group->attributes = attrs;
    bd_group_log_attribute_sync(group);
    return group;
}

void bd_close_group_log(bd_group_log_t group){
    if( group->parent->rw ){
        bd_group_log_attribute_sync(group);
    }
    //free attributes
    free(group->attributes.sizes);
    free(group->attributes.behaviour);
    free(group->attributes.force_init_x);
    free(group->attributes.force_init_y);
    free(group->attributes.force_init_z);
    free(group->attributes.force_x);
    free(group->attributes.force_y);
    free(group->attributes.force_z);

    H5Gclose(group->h5_group);
    free(group->name);
    free(group);
}

bool bd_flush_group_log(bd_group_log_t group){
    if( group->parent->rw ){
        herr_t retval = H5Fflush(group->h5_group,H5F_SCOPE_GLOBAL);
        return retval >= 0;
    }
    return false;
}

bd_table_frames_t bd_open_table_frames(bd_group_log_t group, const char *tname){
    bd_table_frames_t tb = malloc(sizeof(bd_table_frames));
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

void bd_clear_table_frames(bd_table_frames_t tb){
    if( H5TBget_table_info( tb->parent->h5_group, tb->name, &(tb->num_columns), &(tb->num_records) ) < 0 ){
        printf("table %s not found.\n",tb->name);
        return;
    }
    herr_t status = H5TBdelete_record(tb->parent->h5_group, tb->name, 0, tb->num_records);
    if( status < 0 ){
        printf("failed to delete records.\n");
    }
}

void bd_close_table_frames(bd_table_frames_t tb){
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

bool bd_get_records_frames(bd_table_frames_t table, size_t start, bd_table_frames_recordset_t *records, size_t *num_records){
    bool need_free = false;
    if( *num_records == 0 ){
        *num_records = table->num_records-start;
    }
    if( *records == NULL ){
        need_free = true;
        *records = malloc(sizeof(bd_table_frames_recordset));
    }
    (*records)->set = calloc(*num_records,sizeof(bd_table_frames_record));
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
        bd_table_frames_record_t rec = (*records)->set+i;
        void *mydata = data+i*(table->record_size);
        for(int j=0;j<table->num_columns;++j){
            switch(table->column_names[j][0]){
               case 'f':
                  if( strcmp(table->column_names[j],"frame") == 0 ){
                     rec->frame = (unsigned int *)(mydata+table->column_offsets[j]);
                  }
                  break;
               case 'r':
                  if( strcmp(table->column_names[j],"r") == 0 ){
                     rec->r = (double *)(mydata+table->column_offsets[j]);
                  }
                  break;
               case 't':
                  if( strcmp(table->column_names[j],"time") == 0 ){
                     rec->time = (double *)(mydata+table->column_offsets[j]);
                  }
                  break;
               case 'v':
                  if( strcmp(table->column_names[j],"v") == 0 ){
                     rec->v = (double *)(mydata+table->column_offsets[j]);
                  }
                  break;
            }
        }
    }
    return true;
}

void bd_close_table_frames_recordset(bd_table_frames_recordset_t rec){
    if( rec->set ){
        free(rec->set);
    }
    free(rec->data_raw);
    free(rec);
}

bd_table_frames_t bd_create_table_frames(bd_group_log_t group, const char *tname){
    herr_t ret;
    bd_table_frames_t table = malloc(sizeof(bd_table_frames));
    table->name = strdup(tname);
    table->num_columns = 4;
    table->num_records = 0;
    table->column_offsets = calloc(4,sizeof(size_t));
    table->column_sizes = calloc(4,sizeof(size_t));
    table->column_h5types = calloc(4,sizeof(hid_t));
    table->column_names = calloc(4,sizeof(char *));
    table->parent = group;
    table->column_h5types[0] = H5T_NATIVE_UINT;
    table->column_sizes[0] = sizeof(unsigned int)*1;
    table->column_names[0] = strdup("frame");
    {
        hsize_t array_dims[2] = { (group->attributes.num_particles),(group->attributes.dimension) };
        table->column_h5types[1] = H5Tarray_create(H5T_NATIVE_DOUBLE,2,array_dims);
    }
    table->column_sizes[1] = sizeof(double)*(group->attributes.num_particles)*(group->attributes.dimension);
    table->column_names[1] = strdup("r");
    table->column_h5types[2] = H5T_NATIVE_DOUBLE;
    table->column_sizes[2] = sizeof(double)*1;
    table->column_names[2] = strdup("time");
    {
        hsize_t array_dims[2] = { (group->attributes.num_particles),(group->attributes.dimension) };
        table->column_h5types[3] = H5Tarray_create(H5T_NATIVE_DOUBLE,2,array_dims);
    }
    table->column_sizes[3] = sizeof(double)*(group->attributes.num_particles)*(group->attributes.dimension);
    table->column_names[3] = strdup("v");

    table->record_size = 0;
    for(int i=0;i<4;++i){
        table->column_offsets[i] = table->record_size;
        table->record_size += table->column_sizes[i];
    }
    ret = H5TBmake_table(tname, group->h5_group, tname, table->num_columns, 0, table->record_size, (const char **)table->column_names, table->column_offsets, table->column_h5types, 20, NULL, 5, NULL);
    if( ret < 0 ){
        bd_close_table_frames(table);
        return NULL;
    }
    return table;
}

void bd_add_recordset_frames(bd_table_frames_t table,  bd_table_frames_recordset_t recs){
    herr_t ret = H5TBappend_records(table->parent->h5_group,table->name,recs->num_records,table->record_size,table->column_offsets,table->column_sizes,recs->data_raw);
    if( ret < 0 ){
        printf("failed to append records to table %s\n",table->name);
    }
    bd_close_table_frames_recordset(recs);
}

bd_table_frames_recordset_t bd_create_table_frames_recordset(bd_table_frames_t table, size_t num_records){
    bd_table_frames_recordset_t rs = malloc(sizeof(bd_table_frames_recordset));
    rs->set = malloc(num_records*sizeof(bd_table_frames_record));
    rs->data_raw = malloc(num_records*(table->record_size));
    rs->num_records = num_records;
    for(int i=0;i<num_records;++i){
        void *data = rs->data_raw+(i*table->record_size);
        bd_table_frames_record_t record = &(rs->set[i]);
        record->frame = data+table->column_offsets[0];
        record->r = data+table->column_offsets[1];
        record->time = data+table->column_offsets[2];
        record->v = data+table->column_offsets[3];

    }
    return rs;
}

void bd_add_records_frames(bd_table_frames_t table, size_t num_records, bd_table_frames_record_t records){
    void *data_chunk = calloc(num_records,table->record_size);
    for(int i=0;i<num_records;++i){
        void *data = data_chunk+(i*table->record_size);
        memcpy(data+table->column_offsets[0],records[i].frame,table->column_sizes[0]);
        memcpy(data+table->column_offsets[1],records[i].r,table->column_sizes[1]);
        memcpy(data+table->column_offsets[2],records[i].time,table->column_sizes[2]);
        memcpy(data+table->column_offsets[3],records[i].v,table->column_sizes[3]);

    }
    herr_t ret = H5TBappend_records(table->parent->h5_group,table->name,num_records,table->record_size,table->column_offsets,table->column_sizes,data_chunk);
    if( ret < 0 ){
        printf("failed to append records to table %s\n",table->name);
    }
    free(data_chunk);
}

bd_table_vanhove_t bd_open_table_vanhove(bd_group_log_t group, const char *tname){
    bd_table_vanhove_t tb = malloc(sizeof(bd_table_vanhove));
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

void bd_clear_table_vanhove(bd_table_vanhove_t tb){
    if( H5TBget_table_info( tb->parent->h5_group, tb->name, &(tb->num_columns), &(tb->num_records) ) < 0 ){
        printf("table %s not found.\n",tb->name);
        return;
    }
    herr_t status = H5TBdelete_record(tb->parent->h5_group, tb->name, 0, tb->num_records);
    if( status < 0 ){
        printf("failed to delete records.\n");
    }
}

void bd_close_table_vanhove(bd_table_vanhove_t tb){
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

bool bd_get_records_vanhove(bd_table_vanhove_t table, size_t start, bd_table_vanhove_recordset_t *records, size_t *num_records){
    bool need_free = false;
    if( *num_records == 0 ){
        *num_records = table->num_records-start;
    }
    if( *records == NULL ){
        need_free = true;
        *records = malloc(sizeof(bd_table_vanhove_recordset));
    }
    (*records)->set = calloc(*num_records,sizeof(bd_table_vanhove_record));
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
        bd_table_vanhove_record_t rec = (*records)->set+i;
        void *mydata = data+i*(table->record_size);
        for(int j=0;j<table->num_columns;++j){
            switch(table->column_names[j][0]){
               case 'f':
                  if( strcmp(table->column_names[j],"frame") == 0 ){
                     rec->frame = (unsigned int *)(mydata+table->column_offsets[j]);
                  }
                  break;
               case 't':
                  if( strcmp(table->column_names[j],"time") == 0 ){
                     rec->time = (double *)(mydata+table->column_offsets[j]);
                  }
                  break;
               case 'v':
                  if( strcmp(table->column_names[j],"vanhove") == 0 ){
                     rec->vanhove = (double *)(mydata+table->column_offsets[j]);
                  }
                  break;
            }
        }
    }
    return true;
}

void bd_close_table_vanhove_recordset(bd_table_vanhove_recordset_t rec){
    if( rec->set ){
        free(rec->set);
    }
    free(rec->data_raw);
    free(rec);
}

bd_table_vanhove_t bd_create_table_vanhove(bd_group_log_t group, const char *tname){
    herr_t ret;
    bd_table_vanhove_t table = malloc(sizeof(bd_table_vanhove));
    table->name = strdup(tname);
    table->num_columns = 3;
    table->num_records = 0;
    table->column_offsets = calloc(3,sizeof(size_t));
    table->column_sizes = calloc(3,sizeof(size_t));
    table->column_h5types = calloc(3,sizeof(hid_t));
    table->column_names = calloc(3,sizeof(char *));
    table->parent = group;
    table->column_h5types[0] = H5T_NATIVE_UINT;
    table->column_sizes[0] = sizeof(unsigned int)*1;
    table->column_names[0] = strdup("frame");
    table->column_h5types[1] = H5T_NATIVE_DOUBLE;
    table->column_sizes[1] = sizeof(double)*1;
    table->column_names[1] = strdup("time");
    {
        hsize_t array_dims[2] = { 2,(group->attributes.vh_n) };
        table->column_h5types[2] = H5Tarray_create(H5T_NATIVE_DOUBLE,2,array_dims);
    }
    table->column_sizes[2] = sizeof(double)*2*(group->attributes.vh_n);
    table->column_names[2] = strdup("vanhove");

    table->record_size = 0;
    for(int i=0;i<3;++i){
        table->column_offsets[i] = table->record_size;
        table->record_size += table->column_sizes[i];
    }
    ret = H5TBmake_table(tname, group->h5_group, tname, table->num_columns, 0, table->record_size, (const char **)table->column_names, table->column_offsets, table->column_h5types, 20, NULL, 5, NULL);
    if( ret < 0 ){
        bd_close_table_vanhove(table);
        return NULL;
    }
    return table;
}

void bd_add_recordset_vanhove(bd_table_vanhove_t table,  bd_table_vanhove_recordset_t recs){
    herr_t ret = H5TBappend_records(table->parent->h5_group,table->name,recs->num_records,table->record_size,table->column_offsets,table->column_sizes,recs->data_raw);
    if( ret < 0 ){
        printf("failed to append records to table %s\n",table->name);
    }
    bd_close_table_vanhove_recordset(recs);
}

bd_table_vanhove_recordset_t bd_create_table_vanhove_recordset(bd_table_vanhove_t table, size_t num_records){
    bd_table_vanhove_recordset_t rs = malloc(sizeof(bd_table_vanhove_recordset));
    rs->set = malloc(num_records*sizeof(bd_table_vanhove_record));
    rs->data_raw = malloc(num_records*(table->record_size));
    rs->num_records = num_records;
    for(int i=0;i<num_records;++i){
        void *data = rs->data_raw+(i*table->record_size);
        bd_table_vanhove_record_t record = &(rs->set[i]);
        record->frame = data+table->column_offsets[0];
        record->time = data+table->column_offsets[1];
        record->vanhove = data+table->column_offsets[2];

    }
    return rs;
}

void bd_add_records_vanhove(bd_table_vanhove_t table, size_t num_records, bd_table_vanhove_record_t records){
    void *data_chunk = calloc(num_records,table->record_size);
    for(int i=0;i<num_records;++i){
        void *data = data_chunk+(i*table->record_size);
        memcpy(data+table->column_offsets[0],records[i].frame,table->column_sizes[0]);
        memcpy(data+table->column_offsets[1],records[i].time,table->column_sizes[1]);
        memcpy(data+table->column_offsets[2],records[i].vanhove,table->column_sizes[2]);

    }
    herr_t ret = H5TBappend_records(table->parent->h5_group,table->name,num_records,table->record_size,table->column_offsets,table->column_sizes,data_chunk);
    if( ret < 0 ){
        printf("failed to append records to table %s\n",table->name);
    }
    free(data_chunk);
}

bd_table_current_t bd_open_table_current(bd_group_log_t group, const char *tname){
    bd_table_current_t tb = malloc(sizeof(bd_table_current));
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

void bd_clear_table_current(bd_table_current_t tb){
    if( H5TBget_table_info( tb->parent->h5_group, tb->name, &(tb->num_columns), &(tb->num_records) ) < 0 ){
        printf("table %s not found.\n",tb->name);
        return;
    }
    herr_t status = H5TBdelete_record(tb->parent->h5_group, tb->name, 0, tb->num_records);
    if( status < 0 ){
        printf("failed to delete records.\n");
    }
}

void bd_close_table_current(bd_table_current_t tb){
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

bool bd_get_records_current(bd_table_current_t table, size_t start, bd_table_current_recordset_t *records, size_t *num_records){
    bool need_free = false;
    if( *num_records == 0 ){
        *num_records = table->num_records-start;
    }
    if( *records == NULL ){
        need_free = true;
        *records = malloc(sizeof(bd_table_current_recordset));
    }
    (*records)->set = calloc(*num_records,sizeof(bd_table_current_record));
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
        bd_table_current_record_t rec = (*records)->set+i;
        void *mydata = data+i*(table->record_size);
        for(int j=0;j<table->num_columns;++j){
            switch(table->column_names[j][0]){
               case 'c':
                  if( strcmp(table->column_names[j],"current") == 0 ){
                     rec->current = (double *)(mydata+table->column_offsets[j]);
                  }
                  break;
               case 'f':
                  if( strcmp(table->column_names[j],"frame") == 0 ){
                     rec->frame = (unsigned int *)(mydata+table->column_offsets[j]);
                  }
                  break;
               case 't':
                  if( strcmp(table->column_names[j],"time") == 0 ){
                     rec->time = (double *)(mydata+table->column_offsets[j]);
                  }
                  break;
            }
        }
    }
    return true;
}

void bd_close_table_current_recordset(bd_table_current_recordset_t rec){
    if( rec->set ){
        free(rec->set);
    }
    free(rec->data_raw);
    free(rec);
}

bd_table_current_t bd_create_table_current(bd_group_log_t group, const char *tname){
    herr_t ret;
    bd_table_current_t table = malloc(sizeof(bd_table_current));
    table->name = strdup(tname);
    table->num_columns = 3;
    table->num_records = 0;
    table->column_offsets = calloc(3,sizeof(size_t));
    table->column_sizes = calloc(3,sizeof(size_t));
    table->column_h5types = calloc(3,sizeof(hid_t));
    table->column_names = calloc(3,sizeof(char *));
    table->parent = group;
    {
        hsize_t array_dims[2] = { 2,(group->attributes.vh_n) };
        table->column_h5types[0] = H5Tarray_create(H5T_NATIVE_DOUBLE,2,array_dims);
    }
    table->column_sizes[0] = sizeof(double)*2*(group->attributes.vh_n);
    table->column_names[0] = strdup("current");
    table->column_h5types[1] = H5T_NATIVE_UINT;
    table->column_sizes[1] = sizeof(unsigned int)*1;
    table->column_names[1] = strdup("frame");
    table->column_h5types[2] = H5T_NATIVE_DOUBLE;
    table->column_sizes[2] = sizeof(double)*1;
    table->column_names[2] = strdup("time");

    table->record_size = 0;
    for(int i=0;i<3;++i){
        table->column_offsets[i] = table->record_size;
        table->record_size += table->column_sizes[i];
    }
    ret = H5TBmake_table(tname, group->h5_group, tname, table->num_columns, 0, table->record_size, (const char **)table->column_names, table->column_offsets, table->column_h5types, 20, NULL, 5, NULL);
    if( ret < 0 ){
        bd_close_table_current(table);
        return NULL;
    }
    return table;
}

void bd_add_recordset_current(bd_table_current_t table,  bd_table_current_recordset_t recs){
    herr_t ret = H5TBappend_records(table->parent->h5_group,table->name,recs->num_records,table->record_size,table->column_offsets,table->column_sizes,recs->data_raw);
    if( ret < 0 ){
        printf("failed to append records to table %s\n",table->name);
    }
    bd_close_table_current_recordset(recs);
}

bd_table_current_recordset_t bd_create_table_current_recordset(bd_table_current_t table, size_t num_records){
    bd_table_current_recordset_t rs = malloc(sizeof(bd_table_current_recordset));
    rs->set = malloc(num_records*sizeof(bd_table_current_record));
    rs->data_raw = malloc(num_records*(table->record_size));
    rs->num_records = num_records;
    for(int i=0;i<num_records;++i){
        void *data = rs->data_raw+(i*table->record_size);
        bd_table_current_record_t record = &(rs->set[i]);
        record->current = data+table->column_offsets[0];
        record->frame = data+table->column_offsets[1];
        record->time = data+table->column_offsets[2];

    }
    return rs;
}

void bd_add_records_current(bd_table_current_t table, size_t num_records, bd_table_current_record_t records){
    void *data_chunk = calloc(num_records,table->record_size);
    for(int i=0;i<num_records;++i){
        void *data = data_chunk+(i*table->record_size);
        memcpy(data+table->column_offsets[0],records[i].current,table->column_sizes[0]);
        memcpy(data+table->column_offsets[1],records[i].frame,table->column_sizes[1]);
        memcpy(data+table->column_offsets[2],records[i].time,table->column_sizes[2]);

    }
    herr_t ret = H5TBappend_records(table->parent->h5_group,table->name,num_records,table->record_size,table->column_offsets,table->column_sizes,data_chunk);
    if( ret < 0 ){
        printf("failed to append records to table %s\n",table->name);
    }
    free(data_chunk);
}

bd_table_forces_t bd_open_table_forces(bd_group_log_t group, const char *tname){
    bd_table_forces_t tb = malloc(sizeof(bd_table_forces));
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

void bd_clear_table_forces(bd_table_forces_t tb){
    if( H5TBget_table_info( tb->parent->h5_group, tb->name, &(tb->num_columns), &(tb->num_records) ) < 0 ){
        printf("table %s not found.\n",tb->name);
        return;
    }
    herr_t status = H5TBdelete_record(tb->parent->h5_group, tb->name, 0, tb->num_records);
    if( status < 0 ){
        printf("failed to delete records.\n");
    }
}

void bd_close_table_forces(bd_table_forces_t tb){
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

bool bd_get_records_forces(bd_table_forces_t table, size_t start, bd_table_forces_recordset_t *records, size_t *num_records){
    bool need_free = false;
    if( *num_records == 0 ){
        *num_records = table->num_records-start;
    }
    if( *records == NULL ){
        need_free = true;
        *records = malloc(sizeof(bd_table_forces_recordset));
    }
    (*records)->set = calloc(*num_records,sizeof(bd_table_forces_record));
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
        bd_table_forces_record_t rec = (*records)->set+i;
        void *mydata = data+i*(table->record_size);
        for(int j=0;j<table->num_columns;++j){
            switch(table->column_names[j][0]){
               case 'a':
                  if( strcmp(table->column_names[j],"adiabatic") == 0 ){
                     rec->adiabatic = (double *)(mydata+table->column_offsets[j]);
                  }
                  break;
               case 'f':
                  if( strcmp(table->column_names[j],"frame") == 0 ){
                     rec->frame = (unsigned int *)(mydata+table->column_offsets[j]);
                  }
                  break;
               case 's':
                  if( strcmp(table->column_names[j],"superadiabatic") == 0 ){
                     rec->superadiabatic = (double *)(mydata+table->column_offsets[j]);
                  }
                  break;
               case 't':
                  if( strcmp(table->column_names[j],"time") == 0 ){
                     rec->time = (double *)(mydata+table->column_offsets[j]);
                  }
                  break;
            }
        }
    }
    return true;
}

void bd_close_table_forces_recordset(bd_table_forces_recordset_t rec){
    if( rec->set ){
        free(rec->set);
    }
    free(rec->data_raw);
    free(rec);
}

bd_table_forces_t bd_create_table_forces(bd_group_log_t group, const char *tname){
    herr_t ret;
    bd_table_forces_t table = malloc(sizeof(bd_table_forces));
    table->name = strdup(tname);
    table->num_columns = 4;
    table->num_records = 0;
    table->column_offsets = calloc(4,sizeof(size_t));
    table->column_sizes = calloc(4,sizeof(size_t));
    table->column_h5types = calloc(4,sizeof(hid_t));
    table->column_names = calloc(4,sizeof(char *));
    table->parent = group;
    {
        hsize_t array_dims[2] = { 2,(group->attributes.vh_n) };
        table->column_h5types[0] = H5Tarray_create(H5T_NATIVE_DOUBLE,2,array_dims);
    }
    table->column_sizes[0] = sizeof(double)*2*(group->attributes.vh_n);
    table->column_names[0] = strdup("adiabatic");
    table->column_h5types[1] = H5T_NATIVE_UINT;
    table->column_sizes[1] = sizeof(unsigned int)*1;
    table->column_names[1] = strdup("frame");
    {
        hsize_t array_dims[2] = { 2,(group->attributes.vh_n) };
        table->column_h5types[2] = H5Tarray_create(H5T_NATIVE_DOUBLE,2,array_dims);
    }
    table->column_sizes[2] = sizeof(double)*2*(group->attributes.vh_n);
    table->column_names[2] = strdup("superadiabatic");
    table->column_h5types[3] = H5T_NATIVE_DOUBLE;
    table->column_sizes[3] = sizeof(double)*1;
    table->column_names[3] = strdup("time");

    table->record_size = 0;
    for(int i=0;i<4;++i){
        table->column_offsets[i] = table->record_size;
        table->record_size += table->column_sizes[i];
    }
    ret = H5TBmake_table(tname, group->h5_group, tname, table->num_columns, 0, table->record_size, (const char **)table->column_names, table->column_offsets, table->column_h5types, 20, NULL, 5, NULL);
    if( ret < 0 ){
        bd_close_table_forces(table);
        return NULL;
    }
    return table;
}

void bd_add_recordset_forces(bd_table_forces_t table,  bd_table_forces_recordset_t recs){
    herr_t ret = H5TBappend_records(table->parent->h5_group,table->name,recs->num_records,table->record_size,table->column_offsets,table->column_sizes,recs->data_raw);
    if( ret < 0 ){
        printf("failed to append records to table %s\n",table->name);
    }
    bd_close_table_forces_recordset(recs);
}

bd_table_forces_recordset_t bd_create_table_forces_recordset(bd_table_forces_t table, size_t num_records){
    bd_table_forces_recordset_t rs = malloc(sizeof(bd_table_forces_recordset));
    rs->set = malloc(num_records*sizeof(bd_table_forces_record));
    rs->data_raw = malloc(num_records*(table->record_size));
    rs->num_records = num_records;
    for(int i=0;i<num_records;++i){
        void *data = rs->data_raw+(i*table->record_size);
        bd_table_forces_record_t record = &(rs->set[i]);
        record->adiabatic = data+table->column_offsets[0];
        record->frame = data+table->column_offsets[1];
        record->superadiabatic = data+table->column_offsets[2];
        record->time = data+table->column_offsets[3];

    }
    return rs;
}

void bd_add_records_forces(bd_table_forces_t table, size_t num_records, bd_table_forces_record_t records){
    void *data_chunk = calloc(num_records,table->record_size);
    for(int i=0;i<num_records;++i){
        void *data = data_chunk+(i*table->record_size);
        memcpy(data+table->column_offsets[0],records[i].adiabatic,table->column_sizes[0]);
        memcpy(data+table->column_offsets[1],records[i].frame,table->column_sizes[1]);
        memcpy(data+table->column_offsets[2],records[i].superadiabatic,table->column_sizes[2]);
        memcpy(data+table->column_offsets[3],records[i].time,table->column_sizes[3]);

    }
    herr_t ret = H5TBappend_records(table->parent->h5_group,table->name,num_records,table->record_size,table->column_offsets,table->column_sizes,data_chunk);
    if( ret < 0 ){
        printf("failed to append records to table %s\n",table->name);
    }
    free(data_chunk);
}

