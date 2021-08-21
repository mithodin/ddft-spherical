#ifndef _HDF_HL_H
    #include <hdf5_hl.h>
#endif

typedef struct _cutdsynjly {
    hid_t root;
    hid_t h5_file;
    bool rw;
} vh_log_file;

typedef vh_log_file * vh_log_file_t;

typedef struct _znvohjlysr {
        unsigned int num_bins;
        unsigned int bins_sphere;
        double dr;
        double dt;
        double timestep_log;
        double time_simulation;
        double bulk_density;
        char *simulation_description;
} vh_log_group_simulation_attributes;

typedef vh_log_group_simulation_attributes * vh_log_group_simulation_attributes_t;

typedef struct _qjsttfxlyy {
    char *name;
    hid_t h5_group;
    vh_log_file_t parent;
    vh_log_group_simulation_attributes attributes;
} vh_log_group_simulation;

typedef vh_log_group_simulation * vh_log_group_simulation_t;

typedef struct _tkjvjmvelk {
    size_t record_size;
    size_t *column_offsets;
    size_t *column_sizes;
    hid_t *column_h5types;
    char **column_names;
    hsize_t num_columns;
    hsize_t num_records;
    char *name;
    vh_log_group_simulation_t parent;
} vh_log_table_vanhove;

typedef vh_log_table_vanhove * vh_log_table_vanhove_t;

typedef struct _leincgehiz {
    unsigned long *frame;
    double *time;
    double *vanhove_self;
    double *vanhove_distinct;
    double *current_self;
    double *current_distinct;
} vh_log_table_vanhove_record;

typedef vh_log_table_vanhove_record * vh_log_table_vanhove_record_t;

typedef struct _seawggfbww {
    vh_log_table_vanhove_record_t set;
    size_t num_records;
    void *data_raw;
} vh_log_table_vanhove_recordset;

typedef vh_log_table_vanhove_recordset * vh_log_table_vanhove_recordset_t;

vh_log_file_t vh_log_open(const char *, const char *, bool);
vh_log_file_t vh_log_create(const char *);
void vh_log_close(vh_log_file_t);
bool vh_log_flush(vh_log_file_t);
bool vh_log_get_groups(vh_log_file_t, const char *, char ***, size_t *);
vh_log_group_simulation_t vh_log_open_group_simulation(vh_log_file_t, const char[]);
vh_log_group_simulation_t vh_log_create_group_simulation(vh_log_file_t, vh_log_group_simulation_attributes, const char[]);
void vh_log_group_simulation_attribute_sync(vh_log_group_simulation_t);
void vh_log_close_group_simulation(vh_log_group_simulation_t group);
bool vh_log_flush_group_simulation(vh_log_group_simulation_t group);
vh_log_table_vanhove_t vh_log_open_table_vanhove(vh_log_group_simulation_t, const char *);
void vh_log_close_table_vanhove(vh_log_table_vanhove_t);
void vh_log_clear_table_vanhove(vh_log_table_vanhove_t);
bool vh_log_get_records_vanhove(vh_log_table_vanhove_t, size_t, vh_log_table_vanhove_recordset_t *, size_t *);
void vh_log_close_table_vanhove_recordset(vh_log_table_vanhove_recordset_t);
vh_log_table_vanhove_t vh_log_create_table_vanhove(vh_log_group_simulation_t, const char *);
vh_log_table_vanhove_recordset_t vh_log_create_table_vanhove_recordset(vh_log_table_vanhove_t,size_t);
void vh_log_add_records_vanhove(vh_log_table_vanhove_t, size_t, vh_log_table_vanhove_record_t);
void vh_log_add_recordset_vanhove(vh_log_table_vanhove_t, vh_log_table_vanhove_recordset_t);
