#ifndef _HDF_HL_H
    #include <hdf5_hl.h>
#endif

typedef struct _govskplule {
    hid_t root;
    hid_t h5_file;
    bool rw;
} bd_file;

typedef bd_file * bd_file_t;

typedef struct _aidlhcrmrc {
        unsigned int dimension;
        unsigned int num_particles;
        double sigma;
        double zeta0;
        double dt;
        double frame_timestep;
        double time;
        double time_init;
        double *sizes;
        int *behaviour;
        char *force_init_x;
        char *force_init_y;
        char *force_init_z;
        char *force_x;
        char *force_y;
        char *force_z;
        double vh_dr;
        double vh_dt;
        unsigned int vh_n;
} bd_group_log_attributes;

typedef bd_group_log_attributes * bd_group_log_attributes_t;

typedef struct _rinnafttsp {
    char *name;
    hid_t h5_group;
    bd_file_t parent;
    bd_group_log_attributes attributes;
} bd_group_log;

typedef bd_group_log * bd_group_log_t;

typedef struct _vygacshjih {
    size_t record_size;
    size_t *column_offsets;
    size_t *column_sizes;
    hid_t *column_h5types;
    char **column_names;
    hsize_t num_columns;
    hsize_t num_records;
    char *name;
    bd_group_log_t parent;
} bd_table_frames;

typedef bd_table_frames * bd_table_frames_t;

typedef struct _wsbphkreuz {
    unsigned int *frame;
    double *time;
    double *v;
    double *r;
} bd_table_frames_record;

typedef bd_table_frames_record * bd_table_frames_record_t;

typedef struct _lqwfdnlwph {
    bd_table_frames_record_t set;
    size_t num_records;
    void *data_raw;
} bd_table_frames_recordset;

typedef bd_table_frames_recordset * bd_table_frames_recordset_t;

typedef struct _ayyyczkdxd {
    size_t record_size;
    size_t *column_offsets;
    size_t *column_sizes;
    hid_t *column_h5types;
    char **column_names;
    hsize_t num_columns;
    hsize_t num_records;
    char *name;
    bd_group_log_t parent;
} bd_table_vanhove;

typedef bd_table_vanhove * bd_table_vanhove_t;

typedef struct _mededwusvs {
    unsigned int *frame;
    double *time;
    double *vanhove;
} bd_table_vanhove_record;

typedef bd_table_vanhove_record * bd_table_vanhove_record_t;

typedef struct _fvwqbkhadu {
    bd_table_vanhove_record_t set;
    size_t num_records;
    void *data_raw;
} bd_table_vanhove_recordset;

typedef bd_table_vanhove_recordset * bd_table_vanhove_recordset_t;

typedef struct _sohofcqaij {
    size_t record_size;
    size_t *column_offsets;
    size_t *column_sizes;
    hid_t *column_h5types;
    char **column_names;
    hsize_t num_columns;
    hsize_t num_records;
    char *name;
    bd_group_log_t parent;
} bd_table_current;

typedef bd_table_current * bd_table_current_t;

typedef struct _cldtercknq {
    unsigned int *frame;
    double *time;
    double *current;
} bd_table_current_record;

typedef bd_table_current_record * bd_table_current_record_t;

typedef struct _schopsjpgu {
    bd_table_current_record_t set;
    size_t num_records;
    void *data_raw;
} bd_table_current_recordset;

typedef bd_table_current_recordset * bd_table_current_recordset_t;

typedef struct _cmrhsxerez {
    size_t record_size;
    size_t *column_offsets;
    size_t *column_sizes;
    hid_t *column_h5types;
    char **column_names;
    hsize_t num_columns;
    hsize_t num_records;
    char *name;
    bd_group_log_t parent;
} bd_table_forces;

typedef bd_table_forces * bd_table_forces_t;

typedef struct _zvkngtdisv {
    unsigned int *frame;
    double *time;
    double *adiabatic;
    double *superadiabatic;
} bd_table_forces_record;

typedef bd_table_forces_record * bd_table_forces_record_t;

typedef struct _klqpmasebl {
    bd_table_forces_record_t set;
    size_t num_records;
    void *data_raw;
} bd_table_forces_recordset;

typedef bd_table_forces_recordset * bd_table_forces_recordset_t;

bd_file_t bd_open(const char *, const char *, bool);
bd_file_t bd_create(const char *);
void bd_close(bd_file_t);
bool bd_flush(bd_file_t);
bool bd_get_groups(bd_file_t, const char *, char ***, size_t *);
bd_group_log_t bd_open_group_log(bd_file_t, const char[]);
bd_group_log_t bd_create_group_log(bd_file_t, bd_group_log_attributes, const char[]);
void bd_group_log_attribute_sync(bd_group_log_t);
void bd_close_group_log(bd_group_log_t group);
bool bd_flush_group_log(bd_group_log_t group);
bd_table_frames_t bd_open_table_frames(bd_group_log_t, const char *);
void bd_close_table_frames(bd_table_frames_t);
void bd_clear_table_frames(bd_table_frames_t);
bool bd_get_records_frames(bd_table_frames_t, size_t, bd_table_frames_recordset_t *, size_t *);
void bd_close_table_frames_recordset(bd_table_frames_recordset_t);
bd_table_frames_t bd_create_table_frames(bd_group_log_t, const char *);
bd_table_frames_recordset_t bd_create_table_frames_recordset(bd_table_frames_t,size_t);
void bd_add_records_frames(bd_table_frames_t, size_t, bd_table_frames_record_t);
void bd_add_recordset_frames(bd_table_frames_t, bd_table_frames_recordset_t);
bd_table_vanhove_t bd_open_table_vanhove(bd_group_log_t, const char *);
void bd_close_table_vanhove(bd_table_vanhove_t);
void bd_clear_table_vanhove(bd_table_vanhove_t);
bool bd_get_records_vanhove(bd_table_vanhove_t, size_t, bd_table_vanhove_recordset_t *, size_t *);
void bd_close_table_vanhove_recordset(bd_table_vanhove_recordset_t);
bd_table_vanhove_t bd_create_table_vanhove(bd_group_log_t, const char *);
bd_table_vanhove_recordset_t bd_create_table_vanhove_recordset(bd_table_vanhove_t,size_t);
void bd_add_records_vanhove(bd_table_vanhove_t, size_t, bd_table_vanhove_record_t);
void bd_add_recordset_vanhove(bd_table_vanhove_t, bd_table_vanhove_recordset_t);
bd_table_current_t bd_open_table_current(bd_group_log_t, const char *);
void bd_close_table_current(bd_table_current_t);
void bd_clear_table_current(bd_table_current_t);
bool bd_get_records_current(bd_table_current_t, size_t, bd_table_current_recordset_t *, size_t *);
void bd_close_table_current_recordset(bd_table_current_recordset_t);
bd_table_current_t bd_create_table_current(bd_group_log_t, const char *);
bd_table_current_recordset_t bd_create_table_current_recordset(bd_table_current_t,size_t);
void bd_add_records_current(bd_table_current_t, size_t, bd_table_current_record_t);
void bd_add_recordset_current(bd_table_current_t, bd_table_current_recordset_t);
bd_table_forces_t bd_open_table_forces(bd_group_log_t, const char *);
void bd_close_table_forces(bd_table_forces_t);
void bd_clear_table_forces(bd_table_forces_t);
bool bd_get_records_forces(bd_table_forces_t, size_t, bd_table_forces_recordset_t *, size_t *);
void bd_close_table_forces_recordset(bd_table_forces_recordset_t);
bd_table_forces_t bd_create_table_forces(bd_group_log_t, const char *);
bd_table_forces_recordset_t bd_create_table_forces_recordset(bd_table_forces_t,size_t);
void bd_add_records_forces(bd_table_forces_t, size_t, bd_table_forces_record_t);
void bd_add_recordset_forces(bd_table_forces_t, bd_table_forces_recordset_t);
