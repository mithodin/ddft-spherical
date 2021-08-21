#define str2(x) #x
#define str(x) str2(x)

#ifdef DEBUG
	#define do_check(x) if( !(x) ){ fprintf(stderr,"dfts failed check %s at line %d in %s\n",str2(x),__LINE__,__FILE__);exit(-1); }
#else
	#define do_check(x) if( !(x) ){ fprintf(stderr,"dfts has encountered an problem and is exiting.\n");exit(-1); }
#endif
