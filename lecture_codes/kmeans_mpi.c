// kmeans_mpi_stats_better.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

static inline uint64_t splitmix64(uint64_t x){
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x>>30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x>>27)) * 0x94D049BB133111EBULL;
  return x ^ (x>>31);
}
static inline double rand01_u64(uint64_t seed, uint64_t idx){
  uint64_t z = splitmix64(seed ^ idx);
  return (z >> 11) * (1.0/9007199254740992.0);
}
static inline void box_muller(uint64_t seed, uint64_t tag, double *z0, double *z1){
  double u1 = fmax(1e-12, rand01_u64(seed, tag*2ULL+0));
  double u2 = rand01_u64(seed, tag*2ULL+1);
  double r = sqrt(-2.0*log(u1)), th = 2.0*M_PI*u2;
  *z0 = r*cos(th); *z1 = r*sin(th);
}
static inline double sqdist(const double* a,const double* b,int D){
  double s=0.0; for(int d=0; d<D; ++d){ double t=a[d]-b[d]; s+=t*t; } return s;
}

typedef enum { DATA_UNIFORM, DATA_BLOBS } data_mode_t;
typedef enum { INIT_FIRST, INIT_KPP } init_mode_t;

typedef struct {
  int N,K,D,iters;
  data_mode_t data_mode;
  init_mode_t init_mode;
  double blob_std;
  uint64_t seed;
} cfg_t;

static void parse_args(int argc, char** argv, cfg_t* c){
  c->N=(argc>1)?atoi(argv[1]):200000; c->K=(argc>2)?atoi(argv[2]):8;
  c->D=(argc>3)?atoi(argv[3]):2;      c->iters=(argc>4)?atoi(argv[4]):15;
  c->data_mode=DATA_UNIFORM; c->init_mode=INIT_FIRST; c->blob_std=0.08; c->seed=12345ULL;
  for(int a=5;a<argc;++a){
    if(!strncmp(argv[a],"--data=uniform",14)) c->data_mode=DATA_UNIFORM;
    else if(!strncmp(argv[a],"--data=blobs",12)) c->data_mode=DATA_BLOBS;
    else if(!strncmp(argv[a],"--init=kpp",10)) c->init_mode=INIT_KPP;
    else if(!strncmp(argv[a],"--init=first",12)) c->init_mode=INIT_FIRST;
    else if(!strncmp(argv[a],"--blob-std=",11)) c->blob_std=atof(argv[a]+11);
    else if(!strncmp(argv[a],"--seed=",7)) c->seed=(uint64_t)strtoull(argv[a]+7,NULL,10);
  }
}

static void gen_point_uniform(const cfg_t* c, uint64_t gi, double* out){ // one point by global index
  for(int d=0; d<c->D; ++d){
    uint64_t key = gi*0x9E3779B97F4A7C15ULL ^ (uint64_t)d;
    out[d] = rand01_u64(c->seed, key);
  }
}

static void gen_point_blobs(const cfg_t* c, uint64_t gi, double* out){
  // deterministic centers
  static int centers_ready=0;
  static double *center=NULL; // allocated lazily per process
  if(!centers_ready){
    center=(double*)malloc(sizeof(double)*(size_t)c->K*c->D);
    for(int k=0;k<c->K;++k)
      for(int d=0; d<c->D; ++d){
        uint64_t key = (uint64_t)k*0xDB4F0B9175AE2165ULL ^ (uint64_t)d;
        center[(size_t)k*c->D + d] = 0.15 + 0.70*rand01_u64(c->seed, key);
      }
    centers_ready=1;
  }
  int k = (int)(gi % (uint64_t)c->K);
  for(int d=0; d<c->D; d+=2){
    double z0,z1; box_muller(c->seed ^ (uint64_t)(gi*1315423911u + d), gi, &z0,&z1);
    out[d]   = center[(size_t)k*c->D + d]   + c->blob_std*z0;
    if(d+1<c->D) out[d+1] = center[(size_t)k*c->D + d+1] + c->blob_std*z1;
  }
  for(int d=0; d<c->D; ++d){ if(out[d]<0.0) out[d]=0.0; if(out[d]>1.0) out[d]=1.0; }
}

static void gen_local(const cfg_t* c, int off, int nloc, double* X){
  for(int i=0;i<nloc;++i){
    uint64_t gi=(uint64_t)(off+i);
    if(c->data_mode==DATA_UNIFORM) gen_point_uniform(c, gi, &X[(size_t)i*c->D]);
    else                           gen_point_blobs(c,   gi, &X[(size_t)i*c->D]);
  }
}

static void init_first_global(const cfg_t* c, double* C){ // any rank can call
  for(int k=0;k<c->K;++k){
    uint64_t gi=(uint64_t)k;
    if(c->data_mode==DATA_UNIFORM) gen_point_uniform(c, gi, &C[(size_t)k*c->D]);
    else                           gen_point_blobs(c,   gi, &C[(size_t)k*c->D]);
  }
}

// k-means++ on a reproducible sample of the first S global points (rank 0), then bcast
static void init_kpp_sample_bcast(const cfg_t* c, double* C, int rank){
  const int S = (c->N < 20000 ? c->N : 20000); // sample size
  if(rank==0){
    double* Xs=(double*)malloc(sizeof(double)*(size_t)S*c->D);
    for(int i=0;i<S;++i){
      uint64_t gi=(uint64_t)i;
      if(c->data_mode==DATA_UNIFORM) gen_point_uniform(c, gi, &Xs[(size_t)i*c->D]);
      else                           gen_point_blobs(c,   gi, &Xs[(size_t)i*c->D]);
    }
    int D=c->D, K=c->K;
    int first = (int)((uint64_t)(rand01_u64(c->seed, 0xABCDEF)* (double)S) % S);
    for(int d=0; d<D; ++d) C[d]=Xs[(size_t)first*D + d];
    double* D2=(double*)malloc(sizeof(double)*(size_t)S);
    for(int i=0;i<S;++i) D2[i]=sqdist(&Xs[(size_t)i*D], &C[0], D);
    for(int m=1;m<K;++m){
      double sum=0.0; for(int i=0;i<S;++i) sum+=D2[i];
      double r = rand01_u64(c->seed, 0xABC000 + (uint64_t)m) * sum;
      int idx=0; double acc=0.0; for(; idx<S-1; ++idx){ acc+=D2[idx]; if(acc>=r) break; }
      for(int d=0; d<D; ++d) C[(size_t)m*D + d] = Xs[(size_t)idx*D + d];
      for(int i=0;i<S;++i){
        double d2 = sqdist(&Xs[(size_t)i*D], &C[(size_t)m*D], D);
        if(d2 < D2[i]) D2[i]=d2;
      }
    }
    free(D2); free(Xs);
  }
  MPI_Bcast(C, c->K*c->D, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv){
  MPI_Init(&argc,&argv);
  int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);

  cfg_t c; parse_args(argc, argv, &c);

  // 1D block partition of points
  int base = c.N / size, rem = c.N % size;
  int nloc = base + (rank < rem);
  int off  = rank*base + (rank < rem ? rank : rem);

  double *X=(double*)malloc(sizeof(double)*(size_t)nloc*c.D);
  double *C=(double*)malloc(sizeof(double)*(size_t)c.K*c.D);
  int    *A=(int*)   malloc(sizeof(int)   *(size_t)nloc);
  if(!X||!C||!A){ fprintf(stderr,"alloc fail @%d\n", rank); MPI_Abort(MPI_COMM_WORLD,1); }

  gen_local(&c, off, nloc, X);
  if(c.init_mode==INIT_KPP) init_kpp_sample_bcast(&c, C, rank);
  else                      init_first_global(&c, C);

  if(rank==0){
    printf("Config: N=%d K=%d D=%d iters=%d data=%s init=%s blob_std=%.3f, P=%d\n",
           c.N,c.K,c.D,c.iters,
           c.data_mode==DATA_BLOBS?"blobs":"uniform",
           c.init_mode==INIT_KPP?"kpp(sample)":"first", c.blob_std, size);
    printf("\nIter |   total_SSE    | rel_impr | (per-cluster: k: n, SSE)\n");
    printf("----------------------------------------------------------------\n");
  }

  double prev_total = INFINITY;
  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();

  for(int it=0; it<c.iters; ++it){
    // Assign local
    for(int i=0;i<nloc;++i){
      const double* xi=&X[(size_t)i*c.D];
      double best=1e300; int bestk=0;
      for(int k=0;k<c.K;++k){
        double d2=sqdist(xi, &C[(size_t)k*c.D], c.D);
        if(d2<best){ best=d2; bestk=k; }
      }
      A[i]=bestk;
    }

    // Local partials
    double *sum_loc=(double*)calloc((size_t)c.K*c.D, sizeof(double));
    double *sumsq_loc=(double*)calloc((size_t)c.K,   sizeof(double));
    int    *cnt_loc=(int*)   calloc((size_t)c.K,     sizeof(int));
    for(int i=0;i<nloc;++i){
      int k=A[i]; const double* xi=&X[(size_t)i*c.D];
      double n2=0.0; for(int d=0; d<c.D; ++d){ sum_loc[(size_t)k*c.D+d]+=xi[d]; n2+=xi[d]*xi[d]; }
      sumsq_loc[k]+=n2; cnt_loc[k]+=1;
    }

    // Global reductions
    double *sum_g=(double*)malloc(sizeof(double)*(size_t)c.K*c.D);
    double *sumsq_g=(double*)malloc(sizeof(double)*(size_t)c.K);
    int    *cnt_g=(int*)   malloc(sizeof(int)   *(size_t)c.K);
    MPI_Allreduce(sum_loc,   sum_g,   c.K*c.D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(sumsq_loc, sumsq_g, c.K,     MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(cnt_loc,   cnt_g,   c.K,     MPI_INT,    MPI_SUM, MPI_COMM_WORLD);
    free(sum_loc); free(sumsq_loc); free(cnt_loc);

    // Update centroids (identical on all ranks)
    for(int k=0;k<c.K;++k){
      if(cnt_g[k]>0){
        for(int d=0; d<c.D; ++d) C[(size_t)k*c.D + d] = sum_g[(size_t)k*c.D + d] / (double)cnt_g[k];
      }
    }

    // Stats (rank 0 prints): SSE_k = Σ||x||^2 - n_k ||μ_k||^2
    double total=0.0;
    for(int k=0;k<c.K;++k){
      double mu2=0.0; for(int d=0; d<c.D; ++d){ double m=C[(size_t)k*c.D + d]; mu2+=m*m; }
      double sse = (cnt_g[k]>0)? (sumsq_g[k] - (double)cnt_g[k]*mu2) : 0.0;
      total += sse;
    }
    if(rank==0){
      double rel = (prev_total<INFINITY && prev_total>0.0)? (prev_total-total)/prev_total : 0.0;
      printf("%4d | %13.6e | %8.4f | ", it, total, rel);
      for(int k=0;k<c.K;++k){
        double mu2=0.0; for(int d=0; d<c.D; ++d){ double m=C[(size_t)k*c.D + d]; mu2+=m*m; }
        double sse = (cnt_g[k]>0)? (sumsq_g[k] - (double)cnt_g[k]*mu2) : 0.0;
        printf("%d: %d, %.3e  ", k, cnt_g[k], sse);
      }
      printf("\n");
    }

    free(sum_g); free(sumsq_g); free(cnt_g);
    prev_total = total;
  }

  double t1 = MPI_Wtime();
  if(rank==0){
    printf("----------------------------------------------------------------\n");
    printf("Total time: %.3f s\n", t1-t0);
  }

  free(X); free(C); free(A);
  MPI_Finalize();
  return 0;
}
