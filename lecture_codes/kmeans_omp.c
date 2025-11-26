// kmeans_omp_stats_better.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

static inline double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + 1e-9*t.tv_nsec; }

// -------- reproducible RNG (SplitMix64) --------
static inline uint64_t splitmix64(uint64_t x){
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x>>30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x>>27)) * 0x94D049BB133111EBULL;
  return x ^ (x>>31);
}
static inline double rand01_u64(uint64_t seed, uint64_t idx){
  uint64_t z = splitmix64(seed ^ idx);
  return (z >> 11) * (1.0/9007199254740992.0); // [0,1)
}
static inline void box_muller(uint64_t seed, uint64_t tag, double *z0, double *z1){
  double u1 = fmax(1e-12, rand01_u64(seed, tag*2ULL+0));
  double u2 = rand01_u64(seed, tag*2ULL+1);
  double r = sqrt(-2.0*log(u1)), th = 2.0*M_PI*u2;
  *z0 = r*cos(th); *z1 = r*sin(th);
}
// ------------------------------------------------

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

static void gen_uniform(const cfg_t* c, double* X){
  #pragma omp parallel for schedule(static)
  for(int i=0;i<c->N;++i){
    uint64_t gi=(uint64_t)i;
    for(int d=0; d<c->D; ++d){
      uint64_t key = gi*0x9E3779B97F4A7C15ULL ^ (uint64_t)d;
      X[(size_t)i*c->D + d] = rand01_u64(c->seed, key);
    }
  }
}

static void gen_blobs(const cfg_t* c, double* X){
  // Deterministic centers in [0.15,0.85]^D
  double* center = (double*)malloc(sizeof(double)*(size_t)c->K*c->D);
  for(int k=0;k<c->K;++k)
    for(int d=0; d<c->D; ++d){
      uint64_t key = (uint64_t)k*0xDB4F0B9175AE2165ULL ^ (uint64_t)d;
      center[(size_t)k*c->D + d] = 0.15 + 0.70*rand01_u64(c->seed, key);
    }
  #pragma omp parallel for schedule(static)
  for(int i=0;i<c->N;++i){
    int k = i % c->K; // round-robin cluster membership for data gen
    for(int d=0; d<c->D; d+=2){
      double z0,z1; box_muller(c->seed ^ (uint64_t)(i*1315423911u + d), (uint64_t)i, &z0,&z1);
      X[(size_t)i*c->D + d]   = center[(size_t)k*c->D + d]   + c->blob_std*z0;
      if(d+1<c->D) X[(size_t)i*c->D + d+1] = center[(size_t)k*c->D + d+1] + c->blob_std*z1;
    }
    for(int d=0; d<c->D; ++d){ // clamp to [0,1]
      double *val=&X[(size_t)i*c->D + d];
      if(*val<0.0)*val=0.0; if(*val>1.0)*val=1.0;
    }
  }
  free(center);
}

static void init_first(const cfg_t* c, const double* X, double* C){
  for(int k=0;k<c->K;++k) for(int d=0; d<c->D; ++d) C[(size_t)k*c->D + d] = X[(size_t)k*c->D + d];
}

static void init_kpp(const cfg_t* c, const double* X, double* C){
  int N=c->N,K=c->K,D=c->D;
  int first = (int)((uint64_t)(rand01_u64(c->seed, 0xABCDEF)* (double)N) % N);
  for(int d=0; d<D; ++d) C[d]=X[(size_t)first*D + d];
  double* D2=(double*)malloc(sizeof(double)*(size_t)N);
  #pragma omp parallel for schedule(static)
  for(int i=0;i<N;++i) D2[i]=sqdist(&X[(size_t)i*D], &C[0], D);
  for(int m=1;m<K;++m){
    double sum=0.0; for(int i=0;i<N;++i) sum+=D2[i];
    double r = rand01_u64(c->seed, 0xABC000 + (uint64_t)m) * sum;
    int idx=0; double acc=0.0; for(; idx<N-1; ++idx){ acc+=D2[idx]; if(acc>=r) break; }
    for(int d=0; d<D; ++d) C[(size_t)m*D + d] = X[(size_t)idx*D + d];
    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;++i){
      double d2 = sqdist(&X[(size_t)i*D], &C[(size_t)m*D], D);
      if(d2 < D2[i]) D2[i]=d2;
    }
  }
  free(D2);
}

int main(int argc, char** argv){
  cfg_t c; parse_args(argc, argv, &c);
  double *X=(double*)malloc(sizeof(double)*(size_t)c.N*c.D);
  double *C=(double*)malloc(sizeof(double)*(size_t)c.K*c.D);
  int    *A=(int*)   malloc(sizeof(int)   *(size_t)c.N);
  if(!X||!C||!A){ fprintf(stderr,"alloc failed\n"); return 1; }

  if(c.data_mode==DATA_UNIFORM) gen_uniform(&c, X); else gen_blobs(&c, X);
  if(c.init_mode==INIT_KPP) init_kpp(&c, X, C); else init_first(&c, X, C);

  printf("Config: N=%d K=%d D=%d iters=%d data=%s init=%s blob_std=%.3f\n",
         c.N,c.K,c.D,c.iters,
         c.data_mode==DATA_BLOBS?"blobs":"uniform",
         c.init_mode==INIT_KPP?"kpp":"first", c.blob_std);
  printf("\nIter |   total_SSE    | rel_impr | (per-cluster: k: n, SSE)\n");
  printf("----------------------------------------------------------------\n");

  double t0=now_s(), prev_total=INFINITY;

  for(int it=0; it<c.iters; ++it){
    // Assign
    #pragma omp parallel for schedule(static)
    for(int i=0;i<c.N;++i){
      const double* xi=&X[(size_t)i*c.D];
      double best=INFINITY; int bestk=0;
      for(int k=0;k<c.K;++k){
        double d2=sqdist(xi, &C[(size_t)k*c.D], c.D);
        if(d2<best){ best=d2; bestk=k; }
      }
      A[i]=bestk;
    }

    // Thread-local reductions: sums[D], sumsq, counts
    int T=omp_get_max_threads();
    double *sum   =(double*)calloc((size_t)T*c.K*c.D, sizeof(double));
    double *sumsq =(double*)calloc((size_t)T*c.K,     sizeof(double));
    int    *cnt   =(int*)   calloc((size_t)T*c.K,     sizeof(int));

    #pragma omp parallel
    {
      int tid=omp_get_thread_num();
      double* sum_t=&sum[(size_t)tid*c.K*c.D];
      double* sumsq_t=&sumsq[(size_t)tid*c.K];
      int*    cnt_t=&cnt[(size_t)tid*c.K];

      #pragma omp for schedule(static)
      for(int i=0;i<c.N;++i){
        int k=A[i]; const double* xi=&X[(size_t)i*c.D];
        double n2=0.0; for(int d=0; d<c.D; ++d){ sum_t[(size_t)k*c.D+d]+=xi[d]; n2+=xi[d]*xi[d]; }
        sumsq_t[k]+=n2; cnt_t[k]+=1;
      }
    }

    // Combine to global and update centroids
    double *sum_g=(double*)calloc((size_t)c.K*c.D,sizeof(double));
    double *sumsq_g=(double*)calloc((size_t)c.K,  sizeof(double));
    int    *cnt_g=(int*)   calloc((size_t)c.K,    sizeof(int));
    for(int k=0;k<c.K;++k){
      for(int t=0;t<T;++t){
        cnt_g[k]   += cnt[(size_t)t*c.K + k];
        sumsq_g[k] += sumsq[(size_t)t*c.K + k];
        for(int d=0; d<c.D; ++d) sum_g[(size_t)k*c.D + d] += sum[(size_t)t*c.K*c.D + (size_t)k*c.D + d];
      }
      if(cnt_g[k]>0){
        for(int d=0; d<c.D; ++d) C[(size_t)k*c.D + d] = sum_g[(size_t)k*c.D + d] / (double)cnt_g[k];
      }
    }

    // Stats: SSE_k = Σ||x||^2 - n_k * ||μ_k||^2
    double total=0.0;
    for(int k=0;k<c.K;++k){
      double mu2=0.0; for(int d=0; d<c.D; ++d){ double m=C[(size_t)k*c.D + d]; mu2+=m*m; }
      double sse = (cnt_g[k]>0)? (sumsq_g[k] - (double)cnt_g[k]*mu2) : 0.0;
      total += sse;
    }
    double rel = (prev_total<INFINITY && prev_total>0.0)? (prev_total-total)/prev_total : 0.0;

    printf("%4d | %13.6e | %8.4f | ", it, total, rel);
    for(int k=0;k<c.K;++k){
      double mu2=0.0; for(int d=0; d<c.D; ++d){ double m=C[(size_t)k*c.D + d]; mu2+=m*m; }
      double sse = (cnt_g[k]>0)? (sumsq_g[k] - (double)cnt_g[k]*mu2) : 0.0;
      printf("%d: %d, %.3e  ", k, cnt_g[k], sse);
    }
    printf("\n");

    free(sum); free(sumsq); free(cnt);
    free(sum_g); free(sumsq_g); free(cnt_g);
    prev_total=total;
  }

  double t1=now_s();
  printf("----------------------------------------------------------------\nTotal time: %.3f s\n", t1-t0);
  free(X); free(C); free(A);
  return 0;
}
