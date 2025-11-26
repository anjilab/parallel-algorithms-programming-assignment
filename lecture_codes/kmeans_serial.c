// kmeans_serial_stats_better.c
// Serial K-Means that prints per-iteration terminal stats.
// Adds: k-means++ init and Gaussian "blobs" data to make improvement visible.
// Reproducible via SplitMix64-based RNG.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

static inline double now_s(void){
  struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
  return t.tv_sec + 1e-9*t.tv_nsec;
}

// ------------------ Reproducible RNG ------------------
static inline uint64_t splitmix64(uint64_t x){
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x>>30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x>>27)) * 0x94D049BB133111EBULL;
  return x ^ (x>>31);
}
static inline double rand01_u64(uint64_t seed, uint64_t idx){
  // top 53 bits to [0,1)
  uint64_t z = splitmix64(seed ^ idx);
  return (z >> 11) * (1.0/9007199254740992.0);
}
// Box–Muller from two deterministic uniforms
static inline void box_muller(uint64_t seed, uint64_t i, double *z0, double *z1){
  double u1 = fmax(1e-12, rand01_u64(seed, i*2ULL+0)); // avoid log(0)
  double u2 = rand01_u64(seed, i*2ULL+1);
  double r = sqrt(-2.0*log(u1));
  double th = 2.0*M_PI*u2;
  *z0 = r*cos(th);
  *z1 = r*sin(th);
}
// ------------------------------------------------------

static inline double sqdist(const double* a,const double* b,int D){
  double s=0.0; for(int d=0; d<D; ++d){ double t=a[d]-b[d]; s+=t*t; } return s;
}

typedef enum { DATA_UNIFORM, DATA_BLOBS } data_mode_t;
typedef enum { INIT_FIRST, INIT_KPP } init_mode_t;

typedef struct {
  int N, K, D, iters;
  data_mode_t data_mode;
  init_mode_t init_mode;
  double blob_std;
  uint64_t seed;
} cfg_t;

static void parse_args(int argc, char** argv, cfg_t* c){
  c->N = (argc>1)? atoi(argv[1]): 20000;
  c->K = (argc>2)? atoi(argv[2]): 4;
  c->D = (argc>3)? atoi(argv[3]): 2;
  c->iters = (argc>4)? atoi(argv[4]): 15;
  c->data_mode = DATA_UNIFORM;
  c->init_mode = INIT_FIRST;
  c->blob_std = 0.08;
  c->seed = 12345ULL;

  for (int a=5; a<argc; ++a){
    if (!strncmp(argv[a],"--data=uniform",14)) c->data_mode = DATA_UNIFORM;
    else if (!strncmp(argv[a],"--data=blobs",12)) c->data_mode = DATA_BLOBS;
    else if (!strncmp(argv[a],"--init=kpp",10)) c->init_mode = INIT_KPP;
    else if (!strncmp(argv[a],"--init=first",12)) c->init_mode = INIT_FIRST;
    else if (!strncmp(argv[a],"--blob-std=",11)) c->blob_std = atof(argv[a]+11);
    else if (!strncmp(argv[a],"--seed=",7)) c->seed = (uint64_t)strtoull(argv[a]+7,NULL,10);
  }
}

// Generate data
static void gen_uniform(const cfg_t* c, double* X){
  for (int i=0;i<c->N;++i){
    uint64_t gi=(uint64_t)i;
    for (int d=0; d<c->D; ++d){
      uint64_t key = gi*0x9E3779B97F4A7C15ULL ^ (uint64_t)d;
      X[(size_t)i*c->D + d] = rand01_u64(c->seed, key);
    }
  }
}

// Deterministic Gaussian blobs around K centers in [0.15,0.85]^D
static void gen_blobs(const cfg_t* c, double* X){
  // choose K centers reproducibly
  double* center = (double*)malloc(sizeof(double)*(size_t)c->K*c->D);
  for (int k=0;k<c->K;++k){
    for (int d=0; d<c->D; ++d){
      // spread centers away from borders
      uint64_t key = (uint64_t)k*0xDB4F0B9175AE2165ULL ^ (uint64_t)d;
      center[(size_t)k*c->D + d] = 0.15 + 0.70*rand01_u64(c->seed, key);
    }
  }
  // assign points to clusters round-robin (reproducible) and sample Gaussian
  for (int i=0;i<c->N;++i){
    int k = i % c->K;
    for (int d=0; d<c->D; d+=2){
      double z0, z1; box_muller(c->seed ^ (uint64_t)(i*1315423911u + d), (uint64_t)i, &z0, &z1);
      X[(size_t)i*c->D + d]   = center[(size_t)k*c->D + d]   + c->blob_std*z0;
      if (d+1 < c->D)
        X[(size_t)i*c->D + d+1] = center[(size_t)k*c->D + d+1] + c->blob_std*z1;
    }
    // clamp to [0,1]
    for (int d=0; d<c->D; ++d){
      if (X[(size_t)i*c->D + d] < 0.0) X[(size_t)i*c->D + d] = 0.0;
      if (X[(size_t)i*c->D + d] > 1.0) X[(size_t)i*c->D + d] = 1.0;
    }
  }
  free(center);
}

// Init centroids
static void init_first(const cfg_t* c, const double* X, double* C){
  for(int k=0;k<c->K;++k)
    for(int d=0; d<c->D; ++d)
      C[(size_t)k*c->D + d] = X[(size_t)k*c->D + d];
}

// k-means++ (deterministic)
static void init_kpp(const cfg_t* c, const double* X, double* C){
  int K=c->K, D=c->D, N=c->N;
  // choose first centroid by a reproducible pseudo-random index
  int first = (int)( (uint64_t)(rand01_u64(c->seed, 0xABCDEF) * (double)N) % N );
  for(int d=0; d<D; ++d) C[d] = X[(size_t)first*D + d];

  double* D2 = (double*)malloc(sizeof(double)*(size_t)N);
  for (int i=0;i<N;++i) D2[i] = sqdist(&X[(size_t)i*D], &C[0], D);

  for (int m=1; m<K; ++m){
    // pick new centroid with prob ∝ D2
    double sum=0.0; for(int i=0;i<N;++i) sum += D2[i];
    // deterministic "random" draw using seed and m
    double r = rand01_u64(c->seed, 0xABC000 + (uint64_t)m) * sum;
    int idx=0; double acc=0.0;
    for (; idx<N-1; ++idx){ acc += D2[idx]; if (acc >= r) break; }
    for(int d=0; d<D; ++d) C[(size_t)m*D + d] = X[(size_t)idx*D + d];

    // update D2
    for (int i=0;i<N;++i){
      double d2 = sqdist(&X[(size_t)i*D], &C[(size_t)m*D], D);
      if (d2 < D2[i]) D2[i] = d2;
    }
  }
  free(D2);
}

int main(int argc, char** argv){
  cfg_t c; parse_args(argc, argv, &c);

  double *X = (double*)malloc(sizeof(double)*(size_t)c.N*c.D);
  double *C = (double*)malloc(sizeof(double)*(size_t)c.K*c.D);
  int    *A = (int*)   malloc(sizeof(int)   *(size_t)c.N);
  if(!X||!C||!A){ fprintf(stderr,"alloc failed\n"); return 1; }

  if (c.data_mode==DATA_UNIFORM) gen_uniform(&c, X);
  else                           gen_blobs(&c, X);

  if (c.init_mode==INIT_KPP) init_kpp(&c, X, C);
  else                       init_first(&c, X, C);

  printf("Config: N=%d K=%d D=%d iters=%d data=%s init=%s blob_std=%.3f\n",
         c.N,c.K,c.D,c.iters,
         c.data_mode==DATA_BLOBS?"blobs":"uniform",
         c.init_mode==INIT_KPP?"kpp":"first",
         c.blob_std);

  printf("\nIter |   total_SSE    | rel_impr | (per-cluster: k: n, SSE)\n");
  printf("----------------------------------------------------------------\n");

  double t0 = now_s();
  double prev_total_sse = INFINITY;

  for (int it=0; it<c.iters; ++it){
    // Assign
    for (int i=0;i<c.N;++i){
      const double* xi = &X[(size_t)i*c.D];
      double best = INFINITY; int bestk = 0;
      for (int k=0;k<c.K;++k){
        double d2 = sqdist(xi, &C[(size_t)k*c.D], c.D);
        if (d2 < best){ best = d2; bestk = k; }
      }
      A[i] = bestk;
    }

    // Partial sums for update + stats (sums, sum||x||^2, counts)
    double *sum   = (double*)calloc((size_t)c.K*c.D, sizeof(double));
    double *sumsq = (double*)calloc((size_t)c.K,     sizeof(double));
    int    *cnt   = (int*)   calloc((size_t)c.K,     sizeof(int));
    for (int i=0;i<c.N;++i){
      int k = A[i];
      const double* xi = &X[(size_t)i*c.D];
      double n2 = 0.0;
      for (int d=0; d<c.D; ++d){ sum[(size_t)k*c.D + d] += xi[d]; n2 += xi[d]*xi[d]; }
      sumsq[k] += n2; cnt[k] += 1;
    }

    // Update centroids
    for (int k=0;k<c.K;++k){
      if (cnt[k]>0){
        for (int d=0; d<c.D; ++d)
          C[(size_t)k*c.D + d] = sum[(size_t)k*c.D + d] / (double)cnt[k];
      }
    }

    // Compute per-cluster SSE from sufficient statistics:
    // SSE_k = Σ||x||^2 - n_k * ||μ_k||^2
    double total_sse = 0.0;
    printf("%4d | ", it);
    for (int k=0;k<c.K;++k){
      double sse_k = 0.0;
      if (cnt[k] > 0){
        double mu2 = 0.0;
        for (int d=0; d<c.D; ++d){ double m = C[(size_t)k*c.D + d]; mu2 += m*m; }
        sse_k = sumsq[k] - (double)cnt[k]*mu2;
      }
      total_sse += sse_k;
    }
    double rel = (prev_total_sse<INFINITY && prev_total_sse>0.0)
                 ? (prev_total_sse - total_sse)/prev_total_sse : 0.0;
    printf("%13.6e | %8.4f | ", total_sse, rel);

    for (int k=0;k<c.K;++k){
      double sse_k = 0.0;
      if (cnt[k] > 0){
        double mu2 = 0.0;
        for (int d=0; d<c.D; ++d){ double m = C[(size_t)k*c.D + d]; mu2 += m*m; }
        sse_k = sumsq[k] - (double)cnt[k]*mu2;
      }
      printf("%d: %d, %.3e  ", k, cnt[k], sse_k);
    }
    printf("\n");

    free(sum); free(sumsq); free(cnt);
    prev_total_sse = total_sse;
  }

  double t1 = now_s();
  printf("----------------------------------------------------------------\n");
  printf("Total time: %.3f s\n", t1-t0);
  free(X); free(C); free(A);
  return 0;
}
