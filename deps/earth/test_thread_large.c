#define STANDALONE 1
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "earth.h"

// Use production nPreds/NK to get same buffer sizes as Julia (BetaCache=107MB etc.)
// Small nCases keeps each iteration fast.
#define NCASES   500
#define NPREDS   37
#define NK       601
#define NR       20
#define NTHREADS 8

static int g_earth_threads = 1;

static void* run_earth(void *arg) {
    int id = *(int*)arg;
    size_t nCases = NCASES;
    int nPreds = NPREDS, nMaxTerms = NK;

    // Use pseudorandom data so Earth runs deep into the forward pass.
    // Each thread seeds differently so data isn't identical.
    unsigned seed = (unsigned)(id * 1234567 + 42);
    double *x = calloc(nCases * nPreds, sizeof(double));
    double *y = calloc(nCases,          sizeof(double));
    for (size_t i = 0; i < nCases; i++) {
        for (int j = 0; j < nPreds; j++) {
            seed = seed * 1664525u + 1013904223u;
            x[i + j*nCases] = (seed >> 8) / (double)(1u << 24);
        }
        seed = seed * 1664525u + 1013904223u;
        y[i] = (seed >> 8) / (double)(1u << 24);
    }
    double bestGcv  = 0;
    int    nTermsOut = 0, iTermCond = 0;
    bool  *bestSet    = calloc(nMaxTerms,            sizeof(bool));
    double *bx        = calloc(nCases * nMaxTerms,   sizeof(double));
    int   *dirs       = calloc(nMaxTerms * nPreds,   sizeof(int));
    double *cuts      = calloc(nMaxTerms * nPreds,   sizeof(double));
    double *residuals = calloc(nCases,               sizeof(double));
    double *betas     = calloc(nMaxTerms,            sizeof(double));
    int   *linPreds   = calloc(nPreds,               sizeof(int));

    for (int iter = 0; iter < NR; iter++) {
        memset(bestSet,   0, nMaxTerms * sizeof(bool));
        memset(bx,        0, nCases * nMaxTerms * sizeof(double));
        memset(dirs,      0, nMaxTerms * nPreds  * sizeof(int));
        memset(cuts,      0, nMaxTerms * nPreds  * sizeof(double));
        memset(residuals, 0, nCases * sizeof(double));
        memset(betas,     0, nMaxTerms * sizeof(double));
        Earth(&bestGcv, &nTermsOut, &iTermCond,
              bestSet, bx, dirs, cuts, residuals, betas,
              x, y, NULL,
              nCases, 1, nPreds,
              4, nMaxTerms,
              -1.0,  /* Penalty=-1: terms free, forces full forward pass */
              0.0,   /* Thresh=0: never reject on RSqDelta */
              0, 0,  /* nMinSpan=0, nEndSpan=0 (auto) */
              false, /* Prune */
              20, 0.0, 0.0,
              linPreds,
              2.0, true,   /* AdjustEndSpan, AutoLinPreds */
              true,        /* UseBetaCache */
              g_earth_threads,
              0.0,         /* Trace */
              NULL);
        printf("thread %d iter %d done nTerms=%d\n", id, iter, nTermsOut);
        fflush(stdout);
    }
    free(x); free(y); free(bestSet); free(bx);
    free(dirs); free(cuts); free(residuals); free(betas); free(linPreds);
    return NULL;
}

int main(void) {
    const char *env = getenv("EARTH_THREADS");
    if (env) g_earth_threads = atoi(env);
    if (g_earth_threads < 1) g_earth_threads = 1;
    printf("NTHREADS=%d EARTH_THREADS=%d NR=%d\n", NTHREADS, g_earth_threads, NR);
    pthread_t t[NTHREADS]; int ids[NTHREADS];
    for (int i = 0; i < NTHREADS; i++) { ids[i]=i; pthread_create(&t[i],NULL,run_earth,&ids[i]); }
    for (int i = 0; i < NTHREADS; i++) pthread_join(t[i], NULL);
    printf("all done\n"); return 0;
}
