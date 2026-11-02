#define STANDALONE 1
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "earth.h"

#define NCASES  500
#define NPREDS  10
#define NK      21
#define NR      5
#define NTHREADS 4

static void* run_earth(void *arg) {
    int id = *(int*)arg;
    int nCases = NCASES, nPreds = NPREDS, nMaxTerms = NK;

    double *x = calloc(nCases * nPreds, sizeof(double));
    double *y = calloc(nCases,          sizeof(double));
    for (int i = 0; i < nCases; i++) {
        for (int j = 0; j < nPreds; j++)
            x[i + j*nCases] = (double)(i*nPreds + j) / (nCases*nPreds) + 0.01;
        y[i] = (double)i / nCases;
    }
    double bestGcv = 0;
    int    nTermsOut = 0;
    int   *bestSet    = calloc(nMaxTerms,          sizeof(int));
    double *bx        = calloc(nCases * nMaxTerms,  sizeof(double));
    int   *dirs       = calloc(nMaxTerms * nPreds,  sizeof(int));
    double *cuts      = calloc(nMaxTerms * nPreds,  sizeof(double));
    double *residuals = calloc(nCases,              sizeof(double));
    double *betas     = calloc(nMaxTerms,           sizeof(double));
    int   *linPreds   = calloc(nPreds,              sizeof(int));

    for (int iter = 0; iter < NR; iter++) {
        memset(bestSet, 0, nMaxTerms * sizeof(int));
        memset(bx,      0, nCases * nMaxTerms * sizeof(double));
        memset(dirs,    0, nMaxTerms * nPreds  * sizeof(int));
        memset(cuts,    0, nMaxTerms * nPreds  * sizeof(double));
        memset(residuals,0,nCases * sizeof(double));
        memset(betas,   0, nMaxTerms * sizeof(double));
        Earth(&bestGcv, &nTermsOut,
              (bool*)bestSet, bx, dirs, cuts, residuals, betas,
              x, y, NULL,
              nCases, 1, nPreds,
              4, nMaxTerms,
              2.0, 0.001, 0,
              0,   /* Prune=false */
              20, 0.0, 0.0,
              linPreds, 1, 0.0, NULL);
    }
    printf("thread %d done nTerms=%d\n", id, nTermsOut); fflush(stdout);
    free(x); free(y); free(bestSet); free(bx);
    free(dirs); free(cuts); free(residuals); free(betas); free(linPreds);
    return NULL;
}

int main(void) {
    pthread_t t[NTHREADS]; int ids[NTHREADS];
    for (int i = 0; i < NTHREADS; i++) { ids[i]=i; pthread_create(&t[i],NULL,run_earth,&ids[i]); }
    for (int i = 0; i < NTHREADS; i++) pthread_join(t[i], NULL);
    printf("all done\n"); return 0;
}
