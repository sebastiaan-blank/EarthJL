CC      = gcc
CFLAGS  = -DSTANDALONE -O3 -std=gnu99 -Wall -fPIC -pthread
# daxpy/dcopy/ddot/dnrm2/dscal are BLAS1 — provided by OpenBLAS at runtime
# dqrls/dqrsl/dtrsl/d_sign are LINPACK/Fortran helpers — not in OpenBLAS, bundled here
LINPACK = blas/dqrls.c blas/dqrsl.c blas/dtrsl.c blas/d_sign.c
RSRC    = R/dqrdc2.c

libearth.so: earth.c $(LINPACK) $(RSRC)
	$(CC) $(CFLAGS) -shared -o $@ $^ -l:libopenblas.so.0

# Canary build: wraps every malloc1/calloc1/free1 with overflow guards.
# Aborts immediately with the buffer name when any allocation is overrun.
libearth_canary.so: earth.c $(LINPACK) $(RSRC)
	$(CC) $(CFLAGS) -DCANARY_CHECK -shared -o $@ $^ -l:libopenblas.so.0

clean:
	rm -f libearth.so libearth_canary.so
