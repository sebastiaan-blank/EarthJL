CC      = gcc
CFLAGS  = -DSTANDALONE -O3 -std=gnu99 -Wall -fPIC -pthread
# daxpy/dcopy/ddot/dnrm2/dscal are BLAS1 — provided by OpenBLAS at runtime
# dqrls/dqrsl/dtrsl/d_sign are LINPACK/Fortran helpers — not in OpenBLAS, bundled here
LINPACK = blas/dqrls.c blas/dqrsl.c blas/dtrsl.c blas/d_sign.c
RSRC    = R/dqrdc2.c

libearth.so: earth.c $(LINPACK) $(RSRC)
	$(CC) $(CFLAGS) -shared -o $@ $^ -l:libopenblas.so.0

clean:
	rm -f libearth.so
