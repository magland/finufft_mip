// C++ wrapper around finufft1d1 for WebAssembly.
// Accepts separate real/imaginary arrays for the WASM interface.

#include <complex>
#include <cstdlib>
#include <finufft.h>

extern "C" {

__attribute__((export_name("my_malloc")))
void *my_malloc(int size) {
  return std::malloc(size);
}

__attribute__((export_name("my_free")))
void my_free(void *ptr) {
  std::free(ptr);
}

// 1D type 1 NUFFT wrapper.
// x[nj]       : nonuniform point locations in [-pi, pi)
// cj_re[nj], cj_im[nj] : real/imaginary parts of input strengths
// iflag       : sign of exponent (+1 or -1)
// eps         : tolerance
// ms          : number of output Fourier modes
// fk_re[ms], fk_im[ms] : output (pre-allocated by caller via my_malloc)
// Returns 0 on success.
__attribute__((export_name("nufft1d1")))
int nufft1d1_wrapper(int nj, double *x,
                     double *cj_re, double *cj_im,
                     int iflag, double eps, int ms,
                     double *fk_re, double *fk_im) {
  auto *cj = new std::complex<double>[nj];
  for (int j = 0; j < nj; j++)
    cj[j] = std::complex<double>(cj_re[j], cj_im[j]);

  auto *fk = new std::complex<double>[ms];

  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.nthreads = 1;
  opts.debug = 0;

  int ier = finufft1d1(static_cast<int64_t>(nj), x, cj, iflag, eps,
                       static_cast<int64_t>(ms), fk, &opts);

  for (int k = 0; k < ms; k++) {
    fk_re[k] = fk[k].real();
    fk_im[k] = fk[k].imag();
  }

  delete[] cj;
  delete[] fk;
  return ier;
}

} // extern "C"
