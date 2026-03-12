// C++ wrapper around FINUFFT for WebAssembly.
// Accepts separate real/imaginary arrays for the WASM interface.
// Exports all 9 simple interface functions (1d1..3d3).

#include <complex>
#include <cstdlib>
#include <finufft.h>

using CPX = std::complex<double>;

static finufft_opts default_opts() {
  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.nthreads = 1;
  opts.debug = 0;
  return opts;
}

static void interleave(int n, const double *re, const double *im, CPX *out) {
  for (int j = 0; j < n; j++)
    out[j] = CPX(re[j], im[j]);
}

static void deinterleave(int n, const CPX *in, double *re, double *im) {
  for (int j = 0; j < n; j++) {
    re[j] = in[j].real();
    im[j] = in[j].imag();
  }
}

extern "C" {

__attribute__((export_name("my_malloc")))
void *my_malloc(int size) { return std::malloc(size); }

__attribute__((export_name("my_free")))
void my_free(void *ptr) { std::free(ptr); }

// ── Type 1: nonuniform → uniform ────────────────────────────────────────────

__attribute__((export_name("nufft1d1")))
int nufft1d1_w(int nj, double *x,
               double *cj_re, double *cj_im,
               int iflag, double eps, int ms,
               double *fk_re, double *fk_im) {
  auto *cj = new CPX[nj];
  interleave(nj, cj_re, cj_im, cj);
  auto *fk = new CPX[ms];
  auto opts = default_opts();
  int ier = finufft1d1((int64_t)nj, x, cj, iflag, eps, (int64_t)ms, fk, &opts);
  deinterleave(ms, fk, fk_re, fk_im);
  delete[] cj; delete[] fk;
  return ier;
}

__attribute__((export_name("nufft2d1")))
int nufft2d1_w(int nj, double *x, double *y,
               double *cj_re, double *cj_im,
               int iflag, double eps, int ms, int mt,
               double *fk_re, double *fk_im) {
  auto *cj = new CPX[nj];
  interleave(nj, cj_re, cj_im, cj);
  int nout = ms * mt;
  auto *fk = new CPX[nout];
  auto opts = default_opts();
  int ier = finufft2d1((int64_t)nj, x, y, cj, iflag, eps,
                       (int64_t)ms, (int64_t)mt, fk, &opts);
  deinterleave(nout, fk, fk_re, fk_im);
  delete[] cj; delete[] fk;
  return ier;
}

__attribute__((export_name("nufft3d1")))
int nufft3d1_w(int nj, double *x, double *y, double *z,
               double *cj_re, double *cj_im,
               int iflag, double eps, int ms, int mt, int mu,
               double *fk_re, double *fk_im) {
  auto *cj = new CPX[nj];
  interleave(nj, cj_re, cj_im, cj);
  int nout = ms * mt * mu;
  auto *fk = new CPX[nout];
  auto opts = default_opts();
  int ier = finufft3d1((int64_t)nj, x, y, z, cj, iflag, eps,
                       (int64_t)ms, (int64_t)mt, (int64_t)mu, fk, &opts);
  deinterleave(nout, fk, fk_re, fk_im);
  delete[] cj; delete[] fk;
  return ier;
}

// ── Type 2: uniform → nonuniform ────────────────────────────────────────────

__attribute__((export_name("nufft1d2")))
int nufft1d2_w(int nj, double *x,
               double *cj_re, double *cj_im,
               int iflag, double eps, int ms,
               double *fk_re, double *fk_im) {
  auto *fk = new CPX[ms];
  interleave(ms, fk_re, fk_im, fk);
  auto *cj = new CPX[nj];
  auto opts = default_opts();
  int ier = finufft1d2((int64_t)nj, x, cj, iflag, eps, (int64_t)ms, fk, &opts);
  deinterleave(nj, cj, cj_re, cj_im);
  delete[] cj; delete[] fk;
  return ier;
}

__attribute__((export_name("nufft2d2")))
int nufft2d2_w(int nj, double *x, double *y,
               double *cj_re, double *cj_im,
               int iflag, double eps, int ms, int mt,
               double *fk_re, double *fk_im) {
  int nin = ms * mt;
  auto *fk = new CPX[nin];
  interleave(nin, fk_re, fk_im, fk);
  auto *cj = new CPX[nj];
  auto opts = default_opts();
  int ier = finufft2d2((int64_t)nj, x, y, cj, iflag, eps,
                       (int64_t)ms, (int64_t)mt, fk, &opts);
  deinterleave(nj, cj, cj_re, cj_im);
  delete[] cj; delete[] fk;
  return ier;
}

__attribute__((export_name("nufft3d2")))
int nufft3d2_w(int nj, double *x, double *y, double *z,
               double *cj_re, double *cj_im,
               int iflag, double eps, int ms, int mt, int mu,
               double *fk_re, double *fk_im) {
  int nin = ms * mt * mu;
  auto *fk = new CPX[nin];
  interleave(nin, fk_re, fk_im, fk);
  auto *cj = new CPX[nj];
  auto opts = default_opts();
  int ier = finufft3d2((int64_t)nj, x, y, z, cj, iflag, eps,
                       (int64_t)ms, (int64_t)mt, (int64_t)mu, fk, &opts);
  deinterleave(nj, cj, cj_re, cj_im);
  delete[] cj; delete[] fk;
  return ier;
}

// ── Type 3: nonuniform → nonuniform ─────────────────────────────────────────

__attribute__((export_name("nufft1d3")))
int nufft1d3_w(int nj, double *x,
               double *cj_re, double *cj_im,
               int iflag, double eps,
               int nk, double *s,
               double *fk_re, double *fk_im) {
  auto *cj = new CPX[nj];
  interleave(nj, cj_re, cj_im, cj);
  auto *fk = new CPX[nk];
  auto opts = default_opts();
  int ier = finufft1d3((int64_t)nj, x, cj, iflag, eps,
                       (int64_t)nk, s, fk, &opts);
  deinterleave(nk, fk, fk_re, fk_im);
  delete[] cj; delete[] fk;
  return ier;
}

__attribute__((export_name("nufft2d3")))
int nufft2d3_w(int nj, double *x, double *y,
               double *cj_re, double *cj_im,
               int iflag, double eps,
               int nk, double *s, double *t,
               double *fk_re, double *fk_im) {
  auto *cj = new CPX[nj];
  interleave(nj, cj_re, cj_im, cj);
  auto *fk = new CPX[nk];
  auto opts = default_opts();
  int ier = finufft2d3((int64_t)nj, x, y, cj, iflag, eps,
                       (int64_t)nk, s, t, fk, &opts);
  deinterleave(nk, fk, fk_re, fk_im);
  delete[] cj; delete[] fk;
  return ier;
}

__attribute__((export_name("nufft3d3")))
int nufft3d3_w(int nj, double *x, double *y, double *z,
               double *cj_re, double *cj_im,
               int iflag, double eps,
               int nk, double *s, double *t, double *u,
               double *fk_re, double *fk_im) {
  auto *cj = new CPX[nj];
  interleave(nj, cj_re, cj_im, cj);
  auto *fk = new CPX[nk];
  auto opts = default_opts();
  int ier = finufft3d3((int64_t)nj, x, y, z, cj, iflag, eps,
                       (int64_t)nk, s, t, u, fk, &opts);
  deinterleave(nk, fk, fk_re, fk_im);
  delete[] cj; delete[] fk;
  return ier;
}

} // extern "C"
