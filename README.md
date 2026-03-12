# finufft_mip

[FINUFFT](https://finufft.readthedocs.io) nonuniform FFT compiled to WebAssembly for use as a [numbl](https://github.com/magland/numbl) mip package.

FINUFFT v2.5.0 is included as a git submodule.

## Functions

- `finufft1d1(x, c_re, c_im, iflag, tol, ms)` — 1D type 1 NUFFT. Returns `[fk_re, fk_im]`.

## Building the WASM module

Prerequisites: [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html) (`emcc`, `emcmake`, `emmake` on PATH).

```bash
git clone --recurse-submodules https://github.com/magland/finufft_mip.git
cd finufft_mip
bash build_wasm.sh
```

This produces `finufft.wasm` (~1.9 MB) in the repo root.

## Usage in numbl

Install as a mip package, then:

```matlab
mip load finufft

nj = 100;
x = linspace(-pi, pi - 2*pi/nj, nj);
c_re = randn(1, nj);
c_im = randn(1, nj);
ms = 200;

[fk_re, fk_im] = finufft1d1(x, c_re, c_im, 1, 1e-9, ms);
```

## Running the test

```bash
numbl run test_finufft1d1.m
```
