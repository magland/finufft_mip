% Test 1D type 1 NUFFT via FINUFFT WebAssembly
% Uses a small problem and verifies against brute-force DFT.

nj = 10;            % number of nonuniform points
ms = 15;            % number of output Fourier modes
tol = 1e-9;
iflag = 1;

% Deterministic nonuniform points in [-pi, pi)
x = linspace(-pi, pi - 2*pi/nj, nj);

% Deterministic complex strengths
c_re = sin((1:nj) * 0.7);
c_im = cos((1:nj) * 1.3);

% Call FINUFFT via WASM
[fk_re, fk_im] = finufft1d1(x, c_re, c_im, iflag, tol, ms);

% Brute-force DFT for verification
% For type 1 with iflag=+1: F[k] = sum_j c[j] * exp(+i * k * x[j])
% Output modes: k = -floor(ms/2) : floor((ms-1)/2)
k_min = -floor(ms / 2);
for idx = 1:ms
    k = k_min + (idx - 1);
    Fk_re = 0;
    Fk_im = 0;
    for j = 1:nj
        phase = k * x(j);
        cp = cos(phase);
        sp = sin(phase);
        % (c_re + i*c_im) * (cos + i*sin)
        Fk_re = Fk_re + c_re(j) * cp - c_im(j) * sp;
        Fk_im = Fk_im + c_re(j) * sp + c_im(j) * cp;
    end
    err_re = abs(fk_re(idx) - Fk_re);
    err_im = abs(fk_im(idx) - Fk_im);
    scale = max(abs(Fk_re) + abs(Fk_im), 1e-15);
    assert((err_re + err_im) / scale < 1e-6, ...
        sprintf('Mode k=%d: error too large (re=%.3g, im=%.3g)', k, err_re, err_im));
end

disp('SUCCESS')
