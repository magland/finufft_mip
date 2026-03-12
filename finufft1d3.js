// wasm: finufft
// finufft1d3(x, c_re, c_im, iflag, tol, s) -> [f_re, f_im]
// 1D type 3 non-uniform FFT (nonuniform → nonuniform).
register({
  check: function (argTypes, nargout) {
    return {
      outputTypes: [
        IType.tensor({ isComplex: false }),
        IType.tensor({ isComplex: false }),
      ],
    };
  },
  apply: function (args, nargout) {
    var x = args[0];
    var c_re = args[1];
    var c_im = args[2];
    var iflag = args[3];
    var tol = args[4];
    var s = args[5];

    var nj = x.data.length;
    var nk = s.data.length;
    var BYTES = 8;
    var exports = wasm.exports;
    var mem = exports.memory;

    var x_ptr = exports.my_malloc(nj * BYTES);
    var cre_ptr = exports.my_malloc(nj * BYTES);
    var cim_ptr = exports.my_malloc(nj * BYTES);
    var s_ptr = exports.my_malloc(nk * BYTES);
    var fre_ptr = exports.my_malloc(nk * BYTES);
    var fim_ptr = exports.my_malloc(nk * BYTES);

    var view = new Float64Array(mem.buffer);
    view.set(new Float64Array(x.data.buffer, x.data.byteOffset, nj), x_ptr / BYTES);
    view.set(new Float64Array(c_re.data.buffer, c_re.data.byteOffset, nj), cre_ptr / BYTES);
    view.set(new Float64Array(c_im.data.buffer, c_im.data.byteOffset, nj), cim_ptr / BYTES);
    view.set(new Float64Array(s.data.buffer, s.data.byteOffset, nk), s_ptr / BYTES);

    var ier = exports.nufft1d3(nj, x_ptr, cre_ptr, cim_ptr, iflag, tol, nk, s_ptr, fre_ptr, fim_ptr);
    if (ier !== 0) {
      exports.my_free(x_ptr); exports.my_free(cre_ptr); exports.my_free(cim_ptr);
      exports.my_free(s_ptr); exports.my_free(fre_ptr); exports.my_free(fim_ptr);
      throw new RuntimeError("finufft1d3 failed with error code " + ier);
    }

    view = new Float64Array(mem.buffer);
    var f_re_out = new FloatXArray(nk);
    var f_im_out = new FloatXArray(nk);
    for (var i = 0; i < nk; i++) {
      f_re_out[i] = view[fre_ptr / BYTES + i];
      f_im_out[i] = view[fim_ptr / BYTES + i];
    }

    exports.my_free(x_ptr); exports.my_free(cre_ptr); exports.my_free(cim_ptr);
    exports.my_free(s_ptr); exports.my_free(fre_ptr); exports.my_free(fim_ptr);

    return [RTV.tensor(f_re_out, [1, nk]), RTV.tensor(f_im_out, [1, nk])];
  },
});
