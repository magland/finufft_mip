// wasm: finufft
// finufft2d3(x, y, c, iflag, tol, s, t) -> f
// 2D type 3 non-uniform FFT (nonuniform -> nonuniform).
// c is complex, f is complex.
register({
  check: function (argTypes, nargout) {
    return {
      outputTypes: [IType.tensor({ isComplex: true })],
    };
  },
  apply: function (args, nargout) {
    var x = args[0];
    var y = args[1];
    var c = args[2];
    var iflag = args[3];
    var tol = args[4];
    var s = args[5];
    var t = args[6];

    var nj = x.data.length;
    var nk = s.data.length;
    var BYTES = 8;
    var exports = wasm.exports;
    var mem = exports.memory;

    var x_ptr = exports.my_malloc(nj * BYTES);
    var y_ptr = exports.my_malloc(nj * BYTES);
    var cre_ptr = exports.my_malloc(nj * BYTES);
    var cim_ptr = exports.my_malloc(nj * BYTES);
    var s_ptr = exports.my_malloc(nk * BYTES);
    var t_ptr = exports.my_malloc(nk * BYTES);
    var fre_ptr = exports.my_malloc(nk * BYTES);
    var fim_ptr = exports.my_malloc(nk * BYTES);

    var view = new Float64Array(mem.buffer);
    view.set(new Float64Array(x.data.buffer, x.data.byteOffset, nj), x_ptr / BYTES);
    view.set(new Float64Array(y.data.buffer, y.data.byteOffset, nj), y_ptr / BYTES);
    view.set(new Float64Array(c.data.buffer, c.data.byteOffset, nj), cre_ptr / BYTES);
    if (c.imag) {
      view.set(new Float64Array(c.imag.buffer, c.imag.byteOffset, nj), cim_ptr / BYTES);
    } else {
      view.fill(0, cim_ptr / BYTES, cim_ptr / BYTES + nj);
    }
    view.set(new Float64Array(s.data.buffer, s.data.byteOffset, nk), s_ptr / BYTES);
    view.set(new Float64Array(t.data.buffer, t.data.byteOffset, nk), t_ptr / BYTES);

    var ier = exports.nufft2d3(nj, x_ptr, y_ptr, cre_ptr, cim_ptr, iflag, tol, nk, s_ptr, t_ptr, fre_ptr, fim_ptr);
    if (ier !== 0) {
      exports.my_free(x_ptr); exports.my_free(y_ptr);
      exports.my_free(cre_ptr); exports.my_free(cim_ptr);
      exports.my_free(s_ptr); exports.my_free(t_ptr);
      exports.my_free(fre_ptr); exports.my_free(fim_ptr);
      throw new RuntimeError("finufft2d3 failed with error code " + ier);
    }

    view = new Float64Array(mem.buffer);
    var f_re = new FloatXArray(nk);
    var f_im = new FloatXArray(nk);
    for (var i = 0; i < nk; i++) {
      f_re[i] = view[fre_ptr / BYTES + i];
      f_im[i] = view[fim_ptr / BYTES + i];
    }

    exports.my_free(x_ptr); exports.my_free(y_ptr);
    exports.my_free(cre_ptr); exports.my_free(cim_ptr);
    exports.my_free(s_ptr); exports.my_free(t_ptr);
    exports.my_free(fre_ptr); exports.my_free(fim_ptr);

    return RTV.tensor(f_re, [1, nk], f_im);
  },
});
