// wasm: finufft
// finufft1d2(x, fk, iflag, tol) -> c
// 1D type 2 non-uniform FFT (uniform -> nonuniform).
// fk is complex, c is complex.
register({
  check: function (argTypes, nargout) {
    return {
      outputTypes: [IType.tensor({ isComplex: true })],
    };
  },
  apply: function (args, nargout) {
    var x = args[0];
    var fk = args[1];
    var iflag = args[2];
    var tol = args[3];

    var nj = x.data.length;
    var ms = fk.data.length;
    var BYTES = 8;
    var exports = wasm.exports;
    var mem = exports.memory;

    var x_ptr = exports.my_malloc(nj * BYTES);
    var cre_ptr = exports.my_malloc(nj * BYTES);
    var cim_ptr = exports.my_malloc(nj * BYTES);
    var fkre_ptr = exports.my_malloc(ms * BYTES);
    var fkim_ptr = exports.my_malloc(ms * BYTES);

    var view = new Float64Array(mem.buffer);
    view.set(new Float64Array(x.data.buffer, x.data.byteOffset, nj), x_ptr / BYTES);
    view.set(new Float64Array(fk.data.buffer, fk.data.byteOffset, ms), fkre_ptr / BYTES);
    if (fk.imag) {
      view.set(new Float64Array(fk.imag.buffer, fk.imag.byteOffset, ms), fkim_ptr / BYTES);
    } else {
      view.fill(0, fkim_ptr / BYTES, fkim_ptr / BYTES + ms);
    }

    var ier = exports.nufft1d2(nj, x_ptr, cre_ptr, cim_ptr, iflag, tol, ms, fkre_ptr, fkim_ptr);
    if (ier !== 0) {
      exports.my_free(x_ptr); exports.my_free(cre_ptr); exports.my_free(cim_ptr);
      exports.my_free(fkre_ptr); exports.my_free(fkim_ptr);
      throw new RuntimeError("finufft1d2 failed with error code " + ier);
    }

    view = new Float64Array(mem.buffer);
    var c_re = new FloatXArray(nj);
    var c_im = new FloatXArray(nj);
    for (var i = 0; i < nj; i++) {
      c_re[i] = view[cre_ptr / BYTES + i];
      c_im[i] = view[cim_ptr / BYTES + i];
    }

    exports.my_free(x_ptr); exports.my_free(cre_ptr); exports.my_free(cim_ptr);
    exports.my_free(fkre_ptr); exports.my_free(fkim_ptr);

    return RTV.tensor(c_re, [1, nj], c_im);
  },
});
