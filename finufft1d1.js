// wasm: finufft
// finufft1d1(x, c, iflag, tol, ms) -> fk
// 1D type 1 non-uniform FFT (nonuniform -> uniform).
// c is complex, fk is complex.
register({
  check: function (argTypes, nargout) {
    return {
      outputTypes: [IType.tensor({ isComplex: true })],
    };
  },
  apply: function (args, nargout) {
    var x = args[0];
    var c = args[1];
    var iflag = args[2];
    var tol = args[3];
    var ms = args[4];

    var nj = x.data.length;
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
    view.set(new Float64Array(c.data.buffer, c.data.byteOffset, nj), cre_ptr / BYTES);
    if (c.imag) {
      view.set(new Float64Array(c.imag.buffer, c.imag.byteOffset, nj), cim_ptr / BYTES);
    } else {
      view.fill(0, cim_ptr / BYTES, cim_ptr / BYTES + nj);
    }

    var ier = exports.nufft1d1(nj, x_ptr, cre_ptr, cim_ptr, iflag, tol, ms, fkre_ptr, fkim_ptr);
    if (ier !== 0) {
      exports.my_free(x_ptr); exports.my_free(cre_ptr); exports.my_free(cim_ptr);
      exports.my_free(fkre_ptr); exports.my_free(fkim_ptr);
      throw new RuntimeError("finufft1d1 failed with error code " + ier);
    }

    view = new Float64Array(mem.buffer);
    var fk_re = new FloatXArray(ms);
    var fk_im = new FloatXArray(ms);
    for (var i = 0; i < ms; i++) {
      fk_re[i] = view[fkre_ptr / BYTES + i];
      fk_im[i] = view[fkim_ptr / BYTES + i];
    }

    exports.my_free(x_ptr); exports.my_free(cre_ptr); exports.my_free(cim_ptr);
    exports.my_free(fkre_ptr); exports.my_free(fkim_ptr);

    return RTV.tensor(fk_re, [1, ms], fk_im);
  },
});
