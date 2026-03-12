// wasm: finufft
// finufft2d1(x, y, c, iflag, tol, ms, mt) -> fk
// 2D type 1 non-uniform FFT (nonuniform -> uniform).
// c is complex, fk is complex.
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
    var ms = args[5];
    var mt = args[6];

    var nj = x.data.length;
    var nout = ms * mt;
    var BYTES = 8;
    var exports = wasm.exports;
    var mem = exports.memory;

    var x_ptr = exports.my_malloc(nj * BYTES);
    var y_ptr = exports.my_malloc(nj * BYTES);
    var cre_ptr = exports.my_malloc(nj * BYTES);
    var cim_ptr = exports.my_malloc(nj * BYTES);
    var fkre_ptr = exports.my_malloc(nout * BYTES);
    var fkim_ptr = exports.my_malloc(nout * BYTES);

    var view = new Float64Array(mem.buffer);
    view.set(new Float64Array(x.data.buffer, x.data.byteOffset, nj), x_ptr / BYTES);
    view.set(new Float64Array(y.data.buffer, y.data.byteOffset, nj), y_ptr / BYTES);
    view.set(new Float64Array(c.data.buffer, c.data.byteOffset, nj), cre_ptr / BYTES);
    if (c.imag) {
      view.set(new Float64Array(c.imag.buffer, c.imag.byteOffset, nj), cim_ptr / BYTES);
    } else {
      view.fill(0, cim_ptr / BYTES, cim_ptr / BYTES + nj);
    }

    var ier = exports.nufft2d1(nj, x_ptr, y_ptr, cre_ptr, cim_ptr, iflag, tol, ms, mt, fkre_ptr, fkim_ptr);
    if (ier !== 0) {
      exports.my_free(x_ptr); exports.my_free(y_ptr);
      exports.my_free(cre_ptr); exports.my_free(cim_ptr);
      exports.my_free(fkre_ptr); exports.my_free(fkim_ptr);
      throw new RuntimeError("finufft2d1 failed with error code " + ier);
    }

    view = new Float64Array(mem.buffer);
    var fk_re = new FloatXArray(nout);
    var fk_im = new FloatXArray(nout);
    for (var i = 0; i < nout; i++) {
      fk_re[i] = view[fkre_ptr / BYTES + i];
      fk_im[i] = view[fkim_ptr / BYTES + i];
    }

    exports.my_free(x_ptr); exports.my_free(y_ptr);
    exports.my_free(cre_ptr); exports.my_free(cim_ptr);
    exports.my_free(fkre_ptr); exports.my_free(fkim_ptr);

    return RTV.tensor(fk_re, [ms, mt], fk_im);
  },
});
