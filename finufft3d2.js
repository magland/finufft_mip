// wasm: finufft
// finufft3d2(x, y, z, fk, iflag, tol) -> c
// 3D type 2 non-uniform FFT (uniform -> nonuniform).
// fk is complex, c is complex.
register({
  check: function (argTypes, nargout) {
    return {
      outputTypes: [IType.tensor({ isComplex: true })],
    };
  },
  apply: function (args, nargout) {
    var x = args[0];
    var y = args[1];
    var z = args[2];
    var fk = args[3];
    var iflag = args[4];
    var tol = args[5];

    var nj = x.data.length;
    var ms = fk.shape[0];
    var mt = fk.shape[1];
    var mu = fk.shape[2];
    var nin = ms * mt * mu;
    var BYTES = 8;
    var exports = wasm.exports;
    var mem = exports.memory;

    var x_ptr = exports.my_malloc(nj * BYTES);
    var y_ptr = exports.my_malloc(nj * BYTES);
    var z_ptr = exports.my_malloc(nj * BYTES);
    var cre_ptr = exports.my_malloc(nj * BYTES);
    var cim_ptr = exports.my_malloc(nj * BYTES);
    var fkre_ptr = exports.my_malloc(nin * BYTES);
    var fkim_ptr = exports.my_malloc(nin * BYTES);

    var view = new Float64Array(mem.buffer);
    view.set(new Float64Array(x.data.buffer, x.data.byteOffset, nj), x_ptr / BYTES);
    view.set(new Float64Array(y.data.buffer, y.data.byteOffset, nj), y_ptr / BYTES);
    view.set(new Float64Array(z.data.buffer, z.data.byteOffset, nj), z_ptr / BYTES);
    view.set(new Float64Array(fk.data.buffer, fk.data.byteOffset, nin), fkre_ptr / BYTES);
    if (fk.imag) {
      view.set(new Float64Array(fk.imag.buffer, fk.imag.byteOffset, nin), fkim_ptr / BYTES);
    } else {
      view.fill(0, fkim_ptr / BYTES, fkim_ptr / BYTES + nin);
    }

    var ier = exports.nufft3d2(nj, x_ptr, y_ptr, z_ptr, cre_ptr, cim_ptr, iflag, tol, ms, mt, mu, fkre_ptr, fkim_ptr);
    if (ier !== 0) {
      exports.my_free(x_ptr); exports.my_free(y_ptr); exports.my_free(z_ptr);
      exports.my_free(cre_ptr); exports.my_free(cim_ptr);
      exports.my_free(fkre_ptr); exports.my_free(fkim_ptr);
      throw new RuntimeError("finufft3d2 failed with error code " + ier);
    }

    view = new Float64Array(mem.buffer);
    var c_re = new FloatXArray(nj);
    var c_im = new FloatXArray(nj);
    for (var i = 0; i < nj; i++) {
      c_re[i] = view[cre_ptr / BYTES + i];
      c_im[i] = view[cim_ptr / BYTES + i];
    }

    exports.my_free(x_ptr); exports.my_free(y_ptr); exports.my_free(z_ptr);
    exports.my_free(cre_ptr); exports.my_free(cim_ptr);
    exports.my_free(fkre_ptr); exports.my_free(fkim_ptr);

    return RTV.tensor(c_re, [1, nj], c_im);
  },
});
