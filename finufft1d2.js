// wasm: finufft
// finufft1d2(x, fk_re, fk_im, iflag, tol) -> [c_re, c_im]
// 1D type 2 non-uniform FFT (uniform → nonuniform).
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
    var fk_re = args[1];
    var fk_im = args[2];
    var iflag = args[3];
    var tol = args[4];

    var nj = x.data.length;
    var ms = fk_re.data.length;
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
    view.set(new Float64Array(fk_re.data.buffer, fk_re.data.byteOffset, ms), fkre_ptr / BYTES);
    view.set(new Float64Array(fk_im.data.buffer, fk_im.data.byteOffset, ms), fkim_ptr / BYTES);

    var ier = exports.nufft1d2(nj, x_ptr, cre_ptr, cim_ptr, iflag, tol, ms, fkre_ptr, fkim_ptr);
    if (ier !== 0) {
      exports.my_free(x_ptr); exports.my_free(cre_ptr); exports.my_free(cim_ptr);
      exports.my_free(fkre_ptr); exports.my_free(fkim_ptr);
      throw new RuntimeError("finufft1d2 failed with error code " + ier);
    }

    view = new Float64Array(mem.buffer);
    var c_re_out = new FloatXArray(nj);
    var c_im_out = new FloatXArray(nj);
    for (var i = 0; i < nj; i++) {
      c_re_out[i] = view[cre_ptr / BYTES + i];
      c_im_out[i] = view[cim_ptr / BYTES + i];
    }

    exports.my_free(x_ptr); exports.my_free(cre_ptr); exports.my_free(cim_ptr);
    exports.my_free(fkre_ptr); exports.my_free(fkim_ptr);

    return [RTV.tensor(c_re_out, [1, nj]), RTV.tensor(c_im_out, [1, nj])];
  },
});
