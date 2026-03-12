// wasm: finufft
// finufft1d1(x, c_re, c_im, iflag, tol, ms) -> [fk_re, fk_im]
// 1D type 1 non-uniform FFT via FINUFFT compiled to WebAssembly.
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
    var x = args[0]; // tensor: nonuniform points in [-pi, pi)
    var c_re = args[1]; // tensor: real part of strengths
    var c_im = args[2]; // tensor: imaginary part of strengths
    var iflag = args[3]; // number: +1 or -1
    var tol = args[4]; // number: tolerance
    var ms = args[5]; // number: number of output modes

    var nj = x.data.length;
    var BYTES = 8; // sizeof(double)
    var exports = wasm.exports;
    var mem = exports.memory;

    // Allocate WASM memory for inputs and outputs
    var x_ptr = exports.my_malloc(nj * BYTES);
    var cre_ptr = exports.my_malloc(nj * BYTES);
    var cim_ptr = exports.my_malloc(nj * BYTES);
    var fkre_ptr = exports.my_malloc(ms * BYTES);
    var fkim_ptr = exports.my_malloc(ms * BYTES);

    // Copy input arrays into WASM linear memory
    var view = new Float64Array(mem.buffer);
    view.set(new Float64Array(x.data.buffer, x.data.byteOffset, nj), x_ptr / BYTES);
    view.set(
      new Float64Array(c_re.data.buffer, c_re.data.byteOffset, nj),
      cre_ptr / BYTES,
    );
    view.set(
      new Float64Array(c_im.data.buffer, c_im.data.byteOffset, nj),
      cim_ptr / BYTES,
    );

    // Call FINUFFT
    var ier = exports.nufft1d1(
      nj,
      x_ptr,
      cre_ptr,
      cim_ptr,
      iflag,
      tol,
      ms,
      fkre_ptr,
      fkim_ptr,
    );
    if (ier !== 0) {
      exports.my_free(x_ptr);
      exports.my_free(cre_ptr);
      exports.my_free(cim_ptr);
      exports.my_free(fkre_ptr);
      exports.my_free(fkim_ptr);
      throw new RuntimeError("finufft1d1 failed with error code " + ier);
    }

    // Copy output from WASM memory
    // Re-create view in case memory grew
    view = new Float64Array(mem.buffer);
    var fk_re_out = new FloatXArray(ms);
    var fk_im_out = new FloatXArray(ms);
    for (var i = 0; i < ms; i++) {
      fk_re_out[i] = view[fkre_ptr / BYTES + i];
      fk_im_out[i] = view[fkim_ptr / BYTES + i];
    }

    // Free WASM memory
    exports.my_free(x_ptr);
    exports.my_free(cre_ptr);
    exports.my_free(cim_ptr);
    exports.my_free(fkre_ptr);
    exports.my_free(fkim_ptr);

    return [RTV.tensor(fk_re_out, [1, ms]), RTV.tensor(fk_im_out, [1, ms])];
  },
});
