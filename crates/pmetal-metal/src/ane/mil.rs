//! MIL 1.3 program builder for Apple Neural Engine.
//!
//! Provides a safe, composable builder pattern for generating MIL (Machine
//! Learning Intermediate Language) programs. Each kernel generator produces
//! a complete MIL text string that can be compiled by the ANE runtime.
//!
//! MIL programs follow the format:
//! ```text
//! program(1.3)
//! [buildInfo = dict<string, string>({...})]
//! {
//!     func main<ios18>(tensor<fp16, [1, C, 1, S]> x) {
//!         ...
//!     } -> (output);
//! }
//! ```

use std::fmt::Write;

/// MIL program header with build info matching the ANE reference.
/// Note: MIL dict literals require double-brace wrapping: `dict<K,V>({{k1,v1},{k2,v2}})`.
const MIL_HEADER: &str = concat!(
    "program(1.3)\n",
    "[buildInfo = dict<string, string>({{",
    "\"coremlc-component-MIL\", \"3510.2.1\"}, ",
    "{\"coremlc-version\", \"3505.4.1\"}, ",
    "{\"coremltools-component-milinternal\", \"\"}, ",
    "{\"coremltools-version\", \"9.0\"}})]",
    "\n{\n"
);

/// Shared conv constants used by all kernel generators.
const CONV_CONST: &str = concat!(
    "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n",
    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n",
    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n",
    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n",
    "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n",
);

/// A MIL program under construction.
pub struct MilProgram {
    text: String,
    var_counter: usize,
}

/// The dtype of the MIL program's input tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MilDtype {
    /// 16-bit floating point.
    Fp16,
    /// 32-bit floating point.
    Fp32,
}

// MIL text generation uses explicit \n in write!() — each line is a complete
// MIL statement and the trailing \n is part of the generated program text.
#[allow(clippy::write_with_newline)]
impl MilProgram {
    /// Create a new MIL program with header and fp16 function signature.
    ///
    /// Input is `tensor<fp16, [1, C, 1, S]> x`.
    pub fn new(input_channels: usize, seq_len: usize) -> Self {
        let mut text = String::with_capacity(8192);
        text.push_str(MIL_HEADER);
        write!(
            text,
            "    func main<ios18>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
            input_channels, seq_len
        )
        .unwrap();
        Self {
            text,
            var_counter: 0,
        }
    }

    /// Create a new MIL program with fp32 input signature.
    ///
    /// Input is `tensor<fp32, [1, C, 1, S]> x`. Used for dynamic weight
    /// pipeline where activations + weights are packed in fp32 IOSurfaces.
    pub fn new_fp32(input_channels: usize, spatial: usize) -> Self {
        let mut text = String::with_capacity(16384);
        text.push_str(MIL_HEADER);
        write!(
            text,
            "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
            input_channels, spatial
        )
        .unwrap();
        Self {
            text,
            var_counter: 0,
        }
    }

    /// Emit shared conv constants (pad, strides, dilations, groups).
    pub fn emit_conv_constants(&mut self) {
        self.text.push_str(CONV_CONST);
    }

    /// Emit a const declaration for a BLOBFILE weight.
    ///
    /// The `offset=uint64(64)` points to the blob descriptor header (DEADBEEF magic)
    /// at byte 64, which contains the actual data offset (128) in its own fields.
    pub fn emit_weight_const(&mut self, name: &str, shape: &[usize], blob_path: &str) {
        let shape_str = format_shape(shape);
        write!(
            self.text,
            "        tensor<fp16, {}> {} = const()[name=string(\"{}\"), val=tensor<fp16, {}>(BLOBFILE(path=string(\"{}\"), offset=uint64(64)))];\n",
            shape_str, name, name, shape_str, blob_path
        )
        .unwrap();
    }

    /// Emit a const scalar value.
    pub fn emit_scalar_const(&mut self, name: &str, dtype: &str, value: &str) {
        write!(
            self.text,
            "        {} {} = const()[name=string(\"{}\"), val={}({})];\n",
            dtype, name, name, dtype, value
        )
        .unwrap();
    }

    /// Emit a const tensor value (inline, not BLOBFILE).
    pub fn emit_tensor_const(&mut self, name: &str, shape: &[usize], dtype: &str, value: &str) {
        let shape_str = format_shape_dtype(shape, dtype);
        write!(
            self.text,
            "        {} {} = const()[name=string(\"{}\"), val={}({})];\n",
            shape_str, name, name, shape_str, value
        )
        .unwrap();
    }

    /// Emit a 1x1 convolution (linear layer on ANE).
    pub fn emit_conv(
        &mut self,
        result_name: &str,
        result_shape: &[usize],
        weight_name: &str,
        input_name: &str,
    ) {
        let shape_str = format_shape(result_shape);
        write!(
            self.text,
            "        tensor<fp16, {}> {} = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight={},x={})[name=string(\"{}\")];\n",
            shape_str, result_name, weight_name, input_name, result_name
        )
        .unwrap();
    }

    /// Emit element-wise multiply.
    pub fn emit_mul(&mut self, result_name: &str, shape: &[usize], x: &str, y: &str) {
        let shape_str = format_shape(shape);
        write!(
            self.text,
            "        tensor<fp16, {}> {} = mul(x={},y={})[name=string(\"{}\")];\n",
            shape_str, result_name, x, y, result_name
        )
        .unwrap();
    }

    /// Emit element-wise add.
    pub fn emit_add(&mut self, result_name: &str, shape: &[usize], x: &str, y: &str) {
        let shape_str = format_shape(shape);
        write!(
            self.text,
            "        tensor<fp16, {}> {} = add(x={},y={})[name=string(\"{}\")];\n",
            shape_str, result_name, x, y, result_name
        )
        .unwrap();
    }

    /// Emit element-wise subtract.
    pub fn emit_sub(&mut self, result_name: &str, shape: &[usize], x: &str, y: &str) {
        let shape_str = format_shape(shape);
        write!(
            self.text,
            "        tensor<fp16, {}> {} = sub(x={},y={})[name=string(\"{}\")];\n",
            shape_str, result_name, x, y, result_name
        )
        .unwrap();
    }

    /// Emit reduce_sum along specified axes.
    pub fn emit_reduce_sum(
        &mut self,
        result_name: &str,
        shape: &[usize],
        input: &str,
        axes_name: &str,
        keep_dims_name: &str,
    ) {
        let shape_str = format_shape(shape);
        write!(
            self.text,
            "        tensor<fp16, {}> {} = reduce_sum(x={},axes={},keep_dims={})[name=string(\"{}\")];\n",
            shape_str, result_name, input, axes_name, keep_dims_name, result_name
        )
        .unwrap();
    }

    /// Emit pow operation.
    pub fn emit_pow(&mut self, result_name: &str, shape: &[usize], x: &str, y: &str) {
        let shape_str = format_shape(shape);
        write!(
            self.text,
            "        tensor<fp16, {}> {} = pow(x={},y={})[name=string(\"{}\")];\n",
            shape_str, result_name, x, y, result_name
        )
        .unwrap();
    }

    /// Emit reshape.
    pub fn emit_reshape(
        &mut self,
        result_name: &str,
        shape: &[usize],
        shape_var: &str,
        input: &str,
    ) {
        let shape_str = format_shape(shape);
        write!(
            self.text,
            "        tensor<fp16, {}> {} = reshape(shape={},x={})[name=string(\"{}\")];\n",
            shape_str, result_name, shape_var, input, result_name
        )
        .unwrap();
    }

    /// Emit transpose.
    pub fn emit_transpose(
        &mut self,
        result_name: &str,
        shape: &[usize],
        perm_var: &str,
        input: &str,
    ) {
        let shape_str = format_shape(shape);
        write!(
            self.text,
            "        tensor<fp16, {}> {} = transpose(perm={},x={})[name=string(\"{}\")];\n",
            shape_str, result_name, perm_var, input, result_name
        )
        .unwrap();
    }

    /// Emit matmul.
    pub fn emit_matmul(
        &mut self,
        result_name: &str,
        shape: &[usize],
        transpose_x: &str,
        transpose_y: &str,
        x: &str,
        y: &str,
    ) {
        let shape_str = format_shape(shape);
        write!(
            self.text,
            "        tensor<fp16, {}> {} = matmul(transpose_x={},transpose_y={},x={},y={})[name=string(\"{}\")];\n",
            shape_str, result_name, transpose_x, transpose_y, x, y, result_name
        )
        .unwrap();
    }

    /// Emit softmax.
    pub fn emit_softmax(
        &mut self,
        result_name: &str,
        shape: &[usize],
        axis_var: &str,
        input: &str,
    ) {
        let shape_str = format_shape(shape);
        write!(
            self.text,
            "        tensor<fp16, {}> {} = softmax(axis={},x={})[name=string(\"{}\")];\n",
            shape_str, result_name, axis_var, input, result_name
        )
        .unwrap();
    }

    /// Emit sigmoid.
    pub fn emit_sigmoid(&mut self, result_name: &str, shape: &[usize], input: &str) {
        let shape_str = format_shape(shape);
        write!(
            self.text,
            "        tensor<fp16, {}> {} = sigmoid(x={})[name=string(\"{}\")];\n",
            shape_str, result_name, input, result_name
        )
        .unwrap();
    }

    /// Emit concat along axis.
    pub fn emit_concat(
        &mut self,
        result_name: &str,
        shape: &[usize],
        axis_var: &str,
        interleave_var: &str,
        values: &[&str],
    ) {
        let shape_str = format_shape(shape);
        let vals = values.join(",");
        write!(
            self.text,
            "        tensor<fp16, {}> {} = concat(axis={},interleave={},values=({}))[name=string(\"{}\")];\n",
            shape_str, result_name, axis_var, interleave_var, vals, result_name
        )
        .unwrap();
    }

    /// Emit slice_by_size.
    pub fn emit_slice_by_size(
        &mut self,
        result_name: &str,
        shape: &[usize],
        input: &str,
        begin_var: &str,
        size_var: &str,
    ) {
        let shape_str = format_shape(shape);
        write!(
            self.text,
            "        tensor<fp16, {}> {} = slice_by_size(x={},begin={},size={})[name=string(\"{}\")];\n",
            shape_str, result_name, input, begin_var, size_var, result_name
        )
        .unwrap();
    }

    /// Emit tile operation (repeat tensor along dimensions).
    pub fn emit_tile(&mut self, result_name: &str, shape: &[usize], reps_var: &str, input: &str) {
        let shape_str = format_shape(shape);
        write!(
            self.text,
            "        tensor<fp16, {}> {} = tile(reps={},x={})[name=string(\"{}\")];\n",
            shape_str, result_name, reps_var, input, result_name
        )
        .unwrap();
    }

    /// Emit a dtype cast operation.
    ///
    /// `from_dtype` / `to_dtype` are MIL dtype strings like `"fp16"`, `"fp32"`.
    pub fn emit_cast(&mut self, result_name: &str, shape: &[usize], input: &str, to_dtype: &str) {
        let shape_str = format_shape(shape);
        write!(
            self.text,
            "        tensor<{}, {}> {} = cast(dtype=string(\"{}\"),x={})[name=string(\"{}\")];\n",
            to_dtype, shape_str, result_name, to_dtype, input, result_name
        )
        .unwrap();
    }

    /// Emit a raw line of MIL text.
    pub fn emit_raw(&mut self, line: &str) {
        self.text.push_str(line);
        if !line.ends_with('\n') {
            self.text.push('\n');
        }
    }

    /// Finalize the program with the output variable.
    pub fn finalize(mut self, output_name: &str) -> String {
        write!(self.text, "    }} -> ({});\n}}\n", output_name).unwrap();
        self.text
    }

    /// Finalize the program with multiple output variables (tuple output).
    pub fn finalize_multi(mut self, output_names: &[&str]) -> String {
        let outs = output_names.join(", ");
        write!(self.text, "    }} -> ({});\n}}\n", outs).unwrap();
        self.text
    }

    /// Generate a unique variable name.
    pub fn next_var(&mut self, prefix: &str) -> String {
        self.var_counter += 1;
        format!("{}_{}", prefix, self.var_counter)
    }

    /// Get a reference to the current program text (for debugging).
    pub fn text(&self) -> &str {
        &self.text
    }
}

/// Format a shape array as MIL tensor dimension string: `[1, 768, 1, 256]`.
fn format_shape(shape: &[usize]) -> String {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("[{}]", dims.join(", "))
}

/// Format a shape with dtype: `tensor<fp16, [1, 768, 1, 256]>`.
fn format_shape_dtype(shape: &[usize], dtype: &str) -> String {
    format!("tensor<{}, {}>", dtype, format_shape(shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mil_program_basic() {
        let mut prog = MilProgram::new(768, 256);
        prog.emit_mul("sq", &[1, 768, 1, 256], "x", "x");
        let text = prog.finalize("sq");

        assert!(text.contains("program(1.3)"));
        assert!(text.contains("func main<ios18>"));
        assert!(text.contains("tensor<fp16, [1, 768, 1, 256]> x"));
        assert!(text.contains("mul(x=x,y=x)"));
        assert!(text.contains("} -> (sq);"));
    }

    #[test]
    fn test_mil_conv_constants() {
        let mut prog = MilProgram::new(768, 256);
        prog.emit_conv_constants();
        let text = prog.finalize("x");
        assert!(
            text.contains("string pt"),
            "should contain pad_type var 'pt'"
        );
        assert!(
            text.contains("st = const()"),
            "should contain strides var 'st'"
        );
        assert!(
            text.contains("dl = const()"),
            "should contain dilations var 'dl'"
        );
    }

    #[test]
    fn test_mil_weight_const() {
        let mut prog = MilProgram::new(768, 256);
        prog.emit_weight_const("Wq", &[768, 768, 1, 1], "@model_path/weights/wq.bin");
        let text = prog.finalize("x");
        assert!(text.contains("BLOBFILE(path=string(\"@model_path/weights/wq.bin\")"));
        assert!(text.contains("[768, 768, 1, 1]"));
    }

    #[test]
    fn test_mil_tile() {
        let mut prog = MilProgram::new(768, 256);
        prog.emit_tensor_const("reps", &[4], "int32", "[1,4,1,1]");
        prog.emit_tile("tiled", &[1, 48, 256, 64], "reps", "x");
        let text = prog.finalize("tiled");
        assert!(text.contains("tile(reps=reps,x=x)"));
        assert!(text.contains("[1, 48, 256, 64]"));
    }

    #[test]
    fn test_mil_fp32_input() {
        let prog = MilProgram::new_fp32(768, 512);
        let text = prog.finalize("x");
        assert!(text.contains("tensor<fp32, [1, 768, 1, 512]> x"));
    }

    #[test]
    fn test_mil_cast() {
        let mut prog = MilProgram::new_fp32(768, 512);
        prog.emit_cast("x16", &[1, 768, 1, 512], "x", "fp16");
        let text = prog.finalize("x16");
        assert!(text.contains("cast(dtype=string(\"fp16\"),x=x)"));
        assert!(text.contains("tensor<fp16, [1, 768, 1, 512]> x16"));
    }

    #[test]
    fn test_mil_concat() {
        let mut prog = MilProgram::new(768, 256);
        prog.emit_scalar_const("cax", "int32", "1");
        prog.emit_scalar_const("cid", "bool", "false");
        prog.emit_concat("out", &[1, 1536, 1, 256], "cax", "cid", &["a", "b"]);
        let text = prog.finalize("out");
        assert!(text.contains("concat(axis=cax,interleave=cid,values=(a,b))"));
    }
}
