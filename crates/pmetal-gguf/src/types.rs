//! GGUF format types and constants.

/// GGUF magic number: "GGUF" in bytes.
pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian

/// Current GGUF version (v3 with big-endian support).
pub const GGUF_VERSION: u32 = 3;

/// Default alignment for tensor data.
pub const GGUF_DEFAULT_ALIGNMENT: u32 = 32;

/// GGML tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    /// 32-bit float
    F32 = 0,
    /// 16-bit float
    F16 = 1,
    /// 4-bit quantization (type 0)
    Q4_0 = 2,
    /// 4-bit quantization (type 1)
    Q4_1 = 3,
    /// 5-bit quantization (type 0)
    Q5_0 = 6,
    /// 5-bit quantization (type 1)
    Q5_1 = 7,
    /// 8-bit quantization (type 0)
    Q8_0 = 8,
    /// 8-bit quantization (type 1)
    Q8_1 = 9,
    /// K-quant 2-bit
    Q2K = 10,
    /// K-quant 3-bit
    Q3K = 11,
    /// K-quant 4-bit
    Q4K = 12,
    /// K-quant 5-bit
    Q5K = 13,
    /// K-quant 6-bit
    Q6K = 14,
    /// K-quant 8-bit
    Q8K = 15,
    /// IQ2 extra-extra-small
    Iq2Xxs = 16,
    /// IQ2 extra-small
    Iq2Xs = 17,
    /// IQ3 extra-extra-small
    Iq3Xxs = 18,
    /// IQ1 small
    Iq1S = 19,
    /// IQ4 non-linear
    Iq4Nl = 20,
    /// IQ3 small
    Iq3S = 21,
    /// IQ2 small
    Iq2S = 22,
    /// IQ4 extra-small
    Iq4Xs = 23,
    /// 8-bit integer
    I8 = 24,
    /// 16-bit integer
    I16 = 25,
    /// 32-bit integer
    I32 = 26,
    /// 64-bit integer
    I64 = 27,
    /// 64-bit float
    F64 = 28,
    /// IQ1 medium
    Iq1M = 29,
    /// BFloat16
    Bf16 = 30,
    /// TQ1 type 0
    Tq1_0 = 34,
    /// TQ2 type 0
    Tq2_0 = 35,
}

impl GgmlType {
    /// Get the number of bytes per element for this type.
    /// Note: For quantized types, this returns the block size in bytes.
    pub fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Q4_0 => 18,  // 32 values in 18 bytes (16 + 2 scale)
            Self::Q4_1 => 20,  // 32 values in 20 bytes
            Self::Q5_0 => 22,  // 32 values in 22 bytes
            Self::Q5_1 => 24,  // 32 values in 24 bytes
            Self::Q8_0 => 34,  // 32 values in 34 bytes
            Self::Q8_1 => 36,  // 32 values in 36 bytes
            Self::Q2K => 84,   // 256 values in 84 bytes
            Self::Q3K => 110,  // 256 values in 110 bytes
            Self::Q4K => 144,  // 256 values in 144 bytes
            Self::Q5K => 176,  // 256 values in 176 bytes
            Self::Q6K => 210,  // 256 values in 210 bytes
            Self::Q8K => 292,  // 256 values in 292 bytes
            // IQ types (importance-weighted quantization)
            Self::Iq1S => 50,   // 256 values in 50 bytes (~1.5625 bits/element)
            Self::Iq1M => 56,   // 256 values in 56 bytes (~1.75 bits/element)
            Self::Iq2Xxs => 66, // 256 values in 66 bytes (~2.0625 bits/element)
            Self::Iq2Xs => 74,  // 256 values in 74 bytes (~2.3125 bits/element)
            Self::Iq2S => 82,   // 256 values in 82 bytes (~2.5625 bits/element)
            Self::Iq3Xxs => 98, // 256 values in 98 bytes (~3.0625 bits/element)
            Self::Iq3S => 110,  // 256 values in 110 bytes (~3.4375 bits/element)
            Self::Iq4Nl => 18,  // 32 values in 18 bytes (~4.5 bits/element)
            Self::Iq4Xs => 136, // 256 values in 136 bytes (~4.25 bits/element)
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 | Self::F64 => 8,
            _ => 2, // Default for other quantized types
        }
    }

    /// Get the block size (number of elements per block) for quantized types.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q8K => 256,
            Self::Q4K | Self::Q5K | Self::Q6K => 256,
            // IQ types: IQ4_NL uses 32, all others use 256
            Self::Iq4Nl => 32,
            Self::Iq1S | Self::Iq1M | Self::Iq2Xxs | Self::Iq2Xs | Self::Iq2S
            | Self::Iq3Xxs | Self::Iq3S | Self::Iq4Xs => 256,
            _ => 32,
        }
    }

    /// Calculate the byte size for a tensor with given number of elements.
    pub fn tensor_size(&self, n_elements: usize) -> usize {
        let block_size = self.block_size();
        let n_blocks = (n_elements + block_size - 1) / block_size;
        n_blocks * self.type_size()
    }

    /// Calculate the byte size with checked arithmetic.
    ///
    /// Returns `None` if the calculation would overflow.
    pub fn tensor_size_checked(&self, n_elements: usize) -> Option<usize> {
        let block_size = self.block_size();
        // (n_elements + block_size - 1) / block_size with overflow checking
        let n_blocks = n_elements.checked_add(block_size.checked_sub(1)?)? / block_size;
        n_blocks.checked_mul(self.type_size())
    }
}

/// GGUF metadata value types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetadataValueType {
    /// 8-bit unsigned integer
    Uint8 = 0,
    /// 8-bit signed integer
    Int8 = 1,
    /// 16-bit unsigned integer
    Uint16 = 2,
    /// 16-bit signed integer
    Int16 = 3,
    /// 32-bit unsigned integer
    Uint32 = 4,
    /// 32-bit signed integer
    Int32 = 5,
    /// 32-bit float
    Float32 = 6,
    /// Boolean (1 byte)
    Bool = 7,
    /// UTF-8 string with length prefix
    String = 8,
    /// Array of values
    Array = 9,
    /// 64-bit unsigned integer
    Uint64 = 10,
    /// 64-bit signed integer
    Int64 = 11,
    /// 64-bit float
    Float64 = 12,
}

/// A metadata value in GGUF format.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    /// 8-bit unsigned integer
    Uint8(u8),
    /// 8-bit signed integer
    Int8(i8),
    /// 16-bit unsigned integer
    Uint16(u16),
    /// 16-bit signed integer
    Int16(i16),
    /// 32-bit unsigned integer
    Uint32(u32),
    /// 32-bit signed integer
    Int32(i32),
    /// 32-bit float
    Float32(f32),
    /// Boolean
    Bool(bool),
    /// UTF-8 string
    String(String),
    /// Array of values (all same type)
    Array(Vec<MetadataValue>),
    /// 64-bit unsigned integer
    Uint64(u64),
    /// 64-bit signed integer
    Int64(i64),
    /// 64-bit float
    Float64(f64),
}

impl MetadataValue {
    /// Get the type of this value.
    pub fn value_type(&self) -> MetadataValueType {
        match self {
            Self::Uint8(_) => MetadataValueType::Uint8,
            Self::Int8(_) => MetadataValueType::Int8,
            Self::Uint16(_) => MetadataValueType::Uint16,
            Self::Int16(_) => MetadataValueType::Int16,
            Self::Uint32(_) => MetadataValueType::Uint32,
            Self::Int32(_) => MetadataValueType::Int32,
            Self::Float32(_) => MetadataValueType::Float32,
            Self::Bool(_) => MetadataValueType::Bool,
            Self::String(_) => MetadataValueType::String,
            Self::Array(_) => MetadataValueType::Array,
            Self::Uint64(_) => MetadataValueType::Uint64,
            Self::Int64(_) => MetadataValueType::Int64,
            Self::Float64(_) => MetadataValueType::Float64,
        }
    }

    /// Get the element type for arrays.
    pub fn array_element_type(&self) -> Option<MetadataValueType> {
        match self {
            Self::Array(arr) => arr.first().map(|v| v.value_type()),
            _ => None,
        }
    }
}

/// Information about a tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Name of the tensor (max 64 bytes).
    pub name: String,
    /// Number of dimensions.
    pub n_dimensions: u32,
    /// Dimensions of the tensor.
    pub dimensions: Vec<u64>,
    /// Data type of the tensor.
    pub dtype: GgmlType,
    /// Offset of tensor data (relative to tensor_data section).
    pub offset: u64,
}

/// Error type for tensor size calculations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum TensorSizeError {
    /// Integer overflow in element count calculation.
    #[error("Integer overflow calculating tensor element count")]
    ElementCountOverflow,
    /// Tensor element count exceeds usize limits.
    #[error("Tensor element count {0} exceeds system limits")]
    ElementCountTooLarge(u64),
    /// Integer overflow in byte size calculation.
    #[error("Integer overflow calculating tensor byte size")]
    ByteSizeOverflow,
}

impl TensorInfo {
    /// Create a new tensor info.
    pub fn new(name: impl Into<String>, dimensions: Vec<u64>, dtype: GgmlType) -> Self {
        let name = name.into();
        let n_dimensions = dimensions.len() as u32;
        Self {
            name,
            n_dimensions,
            dimensions,
            dtype,
            offset: 0, // Set by writer
        }
    }

    /// Get the number of elements in the tensor with checked arithmetic.
    ///
    /// Returns an error if the calculation would overflow.
    pub fn n_elements_checked(&self) -> Result<usize, TensorSizeError> {
        let total = self
            .dimensions
            .iter()
            .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
            .ok_or(TensorSizeError::ElementCountOverflow)?;

        usize::try_from(total).map_err(|_| TensorSizeError::ElementCountTooLarge(total))
    }

    /// Get the number of elements in the tensor.
    ///
    /// # Panics
    /// Panics if the calculation would overflow. Use [`n_elements_checked`] for fallible version.
    pub fn n_elements(&self) -> usize {
        self.n_elements_checked()
            .expect("tensor element count overflow")
    }

    /// Get the byte size of the tensor data with checked arithmetic.
    ///
    /// Returns an error if the calculation would overflow.
    pub fn byte_size_checked(&self) -> Result<usize, TensorSizeError> {
        let n_elements = self.n_elements_checked()?;
        self.dtype
            .tensor_size_checked(n_elements)
            .ok_or(TensorSizeError::ByteSizeOverflow)
    }

    /// Get the byte size of the tensor data.
    ///
    /// # Panics
    /// Panics if the calculation would overflow. Use [`byte_size_checked`] for fallible version.
    pub fn byte_size(&self) -> usize {
        self.byte_size_checked()
            .expect("tensor byte size overflow")
    }
}

/// Standard metadata keys for GGUF files.
pub mod keys {
    // General (required)
    /// Model architecture name
    pub const GENERAL_ARCHITECTURE: &str = "general.architecture";
    /// Quantization format version
    pub const GENERAL_QUANTIZATION_VERSION: &str = "general.quantization_version";
    /// Data alignment
    pub const GENERAL_ALIGNMENT: &str = "general.alignment";

    // General metadata (optional)
    /// Model name
    pub const GENERAL_NAME: &str = "general.name";
    /// Model author
    pub const GENERAL_AUTHOR: &str = "general.author";
    /// Model version
    pub const GENERAL_VERSION: &str = "general.version";
    /// Model description
    pub const GENERAL_DESCRIPTION: &str = "general.description";
    /// File type (quantization)
    pub const GENERAL_FILE_TYPE: &str = "general.file_type";

    // LLM architecture (use with architecture prefix, e.g., "llama.")
    /// Context length
    pub const LLM_CONTEXT_LENGTH: &str = "context_length";
    /// Embedding size
    pub const LLM_EMBEDDING_LENGTH: &str = "embedding_length";
    /// Number of transformer blocks
    pub const LLM_BLOCK_COUNT: &str = "block_count";
    /// Feed-forward layer size
    pub const LLM_FEED_FORWARD_LENGTH: &str = "feed_forward_length";
    /// Number of attention heads
    pub const LLM_ATTENTION_HEAD_COUNT: &str = "attention.head_count";
    /// Number of KV attention heads (for GQA)
    pub const LLM_ATTENTION_HEAD_COUNT_KV: &str = "attention.head_count_kv";
    /// Layer norm epsilon
    pub const LLM_ATTENTION_LAYER_NORM_EPSILON: &str = "attention.layer_norm_epsilon";
    /// RMS norm epsilon
    pub const LLM_ATTENTION_LAYER_NORM_RMS_EPSILON: &str = "attention.layer_norm_rms_epsilon";
    /// RoPE dimension count
    pub const LLM_ROPE_DIMENSION_COUNT: &str = "rope.dimension_count";
    /// RoPE frequency base
    pub const LLM_ROPE_FREQ_BASE: &str = "rope.freq_base";
    /// Expert count (MoE)
    pub const LLM_EXPERT_COUNT: &str = "expert_count";
    /// Experts used per token (MoE)
    pub const LLM_EXPERT_USED_COUNT: &str = "expert_used_count";

    // Tokenizer
    /// Tokenizer model type
    pub const TOKENIZER_MODEL: &str = "tokenizer.ggml.model";
    /// Token list
    pub const TOKENIZER_TOKENS: &str = "tokenizer.ggml.tokens";
    /// Token scores
    pub const TOKENIZER_SCORES: &str = "tokenizer.ggml.scores";
    /// Token types
    pub const TOKENIZER_TOKEN_TYPE: &str = "tokenizer.ggml.token_type";
    /// BPE merges
    pub const TOKENIZER_MERGES: &str = "tokenizer.ggml.merges";
    /// Added tokens
    pub const TOKENIZER_ADDED_TOKENS: &str = "tokenizer.ggml.added_tokens";
    /// BOS token ID
    pub const TOKENIZER_BOS_TOKEN_ID: &str = "tokenizer.ggml.bos_token_id";
    /// EOS token ID
    pub const TOKENIZER_EOS_TOKEN_ID: &str = "tokenizer.ggml.eos_token_id";
    /// Unknown token ID
    pub const TOKENIZER_UNK_TOKEN_ID: &str = "tokenizer.ggml.unknown_token_id";
    /// Padding token ID
    pub const TOKENIZER_PAD_TOKEN_ID: &str = "tokenizer.ggml.padding_token_id";
    /// Chat template
    pub const TOKENIZER_CHAT_TEMPLATE: &str = "tokenizer.chat_template";
}

/// Standard tensor names for transformer models.
pub mod tensors {
    /// Token embedding layer
    pub const TOKEN_EMBD: &str = "token_embd.weight";
    /// Position embedding layer
    pub const POS_EMBD: &str = "pos_embd.weight";
    /// Output normalization layer
    pub const OUTPUT_NORM: &str = "output_norm.weight";
    /// Output layer (LM head)
    pub const OUTPUT: &str = "output.weight";

    /// Format block tensor name.
    pub fn block_tensor(block: usize, name: &str) -> String {
        format!("blk.{}.{}", block, name)
    }

    // Block tensor suffixes
    /// Attention normalization
    pub const ATTN_NORM: &str = "attn_norm.weight";
    /// Attention Q projection
    pub const ATTN_Q: &str = "attn_q.weight";
    /// Attention K projection
    pub const ATTN_K: &str = "attn_k.weight";
    /// Attention V projection
    pub const ATTN_V: &str = "attn_v.weight";
    /// Attention output projection
    pub const ATTN_OUTPUT: &str = "attn_output.weight";
    /// FFN normalization
    pub const FFN_NORM: &str = "ffn_norm.weight";
    /// FFN up projection
    pub const FFN_UP: &str = "ffn_up.weight";
    /// FFN gate projection
    pub const FFN_GATE: &str = "ffn_gate.weight";
    /// FFN down projection
    pub const FFN_DOWN: &str = "ffn_down.weight";
}

/// File type enumeration for quantized models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FileType {
    /// All F32
    AllF32 = 0,
    /// Mostly F16
    MostlyF16 = 1,
    /// Mostly Q4_0
    MostlyQ4_0 = 2,
    /// Mostly Q4_1
    MostlyQ4_1 = 3,
    /// Mostly Q8_0
    MostlyQ8_0 = 7,
    /// Mostly Q5_0
    MostlyQ5_0 = 8,
    /// Mostly Q5_1
    MostlyQ5_1 = 9,
    /// Mostly Q2_K
    MostlyQ2K = 10,
    /// Mostly Q3_K_S
    MostlyQ3KS = 11,
    /// Mostly Q3_K_M
    MostlyQ3KM = 12,
    /// Mostly Q3_K_L
    MostlyQ3KL = 13,
    /// Mostly Q4_K_S
    MostlyQ4KS = 14,
    /// Mostly Q4_K_M
    MostlyQ4KM = 15,
    /// Mostly Q5_K_S
    MostlyQ5KS = 16,
    /// Mostly Q5_K_M
    MostlyQ5KM = 17,
    /// Mostly Q6_K
    MostlyQ6K = 18,
}
