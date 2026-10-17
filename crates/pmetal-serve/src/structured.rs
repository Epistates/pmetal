//! Grammar-constrained decoding: output that always parses.
//!
//! `response_format` (`json_object`, `json_schema`) and the tool calls a
//! request requires (`tool_choice: "required"` or a named function) are
//! enforced while the model generates, not checked after: at every step the
//! set of tokens that can continue the text toward a valid output is
//! computed from the grammar, and the logits of every other token are set to
//! negative infinity, so the sampler can only draw a token that keeps the
//! output valid. The output is exact (it always parses, and validates against
//! the schema) at any temperature.
//!
//! The grammar machinery is [llguidance](https://github.com/guidance-ai/llguidance)
//! (MIT): a JSON-schema and Lark-grammar compiler whose Earley parser and
//! derivative-based lexer walk the tokenizer's byte trie to produce a token
//! mask in tens of microseconds. It was chosen over an in-house automaton
//! for its coverage of JSON Schema (`$ref`, `anyOf`, `enum`, `pattern`,
//! bounds, `additionalProperties`, ...) and its speed on large vocabularies;
//! the glue here is the vocabulary ([`GrammarFactory`] builds llguidance's
//! token environment from the model's own tokenizer), the sampler seam
//! (`GrammarProcessor` is a [`LogitsProcessor`]) and what to do around a
//! reasoning block.
//!
//! # Reasoning
//!
//! A model that thinks first writes `<think>...</think>` before its answer,
//! and the grammar describes the answer. The processor leaves the model free
//! while it reasons: from the start it allows either the grammar's first
//! tokens or a reasoning-open marker, and once the reasoning has closed the
//! grammar applies.

use std::sync::{Arc, Mutex};

use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{SimpleVob, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use llguidance::{Matcher, ParserFactory};
use pmetal_data::Tokenizer;
use pmetal_data::stream_format::ReasoningSplitter;
use pmetal_runtime::InlineArray;
use pmetal_runtime::compat::Dtype;
use pmetal_runtime::decode::LogitsProcessor;
use serde_json::Value;

use crate::vocab::ByteVocab;

/// What the output must conform to.
#[derive(Debug, Clone)]
pub enum ConstraintSpec {
    /// Any JSON object.
    JsonObject,
    /// JSON that validates against this schema.
    JsonSchema(Value),
    /// A Lark grammar (llguidance syntax).
    Lark(String),
}

/// How much white space JSON output may have between its tokens: up to a
/// line break and a deep indent. A model writes the layout it was trained on
/// (compact, or indented by nesting depth), and an upper bound keeps one that
/// has lost the thread from filling its budget with blanks, which unbounded
/// white space would let it.
const JSON_WHITESPACE: &str = "[\\n ]{0,24}";

/// `schema` with the layout rule of [`JSON_WHITESPACE`], unless it states its
/// own `x-guidance`.
pub(crate) fn with_layout(mut schema: Value) -> Value {
    if let Value::Object(object) = &mut schema {
        object.entry("x-guidance").or_insert_with(|| {
            serde_json::json!({
                "whitespace_flexible": true,
                "whitespace_pattern": JSON_WHITESPACE,
            })
        });
    }
    schema
}

impl ConstraintSpec {
    fn grammar(&self) -> TopLevelGrammar {
        match self {
            Self::JsonObject => TopLevelGrammar::from_json_schema(with_layout(
                serde_json::json!({"type": "object"}),
            )),
            Self::JsonSchema(schema) => {
                TopLevelGrammar::from_json_schema(with_layout(schema.clone()))
            }
            Self::Lark(lark) => TopLevelGrammar::from_lark(lark.clone()),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// The tokenizer as llguidance sees it
// ────────────────────────────────────────────────────────────────────────────

struct PmetalTokEnv {
    trie: TokTrie,
    tokenizer: Arc<Tokenizer>,
}

impl TokenizerEnv for PmetalTokEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<TokenId> {
        self.trie.tokenize_with_greedy_fallback(bytes, |text| {
            self.tokenizer.encode(text).unwrap_or_default()
        })
    }
}

/// Compiles grammars for one tokenizer; built once per engine.
pub struct GrammarFactory {
    factory: ParserFactory,
    /// Size of the token environment: the larger of the tokenizer's
    /// vocabulary and the model's output width.
    vocab: usize,
    eos: Vec<u32>,
}

impl GrammarFactory {
    /// The factory for `tokenizer`, whose model emits `model_vocab` logits
    /// per step and stops at any of `eos`.
    ///
    /// # Errors
    ///
    /// A tokenizer whose vocabulary has no byte spelling.
    pub fn new(
        tokenizer: Arc<Tokenizer>,
        bytes: &ByteVocab,
        model_vocab: usize,
        eos: &[u32],
    ) -> Result<Self, String> {
        let vocab = bytes.len().max(model_vocab);
        let mut words: Vec<Vec<u8>> = Vec::with_capacity(vocab);
        for id in 0..vocab as u32 {
            let special = |name: String| {
                let mut word = vec![TokTrie::SPECIAL_TOKEN_MARKER];
                word.extend_from_slice(name.as_bytes());
                word
            };
            if (id as usize) >= bytes.len() {
                // A row of the output matrix no token stands for.
                words.push(special(format!(".<[{id}]>")));
            } else if bytes.is_special(id) {
                let name = tokenizer
                    .inner()
                    .id_to_token(id)
                    .unwrap_or_else(|| format!("<[{id}]>"));
                words.push(special(name));
            } else {
                words.push(bytes.bytes(id).to_vec());
            }
        }
        let eos: Vec<u32> = eos
            .iter()
            .copied()
            .filter(|&id| (id as usize) < vocab)
            .collect();
        let first_eos = eos
            .first()
            .copied()
            .or_else(|| tokenizer.eos_token_id())
            .unwrap_or(0);
        let mut info = TokRxInfo::new(vocab as u32, first_eos);
        info.tok_bos = tokenizer.bos_token_id();
        info.tok_unk = tokenizer.unk_token_id();
        info.tok_pad = tokenizer.pad_token_id();
        let mut trie = TokTrie::from(&info, &words);
        if eos.len() > 1 {
            trie = trie.with_eos_tokens(&eos);
        }
        let env: TokEnv = Arc::new(PmetalTokEnv { trie, tokenizer });
        let mut factory = ParserFactory::new_simple(&env).map_err(|e| e.to_string())?;
        factory.quiet();
        Ok(Self {
            factory,
            vocab,
            eos,
        })
    }

    /// Compile `spec` for a generation that continues `prompt`.
    ///
    /// `thinks` says the model reasons before it answers; the grammar then
    /// waits for the reasoning to close (see the module docs).
    ///
    /// # Errors
    ///
    /// The grammar does not compile (an unsupported schema keyword, a
    /// malformed grammar, a limit exceeded); the message is the compiler's.
    pub fn compile(
        &self,
        spec: &ConstraintSpec,
        tokenizer: &Tokenizer,
        prompt: &[u32],
        thinks: bool,
    ) -> Result<Constraint, String> {
        let parser = self
            .factory
            .create_parser(spec.grammar())
            .map_err(|e| e.to_string())?;
        let mut matcher = Matcher::new(Ok(parser));
        // Computing the first mask now surfaces a grammar the lexer rejects
        // as an error to the caller, not as a failure mid-generation.
        matcher.compute_mask().map_err(|e| e.to_string())?;
        let mut splitter = ReasoningSplitter::new(tokenizer);
        splitter.prime(prompt);
        let phase = if splitter.in_reasoning() {
            Phase::Reasoning
        } else if thinks {
            Phase::Start
        } else {
            Phase::Answer
        };
        let think_open = ["<think>", "[THINK]"]
            .iter()
            .filter_map(|name| tokenizer.inner().token_to_id(name))
            .collect();
        Ok(Constraint {
            inner: Arc::new(ConstraintInner {
                template: Mutex::new(matcher),
                splitter,
                phase,
                think_open,
                vocab: self.vocab,
                eos: self.eos.clone(),
            }),
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// The constraint and its processor
// ────────────────────────────────────────────────────────────────────────────

/// Where a generation is relative to its reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Nothing generated yet by a model that thinks first: the grammar's
    /// first tokens or the reasoning marker.
    Start,
    /// Inside the reasoning: free.
    Reasoning,
    /// The answer: the grammar applies.
    Answer,
}

struct ConstraintInner {
    template: Mutex<Matcher>,
    splitter: ReasoningSplitter,
    phase: Phase,
    think_open: Vec<u32>,
    vocab: usize,
    eos: Vec<u32>,
}

/// A compiled grammar for one request. Cloning is cheap; each generation
/// takes its own [`processor`](Self::processor).
#[derive(Clone)]
pub struct Constraint {
    inner: Arc<ConstraintInner>,
}

impl std::fmt::Debug for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Constraint")
            .field("phase", &self.inner.phase)
            .finish_non_exhaustive()
    }
}

impl Constraint {
    /// A fresh processor at the start of the grammar, for one generation.
    pub fn processor(&self) -> Box<dyn LogitsProcessor> {
        let matcher = self
            .inner
            .template
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .deep_clone();
        Box::new(GrammarProcessor {
            matcher,
            splitter: self.inner.splitter.clone(),
            phase: self.inner.phase,
            think_open: self.inner.think_open.clone(),
            vocab: self.inner.vocab,
            eos: self.inner.eos.clone(),
            error: None,
        })
    }
}

/// Masks the logits to the tokens the grammar allows.
struct GrammarProcessor {
    matcher: Matcher,
    splitter: ReasoningSplitter,
    phase: Phase,
    think_open: Vec<u32>,
    vocab: usize,
    eos: Vec<u32>,
    /// The grammar failed mid-generation (a limit was hit): from then on only
    /// the end-of-sequence tokens are allowed, so the reply ends here.
    error: Option<String>,
}

impl GrammarProcessor {
    /// The mask for the next token, `None` while the model is free.
    fn mask(&mut self) -> Option<SimpleVob> {
        if self.phase == Phase::Reasoning {
            return None;
        }
        let mut mask = if self.error.is_some() {
            self.eos_only()
        } else {
            match self.matcher.compute_mask() {
                Ok(mask) => mask,
                Err(e) => {
                    tracing::error!("grammar failed mid-generation: {e}");
                    self.error = Some(e.to_string());
                    self.eos_only()
                }
            }
        };
        if self.phase == Phase::Start && self.error.is_none() {
            for &token in &self.think_open {
                if (token as usize) < mask.len() {
                    mask.allow_token(token);
                }
            }
        }
        Some(mask)
    }

    fn eos_only(&self) -> SimpleVob {
        let mut mask = SimpleVob::alloc(self.vocab);
        for &token in &self.eos {
            mask.allow_token(token);
        }
        mask
    }
}

impl LogitsProcessor for GrammarProcessor {
    fn process(&mut self, logits_2d: &InlineArray) -> InlineArray {
        let logits = logits_2d.as_dtype(Dtype::Float32.as_i32());
        let Some(mask) = self.mask() else {
            return logits;
        };
        let width = logits.dim(-1).max(0) as usize;
        let mut row = vec![f32::NEG_INFINITY; width];
        mask.iter_set_entries(|token| {
            if let Some(slot) = row.get_mut(token) {
                *slot = 0.0;
            }
        });
        logits.add(&InlineArray::from_f32_slice(&row, &[1, width as i32]))
    }

    fn accept(&mut self, token: u32) {
        // Where the splitter puts the token decides the phase.
        self.splitter.route(token);
        match self.phase {
            Phase::Start if self.think_open.contains(&token) => {
                self.phase = Phase::Reasoning;
                return;
            }
            Phase::Reasoning => {
                if !self.splitter.in_reasoning() {
                    self.phase = Phase::Answer;
                }
                return;
            }
            Phase::Start | Phase::Answer => {}
        }
        self.phase = Phase::Answer;
        if self.error.is_none() {
            if let Err(e) = self.matcher.consume_token(token) {
                tracing::error!("grammar rejected a token it allowed: {e}");
                self.error = Some(e.to_string());
            }
        }
    }

    fn needs_host_tokens(&self) -> bool {
        true
    }

    fn fork(&self) -> Box<dyn LogitsProcessor> {
        Box::new(Self {
            matcher: self.matcher.deep_clone(),
            splitter: self.splitter.clone(),
            phase: self.phase,
            think_open: self.think_open.clone(),
            vocab: self.vocab,
            eos: self.eos.clone(),
            error: self.error.clone(),
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Grammars for tool calls
// ────────────────────────────────────────────────────────────────────────────

/// A tool the model must be able to call: its name and the schema of its
/// arguments.
#[derive(Debug, Clone)]
pub struct ToolSchema {
    pub name: String,
    /// JSON Schema of the arguments object; `None` takes any object.
    pub parameters: Option<Value>,
}

/// The JSON of one call, `{"name": N, "<arguments_key>": ARGS}`, for any of
/// `tools`.
fn call_schema(tools: &[ToolSchema], arguments_key: &str) -> Value {
    let alternatives: Vec<Value> = tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"const": tool.name},
                    arguments_key: tool
                        .parameters
                        .clone()
                        .unwrap_or_else(|| serde_json::json!({"type": "object"})),
                },
                "required": ["name", arguments_key],
                "additionalProperties": false,
            })
        })
        .collect();
    with_layout(serde_json::json!({"anyOf": alternatives}))
}

/// The Lark grammar of a reply that is calls to `tools` in the convention of
/// `format`: one call, or (when `parallel`) one or more. `None` for a family
/// whose call syntax cannot be written as JSON the schema compiler checks
/// (its arguments are not JSON).
///
/// `special` gives the token id of a control token by its text (the call
/// markers of Hermes, Mistral and Llama are control tokens in some
/// vocabularies and plain text in others).
pub fn tool_call_grammar(
    format: crate::tools::ToolFormat,
    tools: &[ToolSchema],
    parallel: bool,
    special: &dyn Fn(&str) -> Option<u32>,
) -> Option<String> {
    use crate::tools::ToolFormat;
    // A marker: the control token if the vocabulary has one, else its text.
    let marker = |text: &str| match special(text) {
        Some(id) => format!("<[{id}]>"),
        None => serde_json::to_string(text).unwrap_or_default(),
    };
    let json = |schema: Value| format!("%json {}", schema);
    let repeat = |block: &str| {
        if parallel {
            format!("{block} (\"\\n\" {block})*")
        } else {
            block.to_string()
        }
    };
    match format {
        ToolFormat::Hermes => {
            let block = format!(
                "{} \"\\n\" call \"\\n\" {}",
                marker("<tool_call>"),
                marker("</tool_call>")
            );
            Some(format!(
                "start: {}\ncall: {}\nblock: {block}\n",
                repeat("block"),
                json(call_schema(tools, "arguments"))
            ))
        }
        ToolFormat::Llama3 => {
            let block = format!("{} call", marker("<|python_tag|>"));
            Some(format!(
                "start: {}\ncall: {}\nblock: {block}\n",
                if parallel {
                    "block (\";\" block)*".to_string()
                } else {
                    "block".to_string()
                },
                json(call_schema(tools, "parameters"))
            ))
        }
        ToolFormat::Mistral => {
            // One array holding the calls.
            let mut schema = serde_json::json!({
                "type": "array",
                "items": call_schema(tools, "arguments"),
                "minItems": 1,
            });
            if !parallel {
                schema["maxItems"] = Value::from(1);
            }
            Some(format!(
                "start: {} arr\narr: {}\n",
                marker("[TOOL_CALLS]"),
                json(schema)
            ))
        }
        // These write their arguments in a syntax of their own, not JSON.
        ToolFormat::QwenXml | ToolFormat::Gemma4 | ToolFormat::Harmony => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_data::tokenizer::testing::byte_level_bpe;

    fn setup(specials: &[&str]) -> (Arc<Tokenizer>, GrammarFactory) {
        let tokenizer = Arc::new(byte_level_bpe(&["h i"], specials, None));
        let bytes = ByteVocab::from_tokenizer(&tokenizer).unwrap();
        let eos = tokenizer.inner().token_to_id("<|endoftext|>").unwrap();
        let factory =
            GrammarFactory::new(Arc::clone(&tokenizer), &bytes, bytes.len() + 3, &[eos]).unwrap();
        (tokenizer, factory)
    }

    /// Run the processor greedily over uniform logits (ties broken by the
    /// lowest id) and collect what it generates: every step it allows only
    /// tokens that keep the output valid.
    fn generate(
        constraint: &Constraint,
        tokenizer: &Tokenizer,
        vocab: usize,
        pick: impl Fn(&[u32], usize) -> u32,
        max_steps: usize,
    ) -> (String, bool) {
        let mut processor = constraint.processor();
        let eos = tokenizer.inner().token_to_id("<|endoftext|>").unwrap();
        let mut out: Vec<u32> = Vec::new();
        for step in 0..max_steps {
            let logits = InlineArray::from_f32_slice(&vec![0.0f32; vocab], &[1, vocab as i32]);
            let masked = processor.process(&logits);
            let values: Vec<f32> = (0..vocab as i32)
                .map(|i| masked.slice(&[0, i], &[1, i + 1]).item_f32())
                .collect();
            let allowed: Vec<u32> = (0..vocab as u32)
                .filter(|&i| values[i as usize] > f32::NEG_INFINITY)
                .collect();
            assert!(!allowed.is_empty(), "no token allowed after {out:?}");
            let token = pick(&allowed, step);
            assert!(allowed.contains(&token));
            processor.accept(token);
            if token == eos {
                return (tokenizer.decode(&out).unwrap(), true);
            }
            out.push(token);
        }
        (tokenizer.decode(&out).unwrap(), false)
    }

    #[test]
    fn a_json_schema_forces_valid_output_whatever_the_model_prefers() {
        let (tokenizer, factory) = setup(&["<|endoftext|>"]);
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "tags": {"type": "array", "items": {"enum": ["a", "b"]}}
            },
            "required": ["name", "age"],
            "additionalProperties": false
        });
        let constraint = factory
            .compile(&ConstraintSpec::JsonSchema(schema), &tokenizer, &[], false)
            .unwrap();
        let vocab = ByteVocab::from_tokenizer(&tokenizer).unwrap().len() + 3;
        // Different "model preferences": always the lowest allowed id, the
        // highest, and a hash-driven walk.
        type Pick = Box<dyn Fn(&[u32], usize) -> u32>;
        let pickers: Vec<Pick> = vec![
            Box::new(|allowed, _| *allowed.first().unwrap()),
            Box::new(|allowed, _| *allowed.last().unwrap()),
            Box::new(|allowed, step| allowed[(step * 2654435761) % allowed.len()]),
            Box::new(|allowed, step| allowed[(step * 40503 + 7) % allowed.len()]),
        ];
        for pick in pickers {
            let (text, finished) = generate(&constraint, &tokenizer, vocab, &*pick, 400);
            if !finished {
                // A walk can wander in a string; what it has written so far
                // must still be a valid prefix.
                continue;
            }
            let value: Value = serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("{text:?} does not parse: {e}"));
            let object = value.as_object().unwrap();
            assert!(object["name"].is_string());
            assert!(object["age"].is_i64() || object["age"].is_u64());
            for key in object.keys() {
                assert!(["name", "age", "tags"].contains(&key.as_str()), "{key}");
            }
            if let Some(tags) = object.get("tags") {
                assert!(
                    tags.as_array()
                        .unwrap()
                        .iter()
                        .all(|t| t == "a" || t == "b")
                );
            }
        }
    }

    #[test]
    fn json_object_mode_allows_only_an_object() {
        let (tokenizer, factory) = setup(&["<|endoftext|>"]);
        let constraint = factory
            .compile(&ConstraintSpec::JsonObject, &tokenizer, &[], false)
            .unwrap();
        let vocab = ByteVocab::from_tokenizer(&tokenizer).unwrap().len() + 3;
        let (text, _) = generate(
            &constraint,
            &tokenizer,
            vocab,
            |a, _| *a.first().unwrap(),
            6,
        );
        assert!(text.trim_start().starts_with('{'), "{text:?}");
    }

    #[test]
    fn a_grammar_that_does_not_compile_is_an_error() {
        let (tokenizer, factory) = setup(&["<|endoftext|>"]);
        let err = factory
            .compile(
                &ConstraintSpec::JsonSchema(serde_json::json!({"type": "nonsense"})),
                &tokenizer,
                &[],
                false,
            )
            .unwrap_err();
        assert!(!err.is_empty());
        assert!(
            factory
                .compile(
                    &ConstraintSpec::Lark("start: (".into()),
                    &tokenizer,
                    &[],
                    false
                )
                .is_err()
        );
    }

    #[test]
    fn a_thinking_model_may_reason_before_the_grammar_starts() {
        let (tokenizer, factory) = setup(&["<|endoftext|>", "<think>", "</think>"]);
        let schema = serde_json::json!({"type": "object", "properties": {"a": {"type": "integer"}},
                                        "required": ["a"], "additionalProperties": false});
        let constraint = factory
            .compile(&ConstraintSpec::JsonSchema(schema), &tokenizer, &[], true)
            .unwrap();
        let vocab = ByteVocab::from_tokenizer(&tokenizer).unwrap().len() + 3;
        let open = tokenizer.inner().token_to_id("<think>").unwrap();
        let close = tokenizer.inner().token_to_id("</think>").unwrap();
        let mut processor = constraint.processor();
        let allowed_after = |processor: &mut Box<dyn LogitsProcessor>| -> Vec<u32> {
            let logits = InlineArray::from_f32_slice(&vec![0.0f32; vocab], &[1, vocab as i32]);
            let masked = processor.process(&logits);
            (0..vocab as u32)
                .filter(|&i| {
                    masked.slice(&[0, i as i32], &[1, i as i32 + 1]).item_f32() > f32::NEG_INFINITY
                })
                .collect()
        };
        // At the start: the object's first character, or the reasoning marker.
        let first = allowed_after(&mut processor);
        assert!(first.contains(&open));
        assert!(first.contains(&(b'{' as u32)));
        assert!(!first.contains(&(b'x' as u32)));
        processor.accept(open);
        // Inside the reasoning: anything.
        assert!(allowed_after(&mut processor).len() > 200);
        processor.accept(b'x' as u32);
        processor.accept(close);
        // After it, the grammar.
        let after = allowed_after(&mut processor);
        assert!(after.contains(&(b'{' as u32)));
        assert!(!after.contains(&(b'x' as u32)));
        assert!(!after.contains(&open));
    }

    #[test]
    fn a_prompt_that_opened_the_reasoning_starts_free() {
        let (tokenizer, factory) = setup(&["<|endoftext|>", "<think>", "</think>"]);
        let prompt = tokenizer.encode_with_special_tokens("<think>").unwrap();
        let constraint = factory
            .compile(&ConstraintSpec::JsonObject, &tokenizer, &prompt, true)
            .unwrap();
        let vocab = ByteVocab::from_tokenizer(&tokenizer).unwrap().len() + 3;
        let mut processor = constraint.processor();
        let logits = InlineArray::from_f32_slice(&vec![0.0f32; vocab], &[1, vocab as i32]);
        let masked = processor.process(&logits);
        assert_eq!(
            masked
                .slice(&[0, b'x' as i32], &[1, b'x' as i32 + 1])
                .item_f32(),
            0.0
        );
    }

    #[test]
    fn tool_call_grammars_name_only_the_offered_tools() {
        use crate::tools::ToolFormat;
        let tools = vec![
            ToolSchema {
                name: "get_weather".into(),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": false
                })),
            },
            ToolSchema {
                name: "ping".into(),
                parameters: None,
            },
        ];
        let (tokenizer, factory) = setup(&["<|endoftext|>", "<tool_call>", "</tool_call>"]);
        let special = |text: &str| tokenizer.inner().token_to_id(text);
        let lark = tool_call_grammar(ToolFormat::Hermes, &tools, false, &special).unwrap();
        let constraint = factory
            .compile(&ConstraintSpec::Lark(lark), &tokenizer, &[], false)
            .unwrap();
        let vocab = ByteVocab::from_tokenizer(&tokenizer).unwrap().len() + 3;
        let (text, finished) = generate(
            &constraint,
            &tokenizer,
            vocab,
            |a, step| a[(step * 7919) % a.len()],
            300,
        );
        let _ = finished;
        assert!(tool_call_grammar(ToolFormat::Gemma4, &tools, false, &special).is_none());
        // Whatever it wrote starts with the call marker (control tokens are
        // dropped by decode, so look at the JSON that follows).
        assert!(text.trim_start().starts_with('{'), "{text:?}");
        assert!(text.contains("\"name\""), "{text:?}");
    }
}
