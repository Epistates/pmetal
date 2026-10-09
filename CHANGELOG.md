# Changelog

All notable changes to PMetal will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`pmetal serve` returns a thinking model's reasoning apart from its answer.** Chat completions carry it in `message.reasoning_content`, and when streaming in `delta.reasoning_content` chunks ahead of the `delta.content` ones; `/v1/messages` answers with a `thinking` block (streamed as `thinking_delta`s) ahead of the `text` block. `content` is the answer only, its logprobs are the answer's, and tool calls are parsed from the answer. The split is by token, from the markers the tokenizer has: Qwen3-family and DeepSeek-R1 `<think>…</think>`, Magistral's `[THINK]`, Gemma 4's thought channel and gpt-oss's `analysis` channel. The prompt is read first, so a reply to a chat template that opens the `<think>` block itself (Qwen3.5 with thinking on) starts in the reasoning. `pmetal infer`'s streamed `[thinking]` and `[answer]` labels come from the same splitter (`pmetal_data::stream_format::ReasoningSplitter`). Qwen3.5-0.8B answering "17 + 25" over every endpoint, streamed and not, gives the same reasoning and "42" as the engine's own tokens split
- **`pmetal grpo --loss-type gspo` trains with GSPO** (*Group Sequence Policy Optimization*, arXiv 2507.18071): one importance ratio per completion, the length-normalized sequence likelihood ratio exp(mean over its tokens of log π − log π_old), clipped in [1 − 3e-4, 1 + 4e-4] as in the paper's section 5.1 and averaged over completions (`GrpoConfig::for_gspo`, `ImportanceSampling::Sequence`). A test checks the ratio against the sequences' log-likelihoods worked by hand, with masked prompt tokens left out. Like any clipped ratio it only acts once the policy has moved, so it goes with `--num-iterations 2` or more, and the command says so when it doesn't
- **`pmetal preference` trains a LoRA adapter on preference data** (alias `pmetal dpo`; `PreferenceTrainer` and `KtoTrainer` in `pmetal_trainer::preference`; the `preference` MCP tool, a TUI tab and a GUI page). `--loss` picks DPO, IPO, hinge (SLiC), SimPO or ORPO on prompt/chosen/rejected pairs, or KTO on completions labelled good or bad; `--label-smoothing` turns DPO into Robust DPO for noisy labels. Data loads from JSONL, JSON, Parquet or a Hugging Face dataset ID, as plain strings or chat messages, and goes through the model's chat template as `pmetal train` renders it. Preference training had been library-only, and SimPO, ORPO and KTO sat behind a feature nothing enabled
  - The reference model costs no memory: a fresh adapter starts at zero, so the model before training is the reference, and every example is scored once up front. On a small f32 model the first DPO step's loss is ln 2 to 1e-4, which the tests check
  - Each loss matches its paper and reference implementation on hand-computed examples (TRL's `DPOTrainer` for DPO, Robust DPO, IPO and hinge, including IPO's length-averaged log-probs; the SimPO authors' trainer; TRL's `ORPOTrainer` and `KTOTrainer`). KTO subtracts the KL estimate the paper uses, from mismatched completions in the batch, clamped at zero and kept out of the gradient. Defaults follow the same sources: β 0.1 (2.5 for SimPO, with γ/β 0.5) and a LoRA learning rate of 1e-5
  - Steps accumulate gradients over micro-batches, clip them, follow a warmup-then-cosine schedule and log the loss, both rewards, their margin and the share of pairs ranked correctly
  - Qwen3-0.6B on 48 arithmetic prompts, 12 steps of 4 pairs at lr 1e-4: the reward margin goes from about 0 to 13.2 (DPO), 7.6 (hinge), 0.44 (IPO), 15.6 (SimPO), 0.40 (ORPO) and 14.3 (KTO), with every pair in the batch ranked correctly by step 4 for the paired losses, at 500 to 1,600 tokens/s on an M4 Max. `pmetal infer --lora` loads the saved adapter
- **A new ANE inference engine behind `infer --ane` and `serve --ane`** (`pmetal_metal::ane::lm::AneLm`). The whole model runs on the ANE: several layers per ANE program, each taking a batch of new tokens at once against a KV cache that stays in IOSurfaces between calls, with int8 weights by default (fp16 optional) and the LM head split into pieces sized to the model width. The engine it replaces ran one layer per kernel and kept part of each step on the GPU
  - Each step also checks guesses taken from the prompt (prompt lookup). Checking a few extra tokens costs about the same as computing one, so accepted guesses are free tokens, and the output is identical with or without them
  - Qwen3-4B on an M4 Max: 24.5 tok/s plain, 37.6 tok/s with prompt lookup on a repetitive prompt, about 7 s to load once the system has cached the compiled programs (with the checkpoint in the page cache or on an internal SSD). Attention runs one softmax over the cache's and the new tokens' scores, and a pass carries 16 tokens: attention's cost grows with both, and over a 4096-slot cache it had been 60% of a layer. Greedy output matches the bf16 reference implementation on Qwen3-0.6B (fp16) and on a Qwen3-4B chat prompt (int8 and fp16), and int8 perplexity on Qwen3-4B matches bf16 (NLL 2.287 vs 2.294)
  - **DFlash drafting for the ANE engine**: `infer --ane --draft-model z-lab/Qwen3-4B-DFlash-b16` runs the DFlash draft model on the GPU, reading the hidden states the ANE programs output after the layers it was trained on, and the ANE verifies its 15 guesses in one pass. Output is identical to decoding without it. Qwen3-4B goes from 24 tok/s to 59-137 tok/s (3.1-7.1 tokens per pass on three chat prompts). The draft model runs with 8-bit weights, which guess as well as bf16 in half the memory and draft faster. `serve --ane --draft-model` does the same for every request. `--draft-model` still takes a Gemma 4 MTP assistant everywhere else; a DFlash draft model without `--ane` is refused with a pointer to `pmetal dflash`
  - Output streams token by token in `infer`, and the timing line separates loading from generation
  - Qwen3 only for now; `infer --ane` says why it's using the GPU for anything else
- **Decision models: Clef and Clef-flash answer typed questions about a state in one forward pass** (`pmetal decide`, `pmetal serve` on `POST /v1/systemone`, the `decide` MCP tool, `pmetal_models::decision`). A request names a state (text or any JSON value) and questions that are propositions, named choices or ordered scores. The joint schema head reads the backbone's final hidden states and scores every option of every question at once, with no generation. Records are encoded token for token as the release's reference code encodes them, Python's JSON rendering included, and malformed requests get its error messages. Requests may carry images and videos, read by the backbone's vision tower (see the Qwen3.5-family vision entry below)
  - Checked against the release's PyTorch code: the head matches to 5.7e-6 on logits in f32, 14 records encode to the same tokens and spans, and Clef-flash's probabilities on five requests are within 4.2e-3 of PyTorch in float32, with every choice the same (PyTorch's own bf16 run is 2.8e-3 away)
  - Clef-flash answers a 260 to 480 token request in 330 to 590 ms on an M4 Max, on the native Qwen3.5 engine
  - The `decide` MCP tool takes a `port` to ask a decision model already served there (`start_serve`), as `chat` does, rather than loading the model for every call. Without one it loads `model` once for the call, as before. A request the server refuses comes back as an invalid-params error with the server's message
- **`Linear` can hold its weight packed for MLX's quantized matmul** (`Linear::quantize`, `LinearQuant`), as `mlx.nn.QuantizedLinear` does, with the packed weight's `scales` and `biases` beside it in the parameter tree. A packed layer doesn't train, but a LoRA adapter on it does and computes what it would on the unpacked weight, which is QLoRA; merging the adapter unpacks the layer
- **Qwen3.8-Flash-Next (`model_type: qwen4_exp`) runs**, text only: the hybrid of Gated DeltaNet and full attention behind a sparse-attention indexer, with four hyper-connection streams, a 512-expert MoE and 51B parameters of hashed n-gram embeddings. `infer`, `serve` and `--experts-dir` drive it like any hybrid model. The vision tower and the MTP head are not loaded yet
  - It matches transformers to 2.1e-6 on a small fp32 model with every feature on, over the whole prompt and through the caches one token at a time
  - The n-gram table is read row by row from the checkpoint rather than loaded (a token reads 16 rows of it), and the NVFP4 release's experts stay packed, 68 GB rather than 241 GB unpacked. That puts the NVFP4 release at about 78 GB resident, which fits a 128 GB Mac
  - The bf16, FP8 and NVFP4 releases' layouts are checked against the loader from their safetensors headers: every tensor is loaded or skipped by name
- **Qwen3.8 (`Qwen/Qwen3.8-27B` and its FP8 release) is checked end to end against transformers.** Greedy replies match transformers token for token with thinking on and off, the first token's top five logits agree to bf16 rounding, `--mtp` accepts 61% of its drafts with unchanged output, and the FP8 release loads with its weights kept in 8 bits. Its chat template, reasoning-effort system prompt included, renders exactly as transformers renders it
  - Both engines read a Qwen3.5-family `config.json` one way now, the way transformers resolves it, and refuse what they can't run by name (an unknown `output_gate_type` or `layer_types` entry, `attn_output_gate: false`, a non-default RoPE type) instead of running it with a default
  - `output_gate_type: "sigmoid"` is supported on both engines; `"swish"`, what the Qwen3.6 and 3.8 releases ship, is SiLU
  - A new parity suite checks both engines against transformers on two small fp32 models covering these keys, over a whole prompt and one cached token at a time, along with the MTP head: every check lands within 1.2e-5
- **Qwen3.5-family models see images and videos** (Qwen3.5, 3.6 and 3.8, and the Clef decision models built on them). The vision tower, the image and video preprocessing, the 3-D positions a prompt with media gets and the merge of vision features into the prompt are ported from transformers, on both text engines, prefill and cached decode
  - `pmetal infer --image <path>` (repeatable) on the native engine. Qwen3.5-0.8B prints the same sentence as transformers' greedy generate for a generated picture, and Qwen3.8-27B describes it correctly in bf16
  - `pmetal infer --video <frames dir>` (repeatable, after the images) with `--video-fps` for the rate the frames were extracted at (24 when not given, as the reference assumes, with a warning, since frames extracted at the model's own 2 per second would then mostly be dropped). The directory's image files are the frames, in natural file-name order; they are sampled and time-stamped by the same code as `decide`'s videos, and only the sampled frames are decoded. Video files are not decoded, so extract the frames first. Qwen3.5-0.8B says a red ball in 16 generated frames moves from left to right, and from right to left when the frames are reversed
  - `/v1/systemone` and `pmetal decide` take a record's `images` (base64, data URIs, and file paths in `decide` only) and `videos` (a list of frames, or `{"frames": [...], "fps": N}`), placed in the prompt as the release's reference code places them. Clef-flash with one image, two images and a six-frame clip lands within 9.2e-3 of the reference's probabilities in float32, every choice the same; the reference's own bf16 run is 1.23e-2 away
  - Preprocessing matches the processor transformers loads by default byte for byte: its torchvision resize rounds its taps differently from Pillow and moves up to 1% of output bytes by one or two levels, so `pillow_resample` gains that variant too. Pixel values, grids, sampled video frames, timestamps and prompt tokens are equal to the reference's
  - On a small fp32 model with an image and a three-frame video, vision features match transformers to 8.1e-6, positions exactly, and logits to 6.8e-6 over the prompt and six decoded tokens, where decoding resumes 16 positions short of the prompt's length
  - `pmetal serve` takes them in chat requests (next entry)
- **`pmetal serve` answers chat requests that carry images and videos**, images and videos for the Qwen3.5-family vision models and images for Llama 3.2 Vision. `/v1/chat/completions` takes the OpenAI content parts, an image as an `image_url` part holding a `data:image/...;base64,` URI, plus a `video` part holding a video's frames; `/v1/messages` takes base64 `image` blocks. Both stream or not. Each image goes to the model's own chat template as an item of its message, so its placeholder lands where the reference processor puts it, and the checkpoint's processor expands it to its tokens. The vision tower loads with the first request that needs it
  - Qwen3.5-0.8B answers a picture and a question with the same 96 tokens as `pmetal infer --image`, token for token, from the same prompt tokens, even though the server decodes on the other engine; the text is the same over both endpoints, streamed or not, with continuous batching on
  - Llama 3.2 11B Vision (bf16) answers a picture and a question with transformers' prompt tokens and its greedy reply, token for token to the end of the turn. Its chat template now renders as transformers renders every template, with `trim_blocks` and `lstrip_blocks`, which removes a stray newline after `<|begin_of_text|>` from every Llama 3.2 Vision prompt; and its vision tower and text model now run a bf16 checkpoint in bf16, where f32 pixels had promoted both to f32
  - On a small fp32 model, the server's cached decode after an image or a two-frame video matches recomputing the whole sequence at every step to 1e-4 in each token's log-probability; decoding one position off fails that check
  - A prompt with media runs on the single-request path and never touches the prefix cache, since both key a prompt by token ids alone and an image's tokens are the same placeholder id whatever it shows; a dense model like Llama 3.2 Vision would otherwise have served one image's cached prefix for another
  - Remote URLs, file paths and uploaded file ids are refused with a 400 that says so (the server fetches and reads nothing on a client's behalf), as are video files, audio and `mm_processor_kwargs`. A model that can't read images answers with a 400 naming it instead of dropping them; Gemma 4 is among these for now. `detail` is accepted and has no effect, since the checkpoint's processor sizes every image
  - The body limit is 64 MiB and now holds: axum's own 2 MiB cap on JSON bodies sat inside the outer limit and refused any request carrying a photo, on `/v1/systemone` too. A 13 MB base64 photo goes through
  - Message content may also be a list of text parts, or null for an assistant turn that only calls tools, on every model
- **`pmetal dflash` runs DFlash 2 drafts**, such as `z-lab/Qwen3.8-27B-DFlash2` for Qwen3.8-27B on the native Qwen3.5 engine. A DFlash 2 draft wraps the attention and the MLP of each of its sliding-window layers in two-tap dynamic convolutions, and guesses with a candidate selector that walks one path through each position's 16 likeliest tokens rather than taking each position's argmax. The checkpoint's declared architecture says which generation it is, and the loader refuses a checkpoint with a tensor it has no place for or a parameter it leaves unfilled. Python's `DFlashGenerator` loads them too
  - Checked against the reference implementation's PyTorch model on a small fp32 draft, over five drafts in a row whose context outgrows the window: hidden states and logits agree to 1.4e-6 and 4.3e-6 and the selector picks the same tokens. Ten deliberate faults in the convolutions, the selector, the window, causality and the RoPE base each fail the check
  - Qwen3.8-27B in bf16 on an M4 Max, four chat prompts with thinking off: 7.61, 4.40, 2.60 and 3.15 tokens per verify step on Fibonacci code, a C hash map, a short story and an explanation for a child, at 53.4, 30.2, 17.6 and 20.9 tok/s against 9.1 tok/s for `pmetal infer` (5.9x, 3.3x, 1.9x, 2.3x). The reference implementation's decoder takes 7.61, 4.47, 2.58 and 3.19 on the same prompt tokens, and accepts exactly as many tokens as pmetal at every step until the two targets round a near tie differently: on the Fibonacci prompt that is every step, with the same tokens
  - The output is greedy decoding's: the same tokens on the Fibonacci prompt and on a 2,753-token prompt that runs past the draft's window, and on the other three the same until a token where the target's two likeliest logits are within two bf16 steps of each other (one is an exact tie), which a block-shaped forward and a one-token forward can round either way
  - `--compare-greedy` also decodes the prompt one token at a time with the same target, and reports the speedup, whether the tokens match, and where they don't, the target's two likeliest tokens there
  - First-generation drafts now read `block_size` from `dflash_config` first and honour `layer_types`, `sliding_window` and `is_causal`, as the reference does. `z-lab/Qwen3-4B-DFlash-b16` decodes the same tokens as before
- **`pmetal upload <PATH> <OWNER/NAME>` publishes a model directory to the Hugging Face Hub**: a fused model, an adapter, a quantized export. It creates the repository if needed (`--private`), commits the directory once (`--revision`, `-m`, `--create-pr`), and skips `.git`, `.DS_Store`, AppleDouble `._*` files and anything matching `--exclude`. It authenticates with `HF_TOKEN` or the token `hf auth login` stored
  - `pmetal_hub::upload_model` now runs on `hf-hub`'s upload support, which asks the Hub how each file must be stored and streams large ones to Xet storage. The hand-written client it replaces, which nothing called, sent every file under 10 MiB inline (a LoRA adapter's weights included), read large files whole into memory, and had no Xet path. It takes `UploadOptions` and an optional token and returns the repository and commit URLs
  - A fake Hub on loopback checks the requests an upload makes: the repository settings, the token, and which files are offered and committed, byte for byte
- **Granite 4.0 and 4.0-H run, and so do the Granite 3.x MoE models** (`model_type` `granitemoehybrid`, `granitemoe` and `granitemoeshared`, on the `Granite` architecture beside plain `granite`). The 4.0-H models interleave Mamba-2 layers with attention layers that carry no positional encoding, and the H Tiny and H Small releases add 64 and 72 routed experts beside a shared MLP; the 4.0 models without the H are attention in every layer, with RoPE. The Mamba-2 layers keep their conv and SSM state in the caller's `MambaCache`, so `infer`, `serve`, LoRA training and continuous batching (each slot with its own state) work as they do for the other hybrids
  - Five small fp32 models, one per combination a release uses (hybrid with experts, hybrid without, all attention with RoPE, `granitemoe`, `granitemoeshared`), match transformers to 7.2e-7 on every layer's output and 1.8e-7 on logits, over the whole prompt and through the caches one token at a time, with the Mamba scan crossing chunk boundaries. Eight deliberate faults (a per-group gated norm, no skip term, state dropped between chunks or between calls, RoPE on a NoPE model, no shared MLP, router weights not renormalized, gate and up swapped) each fail the check
  - `ibm-granite/granite-4.0-h-350m` in fp32 gives the first token's top five logits within 1e-5 of transformers', in the same order, on a plain and a chat prompt, and its final hidden states over 640 tokens agree to 2.6e-6 relative. In bf16 its greedy chat replies are transformers' word for word
  - Every released `granite`, `granitemoe` and `granitemoehybrid` checkpoint in the 3.x, 4.0, 4.1 and 4.2 lines, and the MLX conversions of 4.0-H, puts every tensor its safetensors headers list in place, checked from the headers alone. Both layouts load: the released one, and the MLX conversion's split experts, renamed dense MLP and `[C, K, 1]` conv
- **Qwen3.5, 3.6 and 3.8 run static YaRN** (`rope_parameters.rope_type: "yarn"` in `config.json`, or a legacy `rope_scaling`), the long-context setting their cards give for inputs past 262,144 tokens, on both engines and the bundled MTP head, prefill, cached decode, tree verification and prompts with media. The frequencies are transformers' `_compute_yarn_parameters` (defaults filled in, `attention_factor` from the factor or the `mscale` pair), and the attention factor scales only the rotated quarter of each head, as transformers' `cos` and `sin` carry it. The context window the output budget is bounded by becomes `original_max_position_embeddings * factor`. The CPU hybrid engine refuses it by name
  - A third fixture of the Qwen3.5 parity suite carries the card's YaRN block shrunk to an original length of 16, with the prompt running to 70: both engines and the MTP head match transformers within the suite's 5e-5 over the prefill and every cached step. Dropping the attention factor, applying it to every channel, or keeping the unscaled frequencies each fails it
- **Dense Qwen3 runs YaRN on the native engine** (`pmetal infer`, `serve`), from the `rope_scaling` block the Qwen3 card gives for contexts past 32,768 tokens (`factor` 4) or a `rope_parameters` one. It had run every Qwen3 checkpoint with plain RoPE whatever its config said, which for a YaRN config such as DeepSeek-R1-0528-Qwen3-8B's is wrong at every position, not only past the original window. A `rope_type` the engine does not know is refused by name
  - A small fp32 Qwen3 with the card's block shrunk to an original length of 64 matches transformers' `Qwen3ForCausalLM` to 2.8e-6 on the logits over a 96-token prompt and on every step of a cached decode that starts inside the window and crosses it
- **Chat template keyword arguments reach the template, as transformers' `apply_chat_template(messages, **kwargs)` passes them.** Qwen3.8's `reasoning_effort` (`xhigh`, `medium`, `low`) and `preserve_thinking`, gpt-oss's `reasoning_effort` (`low`, `medium`, `high`) and `model_identity`, and anything else a template reads. `pmetal infer` takes `--reasoning-effort`, `--no-preserve-thinking` and `--chat-template-kwargs '<json>'`, as do the infer job spec, the GUI, the MCP `generate` tool and the Python `infer()`. `/v1/chat/completions` takes OpenAI's top-level `reasoning_effort` and a `chat_template_kwargs` object, streaming or not, and `/v1/messages` maps `thinking` to `enable_thinking`. A level the model refuses is an error (a 400 on the server), and on the CLI so is a control the model's template doesn't have
  - An assistant turn's thinking travels as `reasoning_content` (on `/v1/chat/completions` messages too), which is what templates that render thinking themselves read. A `<think>…</think>` left at the head of `content`, as a thinking model generates it, is split off the way Qwen3's own template splits it, for those templates only
  - 28 renders of Qwen3.8's and gpt-oss's templates over a multi-turn history with thinking, tool calls and tool schemas, at every reasoning effort with preserved thinking on and off, match transformers 5.19 byte for byte, and the level Qwen3.8 refuses is refused

### Changed

- **Native decode keeps its compiled single-token attention graph under scaled RoPE.** The graphs of the dense Qwen3, gpt-oss and Llama 4 engines took a scalar RoPE base and nothing else, so a YaRN or Llama 3-band model (every gpt-oss release, Llama 4 Scout, Qwen3 with the card's YaRN block) ran the per-op path on every token. They now take the period table and the attention factor the per-op path rotates with, through MLX `fast::rope`'s `freqs`. On a small fp32 Qwen3 with YaRN the two paths agree to 1.3e-6 on the logits of every cached step and both match transformers; leaving out the attention factor moves them by 0.25. Decode speed on Qwen3-0.6B and Qwen3-4B with YaRN is the same on both paths within run-to-run noise (about 275 and 54 tok/s on an M4 Max), so on those models the graph saves CPU dispatch that asynchronous pipelining already hid. `PMETAL_DISABLE_COMPILED_DECODE=1` runs the per-op path for comparison
- **A model whose maker publishes no sampling settings generates as transformers would, not with a Qwen preset.** Without a family preset, `pmetal infer`, `pmetal serve` and everything that loads sampling defaults used temperature 0.7, top-p 0.8 and top-k 20 for whatever `generation_config.json` left out, which is Qwen's non-thinking card applied to every model. They now read `generation_config.json` the way transformers' `generate` does and take transformers' `GenerationConfig` defaults for the rest: greedy unless the file sets `do_sample: true` (a temperature without it is ignored, as transformers ignores it), and temperature 1.0, top-p 1.0 and top-k 50 when sampling. A model with no `generation_config.json` decodes greedily. Family presets from model cards still come first, and an explicit request or CLI value still wins
- **`pmetal infer` without `--max-tokens` uses the model's output budget, not 256 tokens.** That is `max_new_tokens` from its `generation_config.json`, else its `max_length` (which counts the prompt, as in transformers), else 32,768 tokens for a thinking model, the length Qwen's cards recommend for most queries, else 256, never past the context window. 256 tokens stopped every thinking model mid-thought. `pmetal serve` gives a request without `max_tokens` the same budget (and takes OpenAI's `max_completion_tokens`), as do the infer job spec, the GUI, the MCP `generate` tool and the Python `infer()`; an explicit value always wins
- **Sampling presets follow each model maker's published settings, by model family rather than by chat template.** Qwen3 (0.6 thinking, 0.7 / 0.8 non-thinking), the Qwen3-2507 and Qwen3-Next Instruct and Thinking models, Qwen3.5, 3.6 and 3.8 each get their own card's values, where one Qwen preset used to stand in for all of them (giving Qwen3 thinking a temperature of 1.0); Gemma 4, gpt-oss, Mistral Small 3.x, Magistral, Phi-4-reasoning, DeepSeek-R1, DeepSeek-V3-0324 and Nemotron Nano 2 get theirs. The family is read from `config.json` and the model's name; families whose makers publish nothing keep `generation_config.json` and the fallback. Cards that name an output length (Qwen's 32,768 for most queries, 16,384 for Qwen3-2507 Instruct, Magistral's 40,960) set the default output budget when `generation_config.json` does not
- **`pmetal serve` fills the sampling a request leaves out the same way**, instead of decoding greedily: the maker's settings for the model and its thinking mode, else `generation_config.json`, else the fallback. Qwen's cards say not to decode thinking models greedily. `"temperature": 0` still asks for plain greedy decoding, with no default penalties
- **`pmetal serve` without `--max-seq-len` sizes the context from the model, not 4096 tokens.** It takes the model's context window (YaRN-stretched when configured), capped so the weights plus a full-length fp16 KV cache per sequence (per slot under continuous batching) fit in 70% of the device's working set; never under 4096. Qwen3.5-0.8B gets its whole 262,144-token window on a 128 GB Mac, and a 255,025-token prompt there finds a needle from its middle. 4096 left no room for a thinking model's answer or for a large photo on a Qwen3.5-family vision model (up to 16,384 tokens each). The serve job spec, GUI, TUI and MCP `start_serve` leave it to the server unless set
- **Thinking follows each chat template's own default unless asked otherwise.** Every prompt passed `enable_thinking: true` when no flag said otherwise, which turned thinking on for Gemma 4 and Qwen3.5-0.8B, whose makers ship it off. Now an unset control stays undefined in the template, as it does in transformers, and `--no-thinking` or `chat_template_kwargs` set it. Whether a model thinks (for the output budget and the sampling preset) is read from its template the same way
- **QLoRA trains the model `pmetal serve` runs, on every architecture, with its base packed on the GPU.** `pmetal train --quantization` now packs the projections of the same model LoRA trains and attaches the adapters to it, where it used to build one of fourteen per-architecture QLoRA copies that dequantized each base weight on the CPU and copied it to the GPU in f32 on every forward pass. QLoRA gets what LoRA has: every architecture the dispatcher loads, sequence packing, gradient checkpointing, the fused optimizer and the forward pass inference runs. The LM head, MoE routers and routed experts stay as loaded. On Qwen3-0.6B (24 conversations up to 512 tokens), against the old path run back to back: NF4 went from about 110 to 325 tokens/s, FP4 from 105 to 430 and int8 from 210 to 405 (unquantized LoRA: 440); peak GPU memory fell from 6.3 GB to 2.1, 1.9 and 2.2 GB (unquantized: 2.6 GB) and the process from about 4 GB resident to 1.5 GB
  - `nf4` is NF4 as the QLoRA paper defines it, with the codes unpacked on the GPU (MLX has no NF4 kernel). Its loss on that set is 2.648 against 2.604 unquantized, the same as the old CPU path's 2.650, and better than MLX's 4-bit affine format at the same 4.5 bits (2.684)
  - `fp4` is NVFP4: E2M1 values with an E4M3 scale per 16 weights and an FP32 scale per tensor (loss 2.660). It was E2M1 scaled per 64 weights, a format no kernel reads. `--quant-block-size` doesn't apply to it
  - `int8` is MLX's 8-bit affine format over `--quant-block-size` weights (loss 2.601), where it was symmetric over the block's absmax
  - `--double-quant` packs NF4's per-block absmax values to 8 bits, 4.13 bits a weight against 4.25, at no measurable cost (loss 2.647). It's refused for `fp4` and `int8`, which have no absmax to pack. A QLoRA run on a GGUF checkpoint is refused with a pointer to the safetensors one
- **`TrainingConfig.warmup_steps` defaults to 0**, as `pmetal train --warmup-steps` and the reference trainers do. It defaulted to 100, which the commands that build a `TrainingConfig` from defaults (`grpo`, `rlkd`, `distill`) inherited; now that their schedules span the run, a 12-step GRPO run would have spent all of it warming up. With `--config`, the `warmup_steps` a YAML file sets is no longer replaced by the CLI default of 0
- **`GrpoLossType` is `Grpo`, `Dapo` (the default) and `DrGrpo`**; `Bnpo` and `Reinforce` are gone (no CLI flag named them). `GrpoTrainer::compute_grpo_loss` returns a `GrpoLoss` with the ratio and clip fraction beside the losses, `train_step` takes the learning-rate setter so it can schedule each of its updates, and `GrpoIterationStats` gains `clip_fraction` and `iterations`
- **`max_steps` counts optimizer steps in the SFT loops**, as it does in the preference loop and the reference trainers. With gradient accumulation it used to count micro-batches, so `max_steps: 100` with 4 accumulation steps stopped after 25 updates
- **`AdamWGroups` is `ParamGroupOptimizer`** (with `ParamGroupOptimizerBuilder::new(kind, lr)`), since its groups run whichever optimizer the config names, and `TrainingLoop::build_optimizer` builds the one every loop variant uses. The compat `AdamW`'s public `lr` array is gone: it was never read; use `set_lr` and `lr()`
- **`--ane-max-seq-len` is the largest context the ANE engine compiles for**, prompt plus output, and defaults to 4096 (was 1024). Each request gets the smallest power of two from 512 that fits it, so short requests stay fast; a longer one recompiles the model once at the larger size. A prompt that doesn't fit is refused with the flag named
- **A Qwen3 `config.json` without `tie_word_embeddings` means an untied head on the native engine**, as it does in transformers and on the `DynamicModel` path. The native engine assumed tied, so such a checkpoint ran with the embedding table as its head and its own `lm_head.weight` dropped. Every Qwen3 release sets the key, so they load as before
- **The `infer --benchmark` helpers have plain names**: `benchmark_trial` in each native engine (`pmetal_bridge::{qwen3_native, llama4_native, deepseek_native, gpt_oss_native}`), `pmetal::native_inference::benchmark_native` returning `BenchmarkTrialMetrics`, and `InferenceRunner::benchmark`. They were named after another tool's benchmark, whose workload shape they reproduce. Behaviour and output are unchanged
- **Default models download from their makers' own Hugging Face repositories**: `Qwen/Qwen3.5-0.8B` for `bench-gdn --model`, the `hybrid-qwen3next` and `hybrid-qwen35-steady` bench presets and the DFlash capture example, and `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` for the `moe-nemotronh` preset. They had pointed at a third-party re-upload; the weights are byte for byte the same

- **Every dependency is at its latest release**, with all lockfiles regenerated from scratch. Majors: `hf-hub` 1.0, `safetensors` 0.8, `base64` 0.23, `pyo3` 0.29, `dirs` 7
  - `hf-hub` 1.0 is a rewrite. Downloads now use Hugging Face's Xet transfer protocol and retry 429 and 5xx responses on their own. The Xet client's per-request INFO logging is held to warnings in the CLI
  - Bundled MLX moved to upstream `main` (`09e67c6`), which adds a Metal fence deadlock fix, lower SDPA memory use at head dims 256 and 512, and a faster gather matmul
- **MSRV**: Raised workspace `rust-version` to 1.91. The 1.89 it declared no longer built: `ordered-float` needs 1.90 and `hf-hub`'s Xet client needs 1.91, verified with `cargo +1.91 check`
- **TensorBoard logging is always built** and writes its event files with pmetal's own writer (`pmetal_trainer::tensorboard::EventWriter`, on `prost` and `crc`). The `tensorboard-rs` crate it replaces had been unmaintained since 2022 and pulled in `protobuf` 2.28 (RUSTSEC-2024-0437) and 15 other crates. The `tensorboard` feature is now a no-op, kept so manifests that name it still resolve. Each metric is now its own tag in one event file (`train/loss`, `train/epoch`, `epoch/loss`, `epoch/perplexity`). `tensorboard-rs` wrote each metric to its own run subdirectory (`log_dir/train/loss/`) under the shared tag `train`, which drew loss and epoch number on one chart
- **The GUI uses bun alone.** The release build installed from `pnpm-lock.yaml` while local builds and preflight used `bun.lock`, so a release could ship frontend dependencies nobody had built locally. `pnpm-lock.yaml` is gone, and `packageManager` in `package.json` pins the bun version CI installs
- **Local builds take far less disk.** The root, GUI and fuzz workspaces share one `target/`. MLX is compiled once per configuration and reused by every `pmetal-bridge` variant, where each variant used to compile its own (~800 MB and several minutes apiece). Dev builds keep line tables rather than full debug info (`CARGO_PROFILE_DEV_DEBUG=true` restores it). A full `just preflight` from clean leaves 19 GB
- **Dependency policy is enforced by `cargo-deny`** (`deny.toml`, `just deny`, in `just preflight` and a new CI job) for both the workspace and the GUI. Licenses are permissive-only; the six MPL-2.0 crates in the tree (`option-ext`, `colored`, and in the GUI `cssparser`, `cssparser-macros`, `dtoa-short`, `selectors`) are named exceptions, so a new copyleft crate fails the check. `just audit` now runs the same advisory check with `deny.toml`'s reviewed ignores

### Removed

- **Kernels that computed outside autograd's graph and that nothing called any more**: `pmetal_metal::FusedDistill` and its MPP twin `MppFusedDistill` (with `FusedHiddenAlign`, the backend's `fused_distill_loss` and `has_distill`), whose SIMD forward also kept one SIMD group's share of the softmax normalizer and so returned a KL about four times off for vocabularies over 1024; `pmetal_mlx::kernels::fast_lora` (its MPP GEMM path and the optimized LoRA forwards) and `pmetal_mlx::kernels::quantized_matmul`; and `pmetal_mlx::moe::{MoELayer, MoERouter, MoEConfig}`, which read routing weights back to the host and rebuilt them as constants. DeepSeek, the one model that held a `MoELayer`, held it only for its experts and routed through its own gate; it now keeps the experts directly, under the same parameter paths. A model or trainer that differentiates its losses goes through MLX's graph, which is what these could not do
- **The per-architecture QLoRA models** (`DynamicQloraModel` and the fourteen `*QloraForCausalLM` types for Llama, Mistral, Qwen 2 / 3, Qwen 3.5, Qwen3-MoE, Gemma, Gemma 4, GPT-OSS, Granite, Llama 4, DeepSeek, Nemotron-H, Phi and Cohere) and `QLoraLinear`. `quantize_base` with `QLoraConfig` and `QLoraScheme`, and `DynamicLoraModel::from_pretrained_quantized`, replace them
- **The code only the per-architecture QLoRA models used**: the six LoRA model stacks they were built from (`Gemma4LoraForCausalLM`, `GptOssLoraForCausalLM`, `Llama4LoraForCausalLM`, `NemotronHLoraForCausalLM`, `Qwen3MoELoraForCausalLM`, `Qwen3NextLoraForCausalLM` and their layers), the standalone adapter layers `LoraLinear`, `DoraLinear` and `LinearAdapter`, `lora_helpers`, the hand-written LoRA backward passes in `autograd` (with `pmetal_trainer::CustomAutogradModel`, which nothing implemented), `sanitize_loaded_weights` and `effective_rank`, GPT-OSS's own LoRA types in pmetal-models (`GptOssLoraForCausalLM`, `GptOssForCausalLM::into_lora`), and `pmetal_mlx::quantization`, the CPU NF4, FP4 and int8 quantizers. LoRA and DoRA live on `Linear` and every architecture trains through `AdaptedModel`
- **`pmetal_metal::FusedTrainingCoordinator`.** Nothing constructed it: the training loops drive `FusedAdamW` through `MlxMetalOptimizer`, and the coordinator only bundled that with a gradient clipper and a cross-entropy kernel behind getters
- **The old ANE inference engine** (`pmetal_metal::ane::inference`, `AneInferenceEngine`) and the `inference_ane` example that drove it. `AneLm` replaced it behind `infer --ane` and `serve --ane`, and nothing else called it. Its sampler moved to `pmetal_metal::ane::lm::sample`
- **`infer --ane-real-time` and `serve --ane-real-time`.** The flag reached only the old engine, so with `AneLm` it did nothing, and the private real-time path it asked for never ran on macOS 27 and failed on earlier releases. It's gone from `InferSpec`, `ServeSpec`, the TUI's Serve form, the GUI and the MCP `infer` and `start_serve` tools, along with `GenerationConfig.ane_real_time` and the runtime's real-time and loopback-chaining probes (`AneModel::evaluate_real_time`, `prepare_loopback_chain`)
- **The static ANE trainer and the code only it used** (`pmetal_metal::ane::{trainer, budget, pipeline, profiler}`, the static kernel generators in `ane::kernel`, and the dynamic `ffn_w2`, `wo_bwd`, `qkv_bwd`, `rmsnorm`, `softmax` and attention-only kernels). `DynamicAneTrainer` replaced the static trainer and runs those layers through its projection kernel; nothing called them. `PMetalError::AneCompileBudgetExhausted`, which nothing raised, is gone too
- **`infer --backend dflash` and `DraftBackend`.** The backend only ever refused to run and pointed at `pmetal dflash`, which is still the way to run DFlash on the GPU (`infer --ane --draft-model` drafts for the ANE engine). It's gone from `InferenceBackend`, `InferSpec`'s backend options and the MCP `infer` tool. `DFlashConfig.draft_backend` had one working value; its other, `DraftBackend::Ane`, returned "not yet implemented"
- **`infer --compiled`, `--metal-sampler` and `--minimal`.** Each picked the same path as `--backend compiled`, `metal-sampler` or `minimal`, and they only took effect with `--backend auto`, so two settings could disagree about one choice. `--backend` is the one way to pick a generation path now. The flags are gone from `InferSpec` (with its `compiled`, `metal_sampler` and `minimal` fields) and the MCP `generate` tool's `compiled` option, and `--backend`'s help no longer lists the removed `dflash`
- **`qwen3_native::{QwenDecodeBackend, canonical_decode_backend, generate_canonical, benchmark_trial_canonical}`.** With the C++ decode path gone the backend had one variant, so the two `_canonical` functions only forwarded to `generate` and `benchmark_trial`, which callers now use directly
- **`pmetal_mlx::offloading`** (`ActivationOffloader`, `GradientOffloader`, `FrozenParameterManager`, `OffloadedEmbedding` and their configs). The module described itself as not yet integrated, and nothing in pmetal used it
- **`pmetal_trainer::SftTrainer`**, a legacy trainer whose `train`, `evaluate`, `save_checkpoint` and `load_checkpoint` only returned "not implemented", along with its `TrainingState` (a second type of that name beside `pmetal_core::TrainingState`) and `lm_loss`. `TrainingLoop` is the trainer
- **`CompressionStrategy::PowerSGD`** in pmetal-distributed. It logged a warning and sent the gradients uncompressed, while the crate README listed it as a working strategy
- **pmetal-trainer's compile and fused-step prototypes** (`explicit_state_compile`, `jit_compile`, `ffi_compile`, `metal_fused` and `lora_trainer`, with `ExplicitStateTrainer`, `CompiledTrainStep`, `CompiledTrainingStep` and `LoraTrainer`). Nothing called them. The three compile modules wrapped their step in a `compile` that returned the function unchanged, so nothing was ever compiled, and `TrainingLoop` already says so when JIT is requested and runs its eager fused path. `metal_fused` and `lora_trainer` were earlier versions of what `TrainingLoop` and `MlxMetalOptimizer` do
- **The library-only RL trainers `GspoTrainer`, `DapoTrainer`, `PpoTrainer` and `OnlineDpoTrainer`, and `reasoning_template`.** None could train a model, and the README listed three of them as working. `GspoTrainer` did not implement GSPO: it reweighted GRPO's token losses and swapped the group mean for a median, with no sequence-level importance ratio. `DapoTrainer` was a set of helpers that `GrpoTrainer`'s DAPO mode (`pmetal grpo --dapo`) already covers. `PpoTrainer` computed GAE and the clipped loss but had no rollout or value model to feed them. `OnlineDpoTrainer` sampled without a KV cache, stopped on token id 2 whatever the tokenizer, and ranked completions by length. The code-execution reward in `reasoning_template` was a stub that scored fenced blocks and substrings
- **The LLaDA-style masked-diffusion trainer** (`pmetal_trainer::diffusion`: `DiffusionTrainingLoop`, `DiffusionConfig`, `forward_process_gpu`, `diffusion_loss_gpu` and the sampler). It trained an ordinary causal LM, whose attention can't see the unmasked tokens to the right of a masked one, so it never learned the bidirectional denoiser LLaDA is. It sat behind the `experimental-trainers` feature, which nothing enabled. DiffusionGemma's block-diffusion training (`pmetal train-diffusion`, `DiffusionGemmaTrainer`) is unchanged, and the README's method table now names it
- **`Adam8bit` and `ScheduleFreeOptimizer`**, which no training loop could use and the READMEs listed as optimizers. `Adam8bit` copied every moment to the CPU and back on each step and quantized the second moment linearly, which loses the small values Adam divides by (8-bit Adam quantizes it on a log-like scale for that reason). `ScheduleFreeOptimizer` replaced Schedule-Free's running average of the iterates with a fixed-weight blend and left out the second moment's bias correction, so it was not the paper's algorithm
- **pmetal-lora's unused adapter variants and helpers**: `QBLoraLinear` and `QBLoraConfig`, the GaLore projector (`GaloreProjector`, `GaloreConfig`, `GaloreParamState` and friends), `ModelPatcher`, `AdapterManager` and the `LoraArchitectureConfig` trait. Each described itself as not yet integrated or had no caller. Q-BLoRA was a standalone layer beside the `Linear`-attached adapters every model trains through. GaLore projected gradients but no optimizer or training loop ever called it, so it saved no memory. The crate README's module and architecture tables, which still listed per-architecture LoRA modules deleted earlier, now describe the crate as it is
- **The separate preference trainers `DpoTrainer`, `SimpoTrainer`, `OrpoTrainer`, the old `KtoTrainer`, `PairedPreferenceTrainer`, and the `experimental-trainers` feature.** `PreferenceTrainer` and the new `KtoTrainer` replace them. The old ones each ran a loop of their own with no learning-rate schedule, accumulation or clipping, and several losses were wrong: ORPO added the summed negative log-likelihood, so on a 200-token answer the odds-ratio term counted for about 1/200th of what the paper weights it; IPO multiplied its margin by β before regressing it to 1/(2β); `DpoLossType::Robust` was a sigmoid-plus-hinge mix that is not Robust DPO; and KTO measured rewards against a reference point of zero instead of the KL estimate. `SimpoTrainer` duplicated SimPO, which `DpoLossType::SimPo` also implemented. `preference_data`'s loaders became `preference::{load_preference_pairs, load_kto_samples}`
- **`pmetal cluster train` and `pmetal cluster serve`.** Both only exited with an error: `cluster train` pointed at `pmetal train --distributed-auto`, which is how to train across the cluster, and `cluster serve` needed per-layer model execution that doesn't exist. `cluster up`, `status`, `bench` and `pipeline-bench` are unchanged
- **The Pixtral, Qwen2-VL and Whisper modules** (`pmetal_models::architectures::{pixtral, qwen2_vl, whisper}`). The docs listed them as implemented architectures, but none could run a released model: Pixtral and Qwen2-VL wrapped the Mistral and Qwen2 text models and dropped the vision weights, and Whisper had no weight loader, audio frontend or decoding loop. Nothing called any of them. CLIP, T5 and the VAE, listed beside them, are the Flux pipeline's components and stay
- **Unused modules in pmetal-mlx**: `attention` (a dispatcher nothing dispatched through), `grouped_gemm_moe` (which copied a whole expert weight per token), `sequence_packing` (pmetal-data packs), `kv_cache::paged` (no attention kernel read its blocks), `quantization::group` with the `QuantScheme::Group*` variants (no GPTQ or AWQ checkpoint could be loaded through it), and in `kernels` the `cross_entropy`, forward-only `metal_cross_entropy`, unfused `metal_norm_lora`, `training_attention` and `swiglu` modules. Training's losses, attention and caches are unchanged
- **pmetal-distributed's `election`, `health`, `collective`, `cloud_bridge` and `solver` modules**, with the error variants and metrics only they would have produced. Nothing started an election, fed the heartbeat monitor, implemented the tree all-reduce or used the cloud bundle export, and `solver` wrapped `layer_assignment` with a made-up latency estimate. The cluster commands, the ring all-reduce and the pipeline harness are unchanged
- **`pmetal_gguf::vec_dot`**, NEON dot products for a CPU matmul pmetal does not have. GGUF weights are dequantized at load and run on the GPU
- **pmetal-mhc's Metal kernels and its `metal` and `cuda` features.** Nothing dispatched the kernels (the mHC layer runs on ndarray), and `cuda` was empty. pmetal's `mhc` feature no longer pulls in `core` and `metal`
- **pmetal-distill's `metal` feature and `is_gpu_available`.** The fused Metal forward they switched on returned each loss as a constant, so it could not train anything (see Fixed), and for any vocabulary over 1024 entries the value was wrong too: its reduction kept one SIMD group's share of the normalizer, which put the KL about four times too high. The losses run on the GPU through MLX
- **pmetal-mlx's FlashAttention training cache**: `compute_attention_gradients`, `AttentionCache` and `differentiable_attention`. Nothing ever called the backward, and against MLX's own attention gradient it was off by 5% on queries and by 40 to 120 times on keys and values for grouped-query heads, while running 1.2 to 5 times slower at 2048 and 4096 tokens. `with_training_mode` and `TrainingContext` remain, as a plain training flag

### Fixed

- **`--cut-cross-entropy` with sequence packing (the default) computed the full logits anyway, and said nothing.** The adapted model had no hidden-state forward with positions, so every packed step fell back to the full-logits loss. It has one now (`DynamicModel::forward_hidden_with_positions`, the trunk `forward_with_positions` runs, on every architecture that takes packed positions), and a test holds cut cross-entropy on a packed batch to the full logits' loss and every adapter's gradient on 16 architectures, with positions that restart and stretch so a forward that dropped them would not match. Every training path now asks one place whether cut cross-entropy applies, and when it can't (NEFTune on an unpacked path, image batches, a model with no usable LM head) the run warns once and uses the full logits; a model that offers a head but no hidden-state forward fails the step instead of switching losses. `nn::value_and_grad` returns the loss function's error rather than a NaN loss that hid it, and the packed path, which adds no NEFTune noise, says so when NEFTune is set
- **Cut cross-entropy used more memory than the full logits it exists to avoid.** Differentiating its chunked forward kept every chunk's logits for the backward pass, in f32, and slicing the head made each chunk's gradient a full-size copy of the head. On Qwen3-0.6B at 4,212 tokens a step took 9.0 GB against 2.6 GB for the full bf16 logits (measured in the machine's GPU queue, gradient with respect to the hidden states). Each vocabulary chunk now runs under a gradient checkpoint and is rebuilt in the backward pass, as the reference implementation (Wijmans et al., arXiv 2411.09009) recomputes its logits; the head is split, so its gradient is one concatenation; the target logit is read out of its chunk instead of gathered from the head; and the chunks are chained, so MLX builds one at a time instead of all of them side by side. The same step now peaks at 1.5 GB (the full logits take 2.6 GB in bf16, 6.4 GB with the f32 softmax that matches its precision), and 3.9 GB instead of 9.0 with an adapter on the head. It is slower: 480 ms against 344 before and 247 ms (bf16) or 297 ms (f32 softmax) for the full logits, the price of rebuilding each chunk in the backward pass and running the chunks one at a time. It also no longer reads the number of valid targets back to the host every step
  - Precision follows the reference implementation: each chunk's logits come out of the matmul in the model's dtype, as the model's own head produces them, and the scale, softcap and logsumexp run in f32. Against an f32 log-softmax over the same tokens, the loss is off by 6e-5 and the gradients by 1.1e-2 (hidden states) and 9.8e-3 (head), the same as the full bf16 logits with an f32 softmax. Before, the head's gradient was off by 1.4e-2 on bf16, and by 7.5e-3 even on an f32 model, where it is now 3e-5. An opt-in test (`PMETAL_CCE_CHECKPOINT`) holds a real checkpoint to this, and reports each variant's memory and time
  - `CutCrossEntropyConfig` lost the fields nothing read (`label_smoothing`, accepted and ignored; `token_chunk_size`; `compute_grad`; `use_online_softmax`), and the hand-written backward and the two unused convenience functions are gone. `CutCrossEntropy::forward` returns the loss
- **Log-probabilities were computed in the model's dtype.** On a bf16 model, serve's `logprobs` and the sampler's top-k, top-p and min-p filters worked on a log-softmax rounded to the logits' own precision, steps of 1/8 for logits between 16 and 32, so reported values came out as -0.125, 0 or -1.5. Log-probabilities are now computed in f32, as transformers' `generate` upcasts next-token logits before scoring them
- **`pmetal infer --image` and `--video` pasted the media's placeholders into the prompt's text instead of giving the media to the chat template.** The template never saw an image or a video item, so `--chat-template-kwargs '{"add_vision_id": true}'` numbered nothing (Qwen's templates write `Picture 1: ` and `Video 1: ` ahead of each), and any template that places its media items another way was bypassed. The media are now items of the prompt's message, ahead of its text, rendered by the model's own template: what `pmetal serve` does with a request's image and video parts, and what transformers' `apply_chat_template` does with `[{"type": "image"}, ..., {"type": "text"}]`. Without a chat template the placeholders still precede the text
  - For a picture and a question, Qwen3.5-0.8B's prompt is the same 321 ids in `infer`, in `serve` and in transformers' processor, and for an eight-frame video the same 197 ids in `infer` and `serve`
  - `infer` and `serve` reply alike to both. They decode on different engines that each round the logits to bf16, and the picture's first token is a tie at that precision (`" A"` over `" The"` by 0.066 in f32, equal in transformers' own bf16 run), which the two break differently; from there on the server agrees with `infer` token for token. The opt-in real-weights test now allows exactly that: wherever `infer`'s token is not the server's argmax it must be within 0.3 of it in the server's log-probabilities, and the server continues from `infer`'s choice
- **Llama 4 decodes from a cache, and attends causally, on the `DynamicModel` path** (training, LoRA, `serve`). Its attention rotated every forward from position 0 and dropped the positions it was given, `forward_with_cache` ignored the cache, so a decode step attended to nothing but its own token, and without a mask attention ran both ways. It now rotates through the shared rope module at the cache's offset or the given positions (`RopePositions`, as the other architectures do), appends to the cache, attends causally unless a mask says otherwise, and scales the NoPE layers' queries by the same positions. The QK-norm runs at `rms_norm_eps`, as transformers and the reference implementation run it, not at 1e-6. A dense layer's `feed_forward.*` weights now load
  - A small fp32 Llama 4 with Llama 3 bands, a NoPE layer and temperature tuning stepping every 4 positions matches transformers' `Llama4ForCausalLM` to 3.6e-7 on the logits of a 20-token forward and of every step of a cached decode after a 7-token prefill. Restarting the rotation or the temperature at 0 in a decode step fails it by 0.3 and 6e-4, and QK-norm at 1e-6 by 1.5e-4. Two records packed into one row now come out as two separate runs
  - The native engine matches the same model to 2.4e-7, compiled single-token graph included. It needed the same QK-norm epsilon, and its prefill had applied no temperature at all: the scale was built by concatenating 0-d arrays, which MLX refuses, and the error was swallowed. It also reads `attn_temperature_tuning: true`, `moe_layers` and `no_rope_layers`, which every transformers Llama 4 config carries
- **The native gpt-oss engine computes gpt-oss.** It had no attention sinks, and it could not run a checkpoint at all: the experts were transposed across every axis, gate and up were taken as the two halves of the fused projection rather than its interleaved columns and then swapped in the GLU, the router ignored its bias and weighted experts by a sigmoid instead of a softmax over the chosen logits, and a prompt longer than the 128-token window attended with the wrong keys on every sliding layer. It now follows the reference implementation and transformers' `GptOssForCausalLM`: each head's sink logit joins its softmax on every path (prefill, cached decode, sliding and full layers, the quantized KV cache), sliding layers keep a ring of `sliding_window` keys and see the banded window `q - window < k <= q`, and the MoE is the biased top-k router with a softmax over the selected logits and the clamped GLU. Both of transformers' dense layout and the stacked one MLX conversions ship load
  - A small fp32 model with seeded sinks, YaRN with `truncate: false`, a window of 4 and a 12-token prompt matches transformers to 3e-7 on the logits from either layout, over a prefill, a cached decode crossing the window and a second prefill chunk starting past it. Dropping the sinks, widening the window by one, swapping gate and up or taking the softmax over all experts each fails it by 0.07 to 0.9
  - Sliding layers now run the compiled single-token graph too, which writes the ring slot and reads the filled ones; it matches the per-op path exactly
  - Packed checkpoints, including the release's MXFP4 experts, are refused by name rather than run: the engine's expert kernels are dense. TurboQuant is refused for gpt-oss, whose sinks its attention kernels cannot express
- **YaRN on Llama, Mistral, Qwen2, Qwen3, Qwen3-MoE, Gemma and the DFlash drafts was not YaRN.** pmetal-mlx's `RopeScaling::Yarn` raised the base by `factor^(d/(d−2))` and also divided every position by `factor`, which is neither YaRN nor any other published scaling, so a YaRN checkpoint (the Qwen2.5 and Qwen3 cards' long-context setting, and DeepSeek-R1-0528-Qwen3-8B as shipped) ran wrong from the first token on every path but the native one. These architectures now rotate through `pmetal_bridge::rope`, the one implementation of transformers' `rope_type`s, and match `Qwen3ForCausalLM` with YaRN to 2.8e-6 on a prefill and a cached decode past the original window. Along the way:
  - Dynamic NTK raised the base at every length; transformers leaves it alone until a forward reaches past `max_position_embeddings`, and so does pmetal now
  - Mistral ignored `rope_scaling` entirely, and Gemma 3 applied its `linear` factor 8 to the sliding-window layers as well as the global ones (transformers scales only `full_attention`)
  - A `rope_type` none of these runs is refused by name when the model loads, instead of running as plain RoPE, and `rope_parameters` (transformers v5's spelling, `rope_theta` inside) is read as well as `rope_scaling`
  - Phi's LongRoPE and Llama 3's frequency bands moved to the shared module with their numbers unchanged; packed (explicit-position) rotation now keeps the input's dtype instead of returning f32 for a bf16 model
  - Continuous batching takes the serial path for any scaled RoPE, since its fused block carries one scalar base, and refuses rather than rotate one as plain RoPE
- **Llama 4 ran without its Llama 3 frequency bands, on both engines.** Scout ships `rope_type: "llama3"` (factor 16), which divides the low frequencies at every position; both the `DynamicModel` path and the native engine rotated it as plain RoPE (the native one with a comment saying chunked attention made it optional). Both now rotate with the shared bands, the native engine leaving its compiled scalar-base decode graph on RoPE layers while the bands are on. Cohere now honours `rope_scaling` instead of ignoring it, and Gemma 4 builds its proportional table from the shared module on both engines and refuses a `rope_type` other than `default` and `proportional` by name instead of running it as one of those
- **The native GPT-OSS and DeepSeek engines rotated with plain RoPE.** Every gpt-oss release ships YaRN (factor 32, `truncate: false`) and DeepSeek-V3/R1 ship YaRN factor 40; the native engines ignored both, DeepSeek keeping only YaRN's softmax term. They now rotate with the release's YaRN frequencies and attention factor from the shared module, and so does DeepSeek-V3.2's indexer. GPT-OSS on the `DynamicModel` path rounded YaRN's correction range although gpt-oss says `truncate: false`, which moved 9 of its 32 frequencies by up to 43%; it now reads the key. Native GPT-OSS decode leaves its compiled single-token graph for the per-op path, since that graph rotates by a scalar base
- **gpt-oss stopped after its reasoning, before its answer.** The stop tokens probed from the vocabulary included `<|end|>`, which ends every harmony message, the `analysis` one included, while the turn ends at `<|return|>`. A harmony vocabulary no longer stops at `<|end|>`
- **A long prompt's prefill chunks shrink as the context deepens, so none runs past the GPU watchdog.** A chunk's attention is one Metal dispatch whose time grows with the chunk's length times the depth it attends over, and macOS stops a command buffer that runs past a few seconds (`kIOGPUCommandBufferCallbackErrorTimeout`). With a fixed 2048-token chunk, the last chunks of a 255k-token prompt did about 7 times the attention of a chunk at the 32k mark. Chunks now stay whole until 2048 times the depth passes 1.6e8 (about 76k tokens) and then halve, down to 128 tokens (`prefill_chunk_len`, used by `pmetal serve`, `pmetal infer` and every cached generator); a restored prefix counts toward the depth
- **LoRA adapted Granite 4's MoE router, and QLoRA packed it.** Adapters and QLoRA's packing leave routers alone, and tell one by its path, but Granite 4 (`granitemoehybrid`) keeps its router's projection at `block_sparse_moe.router.layer`, which matched none of the names checked. With every projection targeted (the default when `target_modules` is empty) the router got an adapter whose top-k picks which experts run, and QLoRA packed it to four bits. It is now recognised as a router
- **Llama 4's MoE layers cut part of the gradient to every adapter below them.** Each layer read the router's gate values back to the host and rebuilt them as constants to scale each expert's input, so the share of the gradient that runs through the routing went nowhere, and every adapter upstream of an MoE layer (its attention and shared expert, and the layers before it) trained on a gradient missing that term: 3 to 5% off on a small model, against finite differences of the loss. Only which expert each token goes to is read on the host now; the gate values stay in the graph, computed in f32 and cast back as `Llama4TextMoe` computes them. A new test adapts every projection of every architecture, takes one training step, and holds each adapter's gradient against a finite difference of the loss along it; reverting the fix fails it on nine of Llama 4's adapters
- **LoRA adapters on an FP8 Nemotron-H checkpoint's Mamba and expert projections never reached the model.** Those projections carry a per-tensor `weight_scale` beside the layer, and with one present the forward multiplied by the scaled weight directly, skipping the layer and the adapter on it: the Mamba `in_proj` and `out_proj` and the shared expert's projections trained against a forward pass that ignored them. A scaled projection now applies its adapter on top of the scaled weight; a test checks the output and `B`'s gradient against the primitive form, and fails with the old path
- **Training carried on after a step failed, and saved the adapter it never trained.** A bridge op that throws inside the gradient step records the exception and returns a placeholder rather than unwinding, so the step finished with a NaN loss and zero gradients (or, with the failure further from the loss, a finite loss and a gradient that was wrong), and every training loop logged it and went on; a Qwen3.5 run logged `loss=NaN, grad_norm=0.00` to the end and saved an untouched adapter. Checking the bridge's error after a step could not have caught it either: every op that succeeds clears that slot, and the op that threw is rarely a step's last. The bridge now also keeps the first error nothing has observed (`pmetal_bridge::check_unobserved_error`), and every trainer (`pmetal train` in all four of its loops, distillation, GRPO, RLKD, the MTP and DiffusionGemma trainers, embedding training and pretraining) reads each step's loss through one check that stops the run with an error naming the step and the failed op, or the non-finite loss or gradient norm. Tests break a model on its third forward pass, by a NaN loss and by an op that throws, and require the run to stop there
- **`--cut-cross-entropy` left an adapter on the LM head untrained, and trained Gemma 2, Gemma 4, Cohere and Granite on the wrong loss.** The step read the head's weight before differentiating, so an adapter on `lm_head` (every projection is adapted when `target_modules` is empty) was a constant to autograd: its update was zero and the loss left its contribution out. The loss was also plain `hidden·Wᵀ` cross-entropy, without the softcap, logit scale or `1 / logits_scaling` those architectures apply after the head, and a checkpoint with a packed head handed CCE its packed `uint32` words as the weight. `DynamicModel::lm_head` now returns the head CCE needs (the dense weight with any adapter folded in, its bias, the scale and the softcap) and the trainer reads it inside the differentiated step. On every architecture's test config, CCE's loss and every adapter's gradient now match the full-logits loss's (dropping the softcap fails Gemma 2), and one CCE step moves an `lm_head` adapter exactly as a full-logits step does, where reading the head early left it unmoved
- **A Qwen 3.5 model with an unmerged adapter decoded as its base model, and one with a packed base aborted.** The one-token decode engine (`qwen3_next_inline`) is built from every projection's stored weight, so with adapters attached it skipped all of them from the second generated token on, and with a QLoRA-packed base it multiplied by the packed words and the process aborted; it was also built once, so weights changed later (a merge) kept decoding with the old ones. Fast paths now ask the layer for its weight through one gate, `Linear::plain_weight`, which refuses for an unmerged adapter, a packed weight or a weight being differentiated; the engine runs only when every projection passes, and is rebuilt when any weight changes. Qwen 3.5's decode projections and shared expert use the same gate. A new test decodes every architecture with an adapter on every projection, unpacked and QLoRA-packed, and requires each step to pick the token the uncached forward picks: Qwen 3.5 failed at its second step and aborted with the packed base before. Llama 4's cached decode disagrees with its forward even with no adapter attached; the test records it as a known defect of its own
- **Differentiating through Gated DeltaNet's cached path failed the step.** The recurrence's inference dispatch takes a custom Metal kernel, which has no VJP, or a chunked path whose `tri_inv` has none either, whenever its caller passes `training = false`, as a cached forward does; under a gradient that raised `[Primitive::vjp] Not implemented for CustomKernel`. The dispatch now asks its inputs: a traced input takes the differentiable ops path, as attention already did. The check is one helper, `pmetal_mlx::kernels::any_traced`, which attention, Gated DeltaNet and Gemma 4's packed experts (whose `gather_qmm` path a traced input now leaves for the exact dequantized one) share. A test differentiates through the inference dispatch at 6 and 48 tokens and matches the ops path's gradients; without the check it fails with that error
- **Gemma 4, the DFlash draft models and Llama 3.2 Vision promoted bf16 to f32 through `tanh`.** `compat::ops::tanh` was `2·σ(2x) − 1` against an f32 constant, so a bf16 input came back f32: Gemma 4's softcapped logits (both engines) and the DFlash drafts' came out in f32, and each of Llama 3.2 Vision's tanh gates turned its residual stream to f32 from the first gated layer on. Each now runs in the model's dtype, as the reference implementations do. The same file held other ops that computed something other than their MLX namesakes, which no model reached with an input that showed it: `round` rounded halves up instead of to even, `floor` went through an i32 cast (so values past ±2³¹, inf and NaN came back wrong, and `ceil` and `round` with it), `log1p` was `log(1 + x)`, `any` ignored `keep_dims` and reduced the wrong axes when given several, `pad` ignored its mode, `tile` repeated each element instead of the whole array, and `split` cut an axis it didn't divide into uneven pieces. They all call MLX's own ops now, and `pad` takes a `PadMode` (constant, edge, reflect or symmetric)
- **`pmetal distill` could not load its teacher in online or offline-generate mode, nor could the app.** The teacher was loaded with rank-0 adapters to mean "no adapters", which adapter attachment rejects, so the run stopped before its first step. The teacher is now loaded as a plain frozen model, as `pmetal rlkd` loads its own, and the distillation trainer takes any teacher that can run forward
- **Distillation trained the student on the hard labels only.** On a Metal machine, the KL, reverse KL, Jensen-Shannon, soft cross-entropy and hidden-state losses ran a fused Metal kernel and returned the result as a scalar rebuilt from host memory, which autograd treats as a constant, so the teacher-matching term gave the student no gradient at all and `alpha` only scaled the label loss. Each loss is now one MLX expression: log-probabilities at temperature `T` from `log_softmax` over f32 logits, reduced per token in the graph, with `Distiller::compute_loss` applying the `T²` of Hinton et al. (2015) once. Autograd now matches finite differences of every loss, and the forward-KL gradient matches its closed form `T · (q − p)` at `T` = 1, 2 and 4 through the trainer's own path, labels and mask included. Distilling Qwen3-4B into Qwen3-0.6B on soft targets alone (`--alpha 1`, `T` = 2, 8 samples, 3 epochs), every sample's loss now falls, the mean from 1.91 to 1.44; before, each sample scored the same in every epoch, because the student never changed
  - Jensen-Shannon's gradient was also wrong wherever teacher and student gave a token the same log-probability, tenfold on a test case: its `log(exp(a) + exp(b))` sent the whole derivative of a tie to the student. It now splits it as the true derivative does
  - `DistillLoss::compute_masked` ignored its mask and returned the mean over every token; it now averages over the unmasked ones
- **Qwen3.5 QLoRA did not train at 2048 tokens or more.** In training mode its attention handed Q, K and V to the Metal FlashAttention kernel, which builds its output from a Metal buffer, so autograd saw a constant and the adapters on `q_proj`, `k_proj` and `v_proj` got nothing. With Qwen3.5's 256-wide heads, which the kernel has no variant for, the call failed inside the gradient step instead, and the run carried on with a NaN loss and a zero gradient: Qwen3.5-0.8B at 2112 tokens logged `loss=NaN, grad_norm=0.00` and saved an untouched adapter. It now trains (loss 3.81, 3.55, 3.41 over three steps, gradient norm 4.8). That path also ignored the mask it was given, so a packed batch attended across its documents. Attention a gradient flows through now always uses MLX's attention, whose backward is part of autograd, under the caller's mask, and `fused_sdpa` checks this from the inputs themselves: before, attention with no mask under autograd could still land on a Metal kernel whenever the cached backend choice for its shape was one, and so train on nothing. A frozen teacher's forward keeps the Metal kernels. A training step at 2048 tokens now gives the same adapter gradients as a reference forward, with finite differences agreeing, for Qwen3.5 LoRA and QLoRA at 128- and 256-wide heads
- **Reading an array back from MLX returned the wrong elements when the array was a transpose, an inner-axis slice or a broadcast.** These are views that share another array's buffer with different strides, and `as_slice`, `data_ptr` and `to_f32_vec` walked that buffer from the start in storage order, so a transpose came back untransposed, a column slice came back as a run of whole rows, and a broadcast read past the end of its buffer. They now pack a view into row order before reading it, and still read an ordinary array in place. Distributed training all-reduced every Linear and LoRA gradient in transposed order, since MLX hands back the gradient of a weight used as `w.T` as a transposed view, so multi-Mac fine-tuning trained on scrambled gradients. The TurboQuant KV cache compressed prefill keys and values with heads and positions interleaved, and computed multi-token attention from them. `/v1/embeddings` with CLS pooling returned the first input's token *b* as the embedding of input *b*. Qwen3.5 LoRA and QLoRA training on sequences of 2048 tokens or more read the values in the wrong layout in its Metal flash-attention path. `as_slice` also checks the dtype now, where it used to read a bf16 array as f32 and run off the end of it: this was the case for serve's `logprobs` on models with bf16 logits, `/v1/embeddings` with CLS or last-token pooling, the Llama 4 expert weights, the native Qwen3 engine's key-channel ranking, and the GPU path of the distillation losses, which now read f32 copies
- **`pmetal_bridge::compat::nn::gelu` computed the sigmoid approximation of GELU.** It was `x·σ(1.702·x)`, which is what `quick_gelu` means, not the exact erf GELU that MLX's `nn.gelu` and PyTorch's `nn.GELU()` compute, and `nn::gelu_approximate` was the same sigmoid formula under the name of the tanh one. Every architecture already called a variant by its explicit name, so no model ran the wrong one, but any new caller of `gelu` would have. `gelu` is now exact, the sigmoid one is `gelu_fast_approximate`, and `gelu_approximate` is gone. The tanh approximation also computed `x³` in bf16 on a bf16 model; it is computed in f32 now, like the rest of the activation
- **A chat template that embedded tool schemas never rendered.** The template engine had no `tojson` filter, so the Qwen3 to Qwen3.8 and Llama 3.1 templates, and any other that writes a tool as JSON, failed whenever a request carried tools, and the prompt fell back to a built-in formatter instead (for the Qwen3.5 family, one that writes tools in the format of an older generation). `tojson` is now Python's `json.dumps` as transformers installs it, and templates get `{% break %}` and dicts in insertion order, as in transformers. Tool-call arguments sent as a JSON string, as OpenAI clients send them, are decoded to the object the templates iterate
- **LoRA adapters on Nemotron-H's projections and on Qwen 3.5's MLPs never reached the model.** Nemotron-H's attention, MLP and Mamba projections and Qwen 3.5's dense MLP and shared expert multiplied by their weights directly instead of through the layer, so an adapter on any of them trained against a forward pass that ignored it. They now go through the layer, and Qwen 3.5's decode fast paths, which concatenate the raw weights, step aside for a layer that carries an adapter. A test now nudges every adapter on every architecture and requires the logits to move
- **Every LoRA step on DeepSeek and Nemotron-H mixture-of-experts models produced a NaN loss.** Their routers handed the expert gathers indices that still carried a gradient path, which MLX refuses to differentiate; the indices now stop the gradient, as every other router's already did
- **Gradient checkpointing differentiated every frozen weight inside Granite, Granite 4.0-H, Llama 4 and DeepSeek layers.** A module held in an `Option` or a `Vec` reported all its parameters as trainable, frozen base weights included, and a checkpointed layer computes a gradient for whatever it is told is trainable. It now reports its children's trainable set
- **Adapters went on mixture-of-experts routers and routed experts, where they were never applied.** The routed experts run through stacked expert kernels that read their weights directly, so GPT-OSS's experts, named `gate_proj`, `up_proj` and `down_proj`, took adapters whenever the MLP was targeted and trained nothing. Routers and routed experts are no longer adapted
- **Fine-tuning Qwen 3 and other reasoning models on chat data taught them a turn their template never writes.** Qwen 3's chat template opens every final assistant turn with a `<think>` block, empty when there is no reasoning, and serving prompts with that empty block in non-thinking mode. Training rendered conversations with a hand-written ChatML formatter that left it out, so the model learned to answer straight after `<|im_start|>assistant`, where it puts nearly all its probability on `<think>`: Qwen3-0.6B's first answer token cost 21 nats, and the first-step loss on a short-answer dataset was 6.22 where the template's own rendering gives 2.00. A conversation ending in an assistant turn is now rendered by the model's template whenever that template writes a `<think>` block, with the loss starting where generation would: after the empty block, or after the opening tag when the answer carries its reasoning. The inflated loss is also why 4-bit QLoRA appeared to start a nat below the unquantized model on such data: the quantization noise softened predictions the model was confidently wrong about, while every token it knew got worse
- **`pmetal embed-train --loss cosent` did not compute CoSENT, and returned infinity on any batch that mixed labels.** It compared each row's similarities to every other row's embedding through a mask that only counted pairs where both labels were 1, so a row labelled 0 had no positive and contributed about 1e9, and its count of positive pairs was taken before the diagonal was masked out. Its softplus, `log(1 + exp(x))`, also overflowed to infinity and NaN gradients once scores were about 88 apart. It is now CoSENT as Su Jianlin defined it and sentence-transformers' `CoSENTLoss` computes it: log(1 + Σ exp(λ(cos_i − cos_j))) over every pair of rows whose labels say i is less similar than j, with λ = 1/`--temperature` (the default 0.05 is the reference λ = 20), so binary and graded labels both work and a batch with one label has loss 0. The loss is one log-sum-exp over the pairs, so it stays finite however far apart the scores are. Tests check a graded batch, mixed binary batches and a batch 200 apart against hand-computed values; the old loss gave infinity on the first two. all-MiniLM-L6-v2 on 64 labelled question pairs, 16 a batch: the loss goes from 4.08 to 0.48 over four epochs
- **GRPO and RLKD rollouts were random tokens after the first.** The rollout generator took each decode step's logits from index 1 of a batch axis with one entry and sampled from whatever that read, so every completion past its first token was noise (Qwen3-0.6B continuing an arithmetic question wrote ` Channel/owl Obviously多位bereった…`) and GRPO trained on rewards for noise, which rarely vary within a group, so most of its updates were zero. It now samples each step from the position it just computed and stops at `max_new_tokens` (it made one token more), and the speculative path checks draft i against row i of the verifier's logits (it read a column across positions). A test drives both paths with a stand-in model whose next token is always one more than its input. On Qwen3-0.6B, three arithmetic prompts with 8 completions each, a reward for the share of digits and 2 updates per batch (`tests/grpo_real_model.rs`, ignored without a model), the mean reward goes from 0.18 to 0.33 over the first three batches; clipping changes 6% of tokens on the first batch's second update with DAPO's ±0.2 range and 25% to 69% with GSPO's 3e-4/4e-4, and the learning rate follows the cosine over the 12 planned steps. The prefix-cache path, which no caller reaches since each rollout builds a new generator, still starts decoding without sampling a first token
- **Adapters from `pmetal train` (and the TUI's and GUI's training) did not record their base model**, where `grpo`, `distill`, `rlkd` and `preference` adapters did: the CLI kept its own copy of the adapter-config writer that took a base model, and the orchestrator's dropped it. There is one `orchestrator::save_adapter_config` now, and every adapter's `adapter_config.json` names its `base_model`
- **The MCP docs listed 54 tools for a server with 53**: `estimate_model_memory` is a helper behind `model_fit` and `memory`, not a tool. A test now checks the docs' tool table and stated count against the server's `#[tool]` methods
- **GRPO took one optimizer update per batch of completions, so the policy ratio was always exactly 1 and clipping never engaged.** `pmetal grpo --num-iterations N` (μ, TRL's `num_iterations`) now takes N updates on each batch, against log-probs captured from the policy that generated it before the first one, so from the second update the ratio measures how far the policy has moved and the clip bounds it. Each step reports the share of tokens clipping changed. On a small model with a clip range of 0.01, one update clips nothing, a second clips tokens, and its weights differ from the same two updates with clipping out of reach. GRPO also now follows the learning-rate schedule (it ran at the peak rate throughout), honours `max_steps`, clips the gradient to `max_grad_norm`, and reports the KL and policy loss of the policy being trained rather than the generating one's KL and the total loss
- **GRPO's loss types computed something other than their names.** `DrGrpo` averaged each completion over its own length, which is the original GRPO loss; Dr. GRPO divides by a constant, completions × the maximum completion length. `Bnpo` and `Reinforce` were both the token-level mean. The loss types are now `Grpo`, `Dapo` (token-level, the default, as in TRL's `GRPOTrainer`) and `DrGrpo`, each checked on a hand-worked batch of a 2- and a 4-token completion (0, 1/3 and 1/4), and `--loss-type grpo|dapo|dr_grpo|gspo` picks one; `dr_grpo` also stops dividing advantages by the group's standard deviation, as the paper does. The DAPO recipe (`--dapo`) set the lower clip ε to 0 rather than the paper's 0.2, which left no lower bound on how far a token with a negative advantage could be pushed down in one update, and its overlong penalty and dynamic sampling only ran when the loss type was DAPO; they are now settings of their own (`dapo_overlong_penalty: Option<f64>`). The GUI's GRPO page ignored `dapo`; it has a DAPO checkbox, a loss type and updates per batch now, as do the TUI form and the MCP `grpo` tool
- **Every training loop trained with AdamW whatever optimizer the config named, and SGD, Lion and Adafactor did not exist.** `TrainingConfig.optimizer` was read by nothing. `pmetal train`, `distill`, `preference` and `grpo` now take `--optimizer adamw|sgd|lion|adafactor` (YAML configs take `training.optimizer`), as do their job specs, TUI forms, GUI pages and MCP tools, and the SFT and LoRA loops, distillation, the preference loop and GRPO run the one named (`pmetal_trainer::optimizer`). Each follows its paper with that paper's defaults: SGD with momentum 0.9 and L2 weight decay; Lion with betas (0.9, 0.99) and decoupled weight decay, where the Lion paper recommends a learning rate 3-10x smaller and a weight decay 3-10x larger than AdamW's, which the flag's help repeats; Adafactor with factored second moments for matrices, ε₁ 1e-30, updates clipped to RMS 1 and the decay 1 − t^-0.8, on the scheduled learning rate as for fine-tuning (no relative step, no parameter scaling, no first moment). Hand-worked examples check each rule, and a test trains two steps of each on the SFT loop and compares the weights with a replay through that optimizer and through every other one. The Metal fused optimizer is an AdamW kernel, so the other three train on the standard loop
- **The learning-rate schedule never reached the optimizer on most training paths.** Warmup, cosine decay and the adaptive controller's reductions were computed and logged, but the parameter-group optimizer behind the standard, packed (the default) and fused SFT loops and distillation, and the `preference`, `grpo` and `rlkd` commands, set a learning-rate field the update never read, so every one of them trained at its peak rate from the first step to the last. Only the Metal fused AdamW path scheduled. `AdamW::set_lr` is now the one way to set the rate, and a test sets it to zero and checks that nothing moves
- **Embeddings and the LM head trained at a fixed 5e-5 under AdamW, whatever `--embedding-lr`, the base rate or the schedule said.** The bridge's AdamW sent any parameter whose name held `embed` or `lm_head` to a hidden embedding rate of its own, which reached the embedding group of LoRA training and every parameter of `pmetal pretrain`. It now trains every parameter at the rate it is given and only exempts bias, norm and scale parameters from weight decay; separate embedding and LoRA+ rates are the trainer's parameter groups, which keep their ratio to the scheduled rate. `AdamWBuilder` also dropped the betas and epsilon it was given
- **The SFT and LoRA loops ran their warmup and decay over 10,000 steps whatever the run's length.** Without `max_steps` the schedule assumed 10,000 steps, so an epoch-based run of a few hundred steps still had most of its learning rate at the end, and `warmup_ratio`, `min_lr`, `cosine_num_restarts`, `wsd_stable_ratio` and `polynomial_power` were ignored. The schedule now spans the run's optimizer steps, counted as the reference trainers count them: the dataloader's batches per epoch times the epochs, over the gradient accumulation (or `max_steps`), with `warmup_ratio` taking ceil(ratio × total) steps, and it advances once per optimizer step rather than per micro-batch. The run logs the plan: `pmetal train --optimizer lion` on Qwen3-0.6B with 32 examples, 2 epochs and 4 micro-batches per step logs `LR schedule: Cosine over 16 optimizer steps (2 epoch(s) of 32 micro-batches, 4 per step), 2 warmup steps, peak lr 2.00e-5`, and its learning rate goes 0, 2.0e-5, 1.9e-5, 1.43e-5, 1.0e-5, 3.8e-6, 9.9e-7 at steps 1 to 60, the cosine's values at those optimizer steps, while the loss falls from 2.26 to 1.76. The preference loop and distillation share it (`LearningRateScheduler::for_training`, `total_training_steps`), and the first step of the packed and fused loops and of distillation now runs on the schedule's first rate instead of the peak
- **The Granite 3.x MoE models loaded and computed noise, and Granite 4.0 and 4.0-H would not load at all.** Any architecture whose name contained "Granite" went to plain Granite, which had no experts, so a `granitemoe` checkpoint loaded with every expert dropped as a name it didn't know. Every 4.0 and 4.0-H config failed to parse, since its `layer_types` say `"mamba"` and `"attention"`, and a hybrid that got past that would have run a Mamba-2 stand-in with no conv, no scan and no state. The Granite loader now takes all of a checkpoint or none of it: a tensor with nowhere to go, a parameter left unfilled or a shape that disagrees is an error naming it. Granite models this doesn't compute are refused by name instead of routed somewhere plausible: the sliding-window `granite_swa` and `granitemoe_swa`, `granite_switch`, and the speech and vision wrappers, along with any Granite config that sets `rope_scaling`, an activation other than SiLU or a `time_step_limit` other than the default
- **Granite QLoRA trained a model that couldn't see token order.** Its attention skipped RoPE, on the grounds that the base model's RoPE was a stub, which it hasn't been for a long time, so an adapter was fitted without positions and served with them. It also built the MoE and hybrid families with the stand-in Mamba layer and no experts; those are refused now, and LoRA trains them. Its cached forward ignored the cache, so it no longer reports KV-cache support
- **Sequence packing trained Qwen3.5 / 3.6 / 3.8, NemotronH and other models with recurrent layers across sequence boundaries.** Packing is on by default, and the block-diagonal mask it relies on only reaches attention: a gated-delta-net or Mamba layer carried the first packed sequence's state into the second, so the second trained on context it never had. `pmetal train` now trains these models on unpacked batches and says so at info level. Models report it through `TrainableModel::has_recurrent_layers`, which for the dispatcher's models is whether they need a recurrent cache to decode
- **`pmetal train --quantization fp4` and `--quantization int8` trained on NF4 weights.** The scheme reached `QLoraConfig`, but `QLoraLinear` built an NF4 quantizer whatever it said. It now stores the base weight in the scheme asked for, and refuses one it can't store (such as FP8) by name. FP4 is now the E2M1 grid (0, 0.5, 1, 1.5, 2, 3, 4, 6, scaled per block so the absmax lands on 6); its bins had run to 4 times the block's absmax, which left two of its eight magnitudes unreachable
- **`pmetal serve --continuous-batch` refused to start on Qwen3.5 / 3.6 / 3.8, Qwen4Exp and NemotronH models**, whose recurrent layers it couldn't give per-request state. Each slot now carries its own recurrent state beside its KV cache, reset when the slot is freed, and these models skip the prefix cache, which can't restore it. On Qwen3.5-0.8B, replies streamed through four slots under six concurrent requests match the single-request path token for token
- **`--continuous-batch` lost tokens when a client read more slowly than the model generated.** The scheduler never waits on one reader, and a token that found the request's 64-event buffer full was dropped, as was the `Done` after it. The buffer now holds a whole reply
- **`--continuous-batch` sent the stop token as part of the reply** and counted it as a completion token; the single-request path ends on it without sending it, and now both do
- **A continuous-batching step that failed was retried on every tick**, so its requests never finished and the log filled with the same error. It now ends those requests with an error and frees their slots
- **Generating on a second thread in one process failed with "There is no Stream(gpu, N) in current thread".** The bridge made one generation stream for the whole process, on whichever thread asked first, and an MLX stream works only on the thread that made it. Each thread now gets its own
- **`pmetal serve` failed requests with "There is no Stream(gpu, N) in current thread"** whenever two ran at once or one came in after the server had sat idle, and on Qwen3.5 / 3.6 / 3.8 it failed every request. MLX ties an unevaluated array to the thread that made it. The server loaded the model on one thread and ran each request on whatever thread tokio's blocking pool handed it, so the arrays loading leaves unevaluated, and the KV snapshots the prefix cache keeps between requests, could not be evaluated where they were needed. The model now lives on a thread of its own: it loads there, and every request, `/v1/embeddings` and the continuous-batching scheduler run there
- **Qwen3.5, 3.6 and 3.8 decoded every sequence after the first one in a process from the first one's state**, so `pmetal serve` answered each request after the first by carrying on the first conversation. The single-token decode step keeps attention keys and recurrent state of its own beside the caller's caches, and it kept one for the whole model, made by the first sequence it saw. It now keeps one per sequence and drops it when that sequence's cache is dropped or reset. A process that generates once was not affected
- **`pmetal dflash` on a Qwen3.5-family target kept rejected guesses in its gated-delta-net layers.** After a partial accept the native engine rewound only its attention caches; each GDN layer's recurrent state and convolution inputs still held the whole verified block, so every later token was computed from tokens never emitted. A verify now records each GDN layer's recurrence inputs, and a rewind replays them over the accepted prefix, as the reference implementation does. On a small Qwen3.5 the next token's logits after a rewind match running only the kept tokens bit for bit, where rewinding the attention caches alone moved them by 2.0. Tree verification, which a GDN layer can't follow, falls back to linear on these targets, and a second generation on the same native target starts from an empty cache instead of the last one's
- **`pmetal infer` on Qwen3.5, 3.6 and 3.8 decoded its first tokens from the wrong state.** The gated-delta-net layers carry their last three inputs into the next step, and after a prompt the native engine kept the first three instead. The error carried on in the layer's recurrent state: on Qwen3.5-0.8B a greedy reply stopped after one sentence where transformers writes two, and now matches it token for token
- **Gated-delta-net layers normalized queries and keys with 128 times transformers' epsilon** on Qwen3.5 / 3.6 / 3.8 (on every path: inference, decode and LoRA training). The fused norm pmetal uses adds its epsilon to the mean of squares, not the sum, and was given transformers' value unchanged. It moved logits by 3e-3 on a small model and matches exactly now
- **`infer --mtp` failed on every released Qwen3.5 / 3.6 / 3.8 checkpoint** with "missing model parameters: [model.embed_tokens.weight]": the MTP loader looked for the embedding table under a name only MLX conversions use. On the native engine the MTP layer also rotated positions with the default RoPE base instead of the checkpoint's. `mtp_use_dedicated_embeddings` is honoured
- **`PMETAL_DISABLE_NATIVE_BRIDGE` decoding of an untied Qwen3.5 model produced garbage** after the first token: its single-token step multiplied by the LM head untransposed, which failed on every untied model (the 27B releases are untied)
- **`PMETAL_DISABLE_NATIVE_BRIDGE` decoding of a dense Qwen3.5 model held a second copy of every weight.** Its single-token step copied each weight into a new buffer before the first token, where a view of the model's own weight serves: 108 GB resident for the 27B releases, which swapped at 3.6 s per token. It now uses the model's weights in place. On Qwen3.5-0.8B, memory after decoding drops from 3.1 GB to 1.6 GB, a decode step from 17.7 to 15.6 ms, and the logits are bit for bit the same
- **The CPU hybrid engine computed a different model from the other two.** `infer --ane` and `serve --ane` run it on a flat dense Qwen3.5 text config (`model_type: qwen3_5_text`). It read the config with its own defaults, ignoring `text_config`, `rope_parameters` and the outer head tie, normalized gated-delta-net queries and keys with 128 times transformers' epsilon, always applied SiLU to the output gate, and paired every value head with the key head of the same index, so any model with more value heads than key heads (Qwen3.8-27B has three per key head) produced garbage. It now reads the config the way both GPU engines do, supports `output_gate_type: "sigmoid"`, and joins the transformers parity suite at 1.0e-5
- **Qwen3.8-Flash-Next (`qwen4_exp`) refused a config that spells SiLU `"swish"`**, in `output_gate_type` or `hidden_act`, which the rest of the Qwen3.5 family accepts. It now resolves the gate the same way they do
- **The native engine ignored a Qwen3.5 config's `layer_types`** and built the layout from `full_attention_interval`, so a checkpoint whose layout differs failed to load. Both engines now also take the head tie from the outer config, as transformers does, and let `rope_parameters` win over a legacy `rope_theta`
- **Qwen3.5 checkpoint tensors the loaders didn't read were ignored.** Both engines and the MTP loader now refuse a checkpoint with tensors nothing consumed, naming them; the vision tower and the MTP predictor are skipped by name. Qwen3.8-27B loads all 851 text tensors of its 1,199
- **Two Qwen models in one process could share a compiled decode step** that baked in the first one's head sizes, RoPE base and epsilons. The native engine's compiled GDN and attention steps are now cached per model shape and settings
- **Models downloaded by recent `hf` releases didn't load.** huggingface_hub's cache-wide shared blob store keeps a file's bytes once under `<cache>/blobs/` and links each repo's copy to it, and the loader's shard path check refused any shard that resolved outside the repo's own directory. A shard in the cache's shared store now loads; one that resolves anywhere else is still refused
- **Qwen 3.5 / 3.6 MoE layers counted their input twice** wherever the `pmetal-models` MoE block ran them: `--experts-dir`, LoRA training and `PMETAL_DISABLE_NATIVE_BRIDGE`. The block returned its expert mixture plus its own input, and the decoder layer then added the residual on top. The default inference path has its own MoE and was not affected
- **`pmetal dflash` accepted none of its draft model's guesses** on Qwen3-4B, so it ran slower than decoding without a drafter. Each draft saw only the few tokens the previous step had accepted as context, where the draft model reads the target's hidden states for the whole context, kept in its own KV cache. It now keeps them: tokens per verify step on a chat prompt went from 1.0 to 8.5, 192 tokens from 12.1 s to 2.2 s, with the same output. Tree verification (`--tree-budget`) had the same fault and the same fix
- **`serve --ane` loaded the model again for many requests.** The loaded ANE model was kept per thread, and the server runs each request on whichever thread its pool hands out, so two requests at once, or any request after ten idle seconds, compiled or loaded the whole model again (seconds per request) and held another copy. The models now live on one thread of their own that every request runs on, one at a time
- **`--seed` didn't make training reproducible** (#33). It seeded the data pipeline but not MLX's random key on `train`, `distill`, `rlkd` and `embed-train`, so LoRA initialisation and step-1 gradients differed between runs with the same seed. `train` is seeded where the shared training entry point starts, which covers the CLI, TUI, GUI, MCP and Python
- **ANE: models the engines don't implement ran anyway and computed the wrong thing** (#34). ANE inference is Qwen3-shaped and now admits only `qwen3`; ANE training is Llama-shaped and admits `llama` and `mistral`. Both refuse MoE, `rope_scaling`, sliding windows, attention/MLP bias and untied `lm_head`, and refuse a checkpoint whose tensors are missing, the wrong size, or in a dtype they can't read (quantized weights used to load as zeros). ANE training now uses the checkpoint's `rope_theta` and `rms_norm_eps`. `infer --ane` warns when it doesn't use the ANE
- **The ANE didn't run on macOS 27** (#34). Kernels with several weight files failed to compile (Code=10, a hash mismatch between the descriptor and Apple's compiler service, which hash the files in different orders); they're now packed into one weight file. Evaluation failed when a model padded its rows to 64 bytes (Code=42); surfaces are now restrided to the layout each model reports, and anything still too small fails with the tensor and sizes named. Two private-API changes aborted the process outright (the performance-stats argument in `train --ane`, the real-time path's retry); real-time evaluation is reported unavailable on macOS 27. ANE training now runs there
- **`infer --ane` on a 4B model filled the boot volume and fell back to the GPU** (#34). Every compiled ANE kernel kept two copies of its weights in `$TMPDIR` until the model was dropped, about 300 MB per layer kernel, so Qwen3-4B needed ~15 GB of free space and failed partway through with the ANE compiler's "Failed to allocate memory for file backing". A kernel's compile directory is now removed as soon as it's loaded, so peak use is one kernel's worth. All 36 layers of Qwen3-4B run on the ANE
- **Every ANE kernel kept two extra copies of its weights in memory.** The weight dictionary handed to the ANE framework was over-retained and never freed, and the framework's model descriptor kept another copy after the program had loaded. Both are released now: `infer --ane` on Qwen3-4B peaks at 22 GB instead of 31 GB
- **ANE kernels are compiled once, not on every run.** Apple's ANE service already caches compiled programs across processes; pmetal now loads from that cache when it has the kernel instead of compiling it again, which was ~0.3 s for each of Qwen3-4B's 72 kernels. `infer --ane` logs how many kernels came from the cache
- **ANE training (`train --ane`) trained the wrong model with the wrong gradients** (#34). Its only test checked that loss falls on random weights, which it did anyway. Measured against an independent CPU reference, now in `tests/ane_training_e2e.rs` (forward loss to fp16 precision, and every sampled gradient's sign via Adam's first step):
  - There was no RoPE anywhere, so attention couldn't see token positions
  - Every projection's weights reached the ANE in the wrong layout, square ones transposed and the rest scrambled; the backward pass had the mirror-image error
  - The gradient skipped the residual connection around each FFN, so the attention block and every layer below got only the FFN branch's share
  - The fused attention backward read Wo from the wrong place in its input and computed dQ, dK and dV as zero
  - The decomposed attention used for long sequences multiplied the attention weights by V with the operands swapped (NaN past 1,024 tokens, wrong before)
  - RMSNorm ran in fp16 on the ANE and overflowed on any real checkpoint's residual stream (NaN); it's computed in f32 on the CPU, as ANE inference already did
  - Per-position gradients underflowed fp16 at long sequences; they're carried scaled by the sequence length and unscaled in f32
  - The loss covered only the tokens present in the training data (a "vocab compaction" the CLI always installed), and its fp16 ANE softmax rounded small probabilities to zero. It's now the full-vocabulary cross-entropy in f32. Padding no longer counts toward it, nor do prompt tokens a dataset masks with `-100`; padding used to be most of a short example's loss

  On SmolLM2-135M, step 1 of `train --ane` now matches the GPU trainer's loss (2.568 vs 2.539); it used to start at 6.51 and climb toward 11, about a uniform guess over the vocabulary
- **`train --ane` never saved what it trained.** Its output directory got `training_state.json` and nothing else, while the result pointed at a `lora_weights.safetensors` that was never written. It now writes the model, `model.safetensors` (f32: at the learning rates ANE training uses, bf16 would round the updates away) beside the base model's config and tokenizer, so `pmetal infer -m <output>` runs it. Checkpoints written every `--save-steps` include the weights too
- **LoRA training on Qwen3.5 gave NaN loss on its first step**: the GDN layers ran their fused Metal kernel during training, and that kernel has no backward pass. An uncached forward now takes the differentiable path
- **MLX quantization metadata** (#30). `qwen3_native` ignored MLX's per-module overrides (`"<module>": {"bits", "group_size"}` under `quantization`), so a mixed-precision checkpoint loaded at the file-level width; it now reads them, group size included. Checkpoints written by pmetal's MLX quantizer couldn't be loaded as MLX-format checkpoints (aux tensors named `<module>.weight.scales`, overrides in a pmetal-only map); it now writes MLX's layout. Checkpoints pmetal quantized before still load
- **The GUI reported a Metal library MLX couldn't load only by failing the first job.** It now loads it at startup and shows an error dialog
- **`init`, `download`, `search`, `dataset` and `tokenize` needed a working Metal library** after the startup check was added; they never use MLX, and now skip it
- **Dataset downloads returned the wrong directory when the first file was nested** (`data/train-….parquet`), because the snapshot root was taken as the parent of that file
- **The CLI ignored `PMETAL_METALLIB_PATH`**, though 0.6.0's notes say it is honoured as an operator override. Only the GUI read it. The CLI now checks it before every other location. Both warn and fall back to the normal search when it doesn't name a usable Metal library, where the GUI used to hand a mistyped path to MLX and fail on the first kernel
- **A bad `mlx.metallib` aborted the process** (`libc++abi: terminating due to uncaught exception`) instead of reporting an error. MLX builds its Metal device on the first allocation, and when the library wouldn't load, the throw escaped into Rust, including from the bridge's own error handlers, whose fallback array allocated through the same failing device. Now a wrong, empty or truncated file is rejected by its header before MLX sees it (a corrupt cached copy is skipped and re-extracted), the CLI loads the library before any command and exits with a clear error if MLX still refuses it, and the bridge's error paths no longer allocate on the GPU
- **`pmetal-py` could not be published to crates.io**: its dependency on the `pmetal` crate was a bare path with no version, which `cargo publish` rejects. It now inherits `pmetal-lib` from the workspace table, so `just bump` keeps the version current
- **`pmetal train --log-events <path>` wrote nothing.** It created the file, then warned that event streaming needed a branch that had long since merged. It now writes every job event to the file as a JSONL line, as its help text says

## [0.6.0] - 2026-09-10

### Fixed

#### Shipped artifacts

- **The v0.5.0 release artifacts could not run off the build machine** ([#21](https://github.com/Epistates/pmetal/issues/21)). The release profile builds MLX as a dylib and stamps its own absolute build path as the install name, so `PMetal.app` died at launch with `Library not loaded: /Users/…/libmlx.dylib`, and `cargo install` left a binary pointing at a build directory it had already deleted. Both release jobs and the Homebrew formula now set `PMETAL_MLX_STATIC=1`, which links MLX into the artifact — no dylib to locate, no install name, no rpath. Each job asserts with `otool` that no `libmlx` dependency survives. Reported and diagnosed by [@imleooooo](https://github.com/imleooooo)
- **Bundled MLX moved to v0.32.2** ([#24](https://github.com/Epistates/pmetal/issues/24)), which fixes the kernel-JIT failure that made every training run panic against the Metal toolchain in Xcode 26.6. Contributed by [@texchi2](https://github.com/texchi2)
  - MLX ≥ 0.32 raises `[gather] Cannot calculate VJP with respect to indices`, so every gather whose indices descend from a differentiable tensor now cuts the trace: embedding lookups, cross-entropy label gathers, and MoE routing in `moe_routing`, `gpt_oss_native`, `qwen3_native`, `deepseek_native` and `grouped_gemm_moe`
  - The metallib override is upstream API now (`mlx::core::metal::set_metallib_path`), so the vendored MLX checkout builds unpatched. The CLI and the GUI both call it; `PMETAL_METALLIB_PATH` is still honoured as an operator override

#### Models

- **Gemma 4 unified multimodal checkpoints** (`model_type: gemma4_unified`, e.g. `mlx-community/gemma-4-12B-it-bf16`) now load: the `language_model.*` prefix is stripped alongside the existing `model.language_model.*`, and `vision_embedder` / `embed_audio` entries are skipped. Contributed by [@texchi2](https://github.com/texchi2)

### Added

- **DoRA, RSLoRA and LoRA dropout are reachable.** All three are `LoraConfig` fields that `pmetal-lora` honours, and none had a way in: no CLI flag, no `TrainSpec` field, no MCP parameter, while the GUI rendered a DoRA checkbox, an RSLoRA checkbox and a dropout input that were never sent. Adds `--dora`, `--rslora` and `--lora-dropout`, which the TUI form and the MCP `train` tool pick up from the spec
- **`pmetal tui` gains a jump-to-tab palette (`Ctrl+P`).** `Alt+1-9` reaches nine of twenty tabs; everything past GRPO was only reachable by cycling
- **The generic weight loader reports checkpoint keys that matched no parameter** ([#19](https://github.com/Epistates/pmetal/issues/19)). `assign_loaded_weights` matched by exact name and dropped the rest in silence, so a checkpoint laid out differently from the parameter tree loads nothing and the model runs on random init while looking healthy. It now logs a warning when nothing matched and an info line when some keys were left over. Reporting approach from [@iilyak](https://github.com/iilyak)

#### New architectures

- **DiffusionGemma (block-diffusion) port**: text trunk config, MoE blocks, 7-norm layer and encoder; decoder with bidirectional canvas and self-conditioning; discrete-diffusion generation engine and sampler; dispatcher plus full tied-weight loader
  - Vision tower and multimodal encoder, with a bit-exact image processor
  - LoRA and QLoRA training: trainable forward, block-diffusion loss, bake-in LoRA on the attention trunk, autograd/AdamW/checkpoint loop
  - **`pmetal train-diffusion`** CLI subcommand, with `--qlora` for quantized attention projections and MoE experts
- **Mllama (Llama 3.2 Vision)**: real port replacing the previous skeleton, tiled image processor, weight loader and dispatcher wiring
- **Gemma 4 MoE**: parallel dense-plus-experts block in `Gemma4ForCausalLM`, vision tower, image processor, encoder-KV attention path
- **MTP speculative decoding workflows** for Qwen3Next/Qwen3.6 assistants

#### GGUF

- Generic **GGUF → `DynamicModel` inference loader** for the dense Llama family
- **Tokenizer reconstruction from GGUF metadata**, so a GGUF file alone is enough for CLI inference
- Authoritative **Gemma 4 GGUF tensor-name map** and architecture detection

#### Parity infrastructure

- **Real-released-config sweep** (`real_config_parity`): every architecture is constructed from its shipped `config.json`
- **Real-weight sweep** (`real_weight_parity`): 12 architectures run against real released checkpoints and are held to a *measured* bf16 noise floor, with a per-position profile alongside the aggregate. Also checks that no parameter is left at random init and that no checkpoint tensor goes unclaimed
- **Pillow-exact image resampler**: preprocessing now matches PIL bit-for-bit
- Shared `ACT2FN` activation resolver and an exact (erf) GELU in the bridge

### Fixed

#### Attention (cross-cutting)

- **Sliding-window masks were built in f32**, which MLX rejects against a bf16 output dtype. The op threw and returned a 0-dim array that broadcast silently, so **no windowed layer of any architecture ever ran its window**
- **Trunk-level blanket causal masks** in Gemma and Mistral dropped every layer to an unwindowed mask. The trunk now builds one only when every layer wants the same mask
- **Attention backend selection compared candidates on an absolute difference.** Metal backends stage through f16, so on a model running at ~1e-17 the f16 path returned all zeros, scored as a perfect match and won on speed. Selection is now relative to the reference magnitude, and the persistent kernel cache carries an epoch so a stale recorded choice cannot short-circuit the fix
- Attention masks are coerced to the query dtype

#### Per-architecture correctness

- **Llama 3.x**: `rope_scaling` had no `"llama3"` arm, so the whole family ran unscaled. Banded scaling now applies in inference, LoRA and QLoRA alike
- **Phi**: LongRoPE picks the short or long factor table by sequence length instead of always using `long_factor`; Phi-3 SuRoPE applies per-dimension frequencies and no longer double-scales `mscale`; the fused `qkv_proj` is mapped; `partial_rotary_factor` defaults to 1.0 as transformers reads it; Phi-4-mini reuses the embedding for its tied head
- **Gemma**: Gemma 3 loads `q_norm`/`k_norm`, honors `rope_local_base_freq`, rounds the embedding scale in bf16 and upcasts RMSNorm; Gemma 2 dispatch and final-logit softcapping; correct GELU variant in the LoRA path
- **DeepSeek**: the entire MoE was unmapped (three name mismatches, 5291 tensors silently dropped and every expert left at random init); the router follows the `scoring_func` the config names; MLA uses traditional interleaved RoPE with YARN per-dimension frequencies and embedding mscale; V3 group-limited expert routing
- **GPT-OSS**: clamped SwiGLU, router top-k softmax, attention sinks, YARN RoPE, biased router, and a gradient trace cut at the router's top-k indices
- **Granite**: the attention field was named so the checkpoint's 160 tensors never loaded (it trained fine and inferred noise); the four scalar multipliers Granite is defined by now apply
- **Llama 4**: iRoPE parity (traditional RoPE, post-RoPE QK-norm, NoPE map) and a sigmoid MoE gate on expert input
- **Cohere**: `logit_scale`, tied head, traditional RoPE
- **Qwen2**: `head_dim` is derived rather than assumed to be 128 (Qwen2.5-0.5B ships `"head_dim": null`)
- **Nemotron-H**: `dt` is no longer clamped to `time_step_max` (a training-init range, not an inference clamp) and attention no longer applies RoPE, which Nemotron-H does not use; the loader tolerates the bare `Infinity` its `config.json` contains
- **GELU variants** across BERT, Whisper, Phi, T5, CLIP, Flux and Gemma now follow the variant each reference config names

#### Other

- Bridge error-slot invariant preserved in the GDN Metal fast paths
- Metal MoE routing guarded against expert-count overflow
- Attention KV cache is built for hybrid LoRA inference
- Python bindings hardened
- Offline and sandboxed builds honor `FETCHCONTENT` environment overrides (#17)

### Changed

#### Shipped artifacts and the desktop app

- **The release binary and the Homebrew formula now build with `--features serve,mcp`.** The default feature set stays lean so library consumers don't inherit axum and rmcp, but no distributed binary previously had `pmetal serve` or `pmetal mcp` — while the TUI ships a Serve tab and the GUI a Serve page that both spawn `pmetal serve`
- **The GUI bundle now carries the `pmetal` CLI as a Tauri sidecar.** Eleven GUI pages drive the CLI as a subprocess, and an app launched from Finder inherits launchd's `PATH`, which contains neither `/opt/homebrew/bin` nor `~/.cargo/bin`. The GUI additionally probes those directories by absolute path, with a `PMETAL_CLI` override
- **The GUI bundle now ships an `mlx.metallib` and points `PMETAL_METALLIB_PATH` at it during startup.** MLX loads no kernels without one, and the only thing that had ever populated `~/.cache/pmetal/lib` was building PMetal from source on the same machine

#### Documentation

- **Every code block in every crate README is compiled as a doctest** (`#[cfg(doctest)] #[doc = include_str!("../README.md")]`), so the crates.io front pages cannot drift from the API again. All 33 blocks were corrected in the process: wrong constructor arity, methods that never existed, and one block that was not valid Rust
- **Removed the `easy` API documentation.** `pmetal::easy` was deleted in 2026-03 (`789d1aa`) but remained the entire Quick Start on crates.io and docs.rs, all of `docs/sdk/easy-api.md`, and the "Rust SDK" link from three other pages. Replaced with `orchestrator::run_training`, which is what the removal commit pointed at and what the CLI itself uses
- Corrected the feature-flag tables in the README, the crate README and `docs/configuration/feature-flags.md`: they listed a non-existent `easy` feature and marked `merge` and `distributed` as non-default when both are in `default`
- Corrected the TUI tab table (9 listed, 20 real) and the GUI page list (10 listed, 19 real)
- The training-method matrix no longer claims `easy::dpo()` and friends; DPO, SimPO, ORPO and KTO are named for what they are, library-only

#### Library

- **`pmetal_mlx::prelude::*` no longer makes the name `Result` unusable.** Six `Result` aliases in `kernels/` were glob-exported into the prelude; four were byte-identical duplicates of `kernels::utils::Result` and now reference it, and the two `Exception`-based ones no longer leak into the glob. `pmetal_mlx::kernels::{metal_cross_entropy,metal_swiglu,training_attention,rms_norm,metal_norm_lora}::Result` are no longer nameable
- **`just preflight`'s lockfile gate could never pass.** It ran `cargo update --locked`, which fails as soon as any transitive dependency publishes a new version; it now runs `cargo metadata --locked`, which fails only when `Cargo.lock` would actually have to change
- `just fmt` and `just fmt-check` now cover `pmetal-gui/src-tauri`, which is excluded from the workspace and had therefore never been formatted
- **The toolchain is pinned** (`rust-toolchain.toml`). CI lints with `-D warnings` against unpinned stable, so any stable that adds a lint turns the build red with no change on our side — 1.98 did exactly that. Bumping is now a reviewable commit
- **Parity oracles migrated to Hugging Face transformers** across every architecture, with shared fixture helpers and `pmetal_mlx::test_utils`
- `MllamaImageProcessor` renamed to `FixedSizeImageProcessor`, reflecting what it actually does
- Gemma 4 caches pre-transposed expert weights
- Clean under Rust 1.98 clippy (`chunks_exact` → `as_chunks`, `for_kv_map`, `drain_collect`, `needless_late_init`)

### Removed

- **`pmetal infer --stream`.** It was accepted, threaded through `main.rs`, and landed in a parameter nothing read. It was hidden on the CLI but not elsewhere: `InferSpec` carried it as a "Stream" toggle, so the TUI's inference form rendered a switch for it, and the MCP `infer` tool advertised it as "Stream tokens to stdout as they are generated"
- **`ServeSpec.lora`.** `pmetal serve` has no `--lora` — the documented workflow is to `pmetal fuse` the adapter first — but the spec still carried a "LoRA Adapter" field, so the TUI's Serve form and the GUI's Serve page both rendered an input that would have produced an unrecognized flag. Also removed from the MCP `start_serve` tool
- Sixteen CLI flags that appeared in the README and `docs/cli/*.md` but were never defined, including `infer --show-thinking`, `merge --models`, `quantize --type`, `train --dora`, `rlkd --teacher`, `ollama modelfile --model` and `search --type`. DoRA is real but library-only (`LoraConfig.use_dora`); the docs no longer claim a CLI flag, GUI control or TUI control for it

## [0.5.0] - 2026-05-07

### Added

#### Distributed inference & training

- **`pmetal-distributed` crate (Phases 1-4 + 7)**: Thunderbolt-fabric-aware multi-Mac cluster runtime with feature-gated tensor, expert, context, ZeRO, and pipeline parallelism modules
  - `pmetal cluster` CLI: per-node launch, ring/mesh topology discovery, fabric handshake
  - Pipeline harness with overlap of computation and Thunderbolt transfers
  - Canonical expert-rank mapping + per-architecture MoE/MLA tensor-parallel plans
  - Ring all-reduce / all-gather with corrected chunk indexing

#### TurboQuant KV cache (production-ready)

- **TurboQuant KV cache quantization**: Provably near-optimal KV cache compression based on random rotation + Lloyd-Max scalar quantization + QJL residual for unbiased inner products (arXiv:2504.19874). Achieves 4-6x KV cache compression with near-zero quality loss. Available via `--kv-turboquant` or presets `--kv-turboquant-preset q3_5` (near-lossless) / `q2_5` (6.4x compression)
  - Separate key/value runtimes with independent bit widths and outlier-aware mixed-precision
  - Direct attention path for single-token decode avoids full cache dequantization
  - Data-oblivious (no calibration data required) — quantizes KV entries online as generated
  - Precomputed codebooks via Lloyd-Max algorithm for Beta distribution (deterministic from seed)
  - Metal kernel backend with CPU fallback
  - **Phase 0**: split monolithic `mod.rs` (6101 → 222 LOC) into config/core/state/bits/math submodules
  - **Phase 3**: GPU-resident hot/cold pipeline + Mixed K/V storage; `mixed_score` as layout oracle
  - **Phase A/B**: QJL ablation harness (feature-gated) + per-row `key_slot_scale` codebook adaptation
  - **Phase C/C′**: Variant F drop-QJL opt-in path; d128/d256 `no_qjl_2pass` fast paths (4..=8 bits)
  - **Phase D**: `TurboQuantPackMode` config + Fullbyte dense-values kernel
  - **Phase E**: `TurboQuantOutlierMode` — encode-side top-K outlier storage, zero pre-quant + decode override, outlier-bias on d128/d256 fullbyte score kernel; CPU mirror in scalar encode/decode
  - **Phase F**: Hamming skip-list dispatch — `skiplist_threshold` config, GPU `sign_hash` buffer, Metal Hamming-distances kernel + FFI, GQA support
  - Mixed-precision attention parity baseline; defensive residual-norm clamp + NaN-safe encode

- **Asymmetric K/V head dimensions**: KV cache, TurboQuant, and fused attention now support models where key and value projections have different widths (e.g. DeepSeek MLA with `qk_head_dim != v_head_dim`)

- **`pmetal serve --kv-turboquant`**: TurboQuant KV cache in the serving engine with `--kv-turboquant-preset q3_5` for near-lossless 4.6x KV compression in production

#### Quantization & model formats

- **Optimized FP8 checkpoint loading**: Hugging Face FP8 `weight_scale_inv` sidecars are dequantized or repacked into MLX `mxfp8` weights for Qwen3-family native paths; mode-aware quantized matmul plumbing handles floating-point quantized weights without dense fallback
- **Expanded GGUF quantization/export**: `pmetal quantize` now writes standard GGUF metadata from Hugging Face configs, tokenizer/pre-tokenizer metadata, HF-to-GGUF tensor names, stacked MoE expert tensors, and method-specific file types
- **Broader GGUF format coverage**: quantization/dequantization support now includes K-quants, legacy Q4/Q5/Q8 variants, Q1_0, TQ1_0/TQ2_0, MXFP4, NVFP4, BF16, F16, and F32 round trips
- **MLX safetensors quantization path**: quality-based bit allocation with `--target-bpw`, GPU-resident weight loading, and tokenizer/config sidecar copy for MLX-format quantized exports

#### Inference server (OpenAI- + Anthropic-compatible)

- **Continuous batching with paged-KV-style admission + shared prefix cache** in `pmetal-serve`: per-request slot scheduling, KV-cache prefix sharing, concurrent decode for many simultaneous chats
  - Token-block admission budget (`--cb-block-size`, `--cb-max-blocks`) prevents over-admitting active contexts and skips head-of-line requests when a smaller queued request fits the remaining block budget
  - Continuous batching now reuses the shared prompt prefix cache, prefills only uncached suffix tokens, and saves extended prefixes after final prefill
  - Continuous batching derives the same cache mode as the single-request serving path, honoring `--kv-quant` and `--kv-turboquant`
  - Hybrid/recurrent models are rejected from continuous batching instead of silently running without recurrent state
- **Anthropic-compatible `/v1/messages` endpoint**: streaming `message_start` → `content_block_start` → `content_block_delta*` → `content_block_stop` → `message_delta` → `message_stop` events; non-streaming JSON path
- **`/v1/embeddings` endpoint**: 17 architectures supported via `forward_hidden` (Llama/Llama4/Qwen2/Qwen3/Qwen3MoE/Qwen3Next/Mistral/Gemma/Gemma4/Phi/Phi4/DeepSeek/Cohere/Granite/GptOss/NemotronH/BERT) — pooling via `pmetal_models::pooling`
- **Token logprobs**: `SamplingParams.logprobs_top_n` plumbed end-to-end through non-streaming and SSE streaming on both `/v1/chat/completions` and `/v1/completions`. New `pmetal_models::generation::token_logprobs` primitive; ANE/CPU paths emit `logprob: None`
- **Best-effort tool calling on `/v1/chat/completions`**: `try_parse_tool_calls` accepts `{name, arguments}` or `{tool_calls: [...]}`. `ChatCompletionRequest.tools` gates the attempt; chat templating threads tool defs into the rendered prompt
- **`IncrementalDecoder<Aux>` SSE buffer**: shared UTF-8 boundary buffer + per-token aux pipelining (used for logprobs alignment) across chat/completions/anthropic streams

#### Job orchestration substrate (TUI / GUI / MCP / CLI parity)

- **`JobSpec` substrate**: 16 canonical spec types in `pmetal-core` (Train, Distill, GRPO, Bench, Eval, Pretrain, Tokenize, Serve, Generate, RLKD, EmbedTrain, DFlash, Memory, Ollama, …) with `#[derive(JobSpec)]` proc-macro
- **`JobEvent` canonical streaming protocol**: progress / metric / log / artefact / complete / failed events emitted by all 4 surfaces (CLI, TUI, GUI, MCP)
- **CLI**: 8 specced `Commands` variants flattened — 613 LOC removed from `main.rs`; `cli/<sub>.rs` Args structs and JobSpec argv round-trip tests; `--log-events` flag stub
- **TUI**: 14 tabs with full CLI parity, `?`-key help overlay, `Ctrl+1..9` tab jump, active-job footer badge, descriptor-driven forms with shared `FormTabState` primitive; channel-based metrics streaming (`ChannelMetricsCallback`) for direct-path train/distill/grpo/bench/eval/pretrain
- **GUI (Tauri)**: complete 9-DTO frontend-lockstep migration to `*Spec` types; Serve, Bench, Eval, Jobs, Pretrain pages; embed-train + rlkd + ollama routes; channel-based metrics streaming
- **MCP**: 51-tool server with migrated train/pretrain/tokenize/memory/dflash/generate coverage, allowlisted CLI passthrough tools for newly added CLI flags, and a JobEvent JSONL consumer for managed background jobs

#### SOTA distillation (`pmetal-distill`)

- **Universal Logit Distillation (ULD)** — Wasserstein-1 over sorted logit distributions for cross-tokenizer KD (Boizard et al. 2024); optional `top_k` truncation; permutation-invariant by design
- **Generalized Knowledge Distillation (GKD)** — λ-weighted off-policy + on-policy KL blend (Agarwal et al. 2024); `OnPolicySampler` trait with `GreedySampler` reference impl; `compute_full(t_off, s_off, t_on, s_on, T)`
- **MiniLLM** — reverse-KL with optional teacher-mix `target = mix·T + (1-mix)·S` (Gu et al. 2024)
- **Skewed JSD (DistiLLM-2)** — `α·KL(T||M_α) + (1-α)·KL(S||M_α)` with `M_α = α·T + (1-α)·S`, log-sum-exp computation; α=0.5 reduces to standard symmetric JSD (Ko et al. 2024)
- **Attention-transfer loss + weighted Metal path** for hidden-state distillation
- **Offline teacher-logit caching**: `pmetal distill --offline-cache <path>` precomputes teacher logits to disk; new `Int8PerToken` compressed-block variant replaces NaN-sentinel scheme with explicit `per_token_meta` field (legacy `Int8` variant retained for read-back)
- **`DistillLossOutput.metrics: HashMap<&'static str, f32>`**: lazily-evaluated `teacher_entropy`, `student_entropy`, `kl_per_token`, `top1_agreement` exposed to trainer JSONL/TUI streaming
- **TAID difficulty-aware** observability: `alpha_var` surfaced for per-step monitoring
- **Configurable `ignore_index`**: PyTorch-standard `-100` default on `TrainingConfig`; safe label clamping before gather
- **Hidden-state shape assertions** before matmul (clear error vs. silent broadcast bug)

#### SOTA model merging (`pmetal-merge`)

- **Fisher merging** (Matena & Raffel 2022): diagonal-Fisher-weighted average `θ = Σ F_i⊙θ_i / (Σ F_i + ε)`; lazy-loaded Fisher safetensors; `fallback_to_mean` for tensors without Fisher entries
- **RegMean** (Jin et al. 2023): closed-form linear-layer merge `W = (Σ G_i)⁻¹ · (Σ G_i W_i)` via hand-rolled Gauss-Jordan `pseudo_inverse_2d` with Tikhonov ridge; falls back to mean for non-2D weights
- **MoE expert permutation alignment**: per-(model, layer) Hungarian solver (Jonker-Volgenant style, O(N³)) over L2-normalized cosine similarity of expert fingerprints; tensor-name remapping `experts.{i}.` → `experts.{π(i)}.` before merge; gated by `align_moe_experts`
- **Honor `config.dtype` in save path**: `MergeBuilder.dtype` builder, `TensorWriter::with_dtype` plumbing, per-dtype byte packing for F16/BF16/F32; previously hardcoded to F16
- **Cross-model dtype consistency check**: `verify_source_dtypes` errors on mismatch unless `allow_mixed_dtype` is set
- **Tied-embedding detection**: `lm_head.weight` and `embed_tokens.weight` aliasing detected and merged once under canonical name
- **Tokenizer + config sidecar copy**: `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `config.json`, `generation_config.json` copied on full-model merge; `config.json.torch_dtype` patched to match output dtype
- **Post-merge sanity sweep** (`SanityLevel::{Off,Quick,Full}`, default `Quick`): NaN/inf detection aborts save; full mode reports per-tensor `mean/std/abs_max/sparsity`
- **`MergeConfig.dry_run`**: short-circuits write phase, logs would-write summary

#### LoRA / QLoRA — full text-architecture coverage

- New LoRA adapters: **Granite, Llama4, DeepSeek, NemotronH, MLlama, Cohere, Phi, Gemma4, GPT-OSS, Qwen3-MoE, Qwen3-Next**
- New QLoRA adapters: **Granite, Llama4, DeepSeek, NemotronH, Cohere, Phi, Gemma4, GPT-OSS, Qwen3-MoE, Qwen3-Next**
- **LoRA+** wired into `run_compiled` training path; Gemma4 QLoRA KV-cache path
- **DeepSeek `merge_lora`** properly implemented; Phi4 dispatched to existing `PhiLoraForCausalLM`
- Interface-parity gradient-checkpointing hooks across 7 adapters

#### Bridge & native paths

- **`pmetal-bridge` crate**: Zero-allocation MLX C++ bridge replacing mlx-rs as the core runtime. Native inference at 201 tok/s (Qwen3.5 0.8B), 4-bit quantized inference (28 tok/s on 27B), compiled attention, KV cache trimming, and full training ops (autograd, optimizer, random, math, reduction, comparison) — all without mlx-rs overhead
- **Fused [T=1] decode kernels** for `gpt_oss` and `llama4` (Bridge Phase 4)
- **Fused [N,1] batched-decode** path for Tier-1/2 architectures
- **Cheap native KV cache fork support** for Qwen3, GPT-OSS, Llama4, DeepSeek, and generic `KVCache`, preserving dense, quantized, and TurboQuant cache state for serving prefix reuse
- **`BRIDGE_TRY_{DST,VOID}` error coverage**: thread-local exception slot replaces process-abort across most ops; `pmetal_bridge::check_last_error()?` surfaces `BridgeError::CxxException` after any op; `InlineArray::try_*` variants for matmul/softmax/reshape/sdpa/gather_mm/dequantize/etc.
- **Scalar dtype footgun fix**: `InlineArray::scalar_like(value, peer)` + `mul_scalar`/`add_scalar`/`sub_scalar`/`div_scalar` eliminate manual `.as_dtype(model_dtype)` calls
- **`async_eval` actually async**: prior implementation blocked the calling thread
- Bridge file splits: `bridge.h` and `bridge.cpp` carved into `cpp/bridge/` sub-headers + 6 source files; `bridge_turboquant.cpp` split by kernel family; `inline_array.rs`, `qwen3_native.rs`, `deepseek_native.rs`, `llama4_native.rs`, `gpt_oss_native.rs` split into submodule directories
- **`forward_hidden`** for 17 architectures (embeddings + retrieval support)

#### Preference & RL trainers (`pmetal-trainer`)

- **`PairedPreferenceTrainer<L>` trait + `DpoLoss` kernel**: `DpoTrainer::train` and `OnlineDpoTrainer::train_step` now delegate to the shared trainer; `ReferenceStrategy::{StopGradient, Zero, Precomputed}` covers the three reference-logp sources
- **Shared log-prob helpers** fanned out to KTO/ORPO/GRPO/OnlineDPO via `logprob_utils::{compute_log_probs, compute_log_probs_with_avg, shifted_selective_log_softmax}`

#### Model, training & data

- **Full-parameter pretraining**: End-to-end `pmetal pretrain` pipeline for training models from scratch
  - Model factory supporting llama, qwen, gemma, mistral, phi, and gpt-oss architectures
  - Gradient accumulation, cosine/linear/constant LR scheduling, gradient clipping
  - Full model + optimizer checkpoint save/restore for resumable runs
  - Memory-mapped streaming shard reader (`StreamingShardReader`) with zero-copy I/O via memmap2
  - `pmetal tokenize` command for converting JSONL corpora to binary shards
  - Pretrain tab in TUI and GUI with real-time loss/throughput/ETA monitoring

- **Gradient checkpointing**: `checkpoint_apply()` wraps forward functions via `mlx::core::checkpoint()` to recompute activations during backward, reducing peak training memory from O(layers) to O(1)

- **Optimizer checkpoint/resume**: `AdamW` gains `step_count()`, `set_lr()`, and `restore_state()` for saving and restoring optimizer state across training runs

- **Gemma 4 architecture**: Full Gemma 4 model support with sliding-window attention and per-layer KV head configuration

- **DFlash speculative decoding**: Native Rust implementation of the DFlash speculative decoding pipeline for accelerated generation

- **Jinja chat templates**: Real upstream jinja rendering via minijinja with 16 parity-audited template types (ChatML, Llama2/3/4, Mistral, Gemma/Gemma4, Phi3/4, Qwen, DeepSeek, Cohere, Alpaca, Vicuna, Zephyr, GptOss)

- **Qwen3.5 MoE dispatch improvements**: Expert prefetch reset per generation, configurable GDN chunk size, chunked prefill, and generation helpers

- **Adaptive sequence packing**: `compute_pack_seq_len()` uses p99 of actual dataset sequence lengths instead of `max_position_embeddings` — up to 256x reduction in wasted compute for short-sequence datasets. `--pack-max-seq-len` for explicit override

#### Metal 4 / MPP backend

- **Metal 4 / MPP kernel backend** (Epistates/pmetal#14): Trait-based kernel dispatch with `Metal3Backend` and `Metal4Backend` for M5+ (Apple10/NAX) GPUs
  - `KernelBackend` trait with 16 methods covering GEMM, attention, fused linear, training, MoE, distillation
  - `KernelDispatch` router on `MetalContext` — selects Metal 4 for large GEMMs (M>1, K%32==0) on M5, Metal 3 for everything else
  - `Metal4CommandBuffer` with correct begin/end lifecycle, `CommandAllocatorPool`, `ResidencyManager`
  - Compile-time `#[cfg(has_metal4)]` gating + runtime `has_nax` check — zero overhead on M1-M4

- **15 MPP-optimized Metal 4 shaders**: All following Apple MPP best practices (single simdgroup execution, Morton-order threadgroup walk, K-dimension alignment to 32, accumulation-loop barriers at BK=128)
  - 8 existing shaders optimized: `mpp_gemm`, `mpp_flash_attention`, `mpp_quantized`, `mpp_fused_swiglu`, `mpp_fused_norm_lora`, `mpp_dw_gemm`, `mpp_grouped_gemm`, `mpp_fused_lora`
  - 5 new shaders: `mpp_fused_training` (AdamW), `mpp_fused_cross_entropy`, `mpp_fused_rope`, `mpp_fused_moe`, `mpp_fused_distill`
  - 2 additional: `mpp_fused_mlp` (gate+up+down combined), quantized MoE expert variants

#### Benchmarking & inference UX

- **Benchmark enhancements**: workload presets, custom dataset/expert-dir controls, inference session repeats, train sample/step/batch/sequence controls, warmup passes, GDN prefill stage profiling, TurboQuant flag for bench commands, and fused gate/up expert packing with auto-detected tensor layout

- **`--mode` sampling presets**: Per-model-family recommended sampling parameters (Qwen3/3.5 thinking/instruct modes). `--mode auto` selects based on `--no-thinking` flag

- **`--detect-repetition`**: Opt-in n-gram repetition loop detection (8-token pattern x 4 repeats), force-stops infinite loops

- **Chip name in decode stats**: Inference output now shows the Apple Silicon chip (e.g., `[M4 Max]`)

### Changed

- **mlx-rs removed**: All crates migrated from mlx-rs to the `pmetal-bridge` compat API. Entire model, training, serving, and GUI stacks now use the zero-allocation C++ bridge
- Thinking trace shown by default for thinking models; use `--hide-thinking` to suppress
- Migrated scattered `has_nax()` checks to `MetalContext::dispatch()` for centralized backend routing
- Split `compat.rs` (3620 lines) into 7-file `compat/` module directory
- Split `bridge.cpp` (6749 lines) into 6 C++ source files with shared `bridge_internal.h`
- `Array::id()` replaces `data_ptr()` for weight change detection — safe with lazy evaluation
- **Persistent MLX build cache** in CI and `build.rs`: skips redundant cmake compilation across CI runs
- **`pmetal_hub::resolve_model_path`** adopted across `core`, `cli`, and `gui` for consistent local-cache → Hub-ID resolution
- **Distillation orchestration stubs removed**: `Distiller::run_online`/`run_offline`/`run_progressive` deleted — orchestration now lives entirely in `pmetal-trainer`
- **MLX MoE routing audit**: confirmed no `argpartition(-scores, -k)` anti-top-k regressions remain; documented as a permanent footgun
- **`MergeMethod` trait** extended with `merge_named(name, …)` (default forwards to `merge`); Fisher and RegMean dispatch through name-aware path
- **MSRV**: Raised workspace `rust-version` to 1.89 to match the updated `turbomcp` dependency

### Fixed

- **Release GUI workflow**: Installs the Tauri frontend with pnpm before `tauri-action`, matching the package manager that the action detects from `pnpm-lock.yaml`
- **MCP adaptive training controls**: LR/checkpoint/stop commands now handle `--output=...` jobs and create the control directory before writing `.lr_control.json`
- **AdamW bias correction**: step counter was advancing per-parameter instead of per-step
- **Cross-entropy loss masking**: ignored labels are masked before gather, using a selective `logsumexp - target_logit` path that avoids materializing full `log_softmax`
- **Gradient clipping** in compiled training path now uses `_clipped` step variants
- **FFI exception safety**: ~33 C++ bridge functions wrapped in try/catch
- **LoRA inference segfault**: put_along_axis crash during generation
- **UTF-8 char boundary panics** in inference/GUI output stream handling
- **Distributed ring reduce**: all-gather chunk indexing used wrong offset, corrupting gradient aggregation
- **Distributed transport/compression audit**: namespace PSK handshakes, TCP fallback hardening, bounded compressed-gradient deserialization, and out-of-range sparse-index guards
- **Serving parameter validation**: OpenAI-compatible routes validate sampling parameters before streaming and non-streaming generation dispatch
- **select_axis parameter order**: standardized (data, index, axis) across all call sites
- **Lazy array segfaults**: diffusion sampler now evals sigmas/timesteps before slice access
- **Sampling penalties**: correctly wired through native bridge decode path
- **Qwen3-Next MoE routing**: corrected anti-top-k expert selection bug (sign/slice pair)
- **Qwen3-Next hybrid cache flag** in LoRA + distillation paths
- **TurboQuant d128 pass-2** cross-simdgroup reduction; lazy-transpose footgun in mixed-precision attention
- **TurboQuant serving prefix-cache compatibility**: prefix cache now stores forked KV caches instead of dense snapshots, preserving compressed TurboQuant history without fp16 re-inflation
- **Native attention correctness/perf audit**: fail-fast TurboQuant dispatch, checked SDPA wrappers, true hot-ring cache behavior, centralized quantized tuple growth, and unsupported-cache rejection in tree verification
- **GKD `compute_weighted`** no longer scales by `(1-λ)` (silently zeroed training at λ=1.0)
- **`pmetal_serve::sse::IncrementalDecoder`** prevents UTF-8 boundary panics on partial codepoint emission
- Zero clippy warnings across entire workspace
- Stale `pmetal-mlx-sys` metallib path in release workflow (renamed to `pmetal-bridge`)

### Removed

- **mlx-rs dependency**: Fully replaced by `pmetal-bridge` — removes ~15K lines of Rust FFI bindings
- 1065 lines of dead code: `qwen3_train.rs`, unused LoRA functions in `qwen3_native.rs`
- 5 superseded LoRA training modules
- **Dropped model support for StarCoder2, FalconH1, RecurrentGemma, and Jamba**
- **`Distiller::run_*` orchestration stubs** (now lives in `pmetal-trainer`)

## [0.4.0] - 2026-03-23

### Added

- **`pmetal-mcp` crate**: Full MCP (Model Context Protocol) server exposing 45 tools for Claude Desktop and other MCP clients. Covers all pmetal functionality — training, inference, distillation, GRPO, RLKD, quantization, model merging, dataset operations, evaluation, benchmarking, model search, and Ollama export
  - **Device & models**: `device_info`, `search_models`, `download_model`, `list_local_models`, `model_fit`, `model_info`
  - **Inference**: `generate` (blocking), `chat` (via running serve instance), `start_serve`, `benchmark`, `bench_train`, `bench_gen`, `bench_corpus`
  - **Training**: `train`, `distill`, `grpo`, `rlkd`, `embed_train` — all as background jobs with full parameter coverage matching the CLI
  - **Runtime training control**: `job_set_lr`, `job_reduce_lr`, `job_reset_lr`, `job_save_checkpoint`, `job_graceful_stop` — LLM-driven adaptive training via the control file protocol
  - **Job management**: `list_jobs`, `job_status`, `job_logs`, `stop_job`
  - **Dataset ops**: `dataset_analyze`, `dataset_preview`, `dataset_validate`, `dataset_download`, `dataset_convert`, `dataset_filter`, `dataset_split`, `dataset_merge`, `dataset_sample`, `dataset_template`, `dataset_prepare`
  - **Quantization & conversion**: `quantize`, `fuse_lora`, `merge_models`, `pack_experts`, `ollama_create`, `ollama_modelfile`
  - **Evaluation**: `eval_perplexity`
  - All tools include rich `#[description]` annotations for parameter documentation in the MCP schema
  - Standalone binary (`pmetal-mcp`) for Claude Desktop + `pmetal mcp` subcommand (behind `mcp` feature flag)
  - Uses `turbomcp` v3.0.7 from crates.io

- **Runtime training control protocol**: Extended the control file protocol (`.lr_control.json`) with `SaveCheckpoint` and `GracefulStop` commands. The adaptive LR controller now polls the control file before checking its `enabled` flag, so external agents (MCP, TUI) can always send commands regardless of whether automatic detection is active

- **`--no-adaptive-lr` flag**: Disables automatic spike/plateau/divergence detection while keeping the control file protocol active. Enables fully LLM-driven learning rate control — the agent observes loss via `job_status` and manually adjusts LR via `job_set_lr`/`job_reduce_lr`

- **UltraFusion execution planner** (`pmetal-distributed`): Per-die stage planner for M-series Ultra Macs with in-memory channel transport backend for same-process links, avoiding TCP overhead on UltraFusion interconnect

- **MPP FlashAttention for head_dim 64/96**: Metal 4 MPP flash attention kernel now supports head_dim 64, 96, and 128 with stride-2/stride-3 SIMD lane packing and causal/non-causal variants

- **Tuna persistent disk cache**: The auto-tuner now persists benchmark results to disk, avoiding re-tuning on restart. Expanded search covers FlashAttention, FusedCrossEntropy, FusedNormLora, and FusedSwiGLU via function constants

- **MoE GPU top-k selection**: Expert top-k selection moved from CPU sort to GPU `argpartition_axis`, eliminating a sync point in the MoE forward path

- **`bench-workload` CLI command**: Benchmark a real cached workload for inference and short LoRA training with named presets (`--preset dense-qwen3`, `--preset hybrid-qwen3next`)

- **KV cache quantization auto-select**: `--kv-quant` is now optional — omitting it auto-selects the fastest quantization mode that fits the device memory budget

- **UltraFusion info display**: `pmetal info` shows UltraFusion topology, die count, and local executor plan on Ultra Macs

- **Qwen3 LoRA RoPE reset**: Qwen3 LoRA and QLoRA gain dense attention and RoPE reset support

- **ANE real-time evaluation**: Experimental `_ANEClient` real-time dispatch with automatic fallback to standard evaluation on failure. Propagated via `--ane-real-time` CLI flag

- **`bench-corpus` CLI command**: Structured kernel benchmarking with device-tier-aware test cases, JSON reporting, and `--quick`/`--output` flags

- **GPU memory bandwidth probing**: Real GPU copy benchmark replaces static spec-table lookup, with disk-cached results and spec-table fallback

- **Persistent runtime kernel backend selection**: Benchmark-and-persist infrastructure races MLX vs MPP backends on Apple10/M5, validates numerical agreement, and caches the winner to disk for 4-bit quantized linear, fused attention, and LoRA matmul

- **MPP kernel tile variants**: Metal 4 GEMM supports parameterized tile variants (32x32, 64x32, 32x64, 64x64) with Tuna auto-tuner selection per device and problem shape

- **Serve ANE/CPU-hybrid engine caching**: Serve engine auto-selects optimal backend (ANE, CPU-hybrid, GPU) at startup with permanent downgrade on failure. Compiled engines cached across requests

- **Rollback enabled by default for LoRA**: Best-loss checkpoint rollback now defaults to on with extended warmup grace period. Persistent snapshot to disk via atomic write. `for_lora()` factory for recommended defaults

- **Extended StepMetrics**: `gpu_fwd_bwd_ms`, `optimizer_ms`, `io_staging_ms`, `overhead_ms` fields for fine-grained training profiling

- **Zero-copy MoE expert dispatch**: `ExpertBufferPool` with `read_experts_aligned` + `encode_expert_aligned` for pread-to-Metal expert weight dispatch. Auto-enable KV-Q8 when memory-constrained

- **ANE dual-die support**: On UltraFusion chips, compile variant-B kernel set with distinct MIL hashes and alternate per step for dual-die thermal distribution. Auto-recompile on throughput degradation (>15% or >25K dispatches)

- **Batched parameter eval**: Model dispatcher evaluates parameters in batches of 128 tensors per sync instead of all-at-once, reducing peak memory during model loading

- **Architecture enhancements**: DeepSeek V3/V3.2, GPT-OSS, Jamba, Llama 4, Qwen3, and Qwen3-MoE model improvements and weight sanitization refinements

- **Third-party attribution**: Complete THIRD_PARTY_NOTICES with entries for all incorporated third-party code

### Changed

- **ANE is now opt-in**: The `--no-ane` flag has been replaced with `--ane` across CLI, TUI, orchestrator, and MCP. ANE training is experimental and limited to small models, so it defaults to off. The orchestrator's `DispatchConfig` now sets `ane: false` by default
- **Gradient checkpointing support corrected**: Qwen3 and Qwen3Next no longer claim gradient checkpointing support (was incorrectly advertised)
- **Training loop refactored**: Gradient checkpointing helper extracted, step logging tracks step numbers correctly, training loop tests expanded

### Removed

- **Merge methods**: Removed merge methods with incompatible licenses. Cleaned up related references across documentation and configuration

### Fixed

- **MetalSampler use-after-free**: Retained source logits array until GPU completion in serve engine
- **Fused merge Tuna cache**: Now uses persistent disk cache instead of ephemeral per-session tuning

## [0.3.13] - 2026-03-22

### Added

- **Warmup-aware adaptive LR**: Grace period now automatically extends to cover the LR scheduler's warmup duration via `set_warmup_steps()`, preventing false divergence triggers during the normal LR ramp. Backed by ZClip/SPAM research — loss increasing during warmup is expected behavior
- **`WarmupCapped` LR event**: Optional early warmup monitoring (disabled by default) for pre-training runs where loss rise during warmup may indicate problems. Enable with `warmup_max_loss_increase: 0.03-0.05`

- **Metal 4 / MPP kernel suite**: 8 Metal Performance Primitives kernels for M5 (Apple10) NAX hardware acceleration, compiled as a separate `pmetal_kernels_metal4.metallib` with automatic runtime dispatch
  - `mpp_gemm.metal`: Core GEMM with Morton ordering, fp16/fp32 variants, and alpha/beta accumulation via cooperative tensor postfix fusion (BK=128 K-loop per MPP Guide Section 2.3.4)
  - `mpp_flash_attention.metal`: FlashAttention-2 with both QK and PV block GEMMs via matmul2d — QK uses 32x32 tiles, PV uses 4 chunks of 32x32 for D=128, P stored as half for PV matmul (30KB threadgroup memory budget)
  - `mpp_fused_swiglu.metal`: Fused SwiGLU MLP — gate and up projections via matmul2d into threadgroup tiles, SwiGLU activation as cooperative post-step
  - `mpp_fused_lora.metal`: Fully fused LoRA forward — base projection via matmul2d, xA computed cooperatively in threadgroup scratch (shared across output elements), LoRA overlay added per-element. Training variant saves xA for backward pass
  - `mpp_fused_norm_lora.metal`: Fused RMSNorm + Linear + LoRA — SIMD cooperative RMS reduction, vectorized norm+dot for base projection, xA computed once and shared via threadgroup scratch
  - `mpp_grouped_gemm.metal`: MoE grouped GEMM with per-expert Morton ordering for LLC cache locality, sequential expert-offset tile lookup
  - `mpp_dw_gemm.metal`: ANE training weight gradient GEMM — simple overwrite and alpha/beta accumulation paths with cooperative tensor postfix fusion
  - `mpp_quantized.metal`: NAX quantized inference — 4-bit (on-the-fly dequant + matmul2d, BK=32) and 8-bit (on-the-fly dequant with per-group scale, BK=64) variants
- **Dual metallib build system**: Conditional Metal 4 compilation (`-std=metal4.0 -target air64-apple-macos26.0`) when Metal compiler >= 400 and SDK >= 26.0, with `has_metal4` cfg flag for Rust-side conditional compilation
- **Metal 4 pipeline cache**: `PipelineCache::load_metal4_library()`, `get_or_create_metal4_pipeline()` with function constant support and `"metal4:"` key prefix
- **NAX detection**: `DeviceProperties::has_nax()` (Apple10+ / architecture gen >= 17), automatic Metal 4 library loading when NAX is available
- **MPP GEMM Rust dispatch** (`mpp_gemm.rs`): `MppGemm` with `is_available()` check, type-erased `execute(&dyn AsMetalBuffer)`, Morton ordering via function constants, linearized 1D grid
- **Benchmark infrastructure** (`mpp_bench.rs`): `bench_gpu_op()` with warmup/timed iterations, `bench_comparative()` for Metal 3 vs Metal 4 side-by-side, `GemmBenchConfig` with standard problem sizes from decode (M=1) to training (M=2048)
- **Generic FP8 quantization** (`fp8_utils.rs`): `quantize_model_linears()` via `ModuleParameters` trait traversal — works for all 18 architectures

### Changed

- **Crate consolidation**: `pmetal-cli` merged into `pmetal` crate — SDK facade + CLI binary in one crate. `cargo build -p pmetal` now builds the binary (feature-gated behind `cli`). Library-only usage: `--no-default-features --features core`
- **Adaptive LR defaults (conservative)**: `divergence_slope_threshold` 0.05 → 0.005, `warmup_fraction` 0.15 → 0.25, warmup monitoring disabled by default. Grace period tied to actual LR scheduler warmup_steps instead of a fixed fraction of total steps
- **Metal 3 fused_swiglu_forward_f16**: Rewritten with SIMD cooperative reduction (was per-thread independent dots)
- **Metal 3 fused_cross_entropy**: Label smoothing enabled in SIMD path, `use_simd` now defaults to true for all vocab sizes
- **Metal 3 grouped_gemm backward_dx**: Replaced untiled O(M*N*K) inner loop with BLOCK_K-sized N-reduction strips using threadgroup staging
- **FP8 dispatcher**: Changed catch-all error to 18 explicit match arms calling generic `quantize_model_linears()`

### Fixed

- **mpp_fused_norm_lora.metal 512KB threadgroup alloc**: `threadgroup half norm_tile[64*4096]` (524KB) exceeded 32KB limit — would crash pipeline creation. Rewrote to use 68-float scratch buffer
- **mpp_fused_norm_lora.metal LoRA O(R*H) per output**: LoRA recomputed `xA` from scratch for every output element. Now computes xA once cooperatively and shares via threadgroup scratch
- **mpp_quantized.metal 8-bit unscaled output**: Per-group scale was deferred to nonexistent "separate pass". Rewrote with accumulation loop applying scale during on-the-fly dequantization
- **mpp_gemm.rs buffer type mismatch**: `execute()` took `MetalBuffer<f32>` but dispatched to f16 kernels. Changed to `&dyn AsMetalBuffer` (type-erased)
- **mpp_gemm.rs accumulate buffer binding**: Accumulate kernel (A=0, B=1, C=2, D=3, params=4) was bound with params at index 3 and missing D buffer. Fixed index mapping
- **mpp_flash_attention.metal scalar PV GEMM**: O += P @ V used per-element scalar loops despite header claiming block GEMM. Implemented via matmul2d with 4 chunks of 32x32 for D=128
- **mpp_fused_lora.metal stub LoRA phases**: Only base projection was implemented. Added cooperative xA computation + per-element LoRA overlay for both training and inference variants
- **matmul2d_descriptor tile size mismatch** (prior diligence): 6 kernels had `desc(32,32)` producing 32x32 threadgroup tiles but Rust dispatch strided by 64 — 75% of output uncomputed. Fixed to `desc(64,64)` except FlashAttention (correctly 32x32 for Bq=Bk=32)
- **Adaptive LR divergence threshold too high**: `divergence_slope_threshold` of 0.05 required ~200% loss increase over 40 steps — effectively dead code. Reduced to 0.005 (20% over 40 steps)
- **Adaptive LR step counter in run_packed**: Accumulated losses were all fed to the adaptive controller with the same batch-end step number. Now retroactively sets the correct per-step value so the grace period and detection windows align properly
- **Clippy `needless_return`**: Removed bare `return` in `pmetal-metal/build.rs` match arm

## [0.3.12] - 2026-03-21

### Added

- **MLX memory management**: Wire real MLX Metal memory API via `mlx_rs::memory` (published in pmetal-mlx-rs 0.25.8). Exposes `clear_cache`, `get_active_memory`, `get_peak_memory`, `get_cache_memory`, `get/set_memory_limit`, `set_cache_limit`, `set_wired_limit`, `reset_peak_memory`. The previous `clear_cache()` was a complete no-op — MLX buffer cache was never freed
- **Memory diagnostics**: `log_memory_stats()` reports active/cache/peak/limit at model load, training start, and completion for visibility into Metal allocator state
- **LoRA**: Implemented dynamic QLoRA and fused metal kernels
- **KV cache quantization** (SOTA inference): q8_0 KV cache is now the default for inference and serving — community benchmarks confirm <0.4% PPL degradation with 12-38% throughput gain
  - Symmetric quantization: `--kv-quant 8` (default), `--kv-quant 4` for aggressive savings
  - Asymmetric K/V quantization: `--kv-k-bits 8 --kv-v-bits 4` — K is more sensitive than V, asymmetric gives near-q4 memory savings with near-q8 quality
  - `--kv-group-size` (default 64), `--no-kv-quant` to disable
  - `CacheMode::Quantized` and `CacheMode::AsymmetricQuantized` variants wired into `KVCache::update_and_fetch()` via per-layer `QuantizedKVCache` delegation
  - `CacheMode::describe()` for human-readable display in logs
  - `DynamicModel::create_cache_with_mode()` with automatic group_size adjustment for non-standard head dimensions (Phi-3 mini head_dim=96, NemotronH head_dim=32)
  - Serve engine defaults to q8_0 KV cache for all requests
- **Context-aware fit estimation**: Efficiency factor is now context-dependent (0.60 dense / 0.50 MoE base, log-linear penalty above 8k context) instead of flat 0.55. KV cache memory calculation accounts for quantization bits. Fit notes recommend q8_0 when memory is tight

### Changed

- **Trainer**: Modularized training loop and configured experimental trainers
- **MLX**: Moved `kv_cache` to a module and updated gated delta
- **Style**: Formatted CLI, Hub, Models, and Serve crates
- **Cleanup**: Removed `easy_reference.rs`

### Fixed

- **Training memory explosion (72 GB peak for 0.6B model → 23 GB)**: Three root causes identified and fixed:
  1. Computation graphs accumulated across deferred evaluation steps. Packed and compiled training paths deferred eval for 10 steps, keeping ~10 full forward+backward graphs in memory simultaneously. Now evaluates each step immediately
  2. Gradient accumulation kept backward graphs alive. Each micro-batch's gradient arrays held references to the entire backward computation graph until gradients were applied. Now evaluates accumulated gradients after each micro-batch
  3. `eval_params(model.parameters())` evaluated all 600M+ frozen base model params every step. Changed to `eval_params(model.trainable_parameters())` — only LoRA adapters (~1-10M params)
- **ANE training attempt doubled memory for LoRA/QLoRA**: The orchestrator always attempted ANE training first (loading full model weights), failed (ANE is incompatible with LoRA adapters), then loaded the GPU LoRA model — keeping both in memory. Now skips ANE entirely for LoRA/QLoRA
- **GUI training defaults misaligned with CLI**: GUI used `batch_size=4` (CLI: 1) and `gradient_checkpointing=false` (CLI: true), causing 5-10x more memory usage for identical models. Defaults now match CLI
- **GUI training logs silently dropped**: Tracing filter only showed `pmetal_gui` crate logs. Training progress from orchestrator, trainer, and model crates was invisible. Now shows all pmetal crate logs
- **MLX buffer cache not freed after training**: `clear_cache()` now runs after training ends (success, error, or cancel) in both orchestrator and GUI
- **KV cache `Quantized` mode was a no-op**: `CacheMode::Quantized` existed in the enum but `KVCache::update_and_fetch()` treated it identically to `Standard`. Now properly delegates to per-layer `QuantizedKVCache` instances with quantize-on-write / dequantize-on-read
- **KV cache quantization crash on non-standard head dimensions**: Models with head_dim not divisible by 64 (Phi-3 mini=96, NemotronH=32, FalconH1=16) would fail in MLX's `quantize` op. `create_cache_with_mode()` now auto-adjusts group_size to the largest compatible power-of-2

## [0.3.11] - 2026-03-20

### Added

- **Production serving engine** (`pmetal-serve`): Complete rewrite from greedy-only PoC to production-quality inference server
  - SOTA GPU-native sampling via `pmetal_models::Sampler`: temperature, top-k, top-p, min-p, repetition/frequency/presence penalties, seeded RNG
  - True token-by-token streaming via `tokio::sync::mpsc` channels — first byte reaches client as GPU produces it (no collect-then-emit)
  - `spawn_blocking` generation — HTTP event loop stays responsive during prefill/decode
  - Client disconnect detection — generation thread stops immediately when receiver drops
  - Input validation: bounds checking on all sampling params (rejects NaN, Inf, out-of-range), max_tokens clamped to max_seq_len
  - OpenAI API: `stop` field accepts both `"string"` and `["array"]` formats, `system_fingerprint` field in responses, request-time `created` timestamps
  - Stop token collection via `collect_all_stop_tokens()` (merges generation_config.json + chat template + tokenizer + well-known probes)
  - Multi-token stop strings filtered with warning (only single-token stop strings supported)
- **qdot MoE Metal kernels** (`fused_moe.metal`): Rewrote all 5 kernels with pre-scaled activation technique — ~30-40% compute reduction, thread-local x caching, register-only design (no threadgroup shared memory overflow), 64-thread threadgroups with function constant specialization
- **Deferred GPU dispatch**: Encode all K experts in single command buffer with one submit instead of K separate GPU flushes
- **Persistent IO thread pool** (`expert_io.rs`): Persistent workers via mpsc channels with zero-copy aligned pread into `AlignedBuffer` (2MB posix_memalign + `newBufferWithBytesNoCopy`)
- **Async expert prefetcher**: Background IO thread with ownership transfer via `Option::take` (no clone)
- **Real benchmark harness**: `--benchmark` / `--benchmark-iters` on `pmetal infer` runs real per-token forward passes with GPU sync, reports mean/min/p50/p99 decode latency
- **forward_offloaded wired end-to-end**: Full pipeline: route → pread K experts → parse → GPU dequant (single CMD buffer) → combine

### Fixed

- **Reasoning dataset training producing no `<think>` tags**: When GUI/CLI set custom text columns (`thinking`, `solution`) with prompt column `problem`, the Custom format path bypassed the Reasoning format's `<think>`/`</think>` tag injection. Auto-detection now routes reasoning-pattern columns to the Reasoning format. Also fixed prompt-column loss masking when prompt isn't in text columns (prompt is now prepended to training text)
- **Adaptive LR controller too aggressive**: Divergence detection was firing on normal training noise, crushing LR from 2e-4 to 1e-7 within 10% of training. SOTA-aligned overhaul:
  - Divergence confirmation: requires 2 consecutive positive-slope windows before triggering (was: single window)
  - Divergence cooldown: 80 steps after reduction before re-checking (was: immediate re-check after 40-step window refill)
  - Max divergence reductions: capped at 4 (was: unlimited cascading)
  - ZClip-style spike exclusion from EMA: detected spikes no longer inflate the EMA threshold
  - Gentler reduction factor: 0.7 per reduction (was: 0.5)
  - Increased grace period: 15% of training steps (was: 10%)
- **GUI LoRA inference producing garbage** (`_framework` token repeated): GUI was not reading `target_modules` from adapter_config.json (all modules got rank=16 instead of only attention) and was not merging LoRA weights before inference. Now reads `target_modules`/`use_rslora`, calls `merge_lora()` + `eval_all()` matching the working CLI path
- **GUI fuse with cached models failing**: "Fuse with remote base models is not supported" error removed — now calls `resolve_model_path()` to download/resolve, matching CLI behavior
- **Fused model not recognized by MLX-format loaders**: Three fixes:
  - Safetensors metadata now uses `format: mlx` (required by MLX-format loaders on macOS)
  - Generates `model.safetensors.index.json` with weight map (required for model discovery)
  - Fused weights preserve base model dtype (bf16/f16) instead of upcasting to f32 — halves output file size
- **`adapter_config.json` missing `base_model`**: Training now saves `base_model` field in adapter config (distillation, GRPO, RLKD paths). Enables auto-detection of base model from LoRA adapter
- **Serve security hardening**: Default bind `127.0.0.1` (was `0.0.0.0`), sanitized error responses (no internal MLX paths leaked), 2MB request body limit, removed unnecessary `unsafe impl Sync for ModelState`
- **Serve extra forward pass**: Decode loop restructured to not run a wasted forward pass after the final token
- **Serve SSE error handling**: `[DONE]` no longer emitted after `TokenEvent::Error` — stream ends with error event only
- **Serve UTF-8 streaming**: Token buffer accumulates and decodes together to prevent garbled multi-byte characters at BPE boundaries

### Changed

- **GUI fuse modal UX**: "Cancel" becomes "Done" after successful fuse with "Fuse Another" option. Base model field is read-only, auto-detected from LoRA adapter's `adapter_config.json`
- **GUI inference auto-select**: Selecting a LoRA adapter in inference automatically selects the matching base model (mirrors training page behavior)
- **Deleted MoE combine bridge**: Removed `FusedMoeCombine` Metal kernel — the 6 MLX ops are already async on GPU; the Metal side-channel added sync barriers making it 5-20x slower
- **GUI streaming inference**: Token-by-token streaming with full stop-token and sampling-config support
- **CLI `--loss-scale` flag**: Gradient scaling for ANE training at >350M params
- **Comprehensive documentation site** (`docs/`): Getting-started, installation, hardware, models, training, CLI reference (21 commands), configuration, SDK, Python, and contributing guides
- **Metal GPU backward kernels** (`dw_gemm.metal`): Tiled fp32 SGEMM for weight gradient GEMMs in ANE training — single `BatchedCommandBuffer` per step
- **GUI adapter discovery**: Scans `~/pmetal-output/` for trained LoRA adapters with rank, alpha, base model metadata
- **GUI adapter dropdowns**: Fuse modal and inference page replace manual path entry with adapter select dropdowns
- **GUI chat template support**: Model-specific chat templates via `detect_chat_template()`
- **`save_adapter_config_with_base()`**: Adapter config now includes `base_model` field

### Removed

- **`pmetal::easy` module** (breaking): Removed in favor of direct `pmetal-models` / `pmetal-hub` / `pmetal-data` usage

## [0.3.10] - 2026-03-18

### Added

- **Training orchestrator** (`pmetal-trainer::orchestrator`): Single `run_training()` entry point replaces four separate training pipeline implementations (CLI ~1000 lines, GUI ~190 lines, easy API ~200 lines, TUI bridge ~70 lines). All consumers now share one canonical pipeline with: ANE training with GPU fallback, QLoRA and standard LoRA, all dispatch modes (packed, compiled, metal-fused, standard), adaptive LR, checkpointing, metrics callbacks, and phase status reporting. Net -1300 lines of duplicated pipeline code
- **`TrainingJobConfig` struct**: Replaces 38 positional parameters with a typed config struct. Includes `DispatchConfig` (optimization flags), `QLoraOrchConfig` (quantization), `TrainingPhase` enum (status reporting), and `PhaseCallback` trait (GUI/TUI status wiring)
- **ANE large-vocab support** (`VocabMap::from_token_ids`, `VocabMap::remap_u32`): ANE training now correctly handles models with vocab > 65536 (e.g. Qwen3 @ 151936). Token IDs are processed as u32 through VocabMap compaction before converting to the u16 format required by ANE IOSurface operations. Previously, u32→u16 casting silently truncated IDs above 65535, corrupting embeddings and gradients
- **ANE first-class metrics**: ANE training path now wires `MetricsJsonCallback` with per-step metrics (loss, tok/s, ANE timing breakdowns), config JSON, and user-provided callbacks (cancel support). Previously ANE produced no metrics output, making GUI/TUI appear stuck during ANE training

### Fixed

- **GUI metrics not updating during training**: Metrics file watcher now detects file truncation (from ANE→GPU fallback) and resets read position. Previously `last_pos` exceeded the new file length after truncation, causing the watcher to skip all new data indefinitely
- **GUI output path relative to process cwd**: Training output now resolves to `~/pmetal-output/` instead of relative to the GUI's working directory (`crates/pmetal-gui/src-tauri/`). Absolute paths from the frontend are preserved as-is
- **GPU metrics callback truncates ANE metrics**: GPU `MetricsJsonCallback` creation moved to after ANE attempt completes, so ANE metrics aren't wiped on fallback
- **GUI drops warmup/lr_schedule/save_steps/logging_steps**: All four fields from the GUI training config DTO are now properly mapped to `TrainingConfig` instead of falling through to defaults
- **Phase status not visible in GUI**: Added `tokio::task::yield_now()` after each phase emit in pre-MLX orchestrator phases and ANE path, allowing the tokio runtime to deliver status events between blocking operations
- **TUI missing `embedding_lr` and `lr_schedule` parsing**: Direct training path now parses `--embedding-lr` and `--lr-schedule` args that were previously ignored
- **Easy API drops `embedding_lr`**: `FinetuneBuilder` now maps `embedding_lr` into `TrainingConfig.embedding_learning_rate`

- **GUI live training dashboard**: Full-screen live view replaces the config form when training is active. Includes real-time loss curve (SVG), metric cards (loss, best loss, tok/s, LR, grad norm, progress %), run details panel with hyperparameters, and progress bar. Config form returns when training stops
- **GUI cached dataset dropdown**: Dataset selector uses a `<select>` dropdown (matching the model selector style) with cached HuggingFace datasets, plus a text input for custom paths or HF dataset IDs
- **GUI dataset column picker**: When a dataset is selected, columns are auto-detected and shown in dropdowns for text, prompt (loss masking), and format selection. Falls back to manual text input when columns can't be detected
- **Multi-column dataset support** (`--text-columns col1,col2`): Concatenate multiple JSONL columns as training text. CLI: `--text-columns thinking,solution --column-separator "\n\n"`. GUI: ordered pill builder with add/remove/reorder. All training paths (Train, Distill, GRPO, RLKD) support column flags uniformly via shared `build_column_config` helper
- **Custom dataset columns** (`--text-column`, `--prompt-column`, `--response-column`): CLI, GUI, TUI, and easy API support arbitrary JSONL column names via `DatasetFormat::Custom`. Prompt column enables loss masking; prompt+response columns concatenate with masking. Distill, GRPO, and RLKD commands now also accept column flags
- **Unified `from_jsonl_tokenized`**: Merged `from_jsonl_tokenized` and `from_jsonl_tokenized_with_columns` into a single method with `columns: Option<&DatasetColumnConfig>`. All 18 call sites updated. DRY across CLI, TUI, GUI, easy API, and Python bindings
- **Dataset statistics and seq len validation**: `DatasetStatistics` with min/max/mean/median/p95/p99 lengths, truncation count/percentage, and suggested `max_seq_len`. `validate_seq_len()` warns when >10% truncated or mean length much shorter than max_seq_len. Logged for all training paths: Train, Distill, GRPO, RLKD, and easy API
- **`peek_dataset_columns`**: Tauri command + API function — reads first JSONL record and returns field names for the GUI column picker
- **GUI training status phases**: Live status messages during setup ("Loading model...", "Loading dataset and tokenising...", "Training...") so users see what's happening before metrics arrive
- **GUI training config summary**: Hyperparameters (LR, batch, seq len, LoRA rank, packing, flash attention) displayed in the active training banner and run detail panel
- **GUI failed-run alerts**: Failed training runs immediately surface with error message in a red banner, no longer hidden until the user clicks stop
- **GUI auto-updater**: Tauri updater plugin with signed update artifacts and `latest.json` manifest in GitHub releases
- **TUI setup phase indicator**: Dashboard shows "Loading model and preparing dataset..." in loss chart and stats panel while model loads, before any metrics arrive
- **TUI `JobPhase` event**: New `AppMsg::JobPhase` message propagates setup status from the command runner to the dashboard
- **TUI dataset peek**: Shows detected columns, estimated token lengths, and seq len warnings when a dataset is selected in the training form
- **GUI seq len warnings**: Contextual warnings under the max seq len input — red (most samples truncated), amber (some truncated), blue (wasteful). Shows "Based on first N rows" with a "check all rows" button that scans the full dataset on the backend
- **GUI retry button**: Completed/cancelled/failed runs show "Retry with these settings" which loads the run's config back into the form for adjustment and re-launch
- **Easy API `on_status()` callback**: Reports granular setup phases (resolving model, resolving dataset, loading tokenizer, tokenizing dataset, loading LoRA adapters, training) — wired to GUI for real-time phase display
- **`find_cached_model` / `find_cached_dataset`** (`pmetal-hub`): Fast local cache lookup for HF repos without network calls

### Fixed

- **Cached models re-downloaded on every training start** (`pmetal-hub`): `download_model` and `download_dataset` now check the local HF cache (`~/.cache/huggingface/hub/`) before making any network calls. If a valid snapshot exists, the cached path is returned instantly. Eliminates ~10s startup latency for cached models across all consumers (CLI, TUI, GUI, easy API, Python SDK)
- **Seq len suggestions use next-multiple-of-64** instead of next-power-of-2, producing practical values (7168 instead of 8192) for GPU-aligned training
- **Non-string dataset columns crash** (`parse_custom_line`): Selecting a column containing an array (e.g. OpenAI `messages` chat format) or number crashed with "not a string". Now handles all JSON types: arrays of message objects auto-extract role+content, numbers/booleans convert to string, other types serialize to JSON. Relates to #2

- **GUI/API trending models and datasets stale**: Changed HuggingFace API sort from `sort=downloads` (all-time) to `sort=trending` for default browse views. Search queries still sort by downloads. Fixed hardcoded User-Agent version string to use `CARGO_PKG_VERSION`
- **HF dataset ID resolution** (`pmetal-data`, `easy.rs`, `commands.rs`, `main.rs`): HuggingFace dataset IDs (e.g., `nohurry/Opus-4.6-Reasoning-3000x-filtered`) and local HF cache directories are now resolved to the actual data file within. Traverses `snapshots/{hash}/` structure, follows symlinks, finds `.jsonl`/`.json`/`.parquet`/`.csv`/`.arrow` in priority order
- **Dataset directory passed as file path**: All three resolution sites (`easy.rs`, GUI `commands.rs`, CLI `main.rs`) now call `resolve_dataset_path_pub` for `DatasetSource::Local` directories instead of passing them as-is to `from_jsonl_tokenized`
- **Metrics not appearing in GUI/TUI**: `log_every` changed from 10 to 1 in `easy.rs` so metrics appear after the first training step. `MetricsJsonCallback` now flushes every step for the first 20 steps (then every 5), ensuring watchers see data promptly
- **`train_start` event handling in GUI**: `apply_metrics_to_training` now recognizes the `train_start` event, sets status message, and reads `total_epochs` from step metrics
- **Watcher task leak on training completion** (GUI + TUI): `finalize_training_run` (and distillation/GRPO variants) now sets `cancel_flag = true` so the 500ms metrics-polling task exits. TUI `CommandRunner::remove()` now calls `job.cancel.cancel()` before dropping
- **QLoRA re-resolves dataset**: `run_qlora_training_in_process` now receives the pre-resolved `PathBuf` instead of re-downloading from HuggingFace on every run
- **Stale status message on failure**: `finalize_training_run` clears `status_message` to `None` so "Loading model..." doesn't overlay the error message
- **README.md failure aborts dataset download** (`pmetal-hub`): README failures are now non-fatal warnings; only data file download failures abort
- **MLX mutex crash on GUI exit**: Added `on_window_event(Destroyed)` handler that calls `std::process::exit(0)` to skip C++ destructor crashes from MLX Arrays dropped on the wrong thread
- **TUI log corruption**: Tracing subscriber suppressed in TUI mode to prevent stderr writes from corrupting the raw terminal. Optional `PMETAL_LOG_FILE` env var for file-based debug logging (with graceful fallback on bad paths)
- **`log_lines` dead code removed**: Removed unused `log_lines` field from `TrainingRun`, `DistillationRun`, and `GrpoRun` GUI state structs

### Changed

- **Release workflow**: Added Tauri signing keys, updater artifacts (`.tar.gz` + `.sig`), and `latest.json` manifest generation for auto-updates
- **GUI Cargo.toml**: Added `tauri-plugin-updater` and `tauri-plugin-process` dependencies

## [0.3.9] - 2026-03-17

### Added

- **RLKD CLI command** (`pmetal rlkd`): Reinforcement Learning with Knowledge Distillation — combines GRPO policy gradient optimization with distillation from a frozen teacher model. CLI exposes `--alpha`, `--final-alpha`, `--anneal-alpha`, `--top-k-distill`, all SFT/LoRA arguments, and `MetricsJsonCallback` integration
- **Embedding training CLI command** (`pmetal embed-train`): Sentence-transformer fine-tuning for BERT/encoder models with contrastive losses (InfoNCE, Triplet, CoSENT). Supports pair and triplet datasets, configurable pooling (CLS, Mean, LastToken), L2 normalization toggle, and automatic tokenizer/config copying to output
- **GRPO VLM mode** (`--vlm`): Vision-Language Model support for GRPO training with image inputs. Loads images from dataset `images` field, passes to reward functions, uses `forward_with_images` for multimodal forward passes. Configurable `--max-image-size`
- **GRPO ML reward model** (`--reward-model`): Pretrained reward model scoring during GRPO. Loads from local path or HuggingFace ID, runs inference-only alongside heuristic rewards. Configurable `--reward-model-weight`, `--reward-model-max-length`, and `--reward-model-template`
- **GRPO speculative decoding** (`--speculative`): Draft/verify rollout generation with 2-4x throughput improvement. Configurable `--speculative-draft-tokens` (default 3). Greedy verification for correctness guarantees
- **GRPO async reward pipelining** (`--async-rewards`): Background reward scoring concurrent with GPU training for ML reward models
- **Cut Cross-Entropy CLI flag** (`--cut-cross-entropy`): Memory-efficient loss computation for SFT training, avoiding full [batch, seq, vocab] logit materialization
- **KL-calibrated GGUF quantization** (`--kl-calibrate`): Per-tensor quantization type selection via NRMSE + cosine distance calibration. `--target-bpw` for budget-constrained quantization, `--kl-threshold` for quality control
- **GRPO TUI form fields**: VLM toggle, speculative decoding, async rewards, ML reward model path, and draft tokens exposed in the interactive TUI
- **Training TUI**: Cut Cross-Entropy toggle added to training tab form

### Fixed

- **Cut Cross-Entropy ignore index panic** (`pmetal-mlx`): `take_axis` with -100 (ignore index) targets caused out-of-bounds gather. Targets are now clamped to valid range before gather; loss masking handles ignored positions
- **Cut Cross-Entropy division by zero** (`pmetal-mlx`): `n_valid=0` (all tokens ignored) caused NaN loss. Guarded with `n_valid.max(1)`
- **Llama LoRA position IDs dropped** (`pmetal-lora`): `forward_hidden_with_positions` silently discarded position IDs, breaking packed-sequence training with non-contiguous positions. Added full position-aware path through attention, decoder layer, and model stack using `apply_rope_with_positions`
- **lm_head weight computed twice per CCE step** (`pmetal-trainer`): Training loop called `lm_head_weight()` for probe and again inside the gradient closure. Weight is now computed once and captured into the closure
- **GRPO VLM pixel_values not replicated per-completion** (`pmetal-trainer`): Images were stacked per-group instead of replicated per-completion, causing batch dimension mismatch. Images now repeat `n_completions` times per group
- **GRPO `run_async` flush skips adaptive LR** (`pmetal-trainer`): Final flush step bypassed adaptive LR, rollback logic, and callbacks. Now applies the same post-step processing as the main loop
- **CoSENT loss overflow with no positive pairs** (`pmetal-trainer`): All-zero labels caused `logsumexp(-1e9)` overflow to `+inf` and `NaN` gradients. Returns `0.0` when no positive pairs exist in the batch
- **LastToken pooling O(batch) GPU syncs** (`pmetal-models`): Per-element `.item()` loop forced one GPU-to-CPU synchronization per batch element. Replaced with vectorized `take_along_axis` + `broadcast_to` for a single gather operation
- **EmbeddingDataset silent empty strings** (`pmetal-data`): Missing text keys (`text_a`/`text_b`) silently produced empty-string training pairs. Now returns an explicit parse error with line number and expected key names
- **BERT `hidden_act` always GELU** (`pmetal-models`): `BertIntermediate::forward` ignored the `hidden_act` config field. Now dispatches to `relu`, `silu`/`swish`, `tanh`, or `gelu` (default) based on config
- **GGUF BPW budget silent non-convergence** (`pmetal-gguf`): `apply_bpw_budget` loop exhausted without warning when all tensors were at minimum quality. Emits `tracing::warn!` when target BPW is unreachable
- **Speculative decode cross-sequence early exit** (`pmetal-models`): Outer generation loop exited when any single sequence hit `max_new_tokens`, truncating other in-progress sequences. Removed `max_generated` check; per-sequence `finished` tracking now controls termination
- **Speculative decode O(seq_len) draft warm-up** (`pmetal-models`): Draft cache was rebuilt from full sequence prefix every step, making total cost O(seq_len^2). Draft caches are now persisted and incrementally advanced with only newly accepted tokens
- **Fused LoRA backward threadgroup memory** (`pmetal-metal`): `fused_lora_backward_a` kernel missing threadgroup memory size check. Added allocation guard with fallback to MLX for large `out_features`
- **LoRA+ double scaling** (`pmetal-lora`): Fused kernel and `AdamWGroups` optimizer could both apply the LoRA+ differential learning rate. Added `kernel_loraplus` flag to prevent double scaling
- **Clippy compliance**: Fixed `field_reassign_with_default` in GGUF calibration summary, `doc_overindented_list_items` in speculative decode docs

### Changed

- **RLKD stats**: Documented that `grpo_component` and `distill_component` in training stats are proportional approximations (`total_loss * (1-alpha)` and `total_loss * alpha`), not true decomposed values
- **Speculative decode bonus token**: Documented that greedy argmax for the bonus token is by design (required for speculative decoding correctness), not a sampling oversight
- **GGUF prefix subsampling**: Expanded documentation warning that prefix subsample assumes i.i.d. weight distribution, which may not hold for structured tensors
- **EmbeddingTrainer**: Added doc warnings that `encode` requires models returning hidden states (not logits) — causal LMs produce `[batch, vocab]` after pooling, which is nonsensical as an embedding

## [0.3.8] - 2026-03-17

### Added

- **Distributed training** (`pmetal-trainer`): Data-parallel gradient synchronization across Apple Silicon clusters via `DistributedGradientSync`. Flatten/all-reduce(Mean)/scatter pipeline with optional gradient compression (fp16, top-k sparsity). Integrated at all 4 training loop sites (run, run_metal_fused, run_jit_compiled, run_packed) with loss sync, epoch barriers, and rank-0-only checkpointing. Feature-gated behind `distributed`
- **Pipeline-parallel inference** (`pmetal-distributed`): Layer-range pipeline parallelism enabling models larger than single-device memory. `ShardableModel` trait decomposes forward pass into embed/apply_layer/normalize/lm_head stages. `PipelineGenerationLoop` for end-to-end autoregressive generation with `StreamMultiplexer` for concurrent request routing
- **Activation transport**: Length-prefixed wire format for hidden state transfer between pipeline stages with fp16 compression codec. `TransportReceiver::recv_vec` for dynamic-size message reception
- **Topology-aware layer assignment**: Proportional (RAM-based) and bandwidth-aware (exhaustive search for 2-3 nodes) solvers with automatic strategy selection based on cluster topology
- **Weight cache**: LRU eviction with reference counting to prevent in-use eviction, per-layer loading, and prefetch support for pipeline stages
- **OpenAI-compatible inference server** (`pmetal-serve`): Drop-in local inference backend with `POST /v1/chat/completions` (streaming SSE and non-streaming), `POST /v1/completions`, `GET /v1/models`, `GET /v1/metrics`, `GET /health`. Chat template auto-detection, stop token collection, and greedy sampling
- **Serving metrics**: Per-request timing (`RequestMetrics`) with first-token latency, total latency, and tok/s. `ServingMetrics` atomic aggregation exposed via `/v1/metrics` endpoint
- **SSE streaming**: Token-by-token Server-Sent Events with role announcement, per-token content deltas, finish_reason, and `[DONE]` sentinel per OpenAI spec
- **Speculative decoding** (`pmetal-models`): Layer-split draft+verify decoder via `SpeculativeDecoder<M: ShardableModel>`. Draft phase uses early layers (default: num_layers/3) for N-token proposals, verify phase runs full model with accept/reject on consecutive matches. `SpeculativeStats` tracks acceptance rate and tokens-per-step
- **f64-accurate LoRA merge** (`pmetal-merge`): Streaming f64 matmul via ndarray for bit-accurate delta computation. Row-by-row fused base+delta+downcast, tiled low-memory path (512-row chunks), bias merging, fan_in_fan_out transpose, overflow clamping before dtype downcast
- **RAM/RAM+ merge method**: Reinforced Agent Merging with unique/shared parameter classification and adaptive tensor-local lambda rescaling
- **Multi-SLERP merge method**: Barycentric spherical interpolation for 3+ models with iterative pairwise SLERP and weight renormalization
- **Frankenmerging config**: `OutputSlice`/`InputSlice` layer-range-based merging with per-slice merge methods, base models, and parameters. `run_merge_sliced()` execution engine with tensor name remapping
- **`ParameterSetting`**: Scalar or conditional (tensor-name filtered) merge parameters enabling per-tensor-type weight variation (attention vs mlp layers)
- **TVD distillation loss**: Total Variation Distance (`0.5 * Σ|P_teacher - P_student|`), bounded [0,1], symmetric proper distance metric
- **Hinge ranking distillation loss**: Pairwise margin-based ranking preservation over top-k teacher tokens with configurable margin
- **Logistic ranking distillation loss**: Softplus-based smooth ranking loss with better gradient flow than hinge, operates on logits for numerical stability
- **CLI `--distributed-peers`, `--distributed-auto`, `--compression-strategy`**: Distributed training flags behind `distributed` feature
- **CLI `pmetal serve --model <path> --port 8080`**: Inference server command behind `serve` feature
- **CLI `--accurate` and `--low-memory`**: Flags for f64 LoRA merge path
- **New merge methods in CLI**: `ram`, `ram_plus`, `multislerp` registered as merge method options

### Fixed

- **Alignment violation in distributed gradient sync**: `sync_gradients` and `sync_loss` previously created `Vec<u8>` buffers with align-1, but the ring backend requires align-4 for f32 operations. Fixed by reinterpreting the `Vec<f32>` buffer directly via aligned pointer cast
- **Double-framing deadlock in activation transport**: `serialize()` embedded its own length prefix AND `TransportSender::send()` added another, causing `recv_activation` to misparse messages. Removed embedded prefix; transport layer handles all framing
- **Double EMA on `running_loss` in distributed mode**: Distributed sync block re-applied EMA that `train_step` already applied, causing doubly-decayed loss values for the adaptive LR controller. Removed manual EMA update in distributed block
- **Zero weights in bandwidth-aware layer assignment**: 3+-node fallback computed `(ram / 1M) * (bw / 1M)` which produced zero for small values, causing NaN proportions. Added `.max(1)` guards
- **`argpartition` panic on ranking losses**: `k.min(vocab - 1)` could underflow when vocab=0. Added `.max(0)` guard

### Changed

- **DataLoader sharding**: `rank` and `world_size` fields for modular-arithmetic data partitioning across distributed nodes
- **Merge config system**: `ParameterSetting` type propagated to CLI merge parameter construction, supporting both scalar and conditional forms

## [0.3.7] - 2026-03-16

### Added

- **`pmetal merge` CLI command**: Model merging exposed as a first-class CLI command supporting all merge methods (Linear, SLERP, TIES, DARE, DELLA, NearSwap, Model Stock) with `--method`, `--base`, `--t`, `--weight-a`, `--weight-b`, `--density`, and `--dtype` flags
- **`pmetal eval` CLI command**: Dataset evaluation command — measures loss/perplexity over a validation set with optional LoRA adapter, `--num-samples` cap, and `--json` output
- **`pmetal info` CLI command**: Prints device and runtime information; `--json` flag emits structured JSON for scripting
- **`pmetal search --json` output**: Structured JSON output mode for search results including fit estimates, download counts, parameter estimates, and tags — enables scripting and GUI integration
- **`QuantizeMethod` enum**: Replaces the string `--method` argument for `pmetal quantize` with a typed enum (`dynamic`, `q8_0`, `q4_k_m`, etc.) — invalid methods now fail at argument parsing rather than deep inside the quantizer
- **GRPO CLI arguments**: `--epochs`, `--lora-r`, `--lora-alpha`, `--max-completion-length`, and `--seed` exposed as CLI arguments, replacing previous hardcoded defaults
- **`loraplus_lr_ratio` and `neftune_noise_alpha`**: New fields on training loop configurations — enables LoRA+ differential learning rates and NEFTune noise injection directly from config
- **`trainable_params()` helper**: New utility in `pmetal-lora` for counting total vs. trainable parameter counts, useful for logging and memory estimation
- **`lora_alpha: f32`**: Distillation CLI and `run_distillation_cli` now accept `lora_alpha` as `f32` instead of `usize` for finer-grained scaling control
- **`seed` parameter in distillation and GRPO CLI**: Reproducible runs via explicit `--seed` flag in all training entry points
- **Gemma3 sliding window auto-detection**: `DynamicModel` loader now reads `model_type == "gemma3"` and sets `is_gemma3 = true` on the config, enabling the correct every-6th-layer global attention pattern without manual config overrides
- **KV cache support for more architectures**: `DynamicModel::forward_with_cache` now routes DeepSeek, Cohere, StarCoder2, and Llama4 to their native caching paths; RecurrentGemma and Jamba now get clear error messages that they require `forward()` directly; hybrid models (NemotronH, Qwen3Next) get a descriptive error directing to `forward_with_hybrid_cache`
- **Speculative decoding greedy path**: `SpeculativeDecoder::verify_greedy()` — exact-correct verification for temperature=0 decoding using argmax equality; avoids the numerically unstable rejection-sampling limit as temperature→0
- **Hub cache management** (`pmetal-hub`): New `cache.rs` module with cache inspection, eviction, and size-reporting helpers
- **Shared model utilities** (`pmetal-models/utils.rs`): Common helpers extracted from per-architecture modules to reduce duplication

### Fixed

- **Scale factor broadcasting in distillation**: `squeeze` applied to the scale factor dimension so it broadcasts correctly across batch and sequence axes — previously caused shape mismatches on non-unit batch sizes
- **TAID `mean_alpha` forcing GPU sync**: `TaidLossOutput::mean_alpha` changed from `f32` to a lazy `Array` — the `.eval()` call is deferred until callers explicitly call `.item::<f32>()`, removing a forced GPU-CPU sync before the backward pass
- **SLERP numerical stability**: Added epsilon clamping in the SLERP merge path to prevent NaN when interpolation parameter is at the boundary values (0.0 or 1.0)
- **Llama LoRA `trainable_params` / gradient application**: Replaced 100+ lines of repeated field accesses with an `insert_adapter!` macro and loop over projection names, fixing DoRA `magnitude` parameter that was silently dropped from gradient maps
- **GaLore improvements**: Corrected projection matrix update schedule and subspace dimensionality handling
- **Distillation hidden-state loss**: Refactored alignment computation to correctly handle variable-rank teacher/student hidden state tensors
- **Jensen-Shannon / KL divergence loss**: Numerical stability improvements — log-sum-exp stabilization applied consistently across all reduction paths
- **Offline distillation**: Fixed logit cache loading to handle both single-file and sharded cache layouts

### Changed

- **`lm_groups.rs` / LoRA+ optimizer groups**: `build_lora_param_groups` significantly reworked — LoRA+ differential LR ratio (`loraplus_lr_ratio`) applied to `lora_b` parameters, NEFTune noise injection integrated into group construction
- **GRPO trainer**: `epochs`, `lora_r`, `lora_alpha`, `max_completion_length`, and `seed` plumbed through from CLI args; previously these were hardcoded to `1`, `16`, `32`, `512`, and a fixed seed
- **Training loop**: `loraplus_lr_ratio` and `neftune_noise_alpha` read from config and forwarded to optimizer group construction
- **`pmetal-core` config / scheduler / traits**: Config structs gained `loraplus_lr_ratio` and `neftune_noise_alpha` fields; scheduler types and learning rate trait bounds refined; `TrainingCallback` trait extended with blanket impls for boxed callbacks
- **Data pipeline**: Tokenizer, packing, `vocab_compact`, dataset, and chat template modules updated — minor correctness and efficiency fixes accumulated across the release cycle
- **GGUF reader / writer / quantize**: Reader handles additional tensor metadata fields; writer improves alignment padding; quantize module uses `QuantizeMethod` enum instead of string matching
- **Hub search**: `search_models` returns richer result structs used by both the human-readable table and the new `--json` output path; upload path fixes for large model shards
- **Metal kernels**: GDN, LoRA, grouped GEMM, and fused SwiGLU Metal shaders updated — improved numerical correctness and register pressure
- **GUI app icons and Tauri config**: Updated icons (32×32, 128×128, 128×128@2x, icns, ico) and `tauri.conf.json` for the 0.3.7 release build; Python vocoder `easy` API additions and mel spectrogram fix

## [0.3.6] - 2026-03-15

### Added

- **Desktop GUI (Tauri + Svelte)**: Full desktop application for model management, training, distillation, GRPO, inference, merging, and quantization. 10 pages: Dashboard, Models, Datasets, Training, Distillation, GRPO, Inference, Merging, Quantize, Settings. Real-time training metrics with live loss charts via broadcast events. Model download with HuggingFace Hub integration, dataset browser, and inference chat interface with streaming token display
- **GUI in-process execution**: Training, distillation, GRPO, inference, model merging, LoRA fuse, and quantization run as direct library calls instead of shelling out to the `pmetal` binary. Eliminates binary discovery issues, reduces process overhead, and enables richer progress reporting. Device info and model metadata also read from library APIs
- **`easy::dpo()` / `easy::simpo()` / `easy::orpo()` / `easy::kto()` builders**: `PreferenceTuneBuilder` in `easy.rs` for preference optimization methods. Full pipeline: model download → tokenizer → dataset loading → LoRA setup → training loop → weight saving. Supports method-specific config (DPO beta/loss type, SimPO gamma/CPO, ORPO beta, KTO desirable/undesirable weights)
- **`easy::infer().generate_streaming()`**: Streaming inference API with per-delta callback. Supports both base models and LoRA adapters. Returns `false` from callback to cancel early. ANE fallback emits full result as single delta
- **Preference trainer `train()` methods**: DPO, KTO, ORPO, and SimPO trainers now have self-contained `train()` methods with optimizer integration, batching, epoch loops, callback lifecycle, and metrics collection. Previously only exposed per-step primitives
- **`TrainingCallback::should_stop()`**: Clean cancellation mechanism — callbacks return `true` to request training loop to finish the current step and exit with `Cancelled` error. Checked after every step in all 5 `TrainingLoop::run*` methods, all 4 preference trainer `train()` loops, and `GrpoTrainer::run()`
- **`PMetalError::Cancelled`**: New error variant for clean training cancellation. Corresponding `Cancelled` variants added to `SftError`, `DpoError`, `KtoError`, `OrpoError`, `SimpoError`, and `GrpoError`
- **Preference batch padding utilities**: `pad_u32_sequences`, `pad_i64_sequences`, `pad_f32_sequences` in `preference_batch.rs` for batching variable-length preference pairs
- **NemotronH runtime FP8 quantization**: `quantize_fp8()` converts float weights to FP8 (E4M3) at runtime for all four block types (Mamba, attention, MLP, MoE). Shared helpers `materialize_linear_weight` and `linear_forward_with_optional_fp8` consolidate FP8 dequantization across the model. MoE weights are restacked after quantization for batched dispatch
- **FluxPipeline::from_pretrained**: Load Flux diffusion pipelines from HuggingFace-style model directories. Discovers components via `model_index.json`, parses both native and diffusers-style config keys for CLIP, T5, FluxDiT, and VAE
- **Python training callbacks**: `Trainer.add_callback()` now wires callbacks into the training loop. Built-in `ProgressCallback`, `LoggingCallback`, and `MetricsJsonCallback` map to native Rust implementations; arbitrary Python objects bridge through `PythonCallbackBridge`

### Fixed

- **Training cancellation via `panic_any` replaced**: GUI and TUI previously used `std::panic::panic_any(CancelledRun)` + `catch_unwind` to abort training — fragile, UB-prone through FFI, and could be swallowed by intermediate catch_unwind. Replaced with `TrainingCallback::should_stop()` returning a clean `Err(Cancelled)` from the training loop
- **GUI QLoRA silently failed on non-Llama models**: `run_qlora_training_in_process` hardcoded `LlamaConfig` deserialization, causing confusing errors or silent misconfiguration for Gemma/Qwen/Phi models. Now detects `model_type` from config.json and returns a clear error for unsupported architectures
- **GUI `resume_from` silently ignored**: Training config accepted `resume_from` but discarded it (`let _ = eval`). Now returns an error directing users to the CLI
- **GUI GRPO with no reward function produced noise**: `DummyReward` returning constant 0.1 for all completions made GRPO training meaningless when reasoning rewards were disabled. Now requires explicit reward configuration
- **Preference trainers doubled compute per step**: DPO, KTO, ORPO, and SimPO `train()` methods ran a second full forward pass after the gradient step solely for logging metrics. Replaced with `RefCell` side-channels that capture metric arrays from within the autograd closure — same metrics, zero extra compute
- **Base model thinking mode**: Auto-detect base vs instruct models and disable `<think>` tag prefill for base models. Base models don't understand thinking tags, causing infinite generation without a closing tag
- **Fused model 5x slower than LoRA**: Skip ANE-hybrid path for models under 2B parameters where GPU KV-cache decode is significantly faster (115 vs 20 tok/s). ANE-hybrid benefits larger models where prefill dominates
- **DataLoader panics on bad images**: Replace `panic!()` in VLM batch construction with proper `DataLoaderError` enum and `try_next_batch()` method. Image preprocessing failures and missing-image errors now propagate as `Result` instead of crashing
- **Division by zero with log_every=0**: Clamp `log_every` and `save_every` to minimum 1 across `TrainingLoop`, `LoggingCallback`, `CheckpointCallback`, and CLI
- **LoRA scaling with rank 0**: `LoraConfig::scaling()` returns 0.0 when rank is 0 instead of dividing by zero
- **BF16 LoRA weights**: `sanitize_loaded_weights()` converts BF16 tensors to FP16 since MLX doesn't natively support BF16 on Apple Silicon
- **Qwen3Next silent weight mismatch**: Weight loading now returns errors for unmatched or missing parameters instead of logging a warning and continuing with a partially loaded model
- **Dataset download only fetched README**: `download_dataset()` now enumerates repo files and downloads actual data files (parquet, json, jsonl, csv, arrow, etc.) with split-aware filtering
- **Model download silent failures**: `download_model()` tracks per-file failures and reports them instead of silently skipping failed downloads
- **Flux loading via DynamicModel**: `DynamicModel::load()` for Flux now returns an error directing to `FluxPipeline` instead of incorrectly loading a diffusion model as a causal LM

### Changed

- **GUI architecture: library calls replace subprocess spawning**: Training, distillation, GRPO, inference, merge, fuse, and quantize commands now call `pmetal` library functions directly instead of spawning `pmetal` CLI as a child process. System info reads from `MetalContext::global()` instead of parsing `pmetal memory` stdout. Removes `which` and `futures-util` dependencies
- **TUI direct training execution**: `command_runner.rs` dispatches `train`, `distill`, and `grpo` commands as in-process library calls via `run_direct_command()`, falling back to subprocess for other commands. Training parameters parsed from `CommandSpec` args with `parse_arg`/`required_arg`/`optional_arg` helpers
- **ORPO loss computation refactored**: `compute_orpo_loss_static` now contains the full computation directly instead of creating a throwaway `OrpoTrainer` instance. The instance method `compute_orpo_loss` delegates to it
- **SimPO gradient-safe loss path**: New `compute_loss_with_cpo_for_grad` static method keeps the computation graph lazy (no `.eval()`/`.item()` calls) for correct autograd. The existing `compute_loss_with_cpo` remains for non-grad contexts
- **`FinetuneBuilder` expanded**: New builder methods — `lora_dropout()`, `use_rslora()`, `use_dora()`, `gradient_checkpointing_layers()`, `callback()`, `metrics_path()`. LoRA config now forwards dropout, RSLoRA, and DoRA settings
- **GRPO CLI gains new parameters**: `epochs`, `lora_r`, `lora_alpha`, `max_completion_length` exposed as CLI arguments and TUI form fields. GRPO now saves `adapter_config.json` alongside LoRA weights
- **CLI `emit_console_output` flag**: Training, distillation, and GRPO CLI functions accept `emit_console_output: bool` and `extra_callbacks: Vec<Box<dyn TrainingCallback>>` to suppress terminal output when called from GUI/TUI
- **DataLoader error handling**: New `DataLoaderError` enum with `Mlx`, `ImagePreprocess`, and `MissingImages` variants. All 7 training loop entry points migrated from `next_batch()` to `try_next_batch()`
- **AdapterManager validation**: `load()` now validates path existence, checks for adapter artifacts in directories, and rejects unsupported file types
- **Metal shader build isolation**: Shader compiler cache redirected to build output directory, preventing pollution of user's home directory
- **unsafe_code lint scoping**: Moved blanket `#![allow(unsafe_code)]` from crate-level `lib.rs` into individual modules that contain unsafe blocks across pmetal-metal, pmetal-mlx, pmetal-models, pmetal-trainer, pmetal-distill, and pmetal-distributed

## [0.3.5] - 2026-03-15

### Added

- **Tool/function calling support**: Chat templates now support tool definitions and tool call formatting for models that natively support function calling:
  - **Qwen/ChatML**: `<tools>` schema injection, `<tool_call>`/`<tool_response>` tags, consecutive tool message merging
  - **Llama 3.1+/4**: `Environment: ipython` header, JSON function calls, `ipython` role for tool responses
  - **Mistral v3+**: `[AVAILABLE_TOOLS]`/`[TOOL_CALLS]`/`[TOOL_RESULTS]` bracketed format
  - **DeepSeek**: Qwen-style tool tags with DeepSeek's unicode tokens
  - CLI: `pmetal infer --tools tools.json -p "What's the weather?"` accepts OpenAI-format tool definitions
- **Tool calling types**: `ToolDefinition`, `ToolCall`, `FunctionCall`, `FunctionDefinition` — OpenAI-compatible structs with serde support for JSON parsing
- **`Message` tool fields**: `tool_calls: Option<Vec<ToolCall>>` for assistant messages, `tool_call_id: Option<String>` for tool response messages, `Message::tool()` and `Message::assistant_tool_calls()` constructors
- **`ChatTemplate::apply_with_tools()`**: New method accepting optional `&[ToolDefinition]` — injects tools into system prompts using model-native format

### Fixed

- **Premature early stop during LoRA training**: The adaptive LR controller was falsely detecting "divergence" from the normal LoRA initialization loss rise (LoRA B starts at zero → first 5-10% of steps naturally increase loss). This triggered rollback cycles that exhausted `max_rollbacks` and killed training at ~5% progress. Fixed with three changes:
  - **Grace period** (`warmup_fraction: 0.1`): No spike/plateau/divergence detection fires during the first 10% of training steps. EMA and loss window still accumulate during this period so detection is primed when it activates
  - **Rollback disabled by default** (`rollback_enabled: false`): Weight rollback undoes valid LoRA weight updates and causes the same initialization pattern to repeat. Now opt-in for long pre-training runs
  - **Less sensitive thresholds**: `divergence_slope_threshold` 0.01 → 0.05, `divergence_window` 20 → 40, `plateau_patience` 50 → 100, `spike_threshold` 3.0 → 3.5
- **Adaptive LR grace period not applied**: `set_total_steps()` is now called by all 7 training entry points (5 in `TrainingLoop`, 1 in `GrpoTrainer`, 1 in `DistillationTrainer`) to compute the grace period from total steps

### Changed

- **Adaptive LR defaults**: Retuned for LoRA fine-tuning rather than pre-training. The controller now acts as a safety net (catches NaN, true catastrophic divergence) rather than an aggressive optimizer
- **Distillation adaptive LR**: `for_distillation()` config uses shorter 5% grace period (distillation has smoother early loss) and tighter divergence thresholds

## [0.3.4] - 2026-03-14

### Added

- **Mixture-of-Depths (MoD)** for Llama 4: Proper implementation per Raposo et al. (2024) — lightweight router with `argpartition_axis` top-k, gather-before-compute on sub-batch, scatter-after, BCE auxiliary loss. Configurable capacity factor and per-layer selection
- **Llama 4 RoPE**: Real RoPE implementation via `pmetal_mlx::kernels::rope::apply_rope` (Metal-accelerated), replacing the placeholder stub. Correctly wired into iRoPE layer dispatch — RoPE layers get rotary embeddings, NoPE layers skip them
- **Llama 4 temperature scaling**: Per Meta's formula `log(floor((pos+1)/floor_scale) + 1) * attn_scale + 1.0`, applied to Q states in NoPE layers before QK matmul for long-context attention stabilization
- **Llama 4 GQA**: KV-head broadcast expansion for grouped-query attention — enables Scout (40 Q / 8 KV) and Maverick configs
- **MoE top-k > 1**: `Llama4Router` uses `argpartition_axis` for O(n) expert selection with L1-normalized weights and per-slot dispatch loop, replacing hardcoded argmax
- **ANE fused kernels**: `gen_dynamic_sdpa_fwd` (single-kernel attention: RMSNorm + QKV + SDPA + Wo) and `gen_dynamic_ffn_w13` (single-kernel FFN: RMSNorm + W1 + W3 + SiLU), replacing 6+ separate ANE evaluations per layer
- **ANE fused backward**: `gen_dynamic_ffn_bwd_w2t` and `gen_dynamic_ffn_bwd_w13t` for fused FFN backward pass
- **Metal dequantization kernels**: Q4_0 and IQ4_XS Metal compute shaders, verified correct per GGML spec. Bridge methods in `MlxMetalBridge` for GPU-accelerated dequantization
- **Cancellation safety infrastructure**: `CompletionToken::Drop` guard in `AsyncScheduler` waits for in-flight GPU commands; `retain_resource()` / `as_retained()` for Metal buffer lifetime extension
- **IoSurface helpers**: `write_f32_strided_at`, `write_f32_at_col_offset`, `zero_channel_range_f32` for fused backward kernel IO
- **CloudBridge**: Complete training state export (weights, optimizer state, RNG, dataloader position, metadata) with working Python bootstrap scripts for FSDP/DeepSpeed cluster resumption and Rust-side loader functions
- **Formal verification**: `cargo-kani` proofs for ring all-reduce chunk arithmetic (95 checks) and k-ary tree topology consistency (607 checks), with justfile recipes
- **Reasoning templates**: `MathReasoningTemplate` (GRPO + accuracy/format rewards) and `CodeReasoningTemplate` (structural code fence + test case matching)
- **Reasoning dataset auto-detection**: `pmetal dataset prepare` automatically detects `problem`/`thinking`/`solution` columns and formats them as `<think>` tagged ChatML conversations
- **`--columns` flag**: General column remapping for `dataset prepare` (e.g., `--columns "instruction=question,output=answer"`)
- **`adapter_config.json`**: Saved alongside LoRA weights during training (r, alpha, target_modules, use_rslora). Loaded automatically at inference and fuse time — eliminates config guesswork
- **Supply chain**: `cargo-vet` initialized with Mozilla, Google, and Bytecode Alliance audit imports; 17 workspace crates covered; 5 transitive dependency exemptions with exact lockfile versions
- **Tracing spans**: 6 `info_span!` markers in Python trainer for phase-level observability (model_resolve, load_tokenizer, load_dataset, load_model, training_loop, save_weights)

### Fixed

- **LoRA inference garbage output**: Merged LoRA weights into base model at inference time (`W += scale*B@A`). The separate-forward path had dtype mismatch issues (BF16 base × F32 LoRA)
- **Auto-chat mode regression**: Removed heuristic that forced chat template on base models just because their tokenizer has `<|im_end|>`. Chat mode now requires explicit `--chat` or an instruction-tuned model
- **Missing EOS in training data**: Training sequences now end with the model's actual EOS token (e.g., `<|endoftext|>` for Qwen). Previously only had turn delimiter (`<|im_end|>`) — model never learned to stop generating
- **Fuse command wrong alpha/rank**: `pmetal fuse` now reads `adapter_config.json` for correct alpha and rank instead of defaulting to `scale=1.0`. Also filters MLP LoRA weights (rank=0) when auto-detecting rank from shapes
- **ANE `x2norm` backward bug**: FFN weight gradients (`dW1`, `dW3`) were computed against the wrong pre-norm tensor (`xnorm` from attention block instead of `x2norm` from FFN block). Restored `x2norm` field and CPU RMSNorm recomputation for gradient correctness
- **ANE `sdpa_bwd` surface dtype**: Backward SDPA output surfaces were allocated as fp32 but ANE kernels produce fp16 — stride mismatch corrupted dV/dQ/dK gradients. Fixed to `IoSurface::for_tensor()` (fp16)
- **MoD argpartition sign**: Router negated weights before `argpartition_axis`, selecting bottom-k (least important) tokens instead of top-k. Removed negation
- **MLX bridge `copy_as_f32` regression**: Renamed methods dropped auto dtype conversion — callers passing wrong dtype would panic. Restored `copy_as_f32` / `copy_as_f16` with auto-conversion
- **MLX bridge `view_f32` eval**: Removed `.eval()` call before accessing data pointer — unevaluated arrays returned null. Restored defensive eval
- **Python API surface**: Restored `ProgressCallback`, `LoggingCallback(log_every=10)`, `__version__`, and `PythonCallbackBridge` that were deleted during PyO3 migration
- **TUI training completion**: Reads final metrics from JSONL file on disk (immune to polling lag). Shows actual loss and step count instead of `0.0000` / sample count
- **TUI Steps/min overflow**: Guards against divide-by-zero when `total_ms=0` — shows `—` instead of `60000`
- **Dataset prepare panic**: Empty results no longer crash with index-out-of-bounds. Shows diagnostic message with format hints

### Changed

- **LoRA inference uses merge**: `merge_lora()` is called before generation, producing a single merged weight matrix per layer. This is equivalent to the fuse command but happens in-memory without saving
- **PyO3 0.23 → 0.28**: `allow_threads` → `detach`, `with_gil` → `attach`, `from_py_object` on all pyclass types, `Bound<'py, PyDict>` return types
- **tokio 1.49 → 1.50**
- **`unsafe_code` lint**: Escalated from `warn` to `deny` workspace-wide

## [0.3.3] - 2026-03-12

### Added

- **Self-contained binary**: `mlx.metallib` is now gzip-compressed and embedded into the `pmetal` binary at build time via `build.rs` + `include_bytes!`. On first run it extracts to `~/.cache/pmetal/lib/` if not already present. `cargo install pmetal-cli` now produces a fully self-contained binary with no external metallib dependency (~31MB added to binary, 70% smaller than the raw 102MB metallib)
- **Adaptive LR rollback**: When divergence is detected and `rollback_enabled = true`, the adaptive LR controller emits `LrEvent::RollbackTriggered` — the training loop restores LoRA weights from the best in-memory EMA snapshot, resets optimizer momentum, and continues with a halved LR multiplier
- **Early-stop on repeated divergence**: After `max_rollbacks` exhausted rollbacks, the controller emits `LrEvent::EarlyStop` — the training loop saves a final checkpoint and exits cleanly instead of spiraling deeper into loss divergence
- **In-memory LoRA snapshot**: `TrainingLoop` holds the best LoRA weight snapshot in RAM via `snapshot_best_weights()` / `restore_best_weights()`. LoRA params are typically 1–20 MB, making this negligible overhead vs checkpoint I/O
- **`AdaptiveAction` enum**: `apply_adaptive_lr()` now returns `AdaptiveAction::Continue | Rollback | EarlyStop` so training loops can react to controller decisions without re-parsing event strings

### Fixed

- **`apply_adaptive_lr` return type**: Previously returned `()`, discarding rollback/early-stop events — callers had no way to react. Now returns `AdaptiveAction`
- **Divergence rollback vs plain reduction ambiguity**: Divergence path now checks `rollback_enabled` and `has_best_snapshot` before deciding between rollback and plain LR reduction — prevents silent rollback when no snapshot exists
- **EMA state reset on rollback**: Spike EMA and variance are reset alongside LR multiplier on rollback so z-score anomaly detection re-stabilizes correctly after weight restoration
- **`total_steps` in metrics**: `run_standard()` and `run_jit_compiled()` computed `total_steps: max_steps.unwrap_or(0)` — now estimates from `dataset.len() / batch_size * epochs` when `max_steps` is `None`, giving accurate progress in the TUI
- **`stats_summary` missing rollback count**: `AdaptiveLrController::stats_summary()` now includes `rollbacks=N` in its output string

### Improved

- **Rollback tests**: Four new unit tests — `test_rollback_triggered_on_divergence`, `test_early_stop_after_max_rollbacks`, `test_rollback_disabled_falls_through_to_divergence`, `test_should_snapshot_best_tracks_ema_improvement`

## [0.3.2] - 2026-03-11

### Added

- **Adaptive learning rate controller**: EMA-based z-score spike detection, patience-based plateau detection, and linear regression divergence detection — automatically adjusts LR multiplier during training to recover from loss spikes, reduce LR on plateaus, and halt on divergence
- **Manual LR override via TUI**: Press `L` in Training, Distillation, or GRPO tabs to set a custom learning rate mid-run; uses atomic control file protocol (`{output_dir}/.lr_control.json`) for safe subprocess communication
- **WSD (Warmup-Stable-Decay) scheduler**: New `LrSchedulerType::Wsd` with configurable `stable_ratio` — holds peak LR for a plateau phase before linear decay, popular for large-scale pretraining
- **GRPO adaptive LR + callbacks**: `GrpoTrainer` now supports adaptive LR, `TrainingCallback` lifecycle events, and `StepMetrics` emission for live TUI monitoring
- **HuggingFace Hub search** (`pmetal search`): CLI command and TUI integration (press `S` in Models tab) to search HF Hub for text-generation models with download counts, parameter estimates, and memory fit assessment
- **Memory fit estimation**: New `pmetal-hub` module estimates inference/training memory requirements, tok/s throughput, and color-coded fit levels (green/yellow/red) based on device specs and model architecture
- **Model detail panel**: Models tab shows memory breakdown — weights, KV cache, overhead, training estimate, and recommended batch size
- **Distillation metrics callbacks**: `DistillationTrainer` now emits step-by-step metrics via `TrainingCallback`, enabling live TUI dashboard during distillation runs
- **Command logging in Jobs tab**: Spawned commands are logged with the full CLI invocation for easier debugging

### Fixed

- **NaN/Inf loss guard**: Adaptive LR skips EMA updates on non-finite losses to prevent EMA poisoning — returns scheduled LR unchanged
- **EMA variance bias correction**: Early-training z-scores now use bias-corrected variance (`raw_var / (1 - alpha^n)`), matching Adam's moment correction — prevents false spike detection in first ~20 steps
- **Zero-variance z-score fallback**: When loss variance is near zero (std_dev < 1e-8), uses absolute deviation threshold instead of division-by-zero; returns z=10 for >50% deviation, z=0 otherwise
- **Atomic control file protocol**: LR control file is renamed to `.lr_control.claimed` before reading and deleted after — prevents race conditions between TUI writer and training subprocess reader
- **Distillation metrics LR**: Distillation step metrics now report post-adaptive LR instead of pre-adjustment scheduled LR
- **Adaptive LR in all training paths**: `apply_adaptive_lr()` now called in `run_metal_fused()`, `run_compiled()`, `run_jit_compiled()`, and `run_packed()` paths (was only in `run_standard()`)
- **TUI LR override validation**: LR range check now accepts 1.0 (was exclusive upper bound); shows error modal on invalid input instead of silent log warning
- **Distillation/GRPO job routing**: Status updates were always routed to the Training tab regardless of job type. Added `active_job_type` tracking to route metrics, completion, and failure to the correct tab (Distill, GRPO, or Training)
- **Distillation CLI args**: TUI sent `--lora-alpha` and `--log-metrics` flags that the CLI didn't accept, causing immediate exit code 2. Added both args to the `Distill` command and `--log-metrics` to `Grpo`
- **Parquet dataset support in distill/GRPO**: Distillation and GRPO commands only supported JSONL datasets. Now auto-detect `.parquet` files and route to the parquet loader, matching the training command's behavior
- **Tab click targeting**: Mouse clicks on Monitor, Inference, and Jobs tabs selected the wrong tab due to hardcoded fixed-width hit-testing. Now computes actual tab widths from rendered text
- **Error diagnostics**: Failed jobs now show the last 5 stderr lines in the tab status panel instead of just "Process exited with code N", with a hint to check the Jobs tab for full output
- **UTF-8 safe string truncation**: `truncate_str` used byte indexing which panics on multi-byte characters; switched to `chars()` iterator
- **Leaked channel in HF search**: `search_hf()` created a sender/receiver pair even without a CommandRunner, silently dropping results
- **Integer overflow in fit estimation**: `estimate_params_from_config` used plain multiplication; switched to `saturating_mul`/`saturating_add`
- **Context length truncation**: u64→u32 cast could wrap for extreme values; capped at 1M before cast

### Improved

- **TUI tab ordering**: System (formerly Device) is now the default first tab; Dashboard renamed to Monitor
- **Empty state messaging**: Monitor tab shows actionable guidance ("Start a run from Training, Distill, or GRPO tab") instead of "Waiting for training data..."
- **Idle state hint**: Tabs show "Press S to start" instead of "Press S to start training" (generic across all job types)

### Security

- **Bounded API responses**: `bounded_json()` caps HF API response bodies at 4MB to prevent heap exhaustion
- **Model ID validation**: `is_valid_model_id()` rejects path traversal, URL injection, and malformed values in HF API paths

## [0.3.1] - 2026-03-11

### Added

- **M5 / Apple10 device detection**: GPU family `Apple10` with architecture generation 17, NAX (Neural Accelerators in GPU) availability flag, and NAX-aware tile size tuning (M5 Max/Ultra get 128×64×32)
- **UltraFusion topology detection**: `sysctl hw.packages` detects multi-die Ultra chips; `is_ultra_fusion` and `die_count` fields on `DeviceProperties`
- **GPU and ANE core count estimation**: Per-chip core counts derived from device name and tier, with UltraFusion die multiplication
- **Memory bandwidth estimation**: Tier + GPU family lookup table for estimated bandwidth (GB/s)
- **ANE performance stats API**: `evaluate_with_stats()` on `AneModel` uses `_ANEPerformanceStats` with `hwExecutionTime` for nanosecond-precision hardware timing
- **TUI device tab enhancements**: GPU core counts (with per-die breakdown for Ultra), ANE core counts, memory bandwidth, architecture generation, NAX and UltraFusion feature flags
- **`crates/pmetal/README.md`**: Crate-level README with feature flags table, quick start examples, hardware support summary, and re-export reference

### Fixed

- **`AppleGPUFamily::Unknown` ordering bug**: `Unknown` was declared last in the enum, causing derived `Ord` to rank it above `Apple10` — unknown GPUs incorrectly got `has_dynamic_caching`, `has_nax`, etc. set to `true`. Fixed by moving `Unknown` to first position
- **Future chip name collision**: `name.contains("M1")` matched "M10"; replaced with `has_chip_id()` that checks the character after the match isn't a digit
- **Dead `sysctl` subprocess in `query_memory_bandwidth`**: Spawned `sysctl` whose result was discarded; removed and renamed to `estimate_memory_bandwidth()` using tier-based lookup

### Improved

- **README updates**: Root README now documents hardware support matrix (M1–M5), 9 TUI tabs (was 7), 16 crates (was 15), all fused Metal kernels (GDN, SwiGLU, RMSNorm+LoRA), ANE perf stats and M1–M5 compatibility
- **Hardware support docs**: Complete M1–M5 chip matrix with arch gen, core counts, bandwidth, ANE TFLOPS measurements; NAX kernel integration roadmap; UltraFusion distributed roadmap

## [0.3.0] - 2026-03-10

### Added

- **TUI Control Center** (`pmetal tui`): Full terminal interface with 9 tabs — Dashboard, Device, Models, Datasets, Training, Distillation, GRPO, Inference, Jobs. Async event loop with crossterm/ratatui, modal system (confirm, text input, model picker, dataset picker, error, progress), and reusable form field widgets
- **Live job integration**: Training, distillation, and GRPO tabs spawn pmetal subprocesses and stream metrics in real time via `CommandRunner` + JSONL polling
- **LoRA fuse command** (`pmetal fuse`): Merge LoRA adapter weights into base model, with optional fuse-then-quantize pipeline
- **Chat template support for Llama 4, DeepSeek, and Cohere**: Full template formatting, Jinja detection, model name heuristics, stop tokens, and inference formatting for all three model families
- **Llama 4 template**: `<|header_start|>`/`<|header_end|>`/`<|eot|>` tokens (distinct from Llama 3's `<|start_header_id|>`/`<|end_header_id|>`/`<|eot_id|>`)
- **DeepSeek template**: Full-width unicode tokens (`<｜begin▁of▁sentence｜>`, `<｜User｜>`, `<｜Assistant｜>`) with thinking mode support (`<think>`/`</think>` prefill)
- **Cohere Command R template**: `<|START_OF_TURN_TOKEN|>`, `<|USER_TOKEN|>`, `<|CHATBOT_TOKEN|>`, `<|END_OF_TURN_TOKEN|>` tokens
- **Comprehensive stop token collection**: `collect_all_stop_tokens()` now probes 11 well-known special tokens across all model families (added `<|eot|>`, `<|end|>`, `<|return|>`, `<|END_OF_TURN_TOKEN|>`, `<｜end▁of▁sentence｜>`)
- **LoRA inference auto-chat detection**: Probes vocabulary for `<|im_end|>`/`<|eot_id|>` to auto-enable chat mode on base models fine-tuned with LoRA
- **Streaming generation support**: `GenerationConfig` streaming extensions in `pmetal-models`
- **Epoch/total_steps in StepMetrics**: Training progress now flows through entire pipeline (training loop → JSONL callback → TUI) showing step X/Y and epoch M/N
- **Hardware support documentation**: Apple Silicon hardware matrix and tuning reference (`docs/hardware-support.md`)

### Fixed

- **TUI inference word wrap**: Model output now wraps correctly within the terminal width instead of clipping off-screen; `normalize_code_fences()` preprocessor ensures ``` markers always appear on their own line even when the model emits text without newlines
- **TUI inference code block rendering**: Fenced code blocks (```python, etc.) now render properly with distinct styling even when the token stream lacks explicit newline characters
- **TUI UTF-8 safe text handling**: Word wrap and code block truncation now use char-count width instead of byte length, preventing panics on multi-byte characters
- **GRPO accuracy reward — last-occurrence extraction**: `AccuracyReward` now uses `rfind()` for `<answer>` tags and `\boxed{}`, correctly grabbing the final answer when the model retries within chain-of-thought
- **GRPO accuracy reward — broken fallback**: Old code compared the entire completion (including reasoning) against the answer when no `<answer>` tags were found; now falls back to last non-empty line
- **GRPO accuracy reward — whitespace normalization**: Answer comparison now collapses internal whitespace runs to single space, preventing false negatives from formatting differences
- **LoRA inference stop tokens**: `run_inference_with_lora` now uses full chat template + comprehensive stop token collection instead of just tokenizer EOS — fixes infinite generation on chat-finetuned models
- **LoRA inference missing parameters**: All sampling parameters (top_k, top_p, min_p, penalties, seed) now passed through to LoRA inference path
- **Llama 4 misdetection**: Model name heuristic now correctly routes `llama-4`/`llama4` to Llama 4 template (was incorrectly using Llama 3 tokens)

### Added

- **GRPO `\boxed{}` answer extraction**: `AccuracyReward` now extracts answers from LaTeX `\boxed{...}` expressions with brace-depth tracking, standard for math GRPO (DeepSeek-R1 style)

### Improved

- **TUI replaces legacy dashboard**: `pmetal tui` provides full control center; legacy `pmetal dashboard` retained for simple metrics monitoring
- **Chat template Jinja detection**: Ordered detection ensures DeepSeek (full-width unicode), Cohere, Llama 4 are matched before generic patterns
- **EOS token stripping**: `strip_eos_tokens()` now handles all model-family EOS tokens

## [0.2.1] - 2026-03-09

### Added

- **Cross-vocabulary distillation**: Sparse top-k alignment (k=128) enables teacher/student with different vocab sizes; implemented in KL divergence, soft cross-entropy, and Jensen-Shannon losses
- **Fused GDN Metal kernel**: Gated Delta Network forward pass for Qwen 3.5 hybrid layers (`fused_gdn.metal` + `fused_gdn.rs`)
- **Gated delta MLX kernel**: Forward and backward passes for GDN in `pmetal-mlx`
- **CPU RMSNorm for ANE inference**: Compute RMSNorm on CPU in f32 to avoid fp16 overflow/saturation on ANE; per-head QK-norm stays on ANE where values are safe
- **`cpu_rmsnorm` flag in kernel generators**: `gen_sdpa_fwd_kv()` and `gen_ffn_fwd()` accept `cpu_rmsnorm: bool` — when true, emits identity instead of RMSNorm and omits weight blobs
- **Test serialization config**: `.cargo/config.toml` sets `RUST_TEST_THREADS=1` to prevent MLX GPU memory races

### Fixed

- **ANE inference garbage output**: fp16 `reduce_sum(x², axis=channel)` overflows for residual values > 256 due to ANE saturation arithmetic; CPU RMSNorm in f32 eliminates the corruption
- **Cross-vocab distillation crash**: Mismatched teacher/student vocab sizes (e.g., Qwen3-4B 151,936 → Qwen3.5-0.8B 152,080) no longer panic; `align_vocab()` handles alignment transparently
- **3D tensor indexing in `align_vocab`**: Use `(Ellipsis, ..k)` for correct last-axis slicing of rank-3+ tensors
- **Qwen 3.5 (1+w) RMSNorm**: Weight sanitization adds 1.0 to RMSNorm weights during loading
- **Clippy lints**: Unnecessary parentheses in `fused_gdn.rs`, too-many-arguments on `rmsnorm_backward`, let-and-return in `next_power_of_2`

### Improved

- **ANE inference cleanup**: Removed ~80 lines of diagnostic logging from hot path
- **Metal GPU path gating**: Cross-vocab losses gate Metal GPU path on matching vocabs, fall back to CPU for mismatched
- **Documentation**: Updated all crate READMEs to reflect current architecture support, training methods, and features

## [0.2.0] - 2026-03-06

### Added

- **Apple Neural Engine (ANE) integration** behind `ane` feature flag — MIL 1.3 program generation, private API FFI via dlopen, IOSurface zero-copy, compilation budget tracking, hybrid CPU/ANE trainer with async gradient accumulation
- **`AneInferenceEngine`** — forward-only ANE kernels (no concat taps, ~6x smaller IO vs training) with CPU-side embedding, RMSNorm, sampling (greedy/temperature/top-k), and autoregressive generation via Easy API `.device(Device::Ane)`
- **KV cache for autoregressive generation** — hybrid ANE prefill + CPU decode architecture eliminates O(n²×L) recomputation per token; ANE processes the full prompt, CPU handles single-token decode steps with cached KV pairs via `cblas_sgemv`
- **GQA/MQA support** — `n_kv_heads` config field enables grouped-query attention (Llama 3, Mistral, etc.); concat-based KV head expansion in ANE kernels
- **SafeTensors weight loading** — direct loading of HuggingFace safetensors format (single and multi-file) with automatic bf16/f16/f32 dtype conversion
- **LoRA adapter fusion** — merge adapter weights (`W += (alpha/rank) * B @ A`) before ANE kernel compilation; supports both `self_attn` and `mlp` target modules
- **Dynamic weight pipeline**: 9 MIL kernels compiled once at startup; weights packed alongside activations in IOSurface spatial dimension — zero recompilation during training
- **`DynamicAneTrainer`**: compile-once training loop replacing the static trainer that consumed ~76% of training time in recompilation
- **`DynamicKernelConfig`** and 12 dynamic kernel generators in `dynamic_kernel.rs`
- **MIL program fragment helpers**: `emit_rmsnorm_fuse` and `emit_dyn_matmul_with_act` for composable RMSNorm fusion and dynamic matmul in ANE kernel generation
- **`rmsnorm_fwd` dynamic kernel**: Fused RMSNorm forward pass on ANE
- **fp32 IOSurface support**: `IoSurface::new_f32()` with packed write/read for dynamic weight pipeline
- **MIL builder extensions**: `emit_cast`, `emit_slice_by_size`, `new_fp32_input` for dynamic kernel generation
- **Non-standard `head_dim` support**: Full forward and backward kernel support for models where `head_dim != dim/n_heads` (e.g., Qwen3 with `head_dim=128`, `dim/n_heads=64`)
- **Training dashboard (TUI)**: `pmetal dashboard` subcommand using ratatui for real-time loss curves, timing breakdown, and throughput monitoring
- **`MetricsJsonCallback`**: Emits full `StepMetrics` including ANE timing, Adam timing, and throughput to JSONL
- **GSPO trainer**: Group Sequence Policy Optimization (fixes GRPO length bias)
- **DAPO trainer**: Decoupled Clip and Dynamic Sampling Policy Optimization (all 4 ByteDance innovations)
- **Python bindings** (`pmetal-py`) via PyO3/maturin with type stubs
- **High-level Easy API** (`pmetal::easy`) — builder pattern for fine-tuning and inference
- **Version and device introspection** (`pmetal::version`)
- **Examples**: `device_info`, `finetune_easy`, `finetune_manual`, `inference_easy`
- **Python CI workflow** (`.github/workflows/python.yml`)
- `Device::Ane` variant with feature-gated support
- ANE-specific error types in `pmetal-core` and `pmetal-metal`
- ANE training loop integration in `pmetal-trainer`
- `silu_inplace` in Accelerate wrappers for CPU decode SwiGLU

### Fixed

- **Metal resource exhaustion on long training runs**: `eval_training_state()` now evaluates model params and optimizer states (momentum, velocity) alongside losses, preventing unbounded computation graph growth in deferred-eval mode
- **Gradient checkpointing default**: `CheckpointStrategy` default changed from `Smart` to `None` — MLX backend does not implement it yet; configs remain forward-compatible
- **Training defaults**: batch_size default 4→1, gradient_accumulation_steps default 1→4 (same effective batch size, lower per-step memory pressure)
- ANE inference gibberish output: added RoPE and per-head QK-norm to prefill kernel and CPU decode
- ANE inference missing `compile_kernels()` call in `generate_cached_ane`
- All backward kernels (static + dynamic) now use `q_dim()`/`kv_dim()` instead of hardcoded `dim` — fixes incorrect gradient shapes for non-standard architectures
- `sdpa_bwd1_input_ch`: `4*dim` → `q_dim + 2*kv_dim + dim`
- `sdpa_bwd1_output_ch`: `dim + 2*score_ch` → `kv_dim + 2*score_ch`
- `sdpa_bwd2_input_ch`: `2*score_ch + 2*dim` → `2*score_ch + q_dim + kv_dim`
- Dynamic backward kernels: `wo_bwd`, `sdpa_bwd1`, `sdpa_bwd2`, `qkv_bwd` all updated for q_dim/kv_dim
- SafeTensors dtype/alignment error handling
- Token ID bounds check in CPU decode
- Softmax numerical stability for zero-sum edge case
- ANE GQA inference failure (`status=0x1d`): replaced unreliable `tile` MIL op with concat-based KV head expansion in all 3 SDPA kernels
- Token ID truncation: `embed_lookup`/`embed_backward` changed from `u16` to `u32` (Qwen3 vocab=151936 exceeds u16 max)
- RMSNorm epsilon hardcoded to 1e-5: now configurable via `cfg.rms_norm_eps` (Qwen3 requires 1e-6)
- CI: Exclude `pmetal-py` from CI clippy/build/test (requires Python dev libs not available on runner)

### Improved

- NEON f16↔f32 conversion upgraded from 4-wide to 8-wide (`fcvtn2`/`fcvtl2`)
- Accelerate/vDSP wrappers expanded with 12 new functions: `rmsnorm`, `rmsnorm_backward`, `cross_entropy_loss`, `softmax_inplace`, `adam_update`, `embed_lookup`, `embed_backward`, `matrix_transpose`, `gemm`, `vadd`, `vmul` (with scalar fallbacks on non-macOS)
- `supports_neural_engine()` now performs real ANE detection via framework dlopen
- Easy API ANE path now auto-detects SafeTensors/flat weights, LoRA adapters, and GQA config; uses `generate_cached()` for KV-cached inference
- ANE config validation (`new()` returns `Result`)
- Kernel config validation (`TransformerKernelConfig::validate()`)
- LoRA safety: rank=0 guard and tensor shape validation
- Decode memory efficiency: pooled scores buffer
- 15 new tests for non-standard head_dim kernels (7 static + 8 dynamic)
- MIL debug dump on ANE compile failure (`/tmp/ane_debug_layer{N}_{attn|ffn}.mil`)
- Qwen3 GQA kernel test (n_heads=16, n_kv_heads=8, verifies no `tile` ops)
- Dynamic kernel documentation: All 12 kernels now document detailed input tensor names alongside dimension formulas

## [0.1.2] - 2026-03-02

### Fixed

- **GPU occupancy waste in gradient scaling**: `scale_gradients` grid dispatch was 4x over-provisioned after float4/half4 vectorization — each thread processes 4 elements but the grid still dispatched one thread per element; corrected from `div_ceil(32)` to `div_ceil(128)`
- **Threadgroup memory overallocation in fused LoRA**: Static `threadgroup float[128 * 256]` arrays in `fused_lora_forward` and `fused_lora_backward_x` allocated 128KB each, exceeding Apple Silicon's 32KB threadgroup memory limit; switched to dynamic threadgroup memory via `setThreadgroupMemoryLength` with host-side size calculation based on actual tile and rank dimensions
- **Silent loss of final async checkpoint**: When `TrainingLoop` was dropped, the pending background checkpoint thread was silently detached — if the process exited before the thread finished, the final safetensors file could be truncated or corrupt; added `Drop` impl that joins the pending handle and logs errors
- **LoRA rank validation**: Raised rank limit from 64 to 256 to match `MAX_LORA_RANK` now that dynamic threadgroup memory removes the static allocation constraint

### Improved

- **Checkpoint I/O deduplication**: Extracted shared file write logic (`write_checkpoint_to_dir`) from `save_checkpoint`, `save_checkpoint_owned`, and `save_best_checkpoint` — eliminated ~100 lines of duplicated directory creation, safetensors serialization, and metadata JSON writes
- **Edge case test coverage**: Added tests for NEON fp16↔fp32 conversion (NaN, Inf, -Inf, -0.0, subnormals, exact 4-element alignment, 1M+ element arrays) and Accelerate vDSP wrappers (negative values, single-element arrays, 1M+ element arrays)

## [0.1.1] - 2026-02-27

### Improved

- **Unified chat template detection**: New `detect_chat_template()` inspects `tokenizer_config.json` Jinja strings before falling back to model-name heuristics — training, inference, and distillation now detect templates consistently
- **Broader inference templates**: Added inference formatters for Llama-2, Gemma, Mistral, Phi-3, Phi-4, and GPT-OSS (previously only ChatML and Llama-3 were supported)
- **Template-aware stop tokens**: Inference now encodes the correct EOS token per template type (`<|eot_id|>` for Llama-3, `<end_of_turn>` for Gemma, etc.) instead of hardcoding `<|im_end|>`
- **Array chat_template support**: Handles HuggingFace models that store `chat_template` as an array of `{name, template}` objects (e.g., Command-R)
- **Distillation template detection**: Distillation now applies the student model's chat template during dataset formatting (was `None` before)
- **Distillation completion output**: Summary box with detected template and actionable next-steps command

### Fixed

- **Silent download failures**: Tokenizer and config file download errors now logged with `warn!`/`debug!` instead of silently swallowed with `let _ =`
- **Silent quantize fallback**: Invalid `--method` values now produce a clear error listing valid methods instead of silently falling back to Q4K
- **Dataset directory error**: Passing a directory to `--dataset` now auto-discovers `train.jsonl`/`data.jsonl`/`dataset.jsonl` or suggests `.jsonl` files found, instead of an opaque "Is a directory" error
- **Tokenizer-not-found guidance**: Error now explains that GGUF models don't bundle tokenizers and suggests `pmetal download <model-id>`
- **Memory stats NaN**: `pmetal memory` guards against division by zero when `total_gb()` is 0
- **EOS token stripping**: `extract_final_response` now strips all known EOS tokens (was only `<|im_end|>` and `<|endoftext|>`)
- **Qwen3 LoRA gradient checkpointing warning**: Now emitted once per run instead of per-layer per-step (via `std::sync::Once`)

## [0.1.0] - 2026-02-26

Initial public release.

### Core Framework

- **pmetal-core**: Foundation types, configuration system, and shared traits for the workspace
- **pmetal-cli**: Command-line interface with `train`, `infer`, and `bench` subcommands

### Model Support

- **pmetal-models**: Dynamic architecture loading with support for:
  - Llama (2, 3, 3.1, 3.2, 3.3, 4)
  - Qwen (2, 2.5, 3, 3-MoE)
  - DeepSeek (V3, V3.2, V3.2-Speciale)
  - Mistral (7B, 8x7B)
  - Gemma (2, 3), Phi (3, 4), Granite (3.0, 3.1), Cohere (Command R), GPT-OSS, Nemotron-H
  - Vision: Pixtral 12B, Qwen2-VL, MLlama 3.2-Vision

### Training

- **pmetal-trainer**: SFT, DPO, and GRPO training loops with learning rate schedulers and gradient checkpointing
- **pmetal-lora**: LoRA and QLoRA with configurable rank, alpha, and target modules
- **pmetal-data**: Dataset loading for ShareGPT, Alpaca, Messages, and raw text formats with sequence packing (99.7% efficiency)
- **pmetal-distill**: Knowledge distillation with KL divergence, Jensen-Shannon, soft cross-entropy, hidden state alignment, and offline logit caching

### GPU Acceleration

- **pmetal-metal**: Custom Metal compute kernels:
  - FlashAttention with O(n) memory
  - Fused LoRA forward pass
  - Fused cross-entropy (chunked vocabulary loss)
  - Fused RoPE
  - Fused sampler with JIT compilation
  - Fused DoRA kernels

### Model Operations

- **pmetal-merge**: Model merging via Linear, SLERP, TIES, DARE, DELLA, NearSwap, and Model Stock methods
- **pmetal-gguf**: GGUF format reading, writing, dequantization, and imatrix quantization
- **pmetal-hub**: HuggingFace Hub downloading, caching, and upload support

### Experimental

- **pmetal-mhc**: Manifold-Constrained Hyper-Connections (Sinkhorn-Knopp doubly stochastic projections) with Metal GPU acceleration
- **pmetal-distributed**: Peer-to-peer distributed training with mDNS auto-discovery, ring all-reduce, and gradient compression
- **pmetal-vocoder**: BigVGAN neural vocoder for text-to-speech synthesis
- **pmetal-mlx**: MLX backend integration with KV cache management, quantization, speculative decoding, and NEFTune

### Infrastructure

- Rust edition 2024, minimum supported Rust version 1.85
- Continuous fuzzing for GGUF reader via `cargo-fuzz`
- CI with clippy, fmt, test, and fuzz workflows
- Dual licensed under MIT and Apache-2.0
