//! Chat templates for checkpoints that ship none.
//!
//! A base model has no `chat_template`, yet is fine-tuned into a chat model in
//! one of the conventional formats. Each of those formats is written here as
//! the Jinja template it would ship, and rendered by the same engine as a
//! model's own template, so there is one renderer for every conversation and
//! one place its output is checked against transformers.
//!
//! The templates take the context transformers gives a chat template
//! (`messages`, `tools`, `add_generation_prompt`, `bos_token`, `eos_token`)
//! and one more, `default_system_message`, which stands in for a system
//! message the conversation lacks. They are the formats as their families
//! first published them; a model that ships a newer template (Llama 3.1's
//! dated system preamble, Qwen 2.5's default persona) uses that one, because
//! the model's own template is what [`ChatTemplate`](super::ChatTemplate)
//! renders whenever there is one.
//!
//! Only the ChatML family and Mistral carry tool definitions; the other
//! formats raise on `tools` rather than drop them without a word.

use super::ChatTemplateType;

/// The default system message, or the conversation's own first system
/// message. Shared by the formats that open with a system turn.
macro_rules! system_message {
    () => {
        r#"{%- if messages and messages[0].role in ['system', 'developer'] %}
    {%- set system_message = messages[0].content | trim %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set system_message = default_system_message %}
    {%- set loop_messages = messages %}
{%- endif %}"#
    };
}

/// Refuse tool definitions in a format that has no place for them.
macro_rules! no_tools {
    () => {
        r#"
{%- if tools %}{{- raise_exception('this built-in chat template has no tool support') }}{%- endif %}"#
    };
}

/// The ChatML layout (`<|im_start|>role\n...<|im_end|>`) with the
/// Hermes-style tool block of the Qwen family, up to the generation prompt.
macro_rules! chatml_turns {
    () => {
        r#"{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages and messages[0].role == 'system' %}
        {{- messages[0].content }}
    {%- elif default_system_message %}
        {{- default_system_message }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- elif messages and messages[0].role == 'system' %}
    {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
{%- elif default_system_message %}
    {{- '<|im_start|>system\n' + default_system_message + '<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if message.role == 'system' and loop.first %}
    {%- elif message.role == 'assistant' and message.tool_calls %}
        {{- '<|im_start|>assistant' }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' + tool_call.name + '", "arguments": ' + (tool_call.arguments | tojson) + '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == 'tool' %}
        {%- if loop.first or messages[loop.index0 - 1].role != 'tool' %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' + message.content + '\n</tool_response>' }}
        {%- if loop.last or messages[loop.index0 + 1].role != 'tool' %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- else %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
"#
    };
}

/// ChatML: the generation prompt opens the assistant turn.
const CHATML: &str = concat!(
    chatml_turns!(),
    r#"{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"#
);

/// Qwen's ChatML. With `enable_thinking` set, the generation prompt opens
/// (`true`) or closes (`false`) a `<think>` block, as Qwen3.5 and later do.
const QWEN: &str = concat!(
    chatml_turns!(),
    r#"{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined %}
        {%- if enable_thinking %}
            {{- '<think>\n' }}
        {%- else %}
            {{- '<think>\n\n</think>\n\n' }}
        {%- endif %}
    {%- endif %}
{%- endif %}"#
);

/// Llama 2 chat: the system prompt rides in the first `[INST]` between
/// `<<SYS>>` markers, and each exchange is closed with the EOS token.
const LLAMA2: &str = concat!(
    system_message!(),
    no_tools!(),
    r#"
{%- for message in loop_messages %}
    {%- if (message.role == 'user') != (loop.index0 % 2 == 0) %}
        {{- raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {%- endif %}
    {%- if message.role == 'user' %}
        {%- if loop.first and system_message %}
            {%- set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message.content %}
        {%- else %}
            {%- set content = message.content %}
        {%- endif %}
        {{- '<s>[INST] ' + content | trim + ' [/INST]' }}
    {%- else %}
        {{- ' ' + message.content | trim + ' </s>' }}
    {%- endif %}
{%- endfor %}"#
);

/// Llama 3: header ids around each role, `<|eot_id|>` after each turn.
const LLAMA3: &str = concat!(
    system_message!(),
    no_tools!(),
    r#"
{{- '<|begin_of_text|>' }}
{%- if system_message %}
    {{- '<|start_header_id|>system<|end_header_id|>\n\n' + system_message + '<|eot_id|>' }}
{%- endif %}
{%- for message in loop_messages %}
    {{- '<|start_header_id|>' + message.role + '<|end_header_id|>\n\n' + message.content | trim + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"#
);

/// Llama 4: the Llama 3 layout with `<|header_start|>` / `<|header_end|>` and
/// `<|eot|>`.
const LLAMA4: &str = concat!(
    system_message!(),
    no_tools!(),
    r#"
{{- '<|begin_of_text|>' }}
{%- if system_message %}
    {{- '<|header_start|>system<|header_end|>\n\n' + system_message + '<|eot|>' }}
{%- endif %}
{%- for message in loop_messages %}
    {{- '<|header_start|>' + message.role + '<|header_end|>\n\n' + message.content | trim + '<|eot|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|header_start|>assistant<|header_end|>\n\n' }}
{%- endif %}"#
);

/// Mistral instruct (v0.1 / v0.2): `[INST] user [/INST]assistant</s>`, the
/// system prompt in front of the first user message, and Mistral v3's tool
/// blocks.
const MISTRAL: &str = concat!(
    system_message!(),
    r#"
{{- '<s>' }}
{%- if tools %}
    {{- '[AVAILABLE_TOOLS] ' + (tools | tojson) + ' [/AVAILABLE_TOOLS]' }}
{%- endif %}
{%- for message in loop_messages %}
    {%- if message.role == 'user' %}
        {%- if loop.first and system_message %}
            {{- '[INST] ' + system_message + '\n\n' + message.content + ' [/INST]' }}
        {%- else %}
            {{- '[INST] ' + message.content + ' [/INST]' }}
        {%- endif %}
    {%- elif message.role == 'assistant' and message.tool_calls %}
        {{- '[TOOL_CALLS] [' }}
        {%- for tool_call in message.tool_calls %}
            {%- set function = tool_call.function if tool_call.function is defined else tool_call %}
            {{- {'name': function.name, 'arguments': function.arguments} | tojson }}
            {%- if not loop.last %}
                {{- ', ' }}
            {%- endif %}
        {%- endfor %}
        {{- ']</s>' }}
    {%- elif message.role == 'assistant' %}
        {{- message.content + '</s>' }}
    {%- elif message.role == 'tool' %}
        {{- '[TOOL_RESULTS] ' + ({'content': message.content} | tojson) + ' [/TOOL_RESULTS]' }}
    {%- else %}
        {{- raise_exception('Only system, user, assistant and tool roles are supported') }}
    {%- endif %}
{%- endfor %}"#
);

/// Gemma 2 and 3: `<start_of_turn>role` ... `<end_of_turn>`, the assistant
/// role named `model`, the system prompt in front of the first user message.
const GEMMA: &str = concat!(
    system_message!(),
    no_tools!(),
    r#"
{{- '<bos>' }}
{%- for message in loop_messages %}
    {%- if (message.role == 'assistant') != (loop.index0 % 2 == 1) %}
        {{- raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {%- endif %}
    {%- set role = 'model' if message.role == 'assistant' else message.role %}
    {{- '<start_of_turn>' + role + '\n' }}
    {%- if loop.first and system_message %}
        {{- system_message + '\n\n' }}
    {%- endif %}
    {{- message.content | trim + '<end_of_turn>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<start_of_turn>model\n' }}
{%- endif %}"#
);

/// Gemma 4: `<|turn>role` ... `<turn|>`, a real system turn, and a thought
/// channel the generation prompt closes when thinking is off.
const GEMMA4: &str = concat!(
    system_message!(),
    no_tools!(),
    r#"
{{- '<bos>' }}
{%- if system_message %}
    {{- '<|turn>system\n' + system_message + '<turn|>\n' }}
{%- endif %}
{%- for message in loop_messages %}
    {%- set role = 'model' if message.role == 'assistant' else message.role %}
    {{- '<|turn>' + role + '\n' + message.content | trim + '<turn|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|turn>model\n' }}
    {%- if not (enable_thinking is defined and enable_thinking) %}
        {{- '<|channel>thought\n<channel|>' }}
    {%- endif %}
{%- endif %}"#
);

/// Phi-3: `<|role|>\ncontent<|end|>`.
const PHI3: &str = concat!(
    system_message!(),
    no_tools!(),
    r#"
{%- if system_message %}
    {{- '<|system|>\n' + system_message + '<|end|>\n' }}
{%- endif %}
{%- for message in loop_messages %}
    {{- '<|' + message.role + '|>\n' + message.content + '<|end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- endif %}"#
);

/// Phi-4: `<|im_start|>role<|im_sep|>content<|im_end|>`, with no newline
/// between turns.
const PHI4: &str = concat!(
    system_message!(),
    no_tools!(),
    r#"
{%- if system_message %}
    {{- '<|im_start|>system<|im_sep|>' + system_message + '<|im_end|>' }}
{%- endif %}
{%- for message in loop_messages %}
    {{- '<|im_start|>' + message.role + '<|im_sep|>' + message.content + '<|im_end|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant<|im_sep|>' }}
{%- endif %}"#
);

/// Alpaca: one instruction, its input and the response; the last message of
/// each role is the one used.
const ALPACA: &str = concat!(
    no_tools!(),
    r#"
{%- set ns = namespace(instruction='', input='', output='') %}
{%- for message in messages %}
    {%- if message.role in ['system', 'instruction'] %}
        {%- set ns.instruction = message.content %}
    {%- elif message.role in ['user', 'input'] %}
        {%- set ns.input = message.content %}
    {%- elif message.role in ['assistant', 'output'] %}
        {%- set ns.output = message.content %}
    {%- endif %}
{%- endfor %}
{%- if not ns.instruction and default_system_message %}
    {%- set ns.instruction = default_system_message %}
{%- endif %}
{%- if ns.instruction %}
    {{- '### Instruction:\n' + ns.instruction + '\n\n' }}
{%- endif %}
{%- if ns.input %}
    {{- '### Input:\n' + ns.input + '\n\n' }}
{%- endif %}
{{- '### Response:\n' + ns.output }}"#
);

/// Vicuna: `USER: ...` / `ASSISTANT: ...` lines under an optional system
/// prompt.
const VICUNA: &str = concat!(
    system_message!(),
    no_tools!(),
    r#"
{%- if system_message %}
    {{- system_message + '\n\n' }}
{%- endif %}
{%- for message in loop_messages %}
    {%- if message.role == 'user' %}
        {{- 'USER: ' + message.content + '\n' }}
    {%- else %}
        {{- 'ASSISTANT: ' + message.content + '\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- 'ASSISTANT: ' }}
{%- endif %}"#
);

/// Zephyr: `<|role|>\ncontent</s>`.
const ZEPHYR: &str = concat!(
    system_message!(),
    no_tools!(),
    r#"
{%- if system_message %}
    {{- '<|system|>\n' + system_message + '</s>\n' }}
{%- endif %}
{%- for message in loop_messages %}
    {{- '<|' + message.role + '|>\n' + message.content + '</s>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- endif %}"#
);

/// gpt-oss's Harmony format, reduced to the conversation: system and
/// developer turns, user turns, and assistant turns on the `final` channel.
/// The reasoning, tool and date preamble of the released template belong to
/// the model's own template.
const GPT_OSS: &str = concat!(
    system_message!(),
    no_tools!(),
    r#"
{%- if system_message %}
    {{- '<|start|>system<|message|>' + system_message + '<|end|>' }}
{%- endif %}
{%- for message in loop_messages %}
    {%- if message.role == 'assistant' %}
        {{- '<|start|>assistant<|channel|>final<|message|>' + message.content + '<|end|>' }}
    {%- else %}
        {{- '<|start|>' + message.role + '<|message|>' + message.content + '<|end|>' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start|>assistant<|channel|>final<|message|>' }}
{%- endif %}"#
);

/// DeepSeek (V2.5 / V3): the system prompt bare after the BOS, then
/// `<｜User｜>` / `<｜Assistant｜>` turns, each assistant turn closed with
/// `<｜end▁of▁sentence｜>`.
const DEEPSEEK: &str = concat!(
    system_message!(),
    no_tools!(),
    r#"
{{- '<｜begin▁of▁sentence｜>' }}
{%- if system_message %}
    {{- system_message }}
{%- endif %}
{%- for message in loop_messages %}
    {%- if message.role == 'user' %}
        {{- '<｜User｜>' + message.content }}
    {%- else %}
        {{- '<｜Assistant｜>' + message.content + '<｜end▁of▁sentence｜>' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<｜Assistant｜>' }}
{%- endif %}"#
);

/// Cohere Command R: turn tokens around a role token.
const COHERE: &str = concat!(
    system_message!(),
    no_tools!(),
    r#"
{{- '<BOS_TOKEN>' }}
{%- if system_message %}
    {{- '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}
{%- endif %}
{%- for message in loop_messages %}
    {%- if message.role == 'assistant' %}
        {{- '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' + message.content | trim + '<|END_OF_TURN_TOKEN|>' }}
    {%- else %}
        {{- '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + message.content | trim + '<|END_OF_TURN_TOKEN|>' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}
{%- endif %}"#
);

/// The Jinja template for a checkpoint of this family that ships none.
pub(super) fn source(template: ChatTemplateType) -> &'static str {
    match template {
        ChatTemplateType::ChatMl | ChatTemplateType::Custom => CHATML,
        ChatTemplateType::Qwen => QWEN,
        ChatTemplateType::Llama2 => LLAMA2,
        ChatTemplateType::Llama3 => LLAMA3,
        ChatTemplateType::Llama4 => LLAMA4,
        ChatTemplateType::Mistral => MISTRAL,
        ChatTemplateType::Gemma => GEMMA,
        ChatTemplateType::Gemma4 => GEMMA4,
        ChatTemplateType::Phi3 => PHI3,
        ChatTemplateType::Phi4 => PHI4,
        ChatTemplateType::Alpaca => ALPACA,
        ChatTemplateType::Vicuna => VICUNA,
        ChatTemplateType::Zephyr => ZEPHYR,
        ChatTemplateType::GptOss => GPT_OSS,
        ChatTemplateType::DeepSeek => DEEPSEEK,
        ChatTemplateType::Cohere => COHERE,
    }
}
