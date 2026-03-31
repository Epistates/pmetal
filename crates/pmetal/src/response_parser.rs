#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedAssistantResponse {
    pub thinking: Option<String>,
    pub response: String,
    pub truncated_thinking: bool,
}

pub fn parse_assistant_response(text: &str) -> ParsedAssistantResponse {
    ParsedAssistantResponse {
        thinking: extract_thinking_content(text),
        response: extract_final_response(text),
        truncated_thinking: text.contains("<think>") && !text.contains("</think>"),
    }
}

/// Extract the final response after `</think>`, discarding thinking content.
pub fn extract_final_response(text: &str) -> String {
    if let Some(pos) = text.rfind("</think>") {
        let after_think = &text[pos + "</think>".len()..];
        let cleaned = after_think
            .trim()
            .trim_start_matches("<think>")
            .trim_start_matches('\n');
        return strip_eos_tokens(cleaned).to_string();
    }

    if text.contains("<think>") {
        return "[Response truncated - model was still thinking. Use --no-thinking or increase --max-tokens]".to_string();
    }

    strip_eos_tokens(text).to_string()
}

/// Extract thinking content from a response for display purposes.
pub fn extract_thinking_content(text: &str) -> Option<String> {
    let end = text.rfind("</think>")?;
    let search_region = &text[..end];

    let mut last_real_start = None;
    let mut pos = 0;
    while let Some(start) = search_region[pos..].find("<think>") {
        let absolute_start = pos + start;
        let after_tag = &search_region[absolute_start + "<think>".len()..];
        let trimmed = after_tag.trim_start();
        if !trimmed.starts_with("<think>") && !trimmed.is_empty() {
            last_real_start = Some(absolute_start);
        }
        pos = absolute_start + "<think>".len();
    }

    if let Some(start) = last_real_start {
        let thinking = &text[start + "<think>".len()..end];
        let cleaned = thinking
            .trim()
            .trim_start_matches("<think>")
            .trim_start_matches('\n')
            .trim();
        if !cleaned.is_empty() {
            return Some(cleaned.to_string());
        }
    }

    if let Some(start) = text.find("<think>") {
        if end > start {
            let thinking = &text[start + "<think>".len()..end];
            let cleaned = thinking.trim();
            if !cleaned.is_empty() {
                return Some(cleaned.to_string());
            }
        }
    }

    None
}

fn strip_eos_tokens(text: &str) -> &str {
    const EOS_TOKENS: &[&str] = &[
        "<|endoftext|>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|eot|>",
        "<end_of_turn>",
        "<|END_OF_TURN_TOKEN|>",
        "<｜end▁of▁sentence｜>",
        "<|return|>",
        "<|end|>",
        "</s>",
    ];

    let mut s = text.trim();
    loop {
        let before = s;
        for tok in EOS_TOKENS {
            s = s.trim_end_matches(tok).trim();
        }
        if s == before {
            break;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::{extract_final_response, extract_thinking_content, parse_assistant_response};

    #[test]
    fn extract_final_response_returns_tail_after_complete_think_block() {
        assert_eq!(
            extract_final_response("<think>\nplan\n</think>\nParis<|im_end|>"),
            "Paris"
        );
    }

    #[test]
    fn extract_final_response_flags_incomplete_thinking() {
        assert_eq!(
            extract_final_response("<think>\nplanning"),
            "[Response truncated - model was still thinking. Use --no-thinking or increase --max-tokens]"
        );
    }

    #[test]
    fn extract_thinking_content_uses_last_real_think_block() {
        assert_eq!(
            extract_thinking_content("<think>\n<think>\nplan\n</think>\nParis"),
            Some("plan".to_string())
        );
    }

    #[test]
    fn parse_assistant_response_tracks_truncated_thinking() {
        let parsed = parse_assistant_response("<think>\nplanning");
        assert!(parsed.truncated_thinking);
        assert_eq!(parsed.thinking, None);
        assert_eq!(
            parsed.response,
            "[Response truncated - model was still thinking. Use --no-thinking or increase --max-tokens]"
        );
    }
}
