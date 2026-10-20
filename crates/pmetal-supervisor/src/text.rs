//! What a job's output is made of before anything shows it: secrets hidden,
//! terminal control sequences gone.

use std::borrow::Cow;

/// Hide anything in `text` that looks like a credential: Hugging Face tokens
/// (`hf_…`) and bearer tokens. A job's output is shown to people and models,
/// and a token must not reach either.
pub fn redact_secrets(text: &str) -> Cow<'_, str> {
    if !text.contains("hf_") && !text.contains("Bearer ") && !text.contains("bearer ") {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while !rest.is_empty() {
        let hf = rest.find("hf_");
        let bearer = rest.find("Bearer ").or_else(|| rest.find("bearer "));
        let (at, prefix_len) = match (hf, bearer) {
            (Some(h), Some(b)) if b < h => (b, 7),
            (Some(h), _) => (h, 3),
            (None, Some(b)) => (b, 7),
            (None, None) => break,
        };
        let token_len = rest[at + prefix_len..]
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '=')))
            .unwrap_or(rest.len() - at - prefix_len);
        // `hf_` inside a word (`shelf_life`) is not a token, and a short tail is
        // not a credential.
        let preceded_by_word = rest[..at]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_ascii_alphanumeric());
        let is_token = token_len >= 8 && !(prefix_len == 3 && preceded_by_word);
        out.push_str(&rest[..at]);
        if is_token {
            out.push_str(&rest[at..at + prefix_len]);
            out.push_str("[REDACTED]");
        } else {
            out.push_str(&rest[at..at + prefix_len + token_len]);
        }
        rest = &rest[at + prefix_len + token_len..];
    }
    out.push_str(rest);
    Cow::Owned(out)
}

/// Words in a flag's name that say its value is a credential.
const SECRET_FLAG_WORDS: &[&str] = &["key", "token", "secret", "password", "passwd"];

/// `argv` as it may be shown: the value of a credential flag (`--api-key k`,
/// `--hf-token=t`) hidden, and any token inside another argument redacted.
pub fn redact_argv(argv: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(argv.len());
    let mut hide_next = false;
    for arg in argv {
        if hide_next {
            hide_next = false;
            out.push("[REDACTED]".to_string());
            continue;
        }
        if let Some(flag) = arg.strip_prefix("--") {
            let (name, value) = match flag.split_once('=') {
                Some((name, value)) => (name, Some(value)),
                None => (flag, None),
            };
            if is_secret_flag(name) {
                match value {
                    Some(_) => out.push(format!("--{name}=[REDACTED]")),
                    None => {
                        out.push(arg.clone());
                        hide_next = true;
                    }
                }
                continue;
            }
        }
        out.push(redact_secrets(arg).into_owned());
    }
    out
}

fn is_secret_flag(name: &str) -> bool {
    name.split(['-', '_'])
        .any(|word| SECRET_FLAG_WORDS.contains(&word))
}

/// Make one line of output safe to show in a terminal and to store.
///
/// Escape sequences (colours, cursor moves, window titles) are removed, a
/// carriage return keeps only what was written after it (a progress bar
/// redrawing itself leaves its last state, not every frame), tabs become
/// spaces, and any other control character is dropped. What the line says is
/// otherwise untouched.
pub fn sanitize_line(line: &str) -> Cow<'_, str> {
    let clean = line
        .chars()
        .all(|c| !c.is_control() && c != '\u{7f}' && !is_bidi_override(c));
    if clean {
        return Cow::Borrowed(line);
    }
    // Only what follows the last carriage return was visible at the end.
    let visible = line.rsplit('\r').find(|s| !s.is_empty()).unwrap_or("");
    let mut out = String::with_capacity(visible.len());
    for piece in anstream::adapter::strip_str(visible) {
        for c in piece.chars() {
            match c {
                '\t' => out.push_str("    "),
                c if c.is_control() || is_bidi_override(c) => {}
                c => out.push(c),
            }
        }
    }
    Cow::Owned(out)
}

/// Characters that reorder text on screen: a line of output that uses them can
/// show a different string than the one it holds.
fn is_bidi_override(c: char) -> bool {
    matches!(c, '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hugging_face_tokens_and_bearer_tokens_are_hidden() {
        let token = "hf_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789";
        let text = format!("401 for {token}: header Authorization: Bearer abc.def-ghi_jkl123 done");
        let out = redact_secrets(&text);
        assert!(!out.contains(token), "{out}");
        assert!(!out.contains("abc.def-ghi_jkl123"), "{out}");
        assert!(out.contains("hf_[REDACTED]"), "{out}");
        assert!(out.contains("Bearer [REDACTED]"), "{out}");
        assert!(out.contains("done"), "{out}");
    }

    #[test]
    fn ordinary_text_is_left_alone() {
        for text in [
            "loss 2.5 at step 10",
            "shelf_life_days = 30",
            "hf_ alone",
            "the Bearer was here",
            "path /models/hf_cache/x",
        ] {
            assert_eq!(redact_secrets(text), text);
        }
    }

    #[test]
    fn a_credential_flags_value_is_hidden_in_either_spelling() {
        let argv: Vec<String> = [
            "serve",
            "--model",
            "m",
            "--api-key",
            "sk-secret",
            "--hf-token=hf_AbCdEfGhIjKlMnOpQrSt",
            "--max-tokens",
            "64",
            "--note",
            "uses hf_AbCdEfGhIjKlMnOpQrSt inline",
        ]
        .map(String::from)
        .to_vec();
        let shown = redact_argv(&argv).join(" ");
        assert!(!shown.contains("sk-secret"), "{shown}");
        assert!(!shown.contains("AbCdEfGh"), "{shown}");
        assert!(shown.contains("--api-key [REDACTED]"), "{shown}");
        assert!(shown.contains("--hf-token=[REDACTED]"), "{shown}");
        // `--max-tokens` counts tokens; the word is `tokens`, not `token`.
        assert!(shown.contains("--max-tokens 64"), "{shown}");
        assert!(shown.contains("--model m"), "{shown}");
    }

    #[test]
    fn escape_sequences_and_control_characters_do_not_survive() {
        assert_eq!(sanitize_line("plain text"), "plain text");
        assert_eq!(
            sanitize_line("\u{1b}[31mred\u{1b}[0m and \u{1b}[1;32mgreen\u{1b}[0m"),
            "red and green"
        );
        // A window-title sequence, which could retitle the user's terminal.
        assert_eq!(sanitize_line("a\u{1b}]0;owned\u{7}b"), "ab");
        assert_eq!(sanitize_line("a\tb"), "a    b");
        assert_eq!(sanitize_line("bell\u{7}\u{8}x"), "bellx");
    }

    #[test]
    fn a_progress_bar_leaves_its_last_frame() {
        assert_eq!(sanitize_line("10%\r50%\r100% done"), "100% done");
        // A trailing carriage return does not erase the line.
        assert_eq!(sanitize_line("done\r"), "done");
    }

    #[test]
    fn text_that_reorders_on_screen_is_dropped() {
        assert_eq!(sanitize_line("a\u{202e}b"), "ab");
    }
}
