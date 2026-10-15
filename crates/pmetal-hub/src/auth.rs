//! Hub credentials: where a token comes from, and keeping it out of text.

use std::path::{Path, PathBuf};

use pmetal_core::SecretString;

/// Where the standard Hub token file lives under the Hub home directory.
const TOKEN_FILE: &str = "token";

/// The access token to use, by the precedence the Hub's own clients apply:
///
/// 1. the `explicit` token the caller passed (an empty or blank one counts as
///    none, so an unset-but-exported `HF_TOKEN=` cannot turn into a
///    `Bearer ` header that makes the Hub reject even public downloads);
/// 2. unless `HF_HUB_DISABLE_IMPLICIT_TOKEN` is set: the `HF_TOKEN`
///    environment variable;
/// 3. the file named by `HF_TOKEN_PATH`;
/// 4. the token file `hf auth login` writes, `<HF home>/token`.
///
/// Surrounding whitespace is trimmed: a token read from a file or a shell
/// variable usually carries a trailing newline, which is not a legal HTTP
/// header value and would otherwise drop the credential without a word.
pub fn resolve_token(explicit: Option<&SecretString>) -> Option<SecretString> {
    if let Some(token) = explicit.and_then(clean) {
        return Some(token);
    }
    resolve_implicit_token(&|name| std::env::var(name).ok(), &crate::cache::hf_home())
}

/// The implicit part of [`resolve_token`] with its inputs made explicit: `env`
/// reads a variable, `hf_home` is the Hub home directory.
fn resolve_implicit_token(
    env: &dyn Fn(&str) -> Option<String>,
    hf_home: &Path,
) -> Option<SecretString> {
    if env("HF_HUB_DISABLE_IMPLICIT_TOKEN").is_some_and(|v| !v.is_empty()) {
        return None;
    }
    if let Some(token) = env("HF_TOKEN").and_then(|t| clean_str(&t)) {
        return Some(token);
    }
    let from_file = |path: PathBuf| {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|t| clean_str(&t))
    };
    if let Some(token) = env("HF_TOKEN_PATH").and_then(|p| from_file(PathBuf::from(p))) {
        return Some(token);
    }
    from_file(hf_home.join(TOKEN_FILE))
}

fn clean(token: &SecretString) -> Option<SecretString> {
    clean_str(token.expose_secret())
}

fn clean_str(token: &str) -> Option<SecretString> {
    let token = token.trim();
    (!token.is_empty()).then(|| SecretString::new(token))
}

/// Replace anything that looks like a Hub access token or a bearer credential
/// in `text` with `[REDACTED]`.
///
/// Errors are built from server responses, URLs and `Debug` output of lower
/// layers; none of them is meant to carry a credential, and this is the
/// backstop that keeps it that way for text this crate puts in front of a
/// user or a log.
pub(crate) fn redact_secrets(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = find_secret_start(rest) {
        let (head, tail) = rest.split_at(at);
        out.push_str(head);
        let end = tail
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')))
            .unwrap_or(tail.len());
        out.push_str("[REDACTED]");
        rest = &tail[end..];
    }
    out.push_str(rest);
    out
}

/// Byte offset of the next secret in `s`: a Hub token (`hf_` or `api_org_`
/// followed by a run of token characters) or the credential after `Bearer `.
fn find_secret_start(s: &str) -> Option<usize> {
    const MIN_RUN: usize = 12;
    let token_run = |from: usize| {
        s[from..]
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
            .count()
    };
    let mut best: Option<usize> = None;
    for prefix in ["hf_", "api_org_"] {
        let mut from = 0;
        while let Some(i) = s[from..].find(prefix) {
            let start = from + i;
            // Not in the middle of a longer word (`self_`, `shelf_`).
            let boundary = s[..start]
                .chars()
                .next_back()
                .is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_'));
            if boundary && token_run(start + prefix.len()) >= MIN_RUN {
                best = Some(best.map_or(start, |b| b.min(start)));
                break;
            }
            from = start + prefix.len();
        }
    }
    // `Bearer <credential>`: the credential is whatever follows.
    if let Some(i) = s.to_ascii_lowercase().find("bearer ") {
        let start = i + "bearer ".len();
        if token_run(start) >= MIN_RUN {
            best = Some(best.map_or(start, |b| b.min(start)));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        move |name| map.get(name).cloned()
    }

    fn home_with_token(contents: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("token"), contents).unwrap();
        dir
    }

    #[test]
    fn an_explicit_token_wins_and_is_trimmed() {
        let token = SecretString::new("  hf_explicit_token_value\n");
        let resolved = resolve_token(Some(&token)).unwrap();
        assert_eq!(resolved.expose_secret(), "hf_explicit_token_value");
    }

    #[test]
    fn a_blank_explicit_token_is_not_a_token() {
        for blank in ["", "   ", "\n"] {
            assert!(clean(&SecretString::new(blank)).is_none(), "{blank:?}");
        }
    }

    #[test]
    fn the_environment_token_beats_the_token_file() {
        let home = home_with_token("hf_from_file_token\n");
        let token =
            resolve_implicit_token(&env_of(&[("HF_TOKEN", "hf_from_env_token")]), home.path());
        assert_eq!(token.unwrap().expose_secret(), "hf_from_env_token");
    }

    #[test]
    fn the_standard_token_file_is_read_and_trimmed() {
        let home = home_with_token("hf_from_file_token\n");
        let token = resolve_implicit_token(&env_of(&[]), home.path());
        assert_eq!(token.unwrap().expose_secret(), "hf_from_file_token");
    }

    #[test]
    fn an_empty_environment_token_falls_through_to_the_file() {
        let home = home_with_token("hf_from_file_token");
        let token = resolve_implicit_token(&env_of(&[("HF_TOKEN", "")]), home.path());
        assert_eq!(token.unwrap().expose_secret(), "hf_from_file_token");
    }

    #[test]
    fn hf_token_path_names_the_file_to_read() {
        let home = home_with_token("hf_from_default_file");
        let other = tempfile::tempdir().unwrap();
        let path = other.path().join("elsewhere");
        std::fs::write(&path, "hf_from_named_file\n").unwrap();
        let token = resolve_implicit_token(
            &env_of(&[("HF_TOKEN_PATH", path.to_str().unwrap())]),
            home.path(),
        );
        assert_eq!(token.unwrap().expose_secret(), "hf_from_named_file");
    }

    #[test]
    fn disabling_the_implicit_token_disables_every_source() {
        let home = home_with_token("hf_from_file_token");
        let token = resolve_implicit_token(
            &env_of(&[
                ("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1"),
                ("HF_TOKEN", "hf_from_env_token"),
            ]),
            home.path(),
        );
        assert!(token.is_none());
    }

    #[test]
    fn no_source_means_no_token() {
        let home = tempfile::tempdir().unwrap();
        assert!(resolve_implicit_token(&env_of(&[]), home.path()).is_none());
    }

    #[test]
    fn hub_tokens_and_bearer_credentials_are_redacted() {
        let text = "request to https://h/x failed: token hf_AbCdEfGhIjKlMnOpQrStUv rejected; \
                    sent Authorization: Bearer abcdef0123456789xyz; org api_org_0123456789abcdef";
        let redacted = redact_secrets(text);
        assert!(!redacted.contains("AbCdEfGh"), "{redacted}");
        assert!(!redacted.contains("abcdef0123456789xyz"), "{redacted}");
        assert!(!redacted.contains("0123456789abcdef"), "{redacted}");
        assert!(redacted.contains("https://h/x failed"), "{redacted}");
        assert!(redacted.contains("Bearer [REDACTED]"), "{redacted}");
    }

    #[test]
    fn ordinary_text_is_left_alone() {
        let text = "Repository not found: meta-llama/Llama-3.2-1B (self_attn.hf_proj, shelf_xxxxxxxxxxxxxxxx)";
        assert_eq!(redact_secrets(text), text);
    }
}
