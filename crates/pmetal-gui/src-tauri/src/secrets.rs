//! The Hugging Face access token, kept in the system keychain.
//!
//! The token used to live in the app's settings file, in plain text and
//! readable by anything running as the user, and was sent back to the webview
//! whenever the settings page loaded. Now it is written to the keychain (a
//! generic password under this app's identifier) and only ever read by the
//! backend: the webview can set it, clear it, and ask whether one is set. It
//! is handed to jobs through their environment (`HF_TOKEN`), never on a
//! command line.

use std::sync::Arc;

use keyring_core::{CredentialStore, Entry, Error as KeyringError};
use pmetal::core::SecretString;

use crate::error::AppError;

/// The keychain "service": the app's bundle identifier.
const SERVICE: &str = "ai.pmetal.desktop";
/// The keychain "account" of the token.
const ACCOUNT: &str = "huggingface-token";
/// The longest token accepted. Real ones are about forty characters.
const MAX_TOKEN_LEN: usize = 512;

/// Why a token was refused before it reached the keychain.
fn check_token(raw: &str) -> Result<&str, AppError> {
    let token = raw.trim();
    if token.is_empty() {
        return Err(AppError::new(
            "The token is empty. Paste a Hugging Face access token (it starts with hf_), \
             or use Remove to clear the saved one.",
        ));
    }
    if token.len() > MAX_TOKEN_LEN || !token.chars().all(|c| c.is_ascii_graphic()) {
        return Err(AppError::new(
            "That does not look like a Hugging Face access token: it has spaces, \
             line breaks or other unexpected characters. Copy the token again from \
             huggingface.co/settings/tokens.",
        ));
    }
    Ok(token)
}

/// The token's place in the keychain.
#[derive(Clone)]
pub struct HfToken {
    store: Result<Arc<CredentialStore>, String>,
    account: String,
}

impl std::fmt::Debug for HfToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfToken").finish_non_exhaustive()
    }
}

impl HfToken {
    /// The macOS keychain.
    pub fn system() -> Self {
        Self {
            store: system_store(),
            account: ACCOUNT.to_string(),
        }
    }

    /// Any credential store; for tests.
    #[cfg(test)]
    pub fn with_store(store: Arc<CredentialStore>) -> Self {
        Self {
            store: Ok(store),
            account: ACCOUNT.to_string(),
        }
    }

    fn entry(&self) -> Result<Entry, AppError> {
        let store = self.store.as_ref().map_err(|e| {
            AppError::new(format!(
                "The system keychain is not available ({e}), so the token cannot be saved. \
                 Set the HF_TOKEN environment variable instead."
            ))
        })?;
        store
            .build(SERVICE, &self.account, None)
            .map_err(|e| keychain_error("open", &e))
    }

    /// The saved token, if there is one.
    pub fn get(&self) -> Result<Option<SecretString>, AppError> {
        match self.entry()?.get_password() {
            Ok(token) => Ok(Some(SecretString::new(token))),
            Err(KeyringError::NoEntry) => Ok(None),
            Err(e) => Err(keychain_error("read", &e)),
        }
    }

    /// Whether a token is saved. A keychain that cannot be read counts as no.
    pub fn is_set(&self) -> bool {
        matches!(self.get(), Ok(Some(_)))
    }

    /// Save `token`, replacing any saved one.
    pub fn set(&self, token: &str) -> Result<(), AppError> {
        let token = check_token(token)?;
        self.entry()?
            .set_password(token)
            .map_err(|e| keychain_error("save", &e))
    }

    /// Remove the saved token. Nothing saved is not an error.
    pub fn clear(&self) -> Result<(), AppError> {
        match self.entry()?.delete_credential() {
            Ok(()) | Err(KeyringError::NoEntry) => Ok(()),
            Err(e) => Err(keychain_error("remove", &e)),
        }
    }
}

/// What a keychain refusal means to the person who sees it.
fn keychain_error(action: &str, e: &KeyringError) -> AppError {
    let hint = match e {
        KeyringError::NoStorageAccess(_) => {
            " Unlock the keychain, or allow PMetal access to it when macOS asks."
        }
        _ => "",
    };
    AppError::new(format!(
        "Could not {action} the Hugging Face token in the keychain: {e}.{hint}"
    ))
}

#[cfg(target_os = "macos")]
fn system_store() -> Result<Arc<CredentialStore>, String> {
    let store = apple_native_keyring_store::keychain::Store::new().map_err(|e| e.to_string())?;
    Ok(store)
}

#[cfg(not(target_os = "macos"))]
fn system_store() -> Result<Arc<CredentialStore>, String> {
    Err("PMetal keeps credentials in the macOS keychain".to_string())
}

#[cfg(test)]
pub(crate) fn in_memory() -> HfToken {
    HfToken::with_store(keyring_core::mock::Store::new().unwrap())
}

#[cfg(test)]
pub(crate) fn unavailable() -> HfToken {
    HfToken {
        store: Err("locked".to_string()),
        account: ACCOUNT.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real login keychain, under an account of its own so a token saved
    /// in the app is never touched. It can show a macOS prompt, so it is not
    /// run by default: `cargo test -- --ignored real_keychain`.
    #[test]
    #[ignore = "writes to the login keychain"]
    #[cfg(target_os = "macos")]
    fn real_keychain_round_trip() {
        let token = HfToken {
            store: system_store(),
            account: format!("test-{}", std::process::id()),
        };
        token.set("hf_realKeychainRoundTrip").unwrap();
        assert_eq!(
            token.get().unwrap().unwrap().expose_secret(),
            "hf_realKeychainRoundTrip"
        );
        token.set("hf_replaced").unwrap();
        assert_eq!(token.get().unwrap().unwrap().expose_secret(), "hf_replaced");
        token.clear().unwrap();
        assert!(token.get().unwrap().is_none());
    }

    #[test]
    fn a_saved_token_is_read_back_and_cleared() {
        let token = in_memory();
        assert!(token.get().unwrap().is_none());
        assert!(!token.is_set());

        token.set("  hf_exampleToken123 \n").unwrap();
        assert_eq!(
            token.get().unwrap().unwrap().expose_secret(),
            "hf_exampleToken123"
        );
        assert!(token.is_set());

        token.set("hf_replacement").unwrap();
        assert_eq!(
            token.get().unwrap().unwrap().expose_secret(),
            "hf_replacement"
        );

        token.clear().unwrap();
        assert!(token.get().unwrap().is_none());
        // Clearing what is not there is fine.
        token.clear().unwrap();
    }

    #[test]
    fn a_token_that_is_not_one_is_refused_before_it_is_saved() {
        let token = in_memory();
        for bad in [
            "",
            "   ",
            "hf_has space",
            "hf_line\nbreak",
            "hf_\u{e9}",
            &"x".repeat(600),
        ] {
            let err = token.set(bad).unwrap_err();
            assert!(!err.message().is_empty());
            assert!(
                !err.message().contains("hf_has"),
                "an error must not echo the token back: {err}"
            );
        }
        assert!(token.get().unwrap().is_none());
    }

    #[test]
    fn the_token_does_not_show_in_debug_output() {
        let token = in_memory();
        token.set("hf_neverShown").unwrap();
        let secret = token.get().unwrap().unwrap();
        assert!(!format!("{secret:?}").contains("hf_neverShown"));
        assert!(!format!("{token:?}").contains("hf_neverShown"));
    }

    #[test]
    fn a_keychain_that_cannot_open_says_what_to_do_instead() {
        let token = unavailable();
        let err = token.set("hf_x").unwrap_err();
        assert!(err.message().contains("HF_TOKEN"), "{err}");
        assert!(!token.is_set());
    }
}
