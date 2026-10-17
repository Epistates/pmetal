//! A request's parameters, checked and turned into what the engine takes.
//!
//! Chat completions, completions and Anthropic messages ask for the same few
//! things (a sampling temperature, an output budget, stop sequences, a seed)
//! in different words. Each endpoint fills a [`SamplingRequest`]; this module
//! validates it with the error OpenAI would give (the field named, the range
//! stated), fills in what the request leaves out from the model maker's
//! recommendation, and bounds the output by the context that is left.

use std::collections::HashMap;

use serde_json::{Map, Value};

use crate::engine::{InferenceEngine, SamplingParams};
use crate::error::{ServeError, ServeResult};

/// The most stop sequences OpenAI accepts.
pub const MAX_STOP_SEQUENCES: usize = 4;

/// The most alternatives per token `top_logprobs` may ask for.
pub const MAX_TOP_LOGPROBS: usize = 20;

/// The sampling fields of a request, as the client sent them.
#[derive(Debug, Default, Clone)]
pub struct SamplingRequest<'a> {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub max_tokens: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<i64>,
    pub logit_bias: Option<&'a HashMap<String, f32>>,
    /// `Some(n)`: report log-probabilities with `n` alternatives per token.
    pub logprobs_top_n: Option<usize>,
}

/// Stop sequences, resolved: the ones that are a single token end generation
/// before the token is taken, and all of them are matched in the text.
#[derive(Debug, Clone, Default)]
pub struct ResolvedStops {
    pub token_ids: Vec<u32>,
    pub sequences: Vec<String>,
}

fn out_of_range(param: &str, what: &str) -> ServeError {
    ServeError::invalid_with_code(param, "invalid_value", format!("`{param}` must be {what}"))
}

fn check_finite(param: &str, value: Option<f32>) -> ServeResult<()> {
    match value {
        Some(v) if !v.is_finite() => Err(out_of_range(param, "a finite number")),
        _ => Ok(()),
    }
}

/// Check the numeric ranges OpenAI documents.
pub fn validate_sampling(req: &SamplingRequest<'_>) -> ServeResult<()> {
    for (param, value) in [
        ("temperature", req.temperature),
        ("top_p", req.top_p),
        ("min_p", req.min_p),
        ("repetition_penalty", req.repetition_penalty),
        ("frequency_penalty", req.frequency_penalty),
        ("presence_penalty", req.presence_penalty),
    ] {
        check_finite(param, value)?;
    }
    if req.temperature.is_some_and(|t| !(0.0..=2.0).contains(&t)) {
        return Err(out_of_range("temperature", "between 0 and 2"));
    }
    if req.top_p.is_some_and(|p| p <= 0.0 || p > 1.0) {
        return Err(out_of_range("top_p", "in (0, 1]"));
    }
    if req.min_p.is_some_and(|p| !(0.0..1.0).contains(&p)) {
        return Err(out_of_range("min_p", "in [0, 1)"));
    }
    if req.repetition_penalty.is_some_and(|p| p <= 0.0) {
        return Err(out_of_range("repetition_penalty", "greater than 0"));
    }
    for (param, value) in [
        ("frequency_penalty", req.frequency_penalty),
        ("presence_penalty", req.presence_penalty),
    ] {
        if value.is_some_and(|p| !(-2.0..=2.0).contains(&p)) {
            return Err(out_of_range(param, "between -2 and 2"));
        }
    }
    if req.max_tokens == Some(0) {
        return Err(out_of_range("max_tokens", "at least 1"));
    }
    if req.logprobs_top_n.is_some_and(|n| n > MAX_TOP_LOGPROBS) {
        return Err(out_of_range("top_logprobs", "at most 20"));
    }
    Ok(())
}

/// `n`, within the server's limit.
pub fn check_choices(n: Option<usize>, max: usize) -> ServeResult<usize> {
    match n.unwrap_or(1) {
        0 => Err(out_of_range("n", "at least 1")),
        n if n > max => Err(ServeError::invalid_with_code(
            "n",
            "invalid_value",
            format!("`n` is at most {max} on this server"),
        )),
        n => Ok(n),
    }
}

/// A request's `logit_bias` as `(token, bias)` pairs, in token order. The
/// keys are token ids as decimal strings, as OpenAI sends them; anything else
/// is a bad request.
pub fn parse_logit_bias(bias: Option<&HashMap<String, f32>>) -> ServeResult<Vec<(u32, f32)>> {
    let Some(bias) = bias else {
        return Ok(Vec::new());
    };
    let mut pairs = bias
        .iter()
        .map(|(token, &value)| {
            let id = token.trim().parse::<u32>().map_err(|_| {
                ServeError::invalid(
                    "logit_bias",
                    format!("logit_bias keys are token ids; {token:?} is not one"),
                )
            })?;
            if !value.is_finite() || !(-100.0..=100.0).contains(&value) {
                return Err(ServeError::invalid(
                    "logit_bias",
                    format!("the logit_bias of token {id} must be in [-100, 100], got {value}"),
                ));
            }
            Ok((id, value))
        })
        .collect::<ServeResult<Vec<_>>>()?;
    pairs.sort_by_key(|&(id, _)| id);
    Ok(pairs)
}

/// Resolve stop strings to text sequences (and the ids of control tokens
/// among them), refusing more than `limit` (OpenAI's four; `None` for no
/// limit).
///
/// Every non-empty string is a text sequence: the reply is cut at its first
/// match in the text, wherever it falls in the tokens. A string that is a
/// single control token also ends generation by id, since decoding skips
/// control tokens and the text would never show it.
pub fn resolve_stop_sequences(
    stop: Option<&[String]>,
    limit: Option<usize>,
    engine: &InferenceEngine,
) -> ServeResult<ResolvedStops> {
    let stop = stop.unwrap_or(&[]);
    if limit.is_some_and(|limit| stop.len() > limit) {
        return Err(ServeError::invalid_with_code(
            "stop",
            "invalid_value",
            format!("`stop` takes at most {} sequences", limit.unwrap_or(0)),
        ));
    }
    let mut token_ids = Vec::new();
    let mut sequences = Vec::new();

    for stop in stop {
        if stop.is_empty() {
            continue;
        }
        sequences.push(stop.clone());
        // A stop sequence that is one control token (its text is empty, so
        // the text never shows it) ends generation by its id.
        if let Ok(ids) = engine.tokenize_with_special_tokens(stop) {
            if let [id] = ids.as_slice() {
                if engine.decode(&[*id]).is_ok_and(|text| text.is_empty()) {
                    token_ids.push(*id);
                }
            }
        }
    }

    token_ids.sort_unstable();
    token_ids.dedup();
    sequences.sort();
    sequences.dedup();

    Ok(ResolvedStops {
        token_ids,
        sequences,
    })
}

/// The engine's sampling parameters for a request.
///
/// What the request leaves out follows the model maker's recommendation for
/// the mode it runs in (thinking or not), as `pmetal infer` does; an
/// explicitly greedy request (`temperature: 0`) decodes plain argmax, with no
/// default penalty to shift it. The output budget is the request's, else the
/// model's default, and never more than the context has room for after the
/// prompt.
///
/// # Errors
///
/// A bad value, or a prompt that leaves no room to generate.
pub fn sampling_params(
    engine: &InferenceEngine,
    req: &SamplingRequest<'_>,
    stops: &ResolvedStops,
    thinking: bool,
    prompt_len: usize,
) -> ServeResult<SamplingParams> {
    validate_sampling(req)?;
    let room = engine.room_after(prompt_len)?;
    let defaults = engine.sampling_defaults(thinking);
    let temperature = req.temperature.unwrap_or(defaults.temperature);
    let max_tokens = match req.max_tokens {
        Some(requested) => requested.min(room),
        None => engine.default_max_tokens(thinking, prompt_len).min(room),
    };
    Ok(SamplingParams {
        max_tokens,
        temperature,
        top_k: req.top_k.or(Some(defaults.top_k)),
        top_p: req.top_p.or(Some(defaults.top_p)),
        min_p: req.min_p.or(Some(defaults.min_p)),
        repetition_penalty: req
            .repetition_penalty
            .or((temperature > 0.0).then_some(defaults.repetition_penalty)),
        frequency_penalty: req
            .frequency_penalty
            .or((temperature > 0.0).then_some(defaults.frequency_penalty)),
        presence_penalty: req
            .presence_penalty
            .or((temperature > 0.0).then_some(defaults.presence_penalty)),
        seed: req.seed.map(|seed| seed as u64),
        extra_stop_token_ids: stops.token_ids.clone(),
        stop_sequences: stops.sequences.clone(),
        logprobs_top_n: req.logprobs_top_n,
        logit_bias: parse_logit_bias(req.logit_bias)?,
    })
}

/// Refuse the OpenAI fields in `other` that change what is generated and
/// that this server does not implement. A field it does not know at all is
/// left alone (clients add vendor extensions), and so are those that only
/// label a request (`user`, `metadata`, `store`, `safety_identifier`,
/// `prompt_cache_key`, `service_tier`).
pub fn reject_unsupported(other: &Map<String, Value>, fields: &[(&str, &str)]) -> ServeResult<()> {
    for (name, why) in fields {
        match other.get(*name) {
            // Absent and null ask for nothing. (An empty object does: for
            // some fields, `{}` switches the feature on with its defaults.)
            None | Some(Value::Null) => {}
            Some(_) => return Err(ServeError::unsupported(*name, why)),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn invalid_param(result: ServeResult<()>) -> Option<String> {
        match result {
            Err(ServeError::Invalid { param, .. }) => param,
            other => panic!("expected an invalid parameter, got {other:?}"),
        }
    }

    #[test]
    fn ranges_name_their_parameter() {
        let req = |f: &dyn Fn(&mut SamplingRequest<'static>)| {
            let mut req = SamplingRequest::default();
            f(&mut req);
            validate_sampling(&req)
        };
        assert!(req(&|r| r.temperature = Some(2.0)).is_ok());
        assert_eq!(
            invalid_param(req(&|r| r.temperature = Some(2.5))).as_deref(),
            Some("temperature")
        );
        assert_eq!(
            invalid_param(req(&|r| r.temperature = Some(f32::NAN))).as_deref(),
            Some("temperature")
        );
        assert_eq!(
            invalid_param(req(&|r| r.top_p = Some(0.0))).as_deref(),
            Some("top_p")
        );
        assert_eq!(
            invalid_param(req(&|r| r.frequency_penalty = Some(2.5))).as_deref(),
            Some("frequency_penalty")
        );
        assert_eq!(
            invalid_param(req(&|r| r.logprobs_top_n = Some(21))).as_deref(),
            Some("top_logprobs")
        );
        assert_eq!(
            invalid_param(req(&|r| r.max_tokens = Some(0))).as_deref(),
            Some("max_tokens")
        );
    }

    #[test]
    fn choices_are_bounded() {
        assert_eq!(check_choices(None, 4).unwrap(), 1);
        assert_eq!(check_choices(Some(4), 4).unwrap(), 4);
        assert!(check_choices(Some(0), 4).is_err());
        assert!(check_choices(Some(5), 4).is_err());
    }

    #[test]
    fn logit_bias_keys_are_token_ids_and_come_back_in_order() {
        let bias = HashMap::from([("42".to_string(), -100.0), ("7".to_string(), 2.5)]);
        assert_eq!(
            parse_logit_bias(Some(&bias)).unwrap(),
            vec![(7, 2.5), (42, -100.0)]
        );
        assert!(parse_logit_bias(None).unwrap().is_empty());
        let words = HashMap::from([("hello".to_string(), 1.0)]);
        assert!(parse_logit_bias(Some(&words)).is_err());
        let negative = HashMap::from([("-3".to_string(), 1.0)]);
        assert!(parse_logit_bias(Some(&negative)).is_err());
        let huge = HashMap::from([("3".to_string(), 101.0)]);
        assert!(parse_logit_bias(Some(&huge)).is_err());
        let nan = HashMap::from([("3".to_string(), f32::NAN)]);
        assert!(parse_logit_bias(Some(&nan)).is_err());
    }

    #[test]
    fn unsupported_fields_are_named() {
        let other: Map<String, Value> =
            serde_json::from_str(r#"{"audio": {"voice": "x"}, "user": "u", "prediction": null}"#)
                .unwrap();
        let err = reject_unsupported(&other, &[("audio", "no audio output")]).unwrap_err();
        assert!(matches!(
            &err,
            ServeError::Invalid { param: Some(p), code: Some("unsupported_parameter"), .. } if p == "audio"
        ));
        // Null counts as absent; unknown and labelling fields pass.
        assert!(reject_unsupported(&other, &[("prediction", "x"), ("zzz", "x")]).is_ok());
    }
}
