//! Images and videos in chat requests.
//!
//! A chat message's `content` may be a list of parts. Besides
//! `{"type": "text", "text": ...}`, the server reads:
//!
//! * `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`
//!   (OpenAI's form; `image_url` may also be the URL string itself, and the
//!   Responses API's `input_image` spelling is read the same way). The image
//!   is any format the `image` crate decodes: PNG, JPEG, WebP, GIF, BMP, TIFF.
//!   `detail` is accepted and has no effect: the checkpoint's own processor
//!   sets each image's resolution from its pixel budget, as the reference
//!   processor does.
//! * `{"type": "video", "video": [frame, ...]}` or
//!   `{"type": "video", "video": {"frames": [frame, ...], "fps": 30}}`, each
//!   frame a `data:image/...;base64,` URI. This is pmetal's own form, the same
//!   as a `/v1/systemone` record's videos: `fps` is the rate of the frames
//!   given, used to sample them to the processor's rate and to time-stamp them
//!   in the prompt; the reference processor assumes 24 when it is absent.
//!
//! `/v1/messages` image blocks (`{"type": "image", "source": {"type":
//! "base64", "media_type": ..., "data": ...}}`) are translated into the first
//! form by [`crate::anthropic`].
//!
//! Refused with a 400, by name: remote `http(s)` URLs (the server fetches
//! nothing on a client's behalf), file paths and every other URL scheme (the
//! server does not read its own disk for a client), video files (`video_url`:
//! decode the frames and send those), audio, and any other part type.
//!
//! Each image and video goes to the chat template as an item of its message,
//! where the client put it among the text, so the template places the model's
//! own placeholder exactly as the reference processor's `apply_chat_template`
//! does.

use base64::Engine as _;
use pmetal_data::chat_templates::{ContentPart, Message};
use pmetal_data::qwen_vl_processing::{RgbImage, VideoFrames, decode_image};
use serde_json::Value;

use crate::error::{ServeError, ServeResult};
use crate::types::ChatMessage;

/// One image or video as the request sent it: base64 payloads, not yet
/// decoded.
#[derive(Debug, Clone, PartialEq)]
pub enum MediaSource {
    /// An image's base64 payload.
    Image(String),
    /// A video's frames (base64 payloads) and their frame rate.
    Video {
        frames: Vec<String>,
        fps: Option<f64>,
    },
}

/// One image or video, decoded.
#[derive(Debug)]
pub enum DecodedMedia {
    Image(RgbImage),
    Video(VideoFrames),
}

/// A chat request split for a vision model: the messages as the chat template
/// takes them, and the media in prompt order.
#[derive(Debug)]
pub struct ChatMedia {
    pub messages: Vec<Message>,
    pub media: Vec<MediaSource>,
}

/// Whether any message carries content parts other than text.
pub fn has_media(messages: &[ChatMessage]) -> bool {
    messages.iter().any(|m| m.parts.is_some())
}

/// The chat template's view of a message.
pub(crate) fn template_message(message: &ChatMessage, parts: Option<Vec<ContentPart>>) -> Message {
    let mut out = match parts {
        Some(parts) => Message::with_parts(message.role.clone(), parts),
        None => Message::new(message.role.clone(), message.content.clone()),
    };
    out.tool_calls = message.tool_calls.clone();
    out
}

/// Validate every message's parts and split the request into template
/// messages and media. Errors name the offending part.
pub fn split_messages(messages: &[ChatMessage]) -> ServeResult<ChatMedia> {
    let mut media = Vec::new();
    let messages = messages
        .iter()
        .enumerate()
        .map(|(i, message)| {
            let Some(raw) = message.parts.as_ref() else {
                return Ok(template_message(message, None));
            };
            let parts = raw
                .iter()
                .enumerate()
                .map(|(j, part)| parse_part(part, &format!("messages[{i}].content[{j}]")))
                .collect::<ServeResult<Vec<_>>>()?;
            let template_parts = parts
                .into_iter()
                .map(|part| match part {
                    Parsed::Text(text) => ContentPart::Text(text),
                    Parsed::Media(source) => {
                        let part = match source {
                            MediaSource::Image(_) => ContentPart::Image,
                            MediaSource::Video { .. } => ContentPart::Video,
                        };
                        media.push(source);
                        part
                    }
                })
                .collect();
            Ok(template_message(message, Some(template_parts)))
        })
        .collect::<ServeResult<Vec<_>>>()?;
    Ok(ChatMedia { messages, media })
}

enum Parsed {
    Text(String),
    Media(MediaSource),
}

fn bad(at: &str, message: impl std::fmt::Display) -> ServeError {
    ServeError::BadRequest(format!("{at}: {message}"))
}

fn parse_part(part: &Value, at: &str) -> ServeResult<Parsed> {
    let kind = part
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| bad(at, "a content part needs a string \"type\""))?;
    match kind {
        "text" => part
            .get("text")
            .and_then(Value::as_str)
            .map(|text| Parsed::Text(text.to_owned()))
            .ok_or_else(|| bad(at, "a text part needs a string \"text\"")),
        // `input_image` is the Responses API's spelling, `image_url` holding
        // the URL itself; some clients send it here too. `detail` is accepted
        // and has no effect: the checkpoint's processor sets the resolution.
        "image_url" | "input_image" => {
            let url = match part.get("image_url") {
                Some(Value::String(url)) => url.as_str(),
                Some(Value::Object(image)) => image
                    .get("url")
                    .and_then(Value::as_str)
                    .ok_or_else(|| bad(at, "image_url needs a string \"url\""))?,
                _ if part.get("file_id").is_some() => {
                    return Err(bad(
                        at,
                        "uploaded files are not supported; send the image inline as a \
                         data:image/...;base64,... URI",
                    ));
                }
                _ => return Err(bad(at, format!("an {kind} part needs an \"image_url\""))),
            };
            Ok(Parsed::Media(MediaSource::Image(image_payload(url, at)?)))
        }
        "video" => {
            let (frames, fps) = match part.get("video") {
                Some(Value::Array(frames)) => (frames, None),
                Some(Value::Object(video)) => {
                    let frames = video
                        .get("frames")
                        .and_then(Value::as_array)
                        .ok_or_else(|| bad(at, "video.frames must be a list of frames"))?;
                    let fps = match video.get("fps") {
                        None | Some(Value::Null) => None,
                        Some(fps) => Some(
                            fps.as_f64()
                                .filter(|fps| fps.is_finite() && *fps > 0.0)
                                .ok_or_else(|| bad(at, "video.fps must be a positive number"))?,
                        ),
                    };
                    (frames, fps)
                }
                Some(Value::String(_)) => {
                    return Err(bad(
                        at,
                        "video files are not decoded; send the video's frames as a list of \
                         data:image/...;base64 URIs",
                    ));
                }
                _ => {
                    return Err(bad(
                        at,
                        "a video part needs \"video\": a list of frames, or {\"frames\": [...], \
                         \"fps\": ...}",
                    ));
                }
            };
            if frames.is_empty() {
                return Err(bad(at, "a video needs at least one frame"));
            }
            let frames = frames
                .iter()
                .enumerate()
                .map(|(k, frame)| {
                    let at = format!("{at}.video.frames[{k}]");
                    let url = frame
                        .as_str()
                        .ok_or_else(|| bad(&at, "a frame must be a data:image/...;base64 URI"))?;
                    image_payload(url, &at)
                })
                .collect::<ServeResult<Vec<_>>>()?;
            Ok(Parsed::Media(MediaSource::Video { frames, fps }))
        }
        "video_url" => Err(bad(
            at,
            "video files are not decoded; send the video's frames as a \"video\" part \
             ({\"type\": \"video\", \"video\": [data:image/...;base64 URIs]})",
        )),
        "input_audio" | "audio" => Err(bad(at, "audio content is not supported")),
        other => Err(bad(at, format!("unsupported content part type {other:?}"))),
    }
}

/// The base64 payload of an image URL, which must be a
/// `data:image/<format>;base64,<payload>` URI.
fn image_payload(url: &str, at: &str) -> ServeResult<String> {
    let url = url.trim();
    let lower = url.get(..8).unwrap_or(url).to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") {
        return Err(bad(
            at,
            "remote image URLs are not fetched; send the image inline as a \
             data:image/...;base64,... URI",
        ));
    }
    let Some(uri) = url.strip_prefix("data:") else {
        return Err(bad(
            at,
            "an image must be a data:image/...;base64,... URI; the server does not read file \
             paths or other URL schemes",
        ));
    };
    let (header, payload) = uri
        .split_once(',')
        .ok_or_else(|| bad(at, "malformed data URI: no ',' before the payload"))?;
    let mut fields = header.split(';');
    let mime = fields.next().unwrap_or_default();
    if !mime.to_ascii_lowercase().starts_with("image/") {
        return Err(bad(
            at,
            format!("a data URI for an image must have an image/* media type, got {mime:?}"),
        ));
    }
    if !fields.any(|field| field.eq_ignore_ascii_case("base64")) {
        return Err(bad(
            at,
            "the image data URI must be base64-encoded (;base64,)",
        ));
    }
    if payload.trim().is_empty() {
        return Err(bad(at, "the image data URI has no payload"));
    }
    Ok(payload.to_owned())
}

/// Decode every image and video frame. CPU work, for the blocking pool.
pub fn decode_media(sources: &[MediaSource]) -> ServeResult<Vec<DecodedMedia>> {
    let mut images = 0;
    let mut videos = 0;
    sources
        .iter()
        .map(|source| match source {
            MediaSource::Image(payload) => {
                images += 1;
                decode_payload(payload, &format!("image {images}")).map(DecodedMedia::Image)
            }
            MediaSource::Video { frames, fps } => {
                videos += 1;
                let frames = frames
                    .iter()
                    .enumerate()
                    .map(|(k, frame)| decode_payload(frame, &format!("video {videos}, frame {k}")))
                    .collect::<ServeResult<Vec<_>>>()?;
                Ok(DecodedMedia::Video(VideoFrames { frames, fps: *fps }))
            }
        })
        .collect()
}

fn decode_payload(payload: &str, what: &str) -> ServeResult<RgbImage> {
    let compact: String = payload.chars().filter(|c| !c.is_whitespace()).collect();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(compact.as_bytes())
        .map_err(|e| bad(what, format!("invalid base64: {e}")))?;
    decode_image(&bytes).map_err(|e| bad(what, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn png_base64(width: u32, height: u32) -> String {
        let image = RgbImage::from_pixel(width, height, image::Rgb([200, 10, 10]));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(image)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    fn message(value: Value) -> ChatMessage {
        serde_json::from_value(value).unwrap()
    }

    fn bad_request(result: ServeResult<ChatMedia>) -> String {
        match result {
            Err(ServeError::BadRequest(message)) => message,
            other => panic!("expected a 400, got {other:?}"),
        }
    }

    #[test]
    fn string_and_text_only_content_carry_no_parts() {
        let plain = message(json!({"role": "user", "content": "hi"}));
        assert_eq!(plain.content, "hi");
        assert!(plain.parts.is_none());
        let texts = message(json!({"role": "user", "content": [
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"}
        ]}));
        assert_eq!(texts.content, "ab");
        assert!(texts.parts.is_none());
        let null = message(json!({"role": "assistant", "content": null}));
        assert_eq!(null.content, "");
        assert!(!has_media(&[plain, texts, null]));
    }

    #[test]
    fn images_and_videos_keep_their_place_among_the_text() {
        let png = png_base64(3, 2);
        let messages = [
            message(json!({"role": "system", "content": "be brief"})),
            message(json!({"role": "user", "content": [
                {"type": "text", "text": "compare "},
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{png}"), "detail": "high"}},
                {"type": "text", "text": " with"},
                {"type": "video", "video": {"frames": [format!("data:image/png;base64,{png}")], "fps": 2}},
                {"type": "input_image", "image_url": format!("data:image/PNG;charset=x;base64,{png}")}
            ]})),
        ];
        assert!(has_media(&messages));
        let split = split_messages(&messages).unwrap();
        assert!(split.messages[0].parts.is_none());
        assert_eq!(
            split.messages[1].parts.as_deref().unwrap(),
            [
                ContentPart::Text("compare ".into()),
                ContentPart::Image,
                ContentPart::Text(" with".into()),
                ContentPart::Video,
                ContentPart::Image,
            ]
        );
        assert_eq!(split.messages[1].content, "compare  with");
        assert_eq!(
            split.media,
            [
                MediaSource::Image(png.clone()),
                MediaSource::Video {
                    frames: vec![png.clone()],
                    fps: Some(2.0)
                },
                MediaSource::Image(png.clone()),
            ]
        );
        let decoded = decode_media(&split.media).unwrap();
        match &decoded[0] {
            DecodedMedia::Image(image) => assert_eq!(image.dimensions(), (3, 2)),
            other => panic!("{other:?}"),
        }
        match &decoded[1] {
            DecodedMedia::Video(video) => {
                assert_eq!(video.frames.len(), 1);
                assert_eq!(video.fps, Some(2.0));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn unreadable_sources_are_refused_by_name() {
        let png = png_base64(2, 2);
        let cases = [
            (
                json!({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}),
                "remote image URLs are not fetched",
            ),
            (
                json!({"type": "image_url", "image_url": {"url": "HTTP://example.com/a.png"}}),
                "remote image URLs are not fetched",
            ),
            (
                json!({"type": "image_url", "image_url": {"url": "/etc/passwd"}}),
                "does not read file paths",
            ),
            (
                json!({"type": "image_url", "image_url": {"url": "file:///etc/passwd"}}),
                "does not read file paths",
            ),
            (
                json!({"type": "image_url", "image_url": {"url": png.clone()}}),
                "does not read file paths",
            ),
            (
                json!({"type": "image_url", "image_url": {"url": format!("data:text/plain;base64,{png}")}}),
                "image/* media type",
            ),
            (
                json!({"type": "image_url", "image_url": {"url": "data:image/png,rawbytes"}}),
                "base64-encoded",
            ),
            (
                json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}),
                "no payload",
            ),
            (
                json!({"type": "image_url"}),
                "an image_url part needs an \"image_url\"",
            ),
            (
                json!({"type": "input_image", "file_id": "file-abc"}),
                "uploaded files are not supported",
            ),
            (
                json!({"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}),
                "video files are not decoded",
            ),
            (
                json!({"type": "video", "video": "clip.mp4"}),
                "video files are not decoded",
            ),
            (json!({"type": "video", "video": []}), "at least one frame"),
            (
                json!({"type": "video", "video": {"frames": [format!("data:image/png;base64,{png}")], "fps": 0}}),
                "fps must be a positive number",
            ),
            (
                json!({"type": "video", "video": ["https://example.com/f.png"]}),
                "video.frames[0]: remote image URLs",
            ),
            (
                json!({"type": "input_audio", "input_audio": {"data": "AAAA"}}),
                "audio content is not supported",
            ),
            (
                json!({"type": "file", "file": {}}),
                "unsupported content part type \"file\"",
            ),
            (json!({"text": "untyped"}), "needs a string \"type\""),
        ];
        for (part, expected) in cases {
            let messages = [message(json!({"role": "user", "content": [part.clone()]}))];
            let error = bad_request(split_messages(&messages));
            assert!(
                error.starts_with("messages[0].content[0]") && error.contains(expected),
                "{part}: {error}"
            );
        }
    }

    #[test]
    fn undecodable_payloads_are_bad_requests() {
        let not_base64 = [MediaSource::Image("!!!".into())];
        assert!(matches!(
            decode_media(&not_base64),
            Err(ServeError::BadRequest(message)) if message.contains("image 1: invalid base64")
        ));
        let not_an_image = [MediaSource::Image(
            base64::engine::general_purpose::STANDARD.encode(b"plain text"),
        )];
        assert!(matches!(
            decode_media(&not_an_image),
            Err(ServeError::BadRequest(message)) if message.contains("could not decode image")
        ));
    }
}
