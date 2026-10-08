//! Images and videos in a decision record.
//!
//! The reference hands a record's `images` and `videos` to the checkpoint's
//! processor as Python objects (PIL images, frame lists) or as strings it can
//! load. Over JSON only strings exist, so a record names media like this:
//!
//! * `images`: a list of images, each a string: a base64-encoded image file
//!   (PNG, JPEG, WebP, ...), optionally as a `data:image/...;base64,` URI, or,
//!   where the caller allows it ([`MediaSources::AllowPaths`], the CLI), a path
//!   to an image file.
//! * `videos`: a list of videos, each either a list of frames (strings as for
//!   images) or an object `{"frames": [...], "fps": <frames per second>}`.
//!   `fps` is the rate of the frames given, used to sample them to the
//!   processor's rate (2 per second in the released configs) and to time-stamp
//!   them in the prompt, exactly as the reference treats a list of frames; the
//!   reference assumes 24 when it is absent. Video files are not decoded:
//!   decode the frames and send those.
//!
//! URLs are not fetched. The record's media go to the processor once, in
//! order: all images, then all videos, after the prompt's `STATE:` line.

use pmetal_data::qwen_vl_processing::{self, RgbImage, VideoFrames};
use serde_json::{Map, Value};

use super::DecisionError;

/// Which image strings a caller accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MediaSources {
    /// Base64 and data URIs only: what a server should read from a client.
    #[default]
    InlineOnly,
    /// Base64, data URIs and local file paths: the CLI, whose user owns the
    /// disk.
    AllowPaths,
}

/// A record's decoded images and videos.
#[derive(Debug, Default)]
pub struct RecordMediaInput {
    pub images: Vec<RgbImage>,
    pub videos: Vec<VideoFrames>,
}

impl RecordMediaInput {
    /// Whether the record carries no media.
    pub fn is_empty(&self) -> bool {
        self.images.is_empty() && self.videos.is_empty()
    }
}

/// Whether a record carries `images` or `videos` (non-empty, as the reference
/// tests them).
pub fn has_media(record: &Map<String, Value>) -> bool {
    ["images", "videos"]
        .iter()
        .any(|key| record.get(*key).is_some_and(super::encode::is_truthy))
}

/// The prompt text the reference gives the processor for this many images and
/// videos: each one's placeholder, then a newline.
pub fn placeholder_text(images: usize, videos: usize) -> String {
    format!(
        "{}\n",
        qwen_vl_processing::media_placeholders(images, videos)
    )
}

/// Decode a record's `images` and `videos`.
pub fn decode_record_media(
    record: &Map<String, Value>,
    sources: MediaSources,
) -> Result<RecordMediaInput, DecisionError> {
    if record
        .get("media_kwargs")
        .is_some_and(super::encode::is_truthy)
    {
        return Err(DecisionError::Unsupported(
            "media_kwargs is not supported: pmetal preprocesses media with the checkpoint's own \
             processor settings"
                .into(),
        ));
    }
    let list = |key: &str| -> Result<Vec<Value>, DecisionError> {
        match record.get(key) {
            None | Some(Value::Null) => Ok(Vec::new()),
            Some(Value::Array(items)) => Ok(items.clone()),
            Some(_) => Err(DecisionError::Request(format!("{key} must be a list"))),
        }
    };
    let images = list("images")?
        .iter()
        .enumerate()
        .map(|(i, item)| load_image(item, sources, &format!("images[{i}]")))
        .collect::<Result<Vec<_>, _>>()?;
    let videos = list("videos")?
        .iter()
        .enumerate()
        .map(|(i, item)| load_video(item, sources, &format!("videos[{i}]")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RecordMediaInput { images, videos })
}

fn load_image(item: &Value, sources: MediaSources, at: &str) -> Result<RgbImage, DecisionError> {
    let source = item.as_str().ok_or_else(|| {
        DecisionError::Request(format!(
            "{at}: an image must be a string (base64, a data URI, or a file path)"
        ))
    })?;
    if sources == MediaSources::InlineOnly && std::path::Path::new(source).is_file() {
        return Err(DecisionError::Request(format!(
            "{at}: local file paths are not read here; send the image as base64"
        )));
    }
    qwen_vl_processing::load_image_source(source)
        .map_err(|e| DecisionError::Request(format!("{at}: {e}")))
}

fn load_video(item: &Value, sources: MediaSources, at: &str) -> Result<VideoFrames, DecisionError> {
    let (frames, fps) = match item {
        Value::Array(frames) => (frames, None),
        Value::Object(video) => {
            let frames = video
                .get("frames")
                .and_then(Value::as_array)
                .ok_or_else(|| DecisionError::Request(format!("{at}: frames must be a list")))?;
            let fps = match video.get("fps") {
                None | Some(Value::Null) => None,
                Some(value) => Some(value.as_f64().filter(|f| *f > 0.0).ok_or_else(|| {
                    DecisionError::Request(format!("{at}: fps must be a positive number"))
                })?),
            };
            (frames, fps)
        }
        Value::String(_) => {
            return Err(DecisionError::Unsupported(format!(
                "{at}: video files are not decoded; send the video as a list of frames, or as \
                 {{\"frames\": [...], \"fps\": ...}}"
            )));
        }
        _ => {
            return Err(DecisionError::Request(format!(
                "{at}: a video must be a list of frames or an object with frames and fps"
            )));
        }
    };
    if frames.is_empty() {
        return Err(DecisionError::Request(format!(
            "{at}: a video needs frames"
        )));
    }
    let frames = frames
        .iter()
        .enumerate()
        .map(|(i, frame)| load_image(frame, sources, &format!("{at}.frames[{i}]")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(VideoFrames { frames, fps })
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;
    use serde_json::json;

    use super::*;

    fn png(value: u8) -> String {
        let image = RgbImage::from_pixel(4, 3, image::Rgb([value, 0, 0]));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(image)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    fn decode(record: Value, sources: MediaSources) -> Result<RecordMediaInput, DecisionError> {
        decode_record_media(record.as_object().unwrap(), sources)
    }

    #[test]
    fn both_video_forms_decode() {
        let media = decode(
            json!({
                "images": [png(1)],
                "videos": [[png(2), png(3)], {"frames": [png(4)], "fps": 30}]
            }),
            MediaSources::InlineOnly,
        )
        .unwrap();
        assert_eq!(media.images.len(), 1);
        assert_eq!(media.videos[0].frames.len(), 2);
        assert_eq!(media.videos[0].fps, None);
        assert_eq!(media.videos[1].fps, Some(30.0));
    }

    #[test]
    fn servers_do_not_read_paths() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.png");
        RgbImage::from_pixel(2, 2, image::Rgb([9, 9, 9]))
            .save(&path)
            .unwrap();
        let record = json!({"images": [path.to_str().unwrap()]});
        assert!(decode(record.clone(), MediaSources::InlineOnly).is_err());
        assert_eq!(
            decode(record, MediaSources::AllowPaths)
                .unwrap()
                .images
                .len(),
            1
        );
    }

    #[test]
    fn unusable_media_are_refused() {
        for record in [
            json!({"videos": ["clip.mp4"]}),
            json!({"videos": [[]]}),
            json!({"videos": [{"frames": [png(1)], "fps": 0}]}),
            json!({"images": [7]}),
            json!({"images": "a.png"}),
            json!({"images": [png(1)], "media_kwargs": {"fps": 1}}),
        ] {
            assert!(
                decode(record.clone(), MediaSources::AllowPaths).is_err(),
                "{record}"
            );
        }
    }

    #[test]
    fn placeholders_match_the_reference() {
        assert_eq!(
            placeholder_text(2, 1),
            "<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>\
             <|vision_start|><|video_pad|><|vision_end|>\n"
        );
    }
}
