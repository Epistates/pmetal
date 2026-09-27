//! TensorBoard event-file writer for scalar metrics.
//!
//! An event file is a sequence of TFRecords, each holding one serialized
//! `tensorflow.Event`. Only the messages scalars need are declared here, with
//! their upstream field numbers:
//! [event.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/event.proto)
//! and [summary.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto).
//!
//! A TFRecord is `u64 length | u32 masked_crc(length) | data | u32 masked_crc(data)`,
//! little-endian, with CRC-32C (Castagnoli)
//! ([record_writer.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/io/record_writer.cc)).

use crc::{CRC_32_ISCSI, Crc};
use prost::Message;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// `tensorflow.Event`, reduced to the fields a scalar log writes.
///
/// Upstream, `file_version` and `summary` are members of the `what` oneof.
/// Declaring them as optional fields encodes identically.
#[derive(Clone, PartialEq, Message)]
struct Event {
    #[prost(double, tag = "1")]
    wall_time: f64,
    #[prost(int64, tag = "2")]
    step: i64,
    #[prost(string, optional, tag = "3")]
    file_version: Option<String>,
    #[prost(message, optional, tag = "5")]
    summary: Option<Summary>,
}

/// `tensorflow.Summary`.
#[derive(Clone, PartialEq, Message)]
struct Summary {
    #[prost(message, repeated, tag = "1")]
    value: Vec<SummaryValue>,
}

/// `tensorflow.Summary.Value`.
#[derive(Clone, PartialEq, Message)]
struct SummaryValue {
    #[prost(string, tag = "1")]
    tag: String,
    /// A member of the `value` oneof upstream, so it has presence: a plain
    /// proto3 `float` would drop an exact `0.0` and TensorBoard would read a
    /// value with no payload.
    #[prost(float, optional, tag = "2")]
    simple_value: Option<f32>,
}

const CASTAGNOLI: Crc<u32> = Crc::<u32>::new(&CRC_32_ISCSI);

/// TFRecord's CRC mask: rotate right by 15, then add a constant.
fn masked_crc(bytes: &[u8]) -> u32 {
    CASTAGNOLI
        .checksum(bytes)
        .rotate_right(15)
        .wrapping_add(0xa282_ead8)
}

fn wall_time() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// Appends scalar events to one `events.out.tfevents.*` file.
pub struct EventWriter {
    file: BufWriter<File>,
    path: PathBuf,
}

impl EventWriter {
    /// Create a new event file in `log_dir`, creating the directory if needed.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory or file cannot be created.
    pub fn new(log_dir: impl AsRef<Path>) -> io::Result<Self> {
        let log_dir = log_dir.as_ref();
        std::fs::create_dir_all(log_dir)?;

        // TensorBoard only requires "tfevents" in the name. The pid keeps two
        // writers started in the same second from sharing a file.
        let secs = wall_time() as u64;
        let path = log_dir.join(format!(
            "events.out.tfevents.{secs}.pmetal.{}",
            std::process::id()
        ));

        let mut writer = Self {
            file: BufWriter::new(File::create(&path)?),
            path,
        };
        writer.write_event(&Event {
            wall_time: wall_time(),
            step: 0,
            file_version: Some("brain.Event:2".to_string()),
            summary: None,
        })?;
        Ok(writer)
    }

    /// The event file being written.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Record one scalar under `tag` at `step`.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub fn add_scalar(&mut self, tag: &str, value: f32, step: i64) -> io::Result<()> {
        self.add_scalars(&[(tag, value)], step)
    }

    /// Record several scalars at the same `step` in one event.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub fn add_scalars(&mut self, scalars: &[(&str, f32)], step: i64) -> io::Result<()> {
        let value = scalars
            .iter()
            .map(|&(tag, value)| SummaryValue {
                tag: tag.to_string(),
                simple_value: Some(value),
            })
            .collect();
        self.write_event(&Event {
            wall_time: wall_time(),
            step,
            file_version: None,
            summary: Some(Summary { value }),
        })
    }

    /// Flush buffered events to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails.
    pub fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }

    fn write_event(&mut self, event: &Event) -> io::Result<()> {
        let data = event.encode_to_vec();
        let len = (data.len() as u64).to_le_bytes();
        self.file.write_all(&len)?;
        self.file.write_all(&masked_crc(&len).to_le_bytes())?;
        self.file.write_all(&data)?;
        self.file.write_all(&masked_crc(&data).to_le_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CRC-32C's published check value, and the masked value TensorBoard's
    /// own `masked_crc32c` computes for the same input (tensorboard 2.21).
    #[test]
    fn the_record_checksum_is_masked_crc32c() {
        assert_eq!(CASTAGNOLI.checksum(b"123456789"), 0xe306_9283);
        assert_eq!(masked_crc(b"123456789"), 0xc78a_b0e5);
    }

    /// Read back every record, checking both checksums, and decode it.
    fn read_events(path: &Path) -> Vec<Event> {
        let bytes = std::fs::read(path).unwrap();
        let mut events = Vec::new();
        let mut rest = &bytes[..];
        while !rest.is_empty() {
            let (len, tail) = rest.split_at(8);
            let (len_crc, tail) = tail.split_at(4);
            assert_eq!(masked_crc(len).to_le_bytes(), len_crc);
            let n = u64::from_le_bytes(len.try_into().unwrap()) as usize;
            let (data, tail) = tail.split_at(n);
            let (data_crc, tail) = tail.split_at(4);
            assert_eq!(masked_crc(data).to_le_bytes(), data_crc);
            events.push(Event::decode(data).unwrap());
            rest = tail;
        }
        events
    }

    #[test]
    fn scalars_round_trip_through_the_record_format() {
        let dir = std::env::temp_dir().join(format!("pmetal-tb-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut writer = EventWriter::new(&dir).unwrap();
        writer
            .add_scalars(&[("train/loss", 2.5), ("train/epoch", 0.0)], 7)
            .unwrap();
        writer.flush().unwrap();

        let events = read_events(writer.path());
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].file_version.as_deref(), Some("brain.Event:2"));

        let summary = events[1].summary.as_ref().unwrap();
        assert_eq!(events[1].step, 7);
        assert_eq!(summary.value[0].tag, "train/loss");
        assert_eq!(summary.value[0].simple_value, Some(2.5));
        // A zero still carries its payload.
        assert_eq!(summary.value[1].simple_value, Some(0.0));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
