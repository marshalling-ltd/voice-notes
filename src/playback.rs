//! Playing a WAV recording through a chosen output device with `cpal`.
//!
//! The same code runs on Linux and macOS, and the device is the one the user chose (or the
//! system default). A `cpal` stream cannot move between threads, so each playback owns a thread
//! that opens the stream, waits until the sound has been played or [`Playback::stop`] is called,
//! and closes it again.

use crate::devices::{self, Direction};
use anyhow::{Context, Result, anyhow, bail};
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// Extra time after the computed end of the sound before the device is closed.
const MARGIN: Duration = Duration::from_millis(100);

/// The longest wait for the device to play out its buffer, whatever it reports.
const MAX_LEAD: Duration = Duration::from_secs(10);

/// How much longer the device needs to play out the sound, after the last real sample was handed
/// to it. `lead` is how far ahead of the speaker the start of that buffer is, and `remaining`
/// how many frames of real sound the buffer holds. Sound servers often take a buffer of
/// seconds at once, so "the last sample was handed over" is not "the sentence has been heard".
// Frame counts are far below 2^52, so the conversion to seconds is exact in practice.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn time_to_finish(lead: Duration, remaining: usize, rate: u32) -> Duration {
    let playing = Duration::from_secs_f64(remaining as f64 / f64::from(rate.max(1)));
    (lead + playing).min(MAX_LEAD)
}

/// Sound being played. Dropping it (or calling [`Playback::stop`]) cuts the sound off.
pub struct Playback {
    stop: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
    error: Arc<Mutex<Option<String>>>,
    notice: Option<String>,
    thread: Option<JoinHandle<()>>,
}

impl Playback {
    /// Start playing `wav` (any WAV file, any rate, mono or stereo) on the device `device`
    /// (an id or a name; `None` is the system default) and return at once.
    ///
    /// # Errors
    ///
    /// Returns an error if `wav` cannot be read, there is no output device, or the device cannot
    /// be opened. The message says what to do about it.
    pub fn start(device: Option<&str>, wav: &[u8]) -> Result<Self> {
        let (samples, rate) = decode_wav(wav)?;
        let resolved = devices::resolve(Direction::Output, device)?.ok_or_else(|| {
            anyhow!("No speaker found. Connect one and check that it is not muted.")
        })?;
        let notice = resolved.notice;
        let stop = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        let error = Arc::new(Mutex::new(None));
        let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();

        let thread = {
            let (stop, done, error) = (Arc::clone(&stop), Arc::clone(&done), Arc::clone(&error));
            let device = resolved.device;
            std::thread::spawn(move || {
                run(&device, &samples, rate, &stop, &done, &error, &ready_tx);
                done.store(true, Ordering::Relaxed);
            })
        };
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                stop,
                done,
                error,
                notice,
                thread: Some(thread),
            }),
            Ok(Err(e)) => {
                let _ = thread.join();
                Err(e)
            }
            Err(_) => Err(anyhow!("the audio thread exited unexpectedly")),
        }
    }

    /// Whether the sound has finished, or was cut off by an error.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.done.load(Ordering::Relaxed)
    }

    /// A message if the chosen device was not found and the default is being used.
    pub fn take_notice(&mut self) -> Option<String> {
        self.notice.take()
    }

    /// The error that cut the sound off (the speaker was unplugged, for example), once.
    pub fn take_error(&mut self) -> Option<String> {
        self.error.lock().ok().and_then(|mut e| e.take())
    }

    /// Cut the sound off and release the device.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Playback {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Mono samples in -1.0..=1.0 and the sample rate of a WAV file.
// Sample values (at most 32 bits) and frame lengths are far below 2^24, so the conversions to
// `f32` are exact in practice.
///
/// # Errors
///
/// Returns an error if `wav` is not a readable WAV file or has no audio in it.
#[allow(clippy::cast_precision_loss)]
pub fn decode_wav(wav: &[u8]) -> Result<(Vec<f32>, u32)> {
    let mut reader =
        hound::WavReader::new(Cursor::new(wav)).context("the reply is not a readable WAV file")?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    if interleaved.is_empty() {
        bail!("the reply contains no audio");
    }
    let mono = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect();
    Ok((mono, spec.sample_rate))
}

/// Write the next samples of `source`, starting at `*position`, into `out`, giving every channel
/// of a frame the same value. Past the end it writes silence. Returns whether the end was reached.
pub(crate) fn fill<T>(out: &mut [T], channels: usize, source: &[f32], position: &mut usize) -> bool
where
    T: SizedSample + FromSample<f32>,
{
    for frame in out.chunks_mut(channels.max(1)) {
        let value = source.get(*position).copied().unwrap_or(0.0);
        *position += 1;
        frame.fill(T::from_sample(value));
    }
    *position >= source.len()
}

/// What went wrong with an output device, in words a person can act on.
fn output_problem(error: &cpal::Error) -> anyhow::Error {
    use cpal::ErrorKind;
    match error.kind() {
        ErrorKind::DeviceNotAvailable => {
            anyhow!("Speaker disconnected or unavailable. Reconnect it or choose another (d).")
        }
        ErrorKind::DeviceBusy => anyhow!("The speaker is busy: another program is using it."),
        ErrorKind::HostUnavailable => {
            anyhow!("No sound server found. Start PipeWire or PulseAudio.")
        }
        ErrorKind::PermissionDenied => anyhow!("Speaker access denied. Check your sound settings."),
        _ => anyhow!("Speaker error: {error}"),
    }
}

/// Whether an error from the running stream ends the sound (see `audio::ends_recording`).
fn ends_playback(kind: cpal::ErrorKind) -> bool {
    use cpal::ErrorKind;
    !matches!(
        kind,
        ErrorKind::Xrun | ErrorKind::DeviceChanged | ErrorKind::RealtimeDenied
    )
}

/// What the playback thread, the audio callback and [`Playback`] tell each other.
#[derive(Clone)]
struct Signals {
    /// The last sample was handed to the device (it may still be playing it).
    drained: Arc<AtomicBool>,
    /// How much longer, in milliseconds, the device needs to play out what it was given.
    tail_ms: Arc<AtomicU64>,
    /// The sound is over, or was cut off by an error.
    done: Arc<AtomicBool>,
    /// The error that cut the sound off.
    error: Arc<Mutex<Option<String>>>,
}

fn run(
    device: &cpal::Device,
    samples: &[f32],
    rate: u32,
    stop: &Arc<AtomicBool>,
    done: &Arc<AtomicBool>,
    error: &Arc<Mutex<Option<String>>>,
    ready: &mpsc::Sender<Result<()>>,
) {
    let signals = Signals {
        drained: Arc::new(AtomicBool::new(false)),
        tail_ms: Arc::new(AtomicU64::new(0)),
        done: Arc::clone(done),
        error: Arc::clone(error),
    };
    let stream = match open(device, samples, rate, &signals) {
        Ok(stream) => {
            let _ = ready.send(Ok(()));
            stream
        }
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    while !stop.load(Ordering::Relaxed) && !done.load(Ordering::Relaxed) {
        if signals.drained.load(Ordering::Relaxed) {
            // The last samples are with the device; let them play before closing it.
            let wait = Duration::from_millis(signals.tail_ms.load(Ordering::Relaxed)) + MARGIN;
            let mut waited = Duration::ZERO;
            while waited < wait && !stop.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(10));
                waited += Duration::from_millis(10);
            }
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    drop(stream);
}

fn open(
    device: &cpal::Device,
    samples: &[f32],
    rate: u32,
    signals: &Signals,
) -> Result<cpal::Stream> {
    let config = device
        .default_output_config()
        .map_err(|e| output_problem(&e))?;
    let format = config.sample_format();
    let channels = usize::from(config.channels());
    let device_rate = config.sample_rate();
    let config: cpal::StreamConfig = config.into();
    let source = crate::audio::resample(samples, rate, device_rate);

    let stream = match format {
        SampleFormat::I16 => build::<i16>(device, &config, channels, source, signals),
        SampleFormat::I32 => build::<i32>(device, &config, channels, source, signals),
        SampleFormat::F32 => build::<f32>(device, &config, channels, source, signals),
        other => bail!("unsupported speaker sample format: {other}"),
    }?;
    stream.play().map_err(|e| output_problem(&e))?;
    Ok(stream)
}

fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    source: Vec<f32>,
    signals: &Signals,
) -> Result<cpal::Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let (data, failure) = (signals.clone(), signals.clone());
    let rate = config.sample_rate;
    let mut position = 0usize;
    device
        .build_output_stream(
            *config,
            move |out: &mut [T], info: &cpal::OutputCallbackInfo| {
                let before = position;
                if fill(out, channels, &source, &mut position)
                    && !data.drained.load(Ordering::Relaxed)
                {
                    let stamp = info.timestamp();
                    let lead = stamp.playback.saturating_duration_since(stamp.callback);
                    let remaining = source.len().saturating_sub(before);
                    let wait = time_to_finish(lead, remaining, rate);
                    data.tail_ms.store(
                        u64::try_from(wait.as_millis()).unwrap_or(u64::MAX),
                        Ordering::Relaxed,
                    );
                    data.drained.store(true, Ordering::Relaxed);
                }
            },
            move |err| {
                if ends_playback(err.kind()) {
                    if let Ok(mut slot) = failure.error.lock() {
                        slot.get_or_insert_with(|| output_problem(&err).to_string());
                    }
                    failure.done.store(true, Ordering::Relaxed);
                }
            },
            None,
        )
        .map_err(|e| output_problem(&e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(samples: &[i16], channels: u16, rate: u32) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cursor = Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
        for &s in samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
        cursor.into_inner()
    }

    #[test]
    fn decodes_mono_16_bit() {
        let (samples, rate) = decode_wav(&wav(&[0, 16384, -16384, 32767], 1, 24_000)).unwrap();
        assert_eq!(rate, 24_000);
        assert_eq!(samples.len(), 4);
        assert!((samples[1] - 0.5).abs() < 1e-4);
        assert!((samples[2] + 0.5).abs() < 1e-4);
    }

    #[test]
    fn stereo_is_averaged_to_mono_not_summed() {
        let (samples, _) = decode_wav(&wav(&[16384, 16384, 16384, -16384], 2, 48_000)).unwrap();
        assert_eq!(samples.len(), 2);
        assert!((samples[0] - 0.5).abs() < 1e-4);
        assert!(samples[1].abs() < 1e-4);
    }

    #[test]
    fn float_wav_is_read_as_it_is() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut cursor = Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
        writer.write_sample(0.25f32).unwrap();
        writer.finalize().unwrap();
        let (samples, _) = decode_wav(&cursor.into_inner()).unwrap();
        assert!((samples[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn junk_and_empty_files_are_errors_not_panics() {
        assert!(decode_wav(b"definitely not a wav file").is_err());
        assert!(decode_wav(&wav(&[], 1, 16_000)).is_err());
        assert!(decode_wav(&[]).is_err());
    }

    #[test]
    fn the_end_of_the_sound_is_the_lead_plus_what_is_left_to_play() {
        let t = |lead_ms, remaining, rate| {
            time_to_finish(Duration::from_millis(lead_ms), remaining, rate)
        };
        // 1 s of sound left in a buffer that starts playing in 20 ms.
        assert_eq!(t(20, 48_000, 48_000), Duration::from_millis(1020));
        // A server that took the whole sentence at once: 2 s ahead, 0.5 s of it real.
        assert_eq!(t(2000, 24_000, 48_000), Duration::from_millis(2500));
        assert_eq!(t(0, 0, 48_000), Duration::ZERO);
    }

    #[test]
    fn the_wait_is_capped_and_a_zero_rate_does_not_divide_by_zero() {
        assert_eq!(time_to_finish(Duration::from_secs(60), 1, 48_000), MAX_LEAD);
        assert!(time_to_finish(Duration::ZERO, 10, 0) <= MAX_LEAD);
    }

    #[test]
    fn fill_copies_each_sample_to_every_channel() {
        let mut out = [0.0f32; 6];
        let mut position = 0;
        let reached_end = fill(&mut out, 2, &[0.1, 0.2, 0.3, 0.4], &mut position);
        assert_eq!(out, [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]);
        assert_eq!(position, 3);
        assert!(!reached_end);
    }

    #[test]
    fn fill_pads_with_silence_and_reports_the_end() {
        let mut out = [1.0f32; 4];
        let mut position = 0;
        assert!(fill(&mut out, 1, &[0.5, 0.25], &mut position));
        assert_eq!(out, [0.5, 0.25, 0.0, 0.0]);
        let mut again = [1.0f32; 2];
        assert!(fill(&mut again, 1, &[0.5, 0.25], &mut position));
        assert_eq!(
            again,
            [0.0, 0.0],
            "nothing is replayed once the end was reached"
        );
    }

    #[test]
    fn fill_converts_to_the_devices_sample_type() {
        let mut out = [0i16; 2];
        let mut position = 0;
        fill(&mut out, 1, &[1.0, -1.0], &mut position);
        assert!(out[0] > 32_000, "{out:?}");
        assert!(out[1] < -32_000, "{out:?}");
        let mut wide = [0i32; 1];
        let mut at = 0;
        fill(&mut wide, 1, &[0.5], &mut at);
        assert!(wide[0] > 1_000_000_000, "{wide:?}");
    }

    #[test]
    fn fill_survives_a_zero_channel_count() {
        let mut out = [0.0f32; 2];
        let mut position = 0;
        fill(&mut out, 0, &[0.7], &mut position);
        assert!((out[0] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn output_problems_say_what_to_do() {
        use cpal::ErrorKind as K;
        let message = |kind| output_problem(&cpal::Error::new(kind)).to_string();
        assert!(message(K::DeviceNotAvailable).contains("Reconnect"));
        assert!(message(K::DeviceBusy).contains("busy"));
        assert!(message(K::HostUnavailable).contains("sound server"));
        assert!(message(K::PermissionDenied).contains("denied"));
        assert!(message(K::BackendError).starts_with("Speaker error:"));
        for kind in [
            K::DeviceNotAvailable,
            K::DeviceBusy,
            K::HostUnavailable,
            K::PermissionDenied,
        ] {
            assert!(
                message(kind).chars().count() <= 75,
                "{kind:?} must fit the warnings line"
            );
        }
    }

    #[test]
    fn glitches_do_not_end_playback() {
        use cpal::ErrorKind as K;
        for kind in [K::Xrun, K::DeviceChanged, K::RealtimeDenied] {
            assert!(!ends_playback(kind), "{kind:?}");
        }
        for kind in [K::DeviceNotAvailable, K::StreamInvalidated, K::BackendError] {
            assert!(ends_playback(kind), "{kind:?}");
        }
    }
}
