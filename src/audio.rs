use anyhow::{Context, Result, anyhow, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::Duration;

const TARGET_RATE: u32 = 16_000;

/// Audio discarded right after the stream opens.
const STARTUP_SKIP_MS: usize = 300;
/// Ambient noise is measured for this long before segmenting starts.
const CALIBRATION_MS: usize = 1000;
/// Speech must exceed the calibrated noise level (95th percentile) by this factor.
const NOISE_MARGIN: f32 = 1.5;
const FRAME_MS: usize = 20;
/// Silence that ends an utterance.
const SILENCE_MS: usize = 600;
/// Utterances with less speech than this are treated as noise and dropped.
const MIN_SPEECH_MS: usize = 250;
/// Audio kept from before speech starts so the first word isn't clipped.
const PREROLL_MS: usize = 200;
/// Force a cut if someone talks without pausing for this long.
const MAX_SEGMENT_MS: usize = 20_000;

/// Frame and buffer lengths are far below 2^24, so this conversion is exact in practice.
#[allow(clippy::cast_precision_loss)]
fn count(n: usize) -> f32 {
    n as f32
}

/// RMS of a frame with its DC offset removed.
fn ac_rms(frame: &[f32]) -> f32 {
    let n = count(frame.len());
    let mean = frame.iter().sum::<f32>() / n;
    (frame.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / n).sqrt()
}

/// Gate level derived from ambient audio: margin over the 95th-percentile frame level.
fn noise_threshold(ambient: &[f32], sample_rate: u32, min_threshold: f32) -> f32 {
    let frame_len = (sample_rate as usize * FRAME_MS / 1000).max(1);
    let mut levels: Vec<f32> = ambient.chunks_exact(frame_len).map(ac_rms).collect();
    if levels.is_empty() {
        return min_threshold;
    }
    levels.sort_by(f32::total_cmp);
    let p95 = levels[(levels.len() * 95 / 100).min(levels.len() - 1)];
    (p95 * NOISE_MARGIN).max(min_threshold)
}

/// A segment's loudest frame must exceed the gate by this factor, or it is treated as noise.
const PEAK_MARGIN: f32 = 1.5;

/// Splits a continuous mono stream into utterances using an RMS energy gate.
struct Segmenter {
    threshold: f32,
    frame_len: usize,
    silence_limit: usize,
    min_speech: usize,
    preroll: usize,
    max_len: usize,
    partial: Vec<f32>,
    buf: Vec<f32>,
    speech: usize,
    max_rms: f32,
    silence_run: usize,
    in_speech: bool,
}

impl Segmenter {
    fn new(sample_rate: u32, threshold: f32) -> Self {
        let ms = |m: usize| sample_rate as usize * m / 1000;
        Self {
            threshold,
            frame_len: ms(FRAME_MS).max(1),
            silence_limit: ms(SILENCE_MS),
            min_speech: ms(MIN_SPEECH_MS),
            preroll: ms(PREROLL_MS),
            max_len: ms(MAX_SEGMENT_MS),
            partial: Vec::new(),
            buf: Vec::new(),
            speech: 0,
            max_rms: 0.0,
            silence_run: 0,
            in_speech: false,
        }
    }

    fn reset(&mut self) {
        self.buf.clear();
        self.speech = 0;
        self.max_rms = 0.0;
        self.silence_run = 0;
        self.in_speech = false;
    }

    fn take_segment(&mut self) -> Option<Vec<f32>> {
        let keep = self.speech >= self.min_speech && self.max_rms >= self.threshold * PEAK_MARGIN;
        let seg = std::mem::take(&mut self.buf);
        self.reset();
        keep.then_some(seg)
    }

    fn push(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        self.partial.extend_from_slice(samples);
        let mut out = Vec::new();
        let frames = self.partial.len() / self.frame_len;
        let data = std::mem::take(&mut self.partial);
        for frame in data.chunks_exact(self.frame_len).take(frames) {
            let rms = ac_rms(frame);
            self.buf.extend_from_slice(frame);
            self.max_rms = self.max_rms.max(rms);
            if rms > self.threshold {
                self.speech += frame.len();
                self.silence_run = 0;
                self.in_speech = true;
            } else {
                self.silence_run += frame.len();
                if !self.in_speech && self.buf.len() > self.preroll {
                    let excess = self.buf.len() - self.preroll;
                    self.buf.drain(..excess);
                }
            }
            let ended = self.in_speech && self.silence_run >= self.silence_limit;
            if ended || (self.in_speech && self.buf.len() >= self.max_len) {
                out.extend(self.take_segment());
            }
        }
        self.partial = data[frames * self.frame_len..].to_vec();
        out
    }

    /// Flush whatever is buffered at the end of a recording.
    fn finish(&mut self) -> Option<Vec<f32>> {
        let rest = std::mem::take(&mut self.partial);
        self.buf.extend(rest);
        self.take_segment()
    }
}

/// Records from the default microphone and splits the audio into utterances on silence.
///
/// After [`start_recording`](Self::start_recording) the recorder measures about a second of
/// ambient noise to set its speech gate, then [`poll_segments`](Self::poll_segments) returns
/// each finished utterance as 16 kHz mono WAV bytes while recording continues.
///
/// # Examples
///
/// ```no_run
/// use voice_notes::audio::AudioRecorder;
///
/// # fn main() -> anyhow::Result<()> {
/// let mut recorder = AudioRecorder::new(0.01, false);
/// recorder.start_recording()?;
///
/// // Call this regularly (the app polls every 100 ms).
/// for wav in recorder.poll_segments()? {
///     println!("utterance of {} bytes", wav.len());
/// }
///
/// // Stopping returns any speech still in progress.
/// if let Some(wav) = recorder.stop_recording()? {
///     println!("trailing utterance of {} bytes", wav.len());
/// }
/// # Ok(())
/// # }
/// ```
pub struct AudioRecorder {
    min_threshold: f32,
    threshold: Option<f32>,
    ambient: Vec<f32>,
    verbose: bool,
    level: f32,
    skip_remaining: usize,
    stop_flag: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<f32>>>,
    worker: Option<JoinHandle<()>>,
    segmenter: Option<Segmenter>,
    sample_rate: u32,
}

impl AudioRecorder {
    /// `min_threshold` is the lowest RMS level (0.0-1.0) ever treated as speech; the real
    /// gate is calibrated from ambient noise after recording starts. With `verbose`, the live
    /// input level is printed to stderr while polling.
    ///
    /// # Examples
    ///
    /// Creating a recorder does not touch the microphone, so no gate exists yet:
    ///
    /// ```
    /// use voice_notes::audio::AudioRecorder;
    ///
    /// let recorder = AudioRecorder::new(0.01, false);
    /// assert_eq!(recorder.calibrated_threshold(), None);
    /// ```
    #[must_use]
    pub fn new(min_threshold: f32, verbose: bool) -> Self {
        Self {
            min_threshold,
            threshold: None,
            ambient: Vec::new(),
            verbose,
            level: 0.0,
            skip_remaining: 0,
            stop_flag: Arc::new(AtomicBool::new(false)),
            samples: Arc::new(Mutex::new(Vec::new())),
            worker: None,
            segmenter: None,
            sample_rate: TARGET_RATE,
        }
    }

    /// Open the default input device and start capturing in a background thread.
    ///
    /// # Errors
    ///
    /// Returns an error immediately if already recording, or if the input device cannot be
    /// found, configured or started.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voice_notes::audio::AudioRecorder;
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let mut recorder = AudioRecorder::new(0.01, false);
    /// recorder.start_recording()?;
    /// // Starting twice is rejected.
    /// assert!(recorder.start_recording().is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn start_recording(&mut self) -> Result<()> {
        if self.worker.is_some() {
            bail!("already recording");
        }
        self.stop_flag.store(false, Ordering::Relaxed);
        self.samples
            .lock()
            .map_err(|_| anyhow!("poisoned lock"))?
            .clear();

        let stop = Arc::clone(&self.stop_flag);
        let samples = Arc::clone(&self.samples);
        let (ready_tx, ready_rx) = mpsc::channel::<Result<u32>>();

        // cpal streams are not `Send`, so the stream lives entirely on this thread.
        let handle = std::thread::spawn(move || capture(&stop, &ready_tx, &samples));

        match ready_rx.recv() {
            Ok(Ok(rate)) => {
                self.sample_rate = rate;
                // The device emits a loud start-up burst; ignore it.
                self.skip_remaining = rate as usize * STARTUP_SKIP_MS / 1000;
                self.segmenter = None;
                self.threshold = None;
                self.ambient.clear();
                self.worker = Some(handle);
                Ok(())
            }
            Ok(Err(e)) => {
                let _ = handle.join();
                Err(e)
            }
            Err(_) => Err(anyhow!("audio thread exited unexpectedly")),
        }
    }

    fn drain(&mut self) -> Result<Vec<f32>> {
        let mut guard = self.samples.lock().map_err(|_| anyhow!("poisoned lock"))?;
        let mut out = std::mem::take(&mut *guard);
        drop(guard);
        let cut = self.skip_remaining.min(out.len());
        out.drain(..cut);
        self.skip_remaining -= cut;
        Ok(out)
    }

    /// RMS input level (0.0-1.0) of the audio received by the most recent poll.
    ///
    /// # Examples
    ///
    /// ```
    /// use voice_notes::audio::AudioRecorder;
    ///
    /// let recorder = AudioRecorder::new(0.01, false);
    /// assert!(recorder.level().abs() < f32::EPSILON); // nothing recorded yet
    /// ```
    #[must_use]
    pub fn level(&self) -> f32 {
        self.level
    }

    /// Whether speech is currently in progress (the gate has been crossed and no pause ended it).
    ///
    /// # Examples
    ///
    /// ```
    /// use voice_notes::audio::AudioRecorder;
    ///
    /// assert!(!AudioRecorder::new(0.01, false).is_speaking());
    /// ```
    #[must_use]
    pub fn is_speaking(&self) -> bool {
        self.segmenter.as_ref().is_some_and(|s| s.in_speech)
    }

    /// The calibrated speech gate, or `None` while still measuring ambient noise.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voice_notes::audio::AudioRecorder;
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let mut recorder = AudioRecorder::new(0.01, false);
    /// recorder.start_recording()?;
    /// while recorder.calibrated_threshold().is_none() {
    ///     recorder.poll_segments()?; // calibration advances as audio is polled
    ///     std::thread::sleep(std::time::Duration::from_millis(100));
    /// }
    /// println!("speech gate: {:.3}", recorder.calibrated_threshold().unwrap());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn calibrated_threshold(&self) -> Option<f32> {
        self.threshold
    }

    /// Feed new audio to the calibrator, then to the segmenter once calibrated.
    fn process(&mut self, new: Vec<f32>) -> Vec<Vec<f32>> {
        if self.segmenter.is_none() {
            self.ambient.extend(new);
            let needed = self.sample_rate as usize * CALIBRATION_MS / 1000;
            if self.ambient.len() < needed {
                return Vec::new();
            }
            let th = noise_threshold(&self.ambient, self.sample_rate, self.min_threshold);
            self.threshold = Some(th);
            self.segmenter = Some(Segmenter::new(self.sample_rate, th));
            // Audio captured during calibration is ambient, not speech.
            self.ambient.clear();
            return Vec::new();
        }
        self.segmenter
            .as_mut()
            .map_or_else(Vec::new, |s| s.push(&new))
    }

    /// Return WAV bytes (16 kHz mono 16-bit) for each utterance finished since the last call.
    ///
    /// # Errors
    ///
    /// Returns an error if not recording, or if the capture buffer or WAV encoding fails.
    ///
    /// # Examples
    ///
    /// Polling a recorder that was never started is an error:
    ///
    /// ```
    /// use voice_notes::audio::AudioRecorder;
    ///
    /// let mut recorder = AudioRecorder::new(0.01, false);
    /// assert!(recorder.poll_segments().is_err());
    /// ```
    pub fn poll_segments(&mut self) -> Result<Vec<Vec<u8>>> {
        if self.worker.is_none() {
            bail!("not recording");
        }
        let new = self.drain()?;
        if !new.is_empty() {
            self.level = ac_rms(&new);
        }
        if self.verbose && !new.is_empty() {
            let level = self.level;
            eprint!(
                "\r[ac-level {level:.4} / gate {}]   ",
                self.threshold
                    .map_or_else(|| "calibrating".to_string(), |t| format!("{t:.4}"))
            );
        }
        let rate = self.sample_rate;
        self.process(new)
            .iter()
            .map(|s| encode_wav(&resample(s, rate, TARGET_RATE)))
            .collect()
    }

    /// Stop capturing and return the trailing utterance, if it contained speech.
    ///
    /// # Errors
    ///
    /// Returns an error if not recording, the audio thread panicked, or WAV encoding fails.
    ///
    /// # Examples
    ///
    /// Stopping a recorder that was never started is an error:
    ///
    /// ```
    /// use voice_notes::audio::AudioRecorder;
    ///
    /// let mut recorder = AudioRecorder::new(0.01, false);
    /// assert!(recorder.stop_recording().is_err());
    /// ```
    pub fn stop_recording(&mut self) -> Result<Option<Vec<u8>>> {
        self.stop_flag.store(true, Ordering::Relaxed);
        let worker = self.worker.take().context("not recording")?;
        worker
            .join()
            .map_err(|_| anyhow!("audio thread panicked"))?;

        let rest = self.drain()?;
        let rate = self.sample_rate;
        let mut last = self.process(rest);
        let Some(mut seg) = self.segmenter.take() else {
            return Ok(None); // stopped before calibration finished
        };
        last.extend(seg.finish());
        // Anything beyond the first leftover is merged: stopping yields one trailing clip.
        let merged: Vec<f32> = last.into_iter().flatten().collect();
        if merged.is_empty() {
            return Ok(None);
        }
        encode_wav(&resample(&merged, rate, TARGET_RATE)).map(Some)
    }
}

fn capture(stop: &AtomicBool, ready: &mpsc::Sender<Result<u32>>, samples: &Arc<Mutex<Vec<f32>>>) {
    let (stream, sample_rate) = match open_stream(samples) {
        Ok(v) => v,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let _ = ready.send(Ok(sample_rate));

    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(20));
    }
    drop(stream);
}

/// On Linux prefer the `PulseAudio` protocol (served by `PipeWire` and `PulseAudio`); the raw ALSA
/// default often can't be opened when a sound server owns the hardware.
#[cfg(target_os = "linux")]
fn input_host() -> cpal::Host {
    cpal::host_from_id(cpal::HostId::PulseAudio).unwrap_or_else(|_| cpal::default_host())
}

/// Elsewhere (macOS) the default host (`CoreAudio`) is the right one.
#[cfg(not(target_os = "linux"))]
fn input_host() -> cpal::Host {
    cpal::default_host()
}

fn open_stream(samples: &Arc<Mutex<Vec<f32>>>) -> Result<(cpal::Stream, u32)> {
    let host = input_host();
    let device = host
        .default_input_device()
        .context("no default input device found (is a microphone available?)")?;
    let config = device
        .default_input_config()
        .context("could not get the microphone's input configuration")?;

    let sample_rate = config.sample_rate();
    let channels = config.channels() as usize;
    let format = config.sample_format();
    let config: cpal::StreamConfig = config.into();

    let stream = match format {
        SampleFormat::I16 => build::<i16>(&device, config, channels, samples)?,
        SampleFormat::I32 => build::<i32>(&device, config, channels, samples)?,
        SampleFormat::F32 => build::<f32>(&device, config, channels, samples)?,
        other => bail!("unsupported microphone sample format: {other}"),
    };
    stream
        .play()
        .context("could not start the microphone stream")?;
    Ok((stream, sample_rate))
}

fn build<T>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    channels: usize,
    samples: &Arc<Mutex<Vec<f32>>>,
) -> Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let samples = Arc::clone(samples);
    device
        .build_input_stream(
            config,
            move |data: &[T], _: &_| {
                if let Ok(mut buf) = samples.lock() {
                    // Downmix interleaved frames to mono.
                    for frame in data.chunks(channels) {
                        let sum: f32 = frame.iter().map(|&s| f32::from_sample(s)).sum();
                        buf.push(sum / count(frame.len()));
                    }
                }
            },
            |err| eprintln!("Microphone stream error: {err}"),
            None,
        )
        .context("could not open the microphone stream")
}

/// Turn any WAV file (mono or stereo, any sample rate, integer or float samples) into what the
/// microphone path produces: 16 kHz, mono, 16-bit. Lets a recording stand in for the microphone.
///
/// # Errors
///
/// Returns an error if `data` is not a readable WAV file, or contains no audio.
///
/// # Examples
///
/// ```
/// use voice_notes::audio::normalize_wav;
///
/// assert!(normalize_wav(b"definitely not a wav file").is_err());
/// ```
pub fn normalize_wav(data: &[u8]) -> Result<Vec<u8>> {
    // Files streamed from a speech service declare an unknown length; repair those.
    let repaired: Vec<u8>;
    let mut reader = match hound::WavReader::new(Cursor::new(data)) {
        Ok(reader) => reader,
        Err(first) => {
            repaired = repair_wav(data)
                .ok_or(first)
                .context("not a readable WAV file")?;
            hound::WavReader::new(Cursor::new(repaired.as_slice()))
                .context("not a readable WAV file")?
        }
    };
    let spec = reader.spec();
    let channels = usize::from(spec.channels).max(1);
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, _) => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .context("could not read the WAV samples")?,
        (hound::SampleFormat::Int, bits) if (1..=32).contains(&bits) => {
            let scale = 2f64.powi(i32::from(bits) - 1);
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| int_to_unit(v, scale)))
                .collect::<Result<_, _>>()
                .context("could not read the WAV samples")?
        }
        (_, bits) => bail!("unsupported WAV sample size: {bits} bits"),
    };
    let mono: Vec<f32> = samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / count(frame.len()))
        .collect();
    if mono.is_empty() {
        bail!("the WAV file contains no audio");
    }
    encode_wav(&resample(&mono, spec.sample_rate, TARGET_RATE))
}

/// Rebuild a WAV file whose `data` chunk declares a length that does not match the file, as
/// streamed files do (`0xFFFFFFFF` means "unknown, read to the end"). Everything after the header is
/// taken as audio, cut to a whole number of samples; other chunks are dropped. `None` if the
/// bytes are not a RIFF/WAVE file with a format chunk and a data chunk.
fn repair_wav(data: &[u8]) -> Option<Vec<u8>> {
    if data.get(..4)? != b"RIFF" || data.get(8..12)? != b"WAVE" {
        return None;
    }
    let mut pos = 12;
    let mut format: Option<[u8; 16]> = None;
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let size =
            usize::try_from(u32::from_le_bytes(data[pos + 4..pos + 8].try_into().ok()?)).ok()?;
        let body = pos + 8;
        if id == b"fmt " {
            format = Some(data.get(body..body + 16)?.try_into().ok()?);
        } else if id == b"data" {
            let format = format?;
            let block = usize::from(u16::from_le_bytes([format[12], format[13]]));
            if block == 0 {
                return None;
            }
            let available = data.len() - body;
            let length = size.min(available);
            let length = length - length % block;
            let length_field = u32::try_from(length).ok()?;
            let mut fixed = Vec::with_capacity(44 + length);
            fixed.extend_from_slice(b"RIFF");
            fixed.extend_from_slice(&(36 + length_field).to_le_bytes());
            fixed.extend_from_slice(b"WAVEfmt ");
            fixed.extend_from_slice(&16u32.to_le_bytes());
            fixed.extend_from_slice(&format);
            fixed.extend_from_slice(b"data");
            fixed.extend_from_slice(&length_field.to_le_bytes());
            fixed.extend_from_slice(&data[body..body + length]);
            return Some(fixed);
        }
        pos = body.checked_add(size)?.checked_add(size & 1)?;
    }
    None
}

/// An integer sample as a number between -1 and 1.
#[allow(clippy::cast_possible_truncation)]
fn int_to_unit(value: i32, scale: f64) -> f32 {
    (f64::from(value) / scale) as f32
}

/// Linear-interpolation resampler; adequate for speech.
///
/// Indices stay well inside f64's exact integer range and are never negative, so the
/// float/integer casts below cannot lose information.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let ratio = f64::from(from) / f64::from(to);
    let out_len = (input.len() as f64 / ratio) as usize;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos as usize;
            let frac = (pos - idx as f64) as f32;
            let a = input[idx];
            let b = input.get(idx + 1).copied().unwrap_or(a);
            a + (b - a) * frac
        })
        .collect()
}

fn encode_wav(samples: &[f32]) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut wav_buffer = Vec::new();
    {
        let mut writer = hound::WavWriter::new(Cursor::new(&mut wav_buffer), spec)?;
        for &s in samples {
            writer.write_sample(i16::from_sample(s))?;
        }
        writer.finalize()?;
    }
    Ok(wav_buffer)
}

/// Unit tests for the silence-based [`Segmenter`].
#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;

    /// Generate a sine tone of `ms` milliseconds at amplitude `amp`, standing in for speech.
    fn tone(rate: u32, ms: usize, amp: f32) -> Vec<f32> {
        (0..rate as usize * ms / 1000)
            .map(|i| amp * (i as f32 * 0.3).sin())
            .collect()
    }

    /// Two bursts of speech separated by a pause longer than the silence limit must come out
    /// as two separate segments, with nothing left buffered afterwards.
    #[test]
    fn splits_on_silence() {
        let mut seg = Segmenter::new(16_000, 0.01);
        let mut audio = tone(16_000, 500, 0.3);
        audio.extend(vec![0.0; 16_000]);
        audio.extend(tone(16_000, 500, 0.3));
        audio.extend(vec![0.0; 16_000]);
        let out = seg.push(&audio);
        assert_eq!(out.len(), 2);
        assert!(seg.finish().is_none());
    }

    /// Pure silence and a click shorter than the minimum speech length are treated as noise,
    /// so no segment is emitted and nothing is flushed at the end.
    #[test]
    fn drops_noise_blips_and_silence() {
        let mut seg = Segmenter::new(16_000, 0.01);
        let mut audio = vec![0.0; 16_000];
        audio.extend(tone(16_000, 60, 0.3));
        audio.extend(vec![0.0; 16_000]);
        assert_eq!(seg.push(&audio), [] as [std::vec::Vec<f32>; 0]);
        assert!(seg.finish().is_none());
    }

    /// Speech that is still in progress when recording stops has no trailing pause, so it is
    /// not emitted by `push` but must be returned by `finish`.
    #[test]
    fn finish_flushes_trailing_speech() {
        let mut seg = Segmenter::new(16_000, 0.01);
        assert_eq!(
            seg.push(&tone(16_000, 800, 0.3)),
            [] as [std::vec::Vec<f32>; 0]
        );
        assert!(seg.finish().is_some());
    }
}

/// Unit tests for the ambient-noise calibration.
#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod calibration_tests {
    use super::*;

    /// A very quiet room keeps the configured minimum gate, while a noisy room raises the gate
    /// to just above the measured noise (1.5x the 95th-percentile level, about 0.1 to 0.2 here).
    #[test]
    fn gate_scales_with_ambient_noise() {
        let quiet = vec![0.0005_f32; 16_000];
        assert_eq!(noise_threshold(&quiet, 16_000, 0.01), 0.01);

        let noisy: Vec<f32> = (0..16_000).map(|i| 0.1 * (i as f32 * 0.7).sin()).collect();
        let gate = noise_threshold(&noisy, 16_000, 0.01);
        assert!(gate > 0.1 && gate < 0.2, "gate was {gate}");
    }
}

/// Tests for turning any WAV file into the microphone format.
#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
mod wav_tests {
    use super::*;

    /// Write a WAV file in memory: `seconds` of a 440 Hz tone at the given format.
    fn tone_wav(channels: u16, rate: u32, bits: u16, float: bool, seconds: f32) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: bits,
            sample_format: if float {
                hound::SampleFormat::Float
            } else {
                hound::SampleFormat::Int
            },
        };
        let mut bytes = Vec::new();
        {
            let mut writer = hound::WavWriter::new(Cursor::new(&mut bytes), spec).unwrap();
            let frames = (rate as f32 * seconds) as u32;
            for i in 0..frames {
                let v = (i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 0.5;
                for _ in 0..channels {
                    if float {
                        writer.write_sample(v).unwrap();
                    } else {
                        let max = 2f32.powi(i32::from(bits) - 1) - 1.0;
                        writer.write_sample((v * max) as i32).unwrap();
                    }
                }
            }
            writer.finalize().unwrap();
        }
        bytes
    }

    fn read_back(wav: &[u8]) -> (hound::WavSpec, Vec<i16>) {
        let reader = hound::WavReader::new(Cursor::new(wav)).unwrap();
        let spec = reader.spec();
        (
            spec,
            reader.into_samples::<i16>().map(Result::unwrap).collect(),
        )
    }

    /// Whatever the input format, the result is 16 kHz mono 16-bit of the same duration.
    #[test]
    fn normalizes_any_wav_to_16k_mono() {
        for (channels, rate, bits, float) in [
            (1, 16_000, 16, false), // already right
            (2, 44_100, 16, false), // CD quality stereo
            (1, 48_000, 24, false), // studio mono
            (2, 8_000, 32, false),  // telephone-rate, wide samples
            (1, 22_050, 32, true),  // floating point
            (1, 16_000, 8, false),  // 8-bit
        ] {
            let out = normalize_wav(&tone_wav(channels, rate, bits, float, 1.0)).unwrap();
            let (spec, samples) = read_back(&out);
            assert_eq!(
                (spec.channels, spec.sample_rate, spec.bits_per_sample),
                (1, 16_000, 16),
                "{channels}ch {rate}Hz {bits}bit"
            );
            assert!(
                (samples.len() as i64 - 16_000).abs() <= 2,
                "{channels}ch {rate}Hz {bits}bit gave {} samples",
                samples.len()
            );
            assert!(
                samples.iter().any(|s| s.abs() > 8_000),
                "the tone survives ({channels}ch {rate}Hz {bits}bit)"
            );
        }
    }

    /// Stereo channels are averaged, so a tone in both channels keeps its level.
    #[test]
    fn stereo_is_downmixed_not_summed() {
        let out = normalize_wav(&tone_wav(2, 16_000, 16, false, 0.5)).unwrap();
        let (_, samples) = read_back(&out);
        let peak = samples.iter().map(|s| s.abs()).max().unwrap();
        assert!(
            (15_000..=17_500).contains(&peak),
            "a 0.5 amplitude tone peaks near 16384, got {peak}"
        );
    }

    /// Make a WAV the way a streaming speech server does: an extra metadata chunk, a `data` chunk
    /// of unknown length (`0xFFFFFFFF`) and an unknown overall length.
    fn streamed(standard: &[u8]) -> Vec<u8> {
        let mut out = standard[..36].to_vec(); // the header up to the `data` chunk
        out[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        out.extend_from_slice(b"LIST");
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(b"INFO");
        out.extend_from_slice(b"data");
        out.extend_from_slice(&u32::MAX.to_le_bytes());
        out.extend_from_slice(&standard[44..]);
        out
    }

    /// Kokoro's real files look like this: a length of `0xFFFFFFFF` and an extra chunk. They
    /// are read in full instead of being rejected.
    #[test]
    fn reads_streamed_wav_with_unknown_length() {
        let standard = tone_wav(1, 24_000, 16, false, 1.0);
        let file = streamed(&standard);
        assert!(
            hound::WavReader::new(Cursor::new(file.as_slice())).is_err(),
            "the plain reader refuses it"
        );
        let out = normalize_wav(&file).unwrap();
        let (spec, samples) = read_back(&out);
        assert_eq!((spec.channels, spec.sample_rate), (1, 16_000));
        assert!(
            (samples.len() as i64 - 16_000).abs() <= 2,
            "all of the audio is kept, got {}",
            samples.len()
        );
        assert!(samples.iter().any(|s| s.abs() > 8_000));
    }

    /// A streamed file cut in the middle of a sample loses only that half sample.
    #[test]
    fn streamed_wav_cut_mid_sample() {
        let mut file = streamed(&tone_wav(1, 16_000, 16, false, 0.5));
        file.push(0x7f); // a stray odd byte at the end
        let out = normalize_wav(&file).unwrap();
        assert_eq!(read_back(&out).1.len(), 8_000);
    }

    /// Repair only applies to real WAV layouts: other bytes, and WAV files missing a part, are
    /// still rejected.
    #[test]
    fn repair_does_not_accept_junk() {
        assert!(repair_wav(b"").is_none());
        assert!(repair_wav(b"RIFF....WAVE").is_none(), "no chunks");
        let standard = tone_wav(1, 16_000, 16, false, 0.1);
        assert!(repair_wav(&standard[..36]).is_none(), "no data chunk");
        let mut no_fmt = b"RIFF\0\0\0\0WAVE".to_vec();
        no_fmt.extend_from_slice(b"data");
        no_fmt.extend_from_slice(&4u32.to_le_bytes());
        no_fmt.extend_from_slice(&[1, 2, 3, 4]);
        assert!(repair_wav(&no_fmt).is_none(), "data before any format");
        assert!(normalize_wav(b"RIFF....WAVEjunkjunkjunk").is_err());
    }

    /// Garbage, an empty file and a WAV with no audio are all rejected with a reason.
    #[test]
    fn rejects_bad_input() {
        assert!(
            normalize_wav(b"not a wav")
                .unwrap_err()
                .to_string()
                .contains("WAV")
        );
        assert!(normalize_wav(b"").is_err());
        let empty = tone_wav(1, 16_000, 16, false, 0.0);
        assert!(
            normalize_wav(&empty)
                .unwrap_err()
                .to_string()
                .contains("no audio")
        );
    }
}
