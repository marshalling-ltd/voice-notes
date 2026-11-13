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

/// RMS of a frame with its DC offset removed.
fn ac_rms(frame: &[f32]) -> f32 {
    let n = frame.len() as f32;
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

pub struct AudioRecorder {
    min_threshold: f32,
    threshold: Option<f32>,
    ambient: Vec<f32>,
    verbose: bool,
    skip_remaining: usize,
    stop_flag: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<f32>>>,
    worker: Option<JoinHandle<()>>,
    segmenter: Option<Segmenter>,
    sample_rate: u32,
}

impl AudioRecorder {
    /// `min_threshold` is the lowest RMS level (0.0-1.0) ever treated as speech; the real
    /// gate is calibrated from ambient noise after recording starts.
    #[must_use]
    pub fn new(min_threshold: f32, verbose: bool) -> Self {
        Self {
            min_threshold,
            threshold: None,
            ambient: Vec::new(),
            verbose,
            skip_remaining: 0,
            stop_flag: Arc::new(AtomicBool::new(false)),
            samples: Arc::new(Mutex::new(Vec::new())),
            worker: None,
            segmenter: None,
            sample_rate: TARGET_RATE,
        }
    }

    /// Open the default input device and start capturing in a background thread.
    /// Returns an error immediately if the device cannot be opened.
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

    /// The calibrated speech gate, or `None` while still measuring ambient noise.
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
    pub fn poll_segments(&mut self) -> Result<Vec<Vec<u8>>> {
        if self.worker.is_none() {
            bail!("not recording");
        }
        let new = self.drain()?;
        if self.verbose && !new.is_empty() {
            let level = ac_rms(&new);
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

fn open_stream(samples: &Arc<Mutex<Vec<f32>>>) -> Result<(cpal::Stream, u32)> {
    // Prefer the PulseAudio protocol (served by PipeWire and PulseAudio); the raw ALSA
    // default often can't be opened when a sound server owns the hardware.
    let host =
        cpal::host_from_id(cpal::HostId::PulseAudio).unwrap_or_else(|_| cpal::default_host());
    let device = host
        .default_input_device()
        .context("no default input device found (is a microphone available?)")?;
    let config = device
        .default_input_config()
        .context("could not get the microphone's input configuration")?;

    let sample_rate = config.sample_rate() as u32;
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
                        buf.push(sum / frame.len() as f32);
                    }
                }
            },
            |err| eprintln!("Microphone stream error: {err}"),
            None,
        )
        .context("could not open the microphone stream")
}

/// Linear-interpolation resampler; adequate for speech.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, ms: usize, amp: f32) -> Vec<f32> {
        (0..rate as usize * ms / 1000)
            .map(|i| amp * (i as f32 * 0.3).sin())
            .collect()
    }

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

    #[test]
    fn drops_noise_blips_and_silence() {
        let mut seg = Segmenter::new(16_000, 0.01);
        let mut audio = vec![0.0; 16_000];
        audio.extend(tone(16_000, 60, 0.3));
        audio.extend(vec![0.0; 16_000]);
        assert_eq!(seg.push(&audio), [] as [std::vec::Vec<f32>; 0]);
        assert!(seg.finish().is_none());
    }

    #[test]
    fn finish_flushes_trailing_speech() {
        let mut seg = Segmenter::new(16_000, 0.01);
        assert_eq!(seg.push(&tone(16_000, 800, 0.3)), [] as [std::vec::Vec<f32>; 0]);
        assert!(seg.finish().is_some());
    }
}

#[cfg(test)]
mod calibration_tests {
    use super::*;

    #[test]
    fn gate_scales_with_ambient_noise() {
        let quiet = vec![0.0005_f32; 16_000];
        assert_eq!(noise_threshold(&quiet, 16_000, 0.01), 0.01);

        let noisy: Vec<f32> = (0..16_000).map(|i| 0.1 * (i as f32 * 0.7).sin()).collect();
        let gate = noise_threshold(&noisy, 16_000, 0.01);
        assert!(gate > 0.1 && gate < 0.2, "gate was {gate}");
    }
}
