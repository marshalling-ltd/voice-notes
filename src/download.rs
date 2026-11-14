//! Downloading a model: resumable, checked, and safe to interrupt.
//!
//! [`fetch`] saves one [`Model`] from the catalogue into a folder:
//!
//! - It first checks that the disk has room for what is still missing.
//! - The data goes into `NAME.part`, and the file only gets its real name once its size and
//!   sha256 are right, so a half-finished or corrupt file is never where something would load it.
//! - If `NAME.part` is already there (an earlier attempt was interrupted or cancelled), the
//!   download continues from where it stopped, using an HTTP `Range` request. A server that
//!   ignores the request just sends the whole file again, which also works.
//! - A file that does not match its sha256 is deleted, so the next attempt starts clean.
//! - Progress is reported through a callback. Nothing here draws anything.

use crate::catalogue::Model;
use crate::hardware::{self, gib};
use reqwest::header::{CONTENT_RANGE, RANGE};
use sha2::{Digest, Sha256};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// What the download is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Fetching bytes from the server.
    Downloading,
    /// Reading the finished file to check its sha256.
    Verifying,
}

/// How far along a download is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    pub phase: Phase,
    /// Bytes done in this phase.
    pub done: u64,
    /// Bytes in the whole file.
    pub total: u64,
}

/// What [`fetch`] had to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FetchKind {
    /// The file was already there and correct; nothing was downloaded.
    AlreadyPresent,
    /// Downloaded from the start.
    Fresh,
    /// An earlier attempt had `from` bytes; the rest was downloaded.
    Resumed { from: u64 },
}

/// A model that is now on disk, complete and checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fetched {
    pub path: PathBuf,
    pub kind: FetchKind,
}

/// Why a download did not finish.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DownloadError {
    /// The model's file name could not be used safely as a file name.
    BadFileName(String),
    /// The disk has less room than the rest of the download needs.
    NotEnoughSpace { needed: u64, free: u64 },
    /// The server refused: the HTTP status it answered with.
    Http { status: u16 },
    /// The connection failed or dropped; what arrived is kept.
    Network(String),
    /// The disk could not be written to or read.
    Io(String),
    /// The server sent more data than the model's size.
    WrongSize { expected: u64, actual: u64 },
    /// The finished file is not the one the catalogue describes. It was deleted.
    WrongChecksum { expected: String, actual: String },
    /// [`Options::cancel`] was set; what arrived is kept.
    Cancelled,
}

impl DownloadError {
    /// Whether the data downloaded so far is kept, so that trying again continues from there.
    #[must_use]
    pub fn keeps_progress(&self) -> bool {
        match self {
            Self::Network(_) | Self::Cancelled => true,
            Self::Http { status } => *status >= 500,
            _ => false,
        }
    }
}

impl fmt::Display for DownloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadFileName(name) => write!(f, "the file name \"{name}\" cannot be used safely"),
            Self::NotEnoughSpace { needed, free } => write!(
                f,
                "not enough free disk space: {} needed, {} free",
                gib(*needed),
                gib(*free)
            ),
            Self::Http { status } => write!(f, "the download server answered with status {status}"),
            Self::Network(why) => write!(f, "the download was interrupted: {why}"),
            Self::Io(why) => write!(f, "could not write the download: {why}"),
            Self::WrongSize { expected, actual } => write!(
                f,
                "the server sent {actual} bytes or more, but the file has {expected}"
            ),
            Self::WrongChecksum { expected, actual } => write!(
                f,
                "the downloaded file is corrupt (checksum {} instead of {}) and was deleted",
                short(actual),
                short(expected)
            ),
            Self::Cancelled => write!(f, "the download was cancelled; it can be resumed"),
        }
    }
}

impl std::error::Error for DownloadError {}

fn short(checksum: &str) -> &str {
    checksum.get(..12).unwrap_or(checksum)
}

fn io(error: &std::io::Error) -> DownloadError {
    DownloadError::Io(error.to_string())
}

/// What the caller can control.
#[derive(Clone)]
pub struct Options {
    /// Set this (from another task) to stop the download. What arrived is kept.
    pub cancel: Arc<AtomicBool>,
    /// Free bytes on the disk that holds a folder; `None` if unknown (then the check is skipped).
    pub free_space: fn(&Path) -> Option<u64>,
    /// The fewest time between two progress reports, so a screen is not redrawn for every
    /// chunk. The first and the last report are always made.
    pub progress_every: Duration,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            cancel: Arc::new(AtomicBool::new(false)),
            free_space: hardware::free_space,
            progress_every: Duration::from_millis(100),
        }
    }
}

/// Receives progress. Called from the task that runs [`fetch`].
pub type OnProgress<'a> = &'a mut (dyn FnMut(Progress) + Send);

/// Passes progress on, but not more often than asked.
struct Reporter<'a> {
    callback: OnProgress<'a>,
    every: Duration,
    last: Option<Instant>,
}

impl Reporter<'_> {
    fn report(&mut self, phase: Phase, done: u64, total: u64, always: bool) {
        let due = self.last.is_none_or(|t| t.elapsed() >= self.every);
        if always || due {
            self.last = Some(Instant::now());
            (self.callback)(Progress { phase, done, total });
        }
    }
}

/// A file name that is safe to create inside the models folder.
fn safe_file_name(model: &Model) -> Result<&str, DownloadError> {
    let name = model.file_name();
    let unsafe_name = name.is_empty()
        || name == "."
        || name == ".."
        || name.contains(['/', '\\', '\0'])
        || Path::new(name).extension().is_some_and(|e| e == "part");
    if unsafe_name {
        Err(DownloadError::BadFileName(name.to_string()))
    } else {
        Ok(name)
    }
}

/// The first byte of a `Content-Range: bytes START-END/TOTAL` header.
fn content_range_start(response: &reqwest::Response) -> Option<u64> {
    let value = response.headers().get(CONTENT_RANGE)?.to_str().ok()?;
    value
        .strip_prefix("bytes ")?
        .split('-')
        .next()?
        .parse()
        .ok()
}

/// The sha256 of a file, read in pieces so a large model does not fill the memory.
async fn checksum_of(
    path: &Path,
    total: u64,
    reporter: &mut Reporter<'_>,
    cancel: &AtomicBool,
) -> Result<String, DownloadError> {
    let mut file = tokio::fs::File::open(path).await.map_err(|e| io(&e))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut done = 0u64;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(DownloadError::Cancelled);
        }
        let n = file.read(&mut buffer).await.map_err(|e| io(&e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
        done += n as u64;
        reporter.report(Phase::Verifying, done, total, false);
    }
    reporter.report(Phase::Verifying, done, total, true);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Save `model` into `dir` as its file name, and return where it is.
///
/// See the module documentation for what is kept and what is deleted when something goes wrong.
///
/// # Errors
///
/// Returns a [`DownloadError`] saying what went wrong. After [`DownloadError::Network`],
/// [`DownloadError::Cancelled`] and a 5xx [`DownloadError::Http`] the partial download is kept
/// and calling `fetch` again continues it (see [`DownloadError::keeps_progress`]).
pub async fn fetch(
    client: &reqwest::Client,
    model: &Model,
    dir: &Path,
    options: &Options,
    on_progress: OnProgress<'_>,
) -> Result<Fetched, DownloadError> {
    let name = safe_file_name(model)?;
    let total = model.size_bytes;
    let mut reporter = Reporter {
        callback: on_progress,
        every: options.progress_every,
        last: None,
    };
    tokio::fs::create_dir_all(dir).await.map_err(|e| io(&e))?;
    let final_path = dir.join(name);
    let part_path = dir.join(format!("{name}.part"));

    if let Ok(meta) = tokio::fs::metadata(&final_path).await {
        if meta.len() == total
            && checksum_of(&final_path, total, &mut reporter, &options.cancel).await?
                == model.sha256
        {
            return Ok(Fetched {
                path: final_path,
                kind: FetchKind::AlreadyPresent,
            });
        }
        // Not the file the catalogue describes (an older version, or damaged): replace it.
        tokio::fs::remove_file(&final_path)
            .await
            .map_err(|e| io(&e))?;
    }

    let mut have = tokio::fs::metadata(&part_path).await.map_or(0, |m| m.len());
    if have > total {
        tokio::fs::remove_file(&part_path)
            .await
            .map_err(|e| io(&e))?;
        have = 0;
    }
    let needed = total - have;
    if let Some(free) = (options.free_space)(dir)
        && free < needed
    {
        return Err(DownloadError::NotEnoughSpace { needed, free });
    }

    let resumed_from = if have < total {
        stream_into(client, model, &part_path, have, &mut reporter, options).await?
    } else {
        have // an earlier attempt got everything but was never checked
    };

    let actual = checksum_of(&part_path, total, &mut reporter, &options.cancel).await?;
    if actual != model.sha256 {
        let _ = tokio::fs::remove_file(&part_path).await;
        return Err(DownloadError::WrongChecksum {
            expected: model.sha256.clone(),
            actual,
        });
    }
    tokio::fs::rename(&part_path, &final_path)
        .await
        .map_err(|e| io(&e))?;
    Ok(Fetched {
        path: final_path,
        kind: if resumed_from > 0 {
            FetchKind::Resumed { from: resumed_from }
        } else {
            FetchKind::Fresh
        },
    })
}

/// Download what `part_path` is missing, continuing from `have` bytes if the server allows it.
/// Returns how many bytes of the file were already there and were kept.
async fn stream_into(
    client: &reqwest::Client,
    model: &Model,
    part_path: &Path,
    have: u64,
    reporter: &mut Reporter<'_>,
    options: &Options,
) -> Result<u64, DownloadError> {
    let total = model.size_bytes;
    let mut from = have;
    for _ in 0..2 {
        if options.cancel.load(Ordering::Relaxed) {
            return Err(DownloadError::Cancelled);
        }
        let mut request = client.get(&model.url);
        if from > 0 {
            request = request.header(RANGE, format!("bytes={from}-"));
        }
        let response = request
            .send()
            .await
            .map_err(|e| DownloadError::Network(e.without_url().to_string()))?;
        let status = response.status();
        let continuing =
            status.as_u16() == 206 && from > 0 && content_range_start(&response) == Some(from);
        let unusable_partial =
            from > 0 && (status.as_u16() == 416 || (status.as_u16() == 206 && !continuing));
        if unusable_partial {
            // The server cannot continue from there: start again from the beginning.
            let _ = tokio::fs::remove_file(part_path).await;
            from = 0;
            continue;
        }
        if !status.is_success() {
            return Err(DownloadError::Http {
                status: status.as_u16(),
            });
        }
        // `continuing` appends; otherwise the server is sending the whole file.
        let mut file = if continuing {
            tokio::fs::OpenOptions::new()
                .append(true)
                .open(part_path)
                .await
        } else {
            tokio::fs::File::create(part_path).await
        }
        .map_err(|e| io(&e))?;
        let kept = if continuing { from } else { 0 };
        receive(
            response, &mut file, part_path, kept, total, reporter, options,
        )
        .await?;
        return Ok(kept);
    }
    Err(DownloadError::Network(
        "the server would not continue the download".to_string(),
    ))
}

/// Write the body of `response` to `file`, which already holds `done` bytes.
async fn receive(
    mut response: reqwest::Response,
    file: &mut tokio::fs::File,
    part_path: &Path,
    mut done: u64,
    total: u64,
    reporter: &mut Reporter<'_>,
    options: &Options,
) -> Result<(), DownloadError> {
    reporter.report(Phase::Downloading, done, total, true);
    loop {
        if options.cancel.load(Ordering::Relaxed) {
            file.flush().await.map_err(|e| io(&e))?;
            return Err(DownloadError::Cancelled);
        }
        match response.chunk().await {
            Ok(Some(chunk)) => {
                let actual = done + chunk.len() as u64;
                if actual > total {
                    let _ = tokio::fs::remove_file(part_path).await;
                    return Err(DownloadError::WrongSize {
                        expected: total,
                        actual,
                    });
                }
                file.write_all(&chunk).await.map_err(|e| io(&e))?;
                done = actual;
                reporter.report(Phase::Downloading, done, total, false);
            }
            Ok(None) => break,
            Err(e) => {
                file.flush().await.map_err(|e| io(&e))?;
                return Err(DownloadError::Network(e.without_url().to_string()));
            }
        }
    }
    file.flush().await.map_err(|e| io(&e))?;
    if done < total {
        return Err(DownloadError::Network(format!(
            "the connection closed after {done} of {total} bytes"
        )));
    }
    reporter.report(Phase::Downloading, done, total, true);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalogue::Role;
    use crate::hardware::Tier;
    use crate::test_support::{Behaviour, FileServer, serve_file};
    use std::sync::atomic::AtomicUsize;

    const LEN: usize = 300_000;

    /// Bytes that do not repeat, so a wrong prefix or a shifted copy is noticed.
    fn body(len: usize) -> Vec<u8> {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x.to_be_bytes()[3]
            })
            .collect()
    }

    fn sha(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn model(url: &str, data: &[u8]) -> Model {
        Model {
            id: "test-model".into(),
            name: "Test model".into(),
            role: Role::Chat,
            class: Tier::Minimal,
            url: url.into(),
            sha256: sha(data),
            size_bytes: data.len() as u64,
            licence: "MIT".into(),
            licence_url: None,
        }
    }

    /// A folder for one test, removed afterwards.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            let dir = std::env::temp_dir().join(format!(
                "voice-notes-download-test-{}-{n}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            Self(dir)
        }

        fn file(&self) -> PathBuf {
            self.0.join("file.bin")
        }

        fn part(&self) -> PathBuf {
            self.0.join("file.bin.part")
        }

        fn names(&self) -> Vec<String> {
            let mut names: Vec<String> = std::fs::read_dir(&self.0)
                .into_iter()
                .flatten()
                .filter_map(Result::ok)
                .map(|e| e.file_name().to_string_lossy().into())
                .collect();
            names.sort();
            names
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    // The signature is fixed by `Options::free_space`.
    #[allow(clippy::unnecessary_wraps)]
    fn plenty(_: &Path) -> Option<u64> {
        Some(u64::MAX)
    }

    fn every_report() -> Options {
        Options {
            free_space: plenty,
            progress_every: Duration::ZERO,
            ..Options::default()
        }
    }

    async fn run(
        server: &FileServer,
        data: &[u8],
        dir: &Scratch,
        options: &Options,
    ) -> (Result<Fetched, DownloadError>, Vec<Progress>) {
        let mut seen = Vec::new();
        let mut on_progress = |p: Progress| seen.push(p);
        let result = fetch(
            &reqwest::Client::new(),
            &model(&server.url, data),
            &dir.0,
            options,
            &mut on_progress,
        )
        .await;
        (result, seen)
    }

    // ---- the plain download

    #[tokio::test]
    async fn downloads_a_file_and_checks_it() {
        let data = body(LEN);
        let server = serve_file(data.clone(), Behaviour::default()).await;
        let dir = Scratch::new();
        let (result, seen) = run(&server, &data, &dir, &every_report()).await;
        let fetched = result.unwrap();
        assert_eq!(fetched.kind, FetchKind::Fresh);
        assert_eq!(fetched.path, dir.file());
        assert_eq!(std::fs::read(dir.file()).unwrap(), data);
        assert_eq!(dir.names(), ["file.bin"], "no .part file is left behind");
        assert_eq!(
            server.range_starts(),
            [None],
            "a fresh download asks for the whole file"
        );
        let last_download = seen
            .iter()
            .rfind(|p| p.phase == Phase::Downloading)
            .unwrap();
        assert_eq!(
            (last_download.done, last_download.total),
            (LEN as u64, LEN as u64)
        );
        let last = seen.last().unwrap();
        assert_eq!(
            (last.phase, last.done, last.total),
            (Phase::Verifying, LEN as u64, LEN as u64)
        );
    }

    #[tokio::test]
    async fn progress_only_moves_forward_within_each_phase() {
        let data = body(LEN);
        let server = serve_file(data.clone(), Behaviour::default()).await;
        let dir = Scratch::new();
        let (_, seen) = run(&server, &data, &dir, &every_report()).await;
        assert_eq!(
            seen[0],
            Progress {
                phase: Phase::Downloading,
                done: 0,
                total: LEN as u64
            }
        );
        for phase in [Phase::Downloading, Phase::Verifying] {
            let steps: Vec<u64> = seen
                .iter()
                .filter(|p| p.phase == phase)
                .map(|p| p.done)
                .collect();
            assert!(
                steps.windows(2).all(|w| w[0] <= w[1]),
                "{phase:?}: {steps:?}"
            );
        }
        let first_verify = seen
            .iter()
            .position(|p| p.phase == Phase::Verifying)
            .unwrap();
        assert!(
            seen[first_verify..]
                .iter()
                .all(|p| p.phase == Phase::Verifying),
            "verifying comes last"
        );
    }

    #[tokio::test]
    async fn progress_reports_are_not_more_frequent_than_asked() {
        let data = body(LEN);
        let server = serve_file(data.clone(), Behaviour::default()).await;
        let dir = Scratch::new();
        let slow = Options {
            progress_every: Duration::from_secs(3600),
            ..every_report()
        };
        let (result, seen) = run(&server, &data, &dir, &slow).await;
        result.unwrap();
        // Start, end of the download, and the verification's first and last report.
        assert!(seen.len() <= 4, "{} reports: {seen:?}", seen.len());
        assert_eq!(
            seen.last().unwrap().done,
            LEN as u64,
            "the last report is never skipped"
        );
    }

    /// The data is written under a temporary name, and gets its real name only at the end.
    #[tokio::test]
    async fn the_real_name_appears_only_when_the_file_is_complete() {
        let data = body(LEN);
        let server = serve_file(data.clone(), Behaviour::default()).await;
        let dir = Scratch::new();
        let (file, part) = (dir.file(), dir.part());
        let mut during = Vec::new();
        let mut on_progress = |p: Progress| {
            if p.phase == Phase::Downloading && p.done > 0 {
                during.push((file.exists(), part.exists()));
            }
        };
        let options = every_report();
        fetch(
            &reqwest::Client::new(),
            &model(&server.url, &data),
            &dir.0,
            &options,
            &mut on_progress,
        )
        .await
        .unwrap();
        assert!(
            !during.is_empty(),
            "no progress was reported while downloading"
        );
        assert!(
            during
                .iter()
                .all(|&(final_exists, part_exists)| !final_exists && part_exists),
            "{during:?}"
        );
    }

    // ---- resuming

    #[tokio::test]
    async fn an_interrupted_download_resumes_from_where_it_stopped() {
        let data = body(LEN);
        let cut = Behaviour {
            cut_first_after: Some(100_000),
            ..Behaviour::default()
        };
        let server = serve_file(data.clone(), cut).await;
        let dir = Scratch::new();

        let (first, _) = run(&server, &data, &dir, &every_report()).await;
        let error = first.unwrap_err();
        assert!(matches!(error, DownloadError::Network(_)), "{error}");
        assert!(error.keeps_progress());
        assert!(
            !dir.file().exists(),
            "an unfinished file never has the real name"
        );
        let kept = std::fs::metadata(dir.part()).unwrap().len();
        assert!(
            kept > 0 && kept < LEN as u64,
            "part of the file is kept: {kept}"
        );

        let (second, seen) = run(&server, &data, &dir, &every_report()).await;
        assert_eq!(second.unwrap().kind, FetchKind::Resumed { from: kept });
        assert_eq!(
            server.range_starts(),
            [None, Some(kept)],
            "the second request continues"
        );
        assert_eq!(
            seen[0].done, kept,
            "progress starts where the first attempt stopped"
        );
        assert_eq!(std::fs::read(dir.file()).unwrap(), data);
        assert_eq!(dir.names(), ["file.bin"]);
    }

    #[tokio::test]
    async fn an_existing_partial_file_is_continued() {
        let data = body(LEN);
        let server = serve_file(data.clone(), Behaviour::default()).await;
        let dir = Scratch::new();
        std::fs::create_dir_all(&dir.0).unwrap();
        std::fs::write(dir.part(), &data[..120_000]).unwrap();
        let (result, _) = run(&server, &data, &dir, &every_report()).await;
        assert_eq!(result.unwrap().kind, FetchKind::Resumed { from: 120_000 });
        assert_eq!(server.range_starts(), [Some(120_000)]);
        assert_eq!(std::fs::read(dir.file()).unwrap(), data);
    }

    /// A server without range support sends the whole file again; that must still work.
    #[tokio::test]
    async fn a_server_that_ignores_the_range_request_still_gives_a_correct_file() {
        let data = body(LEN);
        let server = serve_file(
            data.clone(),
            Behaviour {
                ignore_range: true,
                ..Behaviour::default()
            },
        )
        .await;
        let dir = Scratch::new();
        std::fs::create_dir_all(&dir.0).unwrap();
        std::fs::write(dir.part(), &data[..50_000]).unwrap();
        let (result, _) = run(&server, &data, &dir, &every_report()).await;
        assert_eq!(
            result.unwrap().kind,
            FetchKind::Fresh,
            "nothing of the partial file was kept"
        );
        assert_eq!(std::fs::read(dir.file()).unwrap(), data);
        assert_eq!(server.range_starts(), [Some(50_000)]);
    }

    #[tokio::test]
    async fn a_partial_file_that_is_complete_is_checked_without_asking_the_server() {
        let data = body(LEN);
        let server = serve_file(data.clone(), Behaviour::default()).await;
        let dir = Scratch::new();
        std::fs::create_dir_all(&dir.0).unwrap();
        std::fs::write(dir.part(), &data).unwrap(); // finished, but the program stopped before checking it
        let (result, _) = run(&server, &data, &dir, &every_report()).await;
        assert!(matches!(result.unwrap().kind, FetchKind::Resumed { .. }));
        assert_eq!(server.request_count(), 0);
        assert_eq!(dir.names(), ["file.bin"]);
    }

    #[tokio::test]
    async fn a_partial_file_that_is_too_long_is_discarded() {
        let data = body(LEN);
        let server = serve_file(data.clone(), Behaviour::default()).await;
        let dir = Scratch::new();
        std::fs::create_dir_all(&dir.0).unwrap();
        std::fs::write(dir.part(), body(LEN + 5000)).unwrap();
        let (result, _) = run(&server, &data, &dir, &every_report()).await;
        assert_eq!(result.unwrap().kind, FetchKind::Fresh);
        assert_eq!(server.range_starts(), [None]);
        assert_eq!(std::fs::read(dir.file()).unwrap(), data);
    }

    /// Asked for a range the server does not have (416): start again from the beginning.
    #[tokio::test]
    async fn a_refused_range_restarts_the_download() {
        let data = body(LEN);
        let server = serve_file(data.clone(), Behaviour::default()).await;
        let dir = Scratch::new();
        std::fs::create_dir_all(&dir.0).unwrap();
        std::fs::write(dir.part(), &data).unwrap();
        // The catalogue says the file is bigger than what the server has, so the partial file
        // (all the server has) looks incomplete, and the range starts past the end.
        let mut bigger = model(&server.url, &data);
        bigger.size_bytes += 100_000;
        let mut on_progress = |_: Progress| {};
        let result = fetch(
            &reqwest::Client::new(),
            &bigger,
            &dir.0,
            &every_report(),
            &mut on_progress,
        )
        .await;
        assert!(
            matches!(result, Err(DownloadError::Network(_))),
            "{result:?}"
        );
        assert_eq!(
            server.range_starts(),
            [Some(LEN as u64), None],
            "416, then again from the start"
        );
    }

    // ---- checking

    #[tokio::test]
    async fn a_file_with_the_wrong_checksum_is_deleted() {
        let data = body(LEN);
        let server = serve_file(body(LEN), Behaviour::default()).await;
        let mut wrong = data.clone();
        wrong[10] ^= 0xFF; // the catalogue describes a different file
        let dir = Scratch::new();
        let (result, _) = run(&server, &wrong, &dir, &every_report()).await;
        match result {
            Err(DownloadError::WrongChecksum { expected, actual }) => {
                assert_eq!(expected, sha(&wrong));
                assert_eq!(actual, sha(&data));
            }
            other => panic!("expected a checksum error, got {other:?}"),
        }
        assert!(dir.names().is_empty(), "nothing is left: {:?}", dir.names());
    }

    #[tokio::test]
    async fn a_bad_partial_file_is_deleted_after_the_check_so_the_retry_starts_clean() {
        let data = body(LEN);
        let server = serve_file(data.clone(), Behaviour::default()).await;
        let dir = Scratch::new();
        std::fs::create_dir_all(&dir.0).unwrap();
        std::fs::write(dir.part(), vec![7u8; 60_000]).unwrap(); // not the start of this file
        let (first, _) = run(&server, &data, &dir, &every_report()).await;
        let error = first.unwrap_err();
        assert!(
            matches!(error, DownloadError::WrongChecksum { .. }),
            "{error}"
        );
        assert!(!error.keeps_progress());
        assert!(dir.names().is_empty(), "nothing is left in the folder");

        let (second, _) = run(&server, &data, &dir, &every_report()).await;
        assert_eq!(second.unwrap().kind, FetchKind::Fresh);
        assert_eq!(std::fs::read(dir.file()).unwrap(), data);
    }

    #[tokio::test]
    async fn a_server_that_sends_too_much_is_rejected_and_nothing_is_kept() {
        let data = body(LEN);
        let server = serve_file(data.clone(), Behaviour::default()).await;
        let mut smaller = model(&server.url, &data);
        smaller.size_bytes = 100_000;
        let dir = Scratch::new();
        let mut on_progress = |_: Progress| {};
        let result = fetch(
            &reqwest::Client::new(),
            &smaller,
            &dir.0,
            &every_report(),
            &mut on_progress,
        )
        .await;
        assert!(
            matches!(
                result,
                Err(DownloadError::WrongSize {
                    expected: 100_000,
                    ..
                })
            ),
            "{result:?}"
        );
        assert!(dir.names().is_empty(), "{:?}", dir.names());
    }

    // ---- files that are already there

    #[tokio::test]
    async fn a_correct_file_is_not_downloaded_again() {
        let data = body(LEN);
        let server = serve_file(data.clone(), Behaviour::default()).await;
        let dir = Scratch::new();
        run(&server, &data, &dir, &every_report()).await.0.unwrap();
        let (again, seen) = run(&server, &data, &dir, &every_report()).await;
        assert_eq!(again.unwrap().kind, FetchKind::AlreadyPresent);
        assert_eq!(server.request_count(), 1, "no second request");
        assert!(
            seen.iter().all(|p| p.phase == Phase::Verifying),
            "it was checked, not downloaded"
        );
    }

    #[tokio::test]
    async fn a_damaged_or_older_file_is_replaced() {
        let data = body(LEN);
        let server = serve_file(data.clone(), Behaviour::default()).await;
        let dir = Scratch::new();
        std::fs::create_dir_all(&dir.0).unwrap();
        let mut damaged = data.clone();
        damaged[1000] ^= 0x55; // right size, wrong content
        std::fs::write(dir.file(), &damaged).unwrap();
        assert_eq!(
            run(&server, &data, &dir, &every_report())
                .await
                .0
                .unwrap()
                .kind,
            FetchKind::Fresh
        );
        assert_eq!(std::fs::read(dir.file()).unwrap(), data);

        std::fs::write(dir.file(), &data[..1000]).unwrap(); // wrong size
        assert_eq!(
            run(&server, &data, &dir, &every_report())
                .await
                .0
                .unwrap()
                .kind,
            FetchKind::Fresh
        );
        assert_eq!(std::fs::read(dir.file()).unwrap(), data);
    }

    // ---- disk space

    #[tokio::test]
    async fn it_refuses_to_start_when_the_disk_is_too_small() {
        let data = body(LEN);
        let server = serve_file(data.clone(), Behaviour::default()).await;
        let dir = Scratch::new();
        let tight = Options {
            free_space: |_| Some(10),
            ..every_report()
        };
        let (result, seen) = run(&server, &data, &dir, &tight).await;
        assert_eq!(
            result,
            Err(DownloadError::NotEnoughSpace {
                needed: LEN as u64,
                free: 10
            })
        );
        assert_eq!(server.request_count(), 0, "nothing was requested");
        assert!(dir.names().is_empty(), "nothing is left in the folder");
        assert!(
            seen.is_empty(),
            "no progress is reported for a download that never starts"
        );
    }

    #[tokio::test]
    async fn space_is_needed_only_for_what_is_still_missing() {
        let data = body(LEN);
        let server = serve_file(data.clone(), Behaviour::default()).await;
        let dir = Scratch::new();
        std::fs::create_dir_all(&dir.0).unwrap();
        std::fs::write(dir.part(), &data[..250_000]).unwrap();
        let just_short = Options {
            free_space: |_| Some(49_999),
            ..every_report()
        };
        let (result, _) = run(&server, &data, &dir, &just_short).await;
        assert_eq!(
            result,
            Err(DownloadError::NotEnoughSpace {
                needed: 50_000,
                free: 49_999
            })
        );
        assert_eq!(
            std::fs::metadata(dir.part()).unwrap().len(),
            250_000,
            "the partial file is untouched"
        );

        let just_enough = Options {
            free_space: |_| Some(50_000),
            ..every_report()
        };
        assert!(run(&server, &data, &dir, &just_enough).await.0.is_ok());
    }

    #[tokio::test]
    async fn unknown_free_space_does_not_stop_the_download() {
        let data = body(LEN);
        let server = serve_file(data.clone(), Behaviour::default()).await;
        let dir = Scratch::new();
        let unknown = Options {
            free_space: |_| None,
            ..every_report()
        };
        assert!(run(&server, &data, &dir, &unknown).await.0.is_ok());
    }

    // ---- cancelling

    #[tokio::test]
    async fn cancelling_keeps_what_arrived_and_the_download_can_be_resumed() {
        let data = body(LEN);
        let server = serve_file(data.clone(), Behaviour::default()).await;
        let dir = Scratch::new();
        let options = every_report();
        let cancel = Arc::clone(&options.cancel);
        let mut on_progress = |p: Progress| {
            if p.phase == Phase::Downloading && p.done > 0 {
                cancel.store(true, Ordering::Relaxed);
            }
        };
        let result = fetch(
            &reqwest::Client::new(),
            &model(&server.url, &data),
            &dir.0,
            &options,
            &mut on_progress,
        )
        .await;
        let error = result.unwrap_err();
        assert_eq!(error, DownloadError::Cancelled);
        assert!(error.keeps_progress());
        assert!(!dir.file().exists());
        let kept = std::fs::metadata(dir.part()).unwrap().len();
        assert!(kept > 0, "what arrived is kept");

        options.cancel.store(false, Ordering::Relaxed);
        let (resumed, _) = run(&server, &data, &dir, &options).await;
        assert!(resumed.is_ok());
        assert_eq!(std::fs::read(dir.file()).unwrap(), data);
    }

    #[tokio::test]
    async fn a_download_cancelled_before_it_starts_makes_no_request() {
        let data = body(LEN);
        let server = serve_file(data.clone(), Behaviour::default()).await;
        let dir = Scratch::new();
        let options = every_report();
        options.cancel.store(true, Ordering::Relaxed);
        let (result, _) = run(&server, &data, &dir, &options).await;
        assert_eq!(result, Err(DownloadError::Cancelled));
        assert_eq!(server.request_count(), 0);
    }

    // ---- server and network errors

    #[tokio::test]
    async fn a_missing_file_is_reported_and_leaves_nothing() {
        let data = body(LEN);
        let server = serve_file(
            data.clone(),
            Behaviour {
                status: Some(404),
                ..Behaviour::default()
            },
        )
        .await;
        let dir = Scratch::new();
        let (result, _) = run(&server, &data, &dir, &every_report()).await;
        let error = result.unwrap_err();
        assert_eq!(error, DownloadError::Http { status: 404 });
        assert!(!error.keeps_progress());
        assert!(dir.names().is_empty(), "nothing is left in the folder");
    }

    #[tokio::test]
    async fn a_server_error_keeps_the_partial_file_for_a_later_try() {
        let data = body(LEN);
        let server = serve_file(
            data.clone(),
            Behaviour {
                status: Some(503),
                ..Behaviour::default()
            },
        )
        .await;
        let dir = Scratch::new();
        std::fs::create_dir_all(&dir.0).unwrap();
        std::fs::write(dir.part(), &data[..1000]).unwrap();
        let (result, _) = run(&server, &data, &dir, &every_report()).await;
        let error = result.unwrap_err();
        assert_eq!(error, DownloadError::Http { status: 503 });
        assert!(error.keeps_progress());
        assert_eq!(std::fs::metadata(dir.part()).unwrap().len(), 1000);
    }

    #[tokio::test]
    async fn an_unreachable_server_is_a_network_error_and_never_a_hang() {
        let data = body(1000);
        let mut gone = model("http://127.0.0.1:1/models/file.bin", &data);
        gone.size_bytes = 1000;
        let dir = Scratch::new();
        let mut on_progress = |_: Progress| {};
        let result = fetch(
            &reqwest::Client::new(),
            &gone,
            &dir.0,
            &every_report(),
            &mut on_progress,
        )
        .await;
        let error = result.unwrap_err();
        assert!(matches!(error, DownloadError::Network(_)), "{error}");
        assert!(
            !error.to_string().contains("127.0.0.1:1"),
            "the address is not repeated in the message"
        );
    }

    // ---- unsafe names

    #[tokio::test]
    async fn file_names_that_could_escape_the_folder_are_refused() {
        let data = body(1000);
        for url in [
            "https://host/models/..",
            "https://host/models/.",
            "https://host/models/",
            "https://host/models/a%5Cb\\c.bin",
            "https://host/models/x.part",
        ] {
            let dir = Scratch::new();
            let mut on_progress = |_: Progress| {};
            let result = fetch(
                &reqwest::Client::new(),
                &model(url, &data),
                &dir.0,
                &every_report(),
                &mut on_progress,
            )
            .await;
            assert!(
                matches!(result, Err(DownloadError::BadFileName(_))),
                "{url}: {result:?}"
            );
            assert!(!dir.0.exists(), "{url}: nothing is created for a bad name");
        }
    }

    // ---- messages

    #[test]
    fn messages_say_what_happened_in_plain_words() {
        let space = DownloadError::NotEnoughSpace {
            needed: 5 << 30,
            free: 1 << 30,
        };
        assert_eq!(
            space.to_string(),
            "not enough free disk space: 5.0 GB needed, 1.0 GB free"
        );
        assert!(
            DownloadError::Http { status: 404 }
                .to_string()
                .contains("404")
        );
        assert!(DownloadError::Cancelled.to_string().contains("resumed"));
        let sum = DownloadError::WrongChecksum {
            expected: "a".repeat(64),
            actual: "b".repeat(64),
        };
        assert!(
            sum.to_string()
                .contains("bbbbbbbbbbbb instead of aaaaaaaaaaaa"),
            "{sum}"
        );
        assert!(sum.to_string().contains("deleted"));
    }

    #[test]
    fn only_interruptions_keep_the_partial_download() {
        let keeps = |e: DownloadError| e.keeps_progress();
        assert!(keeps(DownloadError::Network("x".into())));
        assert!(keeps(DownloadError::Cancelled));
        assert!(keeps(DownloadError::Http { status: 500 }));
        assert!(!keeps(DownloadError::Http { status: 403 }));
        assert!(!keeps(DownloadError::WrongChecksum {
            expected: "a".into(),
            actual: "b".into()
        }));
        assert!(!keeps(DownloadError::WrongSize {
            expected: 1,
            actual: 2
        }));
        assert!(!keeps(DownloadError::NotEnoughSpace { needed: 2, free: 1 }));
    }
}
