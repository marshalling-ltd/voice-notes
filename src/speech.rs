//! Text-to-speech through the `espeak-ng` command.

use anyhow::{Context, Result, anyhow};
use std::io::Write;
use std::process::{Child, Command, Stdio};

/// Longest text spoken in one go.
const MAX_SPOKEN_CHARS: usize = 2000;

/// Turn model output into something pleasant to hear: no code blocks, markdown marks, link
/// targets or URLs, and a sane length.
///
/// # Examples
///
/// ```
/// use voice_notes::speech::speakable;
///
/// let text = "**Sure.** See [the docs](https://example.com) or run:\n```sh\nls -la\n```\nDone_now!";
/// assert_eq!(speakable(text), "Sure. See the docs or run: Done now!");
/// ```
#[must_use]
pub fn speakable(text: &str) -> String {
    let mut prose = String::new();
    let mut in_code = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            in_code = !in_code;
        } else if !in_code {
            prose.push_str(line);
            prose.push('\n');
        }
    }

    let prose = strip_links(&prose);
    let marks_removed: String = prose
        .chars()
        .map(|c| match c {
            '*' | '#' | '`' | '>' | '|' | '~' | '_' => ' ',
            other => other,
        })
        .collect();
    let words: Vec<&str> = marks_removed
        .split_whitespace()
        .filter(|w| !w.starts_with("http://") && !w.starts_with("https://"))
        .collect();
    let mut out = words.join(" ");

    if out.chars().count() > MAX_SPOKEN_CHARS {
        out = out.chars().take(MAX_SPOKEN_CHARS).collect();
        if let Some(cut) = out.rfind(' ') {
            out.truncate(cut);
        }
    }
    out
}

/// Replace `[label](target)` with `label`.
fn strip_links(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find('[') {
        let after = &rest[open + 1..];
        let link = after.find("](").and_then(|close| {
            after[close + 2..]
                .find(')')
                .map(|end| (close, close + 2 + end))
        });
        if let Some((close, end)) = link {
            out.push_str(&rest[..open]);
            out.push_str(&after[..close]);
            rest = &after[end + 1..];
        } else {
            out.push_str(&rest[..=open]);
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// Speaks text aloud, one utterance at a time.
pub struct Speaker {
    voice: String,
    child: Option<Child>,
}

impl Speaker {
    /// `voice` is an `espeak-ng` voice such as `en` or `en-gb`.
    #[must_use]
    pub fn new(voice: impl Into<String>) -> Self {
        Self {
            voice: voice.into(),
            child: None,
        }
    }

    /// Start speaking `text`, interrupting anything already being said. Returns immediately.
    ///
    /// # Errors
    ///
    /// Returns an error if `espeak-ng` is not installed or cannot be started.
    pub fn speak(&mut self, text: &str) -> Result<()> {
        self.stop();
        let text = speakable(text);
        if text.is_empty() {
            return Ok(());
        }
        let mut child = Command::new("espeak-ng")
            .args(["-v", &self.voice, "-s", "165", "--stdin"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => {
                    anyhow!("espeak-ng not found; install it with: sudo apt install espeak-ng")
                }
                _ => anyhow!(e).context("could not start espeak-ng"),
            })?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(text.as_bytes())
                .context("could not send text to espeak-ng")?;
        }
        self.child = Some(child);
        Ok(())
    }

    /// Whether speech is still playing.
    pub fn is_speaking(&mut self) -> bool {
        match self.child.as_mut().map(Child::try_wait) {
            Some(Ok(None)) => true,
            Some(_) => {
                self.child = None;
                false
            }
            None => false,
        }
    }

    /// Cut speech off.
    pub fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for Speaker {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Code blocks, markdown marks, link targets and bare URLs are removed.
    #[test]
    fn cleans_markdown_for_speech() {
        assert_eq!(
            speakable("# Title\n- **bold** and `code`"),
            "Title - bold and code"
        );
        assert_eq!(speakable("see https://example.com now"), "see now");
        assert_eq!(speakable("```\nonly code\n```"), "");
        assert_eq!(speakable("[a](b) [c](d)"), "a c");
        assert_eq!(speakable("a [not a link"), "a [not a link");
    }

    /// Long text is cut at a word boundary.
    #[test]
    fn truncates_long_text() {
        let long = "word ".repeat(1000);
        let out = speakable(&long);
        assert!(out.chars().count() <= MAX_SPOKEN_CHARS);
        assert!(out.ends_with("word"));
    }

    /// A speaker that never spoke is not speaking, and stopping is harmless.
    #[test]
    fn idle_speaker() {
        let mut s = Speaker::new("en");
        assert!(!s.is_speaking());
        s.stop();
        assert!(s.speak("   ").is_ok()); // nothing speakable, so no process needed
        assert!(!s.is_speaking());
    }
}
