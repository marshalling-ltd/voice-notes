//! Voice Notes - Speech-to-Text CLI Application
//!
//! This crate provides speech-to-text transcription with optional text cleaning, and a voice
//! conversation loop: your speech is transcribed, answered by a chat backend, and spoken back.

pub mod audio;
pub mod catalogue;
pub mod chat;
pub mod config;
pub mod devices;
pub mod endpoints;
pub mod hardware;
pub mod headless;
pub mod latency;
pub mod pipeline;
pub mod playback;
pub mod profile;
pub mod setup;
pub mod speech;
pub mod transcribe;

#[cfg(test)]
pub(crate) mod test_support;

#[cfg(test)]
mod tests {
    /// Smoke test confirming the library crate builds and its test harness runs.
    #[test]
    fn test_placeholder() {
        // Placeholder test to ensure the lib.rs compiles
        assert_eq!(1, 1);
    }
}
