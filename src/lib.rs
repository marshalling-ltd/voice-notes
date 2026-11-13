//! Voice Notes - Speech-to-Text CLI Application
//!
//! This crate provides a command-line interface for speech-to-text conversion
//! with optional text cleaning capabilities.

pub mod audio;
pub mod transcribe;

#[cfg(test)]
mod tests {
    /// Smoke test confirming the library crate builds and its test harness runs.
    #[test]
    fn test_placeholder() {
        // Placeholder test to ensure the lib.rs compiles
        assert_eq!(1, 1);
    }
}
