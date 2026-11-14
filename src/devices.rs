//! Audio devices: listing the microphones and speakers, and finding the one a person chose.
//!
//! A choice is saved as the device's id, which `cpal` documents as stable across runs and
//! reboots, and can also be given as the device's name. Both input and output use the same audio
//! host, so ids mean the same thing everywhere.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};
use std::fmt::Write;

/// Which way the sound goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// A microphone.
    Input,
    /// A speaker or headphones.
    Output,
}

impl Direction {
    /// `microphone` or `speaker`, for messages.
    #[must_use]
    pub fn noun(self) -> &'static str {
        match self {
            Self::Input => "microphone",
            Self::Output => "speaker",
        }
    }
}

/// A device as shown to the user and saved in the config.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    /// The stable id; this is what is saved.
    pub id: String,
    /// What people call it, e.g. `Built-in Audio Analog Stereo`.
    pub name: String,
    /// Whether it is what the system uses when nothing is chosen.
    pub is_default: bool,
}

/// How a wanted device relates to the devices that exist.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pick {
    /// Nothing was asked for: use the system default.
    Default,
    /// This device (its id).
    Device(String),
    /// Nothing matches.
    NotFound,
    /// Several devices match; the names are listed so the user can be more specific.
    Ambiguous(Vec<String>),
}

/// Match what the user asked for (an id, a name, or part of a name) against the devices.
///
/// An exact id wins, then an exact name, then a part of a name if exactly one device has it.
/// Matching ignores case. An empty request means the default.
#[must_use]
pub fn plan(devices: &[DeviceInfo], wanted: Option<&str>) -> Pick {
    let Some(wanted) = wanted.map(str::trim).filter(|w| !w.is_empty()) else {
        return Pick::Default;
    };
    if let Some(device) = devices.iter().find(|d| d.id == wanted) {
        return Pick::Device(device.id.clone());
    }
    let lower = wanted.to_lowercase();
    let exact: Vec<&DeviceInfo> = devices
        .iter()
        .filter(|d| d.name.to_lowercase() == lower)
        .collect();
    if let [only] = exact.as_slice() {
        return Pick::Device(only.id.clone());
    }
    let partial: Vec<&DeviceInfo> = if exact.is_empty() {
        devices
            .iter()
            .filter(|d| d.name.to_lowercase().contains(&lower))
            .collect()
    } else {
        exact
    };
    match partial.as_slice() {
        [] => Pick::NotFound,
        [only] => Pick::Device(only.id.clone()),
        several => Pick::Ambiguous(several.iter().map(|d| d.name.clone()).collect()),
    }
}

/// The audio host both directions use. On Linux, prefer the `PulseAudio` protocol (served by
/// `PipeWire` and `PulseAudio`): the raw ALSA default often cannot be opened when a sound server
/// owns the hardware. Elsewhere (macOS) the default host (`CoreAudio`) is the right one.
#[cfg(target_os = "linux")]
pub(crate) fn host() -> cpal::Host {
    cpal::host_from_id(cpal::HostId::PulseAudio).unwrap_or_else(|_| cpal::default_host())
}

/// The audio host both directions use (see the Linux variant).
#[cfg(not(target_os = "linux"))]
pub(crate) fn host() -> cpal::Host {
    cpal::default_host()
}

fn describe(device: &cpal::Device) -> Option<(String, String)> {
    let id = device.id().ok()?.to_string();
    let name = device
        .description()
        .map_or_else(|_| id.clone(), |d| d.name().to_string());
    Some((id, name))
}

/// Every device that can do `direction`, the default one marked.
///
/// # Errors
///
/// Returns an error if the audio system cannot be asked for its devices.
pub fn list(direction: Direction) -> Result<Vec<DeviceInfo>> {
    let host = host();
    let default_id = match direction {
        Direction::Input => host.default_input_device(),
        Direction::Output => host.default_output_device(),
    }
    .and_then(|d| describe(&d))
    .map(|(id, _)| id);
    let devices: Vec<cpal::Device> = match direction {
        Direction::Input => host.input_devices().map(Iterator::collect),
        Direction::Output => host.output_devices().map(Iterator::collect),
    }
    .with_context(|| format!("could not list the {}s", direction.noun()))?;
    Ok(devices
        .iter()
        .filter_map(describe)
        .filter(|(id, _)| direction == Direction::Output || !is_monitor(id))
        .map(|(id, name)| DeviceInfo {
            is_default: default_id.as_deref() == Some(id.as_str()),
            id,
            name,
        })
        .collect())
}

/// `PulseAudio` lists a "monitor" of every speaker as an input. It records what the computer is
/// playing, which is never what someone means by "microphone", so it is not offered.
fn is_monitor(id: &str) -> bool {
    id.ends_with(".monitor")
}

/// The list shown by `voice-notes devices`: one line per device with its name and id, the
/// default and the saved choice marked, and how to choose one.
#[must_use]
pub fn render(direction: Direction, devices: &[DeviceInfo], wanted: Option<&str>) -> String {
    let chosen = match plan(devices, wanted) {
        Pick::Device(id) => Some(id),
        _ => None,
    };
    let title = match direction {
        Direction::Input => "Microphones",
        Direction::Output => "Speakers",
    };
    let mut out = format!("{title}:\n");
    if devices.is_empty() {
        let _ = writeln!(out, "  (no {} found)", direction.noun());
    }
    for device in devices {
        let mut marks = Vec::new();
        if device.is_default {
            marks.push("default");
        }
        if chosen.as_deref() == Some(device.id.as_str()) {
            marks.push("chosen");
        }
        let marks = if marks.is_empty() {
            String::new()
        } else {
            format!("  [{}]", marks.join(", "))
        };
        let _ = write!(out, "  {}{marks}\n      id: {}\n", device.name, device.id);
    }
    if let Some(wanted) = wanted.filter(|w| !w.trim().is_empty())
        && chosen.is_none()
    {
        let _ = writeln!(
            out,
            "  The choice \"{wanted}\" matches no {}; the default is used.",
            direction.noun()
        );
    }
    let flag = match direction {
        Direction::Input => "--input-device",
        Direction::Output => "--output-device",
    };
    let _ = write!(
        out,
        "  Choose one with {flag} ID_OR_NAME, or with d in the app.\n\n"
    );
    out
}

/// A device ready to open, and a message if it is not the one that was asked for.
pub struct Resolved {
    pub device: cpal::Device,
    /// Set when the chosen device was not found and the default is used instead.
    pub notice: Option<String>,
}

/// Find the device for `wanted` (`None` is the default). A device that cannot be found does not
/// stop the app: the default is used and the notice says so, because a USB microphone that is not
/// plugged in today should not make yesterday's choice a dead end.
///
/// `Ok(None)` means the system has no such device at all.
///
/// # Errors
///
/// Returns an error if the audio system cannot list its devices.
pub fn resolve(direction: Direction, wanted: Option<&str>) -> Result<Option<Resolved>> {
    let host = host();
    let default = || match direction {
        Direction::Input => host.default_input_device(),
        Direction::Output => host.default_output_device(),
    };
    let devices = list(direction)?;
    let (device, notice) = match plan(&devices, wanted) {
        Pick::Default => (default(), None),
        Pick::Device(id) => {
            let found = id
                .parse::<cpal::DeviceId>()
                .ok()
                .and_then(|id| host.device_by_id(&id));
            match found {
                Some(device) => (Some(device), None),
                None => (
                    default(),
                    Some(format!(
                        "The chosen {} is not available; using the default",
                        direction.noun()
                    )),
                ),
            }
        }
        Pick::NotFound => (
            default(),
            Some(format!(
                "No {} matches \"{}\"; using the default",
                direction.noun(),
                wanted.unwrap_or_default()
            )),
        ),
        Pick::Ambiguous(names) => (
            default(),
            Some(format!(
                "\"{}\" matches several {}s ({}); using the default",
                wanted.unwrap_or_default(),
                direction.noun(),
                names.join(", ")
            )),
        ),
    };
    Ok(device.map(|device| Resolved { device, notice }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(id: &str, name: &str) -> DeviceInfo {
        DeviceInfo {
            id: id.to_string(),
            name: name.to_string(),
            is_default: false,
        }
    }

    fn some() -> Vec<DeviceInfo> {
        vec![
            device("pulse:alsa_input.pci", "Built-in Audio Analog Stereo"),
            device("pulse:alsa_input.usb", "Blue Yeti USB Microphone"),
            device("pulse:alsa_input.cam", "HD Webcam Microphone"),
        ]
    }

    #[test]
    fn nothing_asked_for_means_the_default() {
        assert_eq!(plan(&some(), None), Pick::Default);
        assert_eq!(plan(&some(), Some("")), Pick::Default);
        assert_eq!(plan(&some(), Some("   ")), Pick::Default);
    }

    #[test]
    fn an_id_matches_exactly() {
        let pick = plan(&some(), Some("pulse:alsa_input.usb"));
        assert_eq!(pick, Pick::Device("pulse:alsa_input.usb".to_string()));
    }

    #[test]
    fn a_name_matches_ignoring_case() {
        let pick = plan(&some(), Some("blue yeti usb microphone"));
        assert_eq!(pick, Pick::Device("pulse:alsa_input.usb".to_string()));
    }

    #[test]
    fn a_part_of_a_name_matches_when_it_is_unique() {
        let pick = plan(&some(), Some("yeti"));
        assert_eq!(pick, Pick::Device("pulse:alsa_input.usb".to_string()));
    }

    #[test]
    fn a_part_of_a_name_that_several_share_is_ambiguous() {
        match plan(&some(), Some("microphone")) {
            Pick::Ambiguous(names) => {
                assert_eq!(names, ["Blue Yeti USB Microphone", "HD Webcam Microphone"]);
            }
            other => panic!("expected ambiguous, got {other:?}"),
        }
    }

    #[test]
    fn an_exact_name_beats_a_longer_name_that_contains_it() {
        let devices = vec![device("a", "USB"), device("b", "USB Headset")];
        assert_eq!(plan(&devices, Some("usb")), Pick::Device("a".to_string()));
    }

    #[test]
    fn an_unknown_device_is_not_found() {
        assert_eq!(plan(&some(), Some("no such thing")), Pick::NotFound);
        assert_eq!(plan(&[], Some("anything")), Pick::NotFound);
        assert_eq!(plan(&[], None), Pick::Default);
    }

    #[test]
    fn speaker_monitors_are_recognised_by_their_id() {
        assert!(is_monitor(
            "pulseaudio:alsa_output.pci-0000_66_00.6.analog-stereo.monitor"
        ));
        assert!(!is_monitor(
            "pulseaudio:alsa_input.pci-0000_66_00.6.analog-stereo"
        ));
    }

    #[test]
    fn the_list_marks_the_default_and_the_chosen_device() {
        let mut devices = some();
        devices[0].is_default = true;
        let text = render(Direction::Input, &devices, Some("yeti"));
        assert!(text.starts_with("Microphones:\n"), "{text}");
        assert!(
            text.contains("Built-in Audio Analog Stereo  [default]\n"),
            "{text}"
        );
        assert!(
            text.contains("Blue Yeti USB Microphone  [chosen]\n"),
            "{text}"
        );
        assert!(text.contains("id: pulse:alsa_input.usb"), "{text}");
        assert!(text.contains("--input-device ID_OR_NAME"), "{text}");
    }

    #[test]
    fn a_device_can_be_both_default_and_chosen() {
        let mut devices = some();
        devices[1].is_default = true;
        let text = render(
            Direction::Output,
            &devices,
            Some("blue yeti usb microphone"),
        );
        assert!(text.starts_with("Speakers:\n"));
        assert!(text.contains("[default, chosen]"), "{text}");
        assert!(text.contains("--output-device"), "{text}");
    }

    #[test]
    fn a_choice_that_matches_nothing_is_called_out() {
        let text = render(Direction::Input, &some(), Some("gone"));
        assert!(
            text.contains("The choice \"gone\" matches no microphone"),
            "{text}"
        );
        assert!(!text.contains("chosen"), "{text}");
        assert!(!render(Direction::Input, &some(), None).contains("matches no"));
    }

    #[test]
    fn an_empty_list_says_so() {
        let text = render(Direction::Output, &[], None);
        assert!(text.contains("(no speaker found)"), "{text}");
    }

    #[test]
    fn directions_have_names_for_messages() {
        assert_eq!(Direction::Input.noun(), "microphone");
        assert_eq!(Direction::Output.noun(), "speaker");
    }
}
