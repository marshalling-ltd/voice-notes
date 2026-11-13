//! Profiles: named combinations of backend, model, thinking and system prompt, so one key press
//! switches between "quick chat", "thinks hard", "developer help" and so on.

use crate::chat::{Backend, DEV_STYLE};
use crate::config::{Config, ProfileConfig};
use crate::speech::find_in_path;
use anyhow::{Result, bail};

/// The model behind `fast` and `dev` when none is configured.
pub const DEFAULT_FAST_MODEL: &str = "qwen3-coder-30b";

/// The profile used when nothing says otherwise.
pub const DEFAULT_PROFILE: &str = "fast";

/// The built-in profiles, in the order they are offered.
pub const BUILTIN: [&str; 5] = ["fast", "dev", "smart", "think", "claude"];

/// One way of talking to an assistant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Profile {
    pub name: String,
    /// One line for the setup screen.
    pub description: String,
    pub backend: Backend,
    /// System prompt; `None` means the default spoken style.
    pub prompt: Option<String>,
}

fn fast_model(config: &Config) -> String {
    config
        .chat_model
        .clone()
        .unwrap_or_else(|| DEFAULT_FAST_MODEL.to_string())
}

/// A built-in profile, if its ingredients are available.
fn builtin(name: &str, config: &Config, claude_installed: bool) -> Option<Profile> {
    let local = |model: String, think: bool| Backend::Local { model, think };
    let (description, backend, prompt) = match name {
        "fast" => (
            "Quick everyday chat",
            local(fast_model(config), false),
            None,
        ),
        "dev" => (
            "Technical answers for developers",
            local(fast_model(config), false),
            Some(DEV_STYLE.to_string()),
        ),
        "smart" => (
            "Slower, nicer answers from a reasoning model (without its thinking)",
            local(config.smart_model.clone()?, false),
            None,
        ),
        "think" => (
            "Thinks before answering: shows the thinking, speaks the answer",
            local(config.smart_model.clone()?, true),
            None,
        ),
        "claude" if claude_installed => ("Claude, with read-only tools", Backend::Claude, None),
        _ => return None,
    };
    Some(Profile {
        name: name.to_string(),
        description: description.to_string(),
        backend,
        prompt,
    })
}

/// Apply a `[profiles.NAME]` table on top of a built-in profile, or build one from scratch.
fn customised(name: &str, pc: &ProfileConfig, base: Option<Profile>, config: &Config) -> Profile {
    let base_backend = base.as_ref().map(|b| b.backend.clone());
    let backend = match (pc.backend.as_deref(), &base_backend) {
        (Some("claude"), _) | (None, Some(Backend::Claude)) => Backend::Claude,
        (_, base_backend) => {
            let (base_model, base_think) = match base_backend {
                Some(Backend::Local { model, think }) => (Some(model.clone()), *think),
                _ => (None, false),
            };
            Backend::Local {
                model: pc
                    .model
                    .clone()
                    .or(base_model)
                    .unwrap_or_else(|| fast_model(config)),
                think: pc.think.unwrap_or(base_think),
            }
        }
    };
    Profile {
        name: name.to_string(),
        description: pc
            .description
            .clone()
            .or_else(|| base.as_ref().map(|b| b.description.clone()))
            .unwrap_or_else(|| "Custom profile".to_string()),
        prompt: pc.prompt.clone().or_else(|| base.and_then(|b| b.prompt)),
        backend,
    }
}

/// Every profile that can be used right now, built-ins first, then your own by name.
/// `claude_installed` says whether the `claude` CLI exists.
#[must_use]
pub fn available_with(config: &Config, claude_installed: bool) -> Vec<Profile> {
    let mut profiles: Vec<Profile> = BUILTIN
        .iter()
        .filter_map(|name| {
            let base = builtin(name, config, claude_installed);
            match config.profiles.get(*name) {
                // An override can also bring a profile to life, e.g. `smart` with its own model.
                Some(pc) if base.is_some() || pc.model.is_some() => {
                    Some(customised(name, pc, base, config))
                }
                _ => base,
            }
        })
        .collect();
    profiles.extend(
        config
            .profiles
            .iter()
            .filter(|(name, _)| !BUILTIN.contains(&name.as_str()))
            .map(|(name, pc)| customised(name, pc, None, config)),
    );
    profiles
}

/// Every profile that can be used on this machine.
#[must_use]
pub fn available(config: &Config) -> Vec<Profile> {
    available_with(config, find_in_path("claude").is_some())
}

/// The profile selected by the config: its `profile`, else `claude` for old configs that chose
/// that backend, else `fast`.
#[must_use]
pub fn default_name(config: &Config) -> String {
    config.profile.clone().unwrap_or_else(|| {
        if config.backend.as_deref() == Some("claude") {
            "claude".to_string()
        } else {
            DEFAULT_PROFILE.to_string()
        }
    })
}

/// Look a profile up by name among `profiles`.
///
/// # Errors
///
/// Returns an error naming the profiles that exist if `name` is not one of them.
pub fn pick(profiles: &[Profile], name: &str) -> Result<Profile> {
    if let Some(profile) = profiles.iter().find(|p| p.name == name) {
        return Ok(profile.clone());
    }
    let names: Vec<&str> = profiles.iter().map(|p| p.name.as_str()).collect();
    bail!(
        "no profile called `{name}` is available (available: {}). `smart` and `think` need a \
         reasoning model: run `voice-notes setup`",
        names.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(profiles: &[Profile]) -> Vec<&str> {
        profiles.iter().map(|p| p.name.as_str()).collect()
    }

    fn with_smart() -> Config {
        Config {
            chat_model: Some("qwen3-coder-30b".into()),
            smart_model: Some("qwen-reasoner".into()),
            ..Config::default()
        }
    }

    /// Without a reasoning model only the profiles that do not need one are offered.
    #[test]
    fn smart_profiles_need_a_reasoning_model() {
        let profiles = available_with(&Config::default(), false);
        assert_eq!(names(&profiles), ["fast", "dev"]);
        let profiles = available_with(&with_smart(), false);
        assert_eq!(names(&profiles), ["fast", "dev", "smart", "think"]);
        let profiles = available_with(&with_smart(), true);
        assert_eq!(
            names(&profiles),
            ["fast", "dev", "smart", "think", "claude"]
        );
    }

    /// Each built-in profile has the model, thinking switch and prompt it promises.
    #[test]
    fn builtin_profiles_do_what_they_say() {
        let profiles = available_with(&with_smart(), true);
        let get = |n: &str| pick(&profiles, n).unwrap();
        let local = |model: &str, think| Backend::Local {
            model: model.into(),
            think,
        };

        assert_eq!(get("fast").backend, local("qwen3-coder-30b", false));
        assert_eq!(get("fast").prompt, None);
        assert_eq!(get("dev").backend, local("qwen3-coder-30b", false));
        assert_eq!(get("dev").prompt.as_deref(), Some(DEV_STYLE));
        assert_eq!(get("smart").backend, local("qwen-reasoner", false));
        assert_eq!(get("think").backend, local("qwen-reasoner", true));
        assert_eq!(get("claude").backend, Backend::Claude);
    }

    /// Built-in profiles fall back to the default fast model when none is saved.
    #[test]
    fn fast_model_default() {
        let fast = pick(&available_with(&Config::default(), false), "fast").unwrap();
        assert_eq!(
            fast.backend,
            Backend::Local {
                model: DEFAULT_FAST_MODEL.into(),
                think: false
            }
        );
    }

    /// A `[profiles.NAME]` table tweaks a built-in profile and keeps what it does not mention.
    #[test]
    fn config_overrides_a_builtin() {
        let mut config = with_smart();
        config.profiles.insert(
            "dev".into(),
            ProfileConfig {
                model: Some("my-coder".into()),
                description: Some("my dev".into()),
                ..ProfileConfig::default()
            },
        );
        let dev = pick(&available_with(&config, false), "dev").unwrap();
        assert_eq!(
            dev.backend,
            Backend::Local {
                model: "my-coder".into(),
                think: false
            }
        );
        assert_eq!(
            dev.prompt.as_deref(),
            Some(DEV_STYLE),
            "the dev prompt is kept"
        );
        assert_eq!(dev.description, "my dev");
    }

    /// Your own profiles are offered after the built-in ones, sorted by name.
    #[test]
    fn custom_profiles_come_last() {
        let mut config = with_smart();
        for (name, model) in [("zed", "z-model"), ("alpha", "a-model")] {
            config.profiles.insert(
                name.into(),
                ProfileConfig {
                    model: Some(model.into()),
                    think: Some(true),
                    prompt: Some("Be brief.".into()),
                    ..ProfileConfig::default()
                },
            );
        }
        let profiles = available_with(&config, false);
        assert_eq!(
            names(&profiles),
            ["fast", "dev", "smart", "think", "alpha", "zed"]
        );
        let alpha = pick(&profiles, "alpha").unwrap();
        assert_eq!(
            alpha.backend,
            Backend::Local {
                model: "a-model".into(),
                think: true
            }
        );
        assert_eq!(alpha.prompt.as_deref(), Some("Be brief."));
        assert_eq!(alpha.description, "Custom profile");
    }

    /// An override can bring `smart` to life on a machine with no detected reasoning model, and a
    /// profile can be pointed at Claude.
    #[test]
    fn overrides_can_create_and_redirect() {
        let mut config = Config::default();
        config.profiles.insert(
            "smart".into(),
            ProfileConfig {
                model: Some("big-model".into()),
                ..ProfileConfig::default()
            },
        );
        config.profiles.insert(
            "helper".into(),
            ProfileConfig {
                backend: Some("claude".into()),
                ..ProfileConfig::default()
            },
        );
        let profiles = available_with(&config, false);
        assert_eq!(
            pick(&profiles, "smart").unwrap().backend,
            Backend::Local {
                model: "big-model".into(),
                think: false
            }
        );
        assert_eq!(pick(&profiles, "helper").unwrap().backend, Backend::Claude);
    }

    /// The launch profile is the configured one, `claude` for old claude configs, else `fast`.
    #[test]
    fn default_profile_name() {
        assert_eq!(default_name(&Config::default()), "fast");
        let claude = Config {
            backend: Some("claude".into()),
            ..Config::default()
        };
        assert_eq!(default_name(&claude), "claude");
        let chosen = Config {
            profile: Some("dev".into()),
            backend: Some("claude".into()),
            ..Config::default()
        };
        assert_eq!(default_name(&chosen), "dev");
    }

    /// An unknown or unavailable profile gives a helpful error.
    #[test]
    fn unknown_profile_lists_the_choices() {
        let profiles = available_with(&Config::default(), false);
        let err = pick(&profiles, "think").unwrap_err().to_string();
        assert!(err.contains("`think`") && err.contains("fast, dev") && err.contains("setup"));
    }
}
