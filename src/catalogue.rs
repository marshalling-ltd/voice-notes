//! The model catalogue: which models exist, where to download them, and what each size class
//! starts with.
//!
//! The catalogue is `models.toml` in the root of the repository, embedded in the program. It is
//! read with [`Catalogue::parse`], which checks everything it can and says what is wrong in plain
//! words, naming the model and the field; all the problems of a file are reported together.
//!
//! ```
//! use voice_notes::catalogue::Catalogue;
//! use voice_notes::hardware::Tier;
//!
//! let catalogue = Catalogue::builtin();
//! let set = catalogue.set(Tier::Light);
//! println!("light: {} and {}", set.speech_to_text.name, set.chat.name);
//! assert!(set.reasoning.is_none());
//! ```

use crate::hardware::Tier;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::OnceLock;
use toml::{Table, Value};

/// The catalogue that ships with the program.
const BUILTIN: &str = include_str!("../models.toml");

/// What a model does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    /// Turns speech into text (Whisper).
    SpeechToText,
    /// Answers in text.
    Chat,
    /// Turns text into speech (Kokoro).
    TextToSpeech,
}

impl Role {
    const ALL: [Self; 3] = [Self::SpeechToText, Self::Chat, Self::TextToSpeech];

    /// The name used in `models.toml`.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::SpeechToText => "speech-to-text",
            Self::Chat => "chat",
            Self::TextToSpeech => "text-to-speech",
        }
    }
}

/// A job in a size class's starting set. Several jobs can use the same kind of model: the
/// reasoning model, which the `smart` and `think` profiles use, is a chat model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Job {
    SpeechToText,
    /// The fast model for everyday answers.
    Chat,
    /// The model that thinks before it answers; only the largest class has room for one.
    Reasoning,
    TextToSpeech,
}

impl Job {
    const ALL: [Self; 4] = [
        Self::SpeechToText,
        Self::Chat,
        Self::Reasoning,
        Self::TextToSpeech,
    ];

    /// The name used in `models.toml`.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::SpeechToText => "speech-to-text",
            Self::Chat => "chat",
            Self::Reasoning => "reasoning",
            Self::TextToSpeech => "text-to-speech",
        }
    }

    /// The kind of model that can do this job.
    #[must_use]
    pub fn role(self) -> Role {
        match self {
            Self::SpeechToText => Role::SpeechToText,
            Self::Chat | Self::Reasoning => Role::Chat,
            Self::TextToSpeech => Role::TextToSpeech,
        }
    }

    fn from_label(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|j| j.label() == text)
    }

    fn required(self) -> bool {
        self != Self::Reasoning
    }
}

/// One downloadable model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model {
    /// Short unique name; sets refer to it.
    pub id: String,
    /// What people see.
    pub name: String,
    pub role: Role,
    /// The smallest size class that should run it.
    pub class: Tier,
    /// Where to download it from, at a fixed commit.
    pub url: String,
    /// The checksum of the file: 64 lowercase hex digits.
    pub sha256: String,
    /// The exact size of the file.
    pub size_bytes: u64,
    /// The licence's name, shown before anything is downloaded.
    pub licence: String,
    /// Where to read the licence.
    pub licence_url: Option<String>,
    /// The model name the stack's server configurations (`stack/profiles`) serve this file
    /// under. The client saves this name as its model once the server serves the file.
    pub served_as: String,
}

impl Model {
    /// The name of the file the download is saved as: the last part of the address.
    #[must_use]
    pub fn file_name(&self) -> &str {
        file_name(&self.url)
    }
}

/// The last part of an address, without a query.
fn file_name(url: &str) -> &str {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    path.rsplit('/').next().unwrap_or(path)
}

/// The ids behind a size class's set, as written in the file.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Set {
    speech_to_text: String,
    chat: String,
    reasoning: Option<String>,
    text_to_speech: String,
}

/// What a size class starts with, with the models looked up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChosenSet<'a> {
    pub speech_to_text: &'a Model,
    pub chat: &'a Model,
    /// `None` for every class except the largest.
    pub reasoning: Option<&'a Model>,
    pub text_to_speech: &'a Model,
}

impl<'a> ChosenSet<'a> {
    /// Every model in the set, by job.
    #[must_use]
    pub fn jobs(&self) -> Vec<(Job, &'a Model)> {
        let mut jobs = vec![
            (Job::SpeechToText, self.speech_to_text),
            (Job::Chat, self.chat),
        ];
        if let Some(model) = self.reasoning {
            jobs.push((Job::Reasoning, model));
        }
        jobs.push((Job::TextToSpeech, self.text_to_speech));
        jobs
    }

    /// The distinct models in the set (the reasoning model can also be another class's chat
    /// model, but within one set nothing appears twice).
    #[must_use]
    // Part of the library's API; the binary compiles this module separately and does not call it.
    #[allow(dead_code)]
    pub fn models(&self) -> Vec<&'a Model> {
        let mut seen = BTreeSet::new();
        self.jobs()
            .into_iter()
            .map(|(_, m)| m)
            .filter(|m| seen.insert(m.id.as_str()))
            .collect()
    }

    /// How many bytes the set takes to download.
    #[must_use]
    // Part of the library's API; the binary compiles this module separately and does not call it.
    #[allow(dead_code)]
    pub fn size_bytes(&self) -> u64 {
        self.models().iter().map(|m| m.size_bytes).sum()
    }
}

/// Everything wrong with a catalogue file, in words a person can act on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatalogueError {
    /// One line per problem.
    pub problems: Vec<String>,
}

impl fmt::Display for CatalogueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.problems.as_slice() {
            [only] => write!(f, "models.toml: {only}"),
            many => {
                write!(f, "models.toml has {} problems:", many.len())?;
                for problem in many {
                    write!(f, "\n  - {problem}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for CatalogueError {}

/// The models and the starting set of every size class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Catalogue {
    models: Vec<Model>,
    sets: BTreeMap<Tier, Set>,
}

const TIERS: [Tier; 4] = [Tier::Minimal, Tier::Light, Tier::Standard, Tier::Power];

const MODEL_FIELDS: [&str; 10] = [
    "id",
    "name",
    "role",
    "class",
    "url",
    "sha256",
    "size_bytes",
    "licence",
    "licence_url",
    "served_as",
];

impl Catalogue {
    /// The catalogue that ships with the program.
    ///
    /// # Panics
    ///
    /// Panics if the embedded `models.toml` is invalid, which a test rules out.
    #[must_use]
    pub fn builtin() -> &'static Self {
        static BUILTIN_CATALOGUE: OnceLock<Catalogue> = OnceLock::new();
        BUILTIN_CATALOGUE.get_or_init(|| {
            Catalogue::parse(BUILTIN).expect("the embedded models.toml is checked by the tests")
        })
    }

    /// Read a catalogue and check it completely.
    ///
    /// # Errors
    ///
    /// Returns every problem found: a missing or malformed field (naming the model and the
    /// field), a duplicate id or address, a set that names a model that does not exist or has the
    /// wrong role, a model that is too big for the class whose set it is in, a size class without
    /// a complete set.
    pub fn parse(text: &str) -> Result<Self, CatalogueError> {
        let document: Table = text.parse().map_err(|e: toml::de::Error| CatalogueError {
            problems: vec![format!("this is not valid TOML: {}", e.message())],
        })?;
        let mut problems = Vec::new();
        for key in document.keys() {
            if !["meta", "model", "sets"].contains(&key.as_str()) {
                problems.push(format!(
                    "unknown section `{key}` (expected `meta`, `model` and `sets`)"
                ));
            }
        }
        let (models, rejected) = parse_models(&document, &mut problems);
        let sets = parse_sets(&document, &models, &rejected, &mut problems);
        if problems.is_empty() {
            Ok(Self { models, sets })
        } else {
            Err(CatalogueError { problems })
        }
    }

    /// Every model, in the order of the file.
    #[must_use]
    // Part of the library's API; the binary compiles this module separately and does not call it.
    #[allow(dead_code)]
    pub fn models(&self) -> &[Model] {
        &self.models
    }

    /// The model with this id.
    #[must_use]
    pub fn model(&self, id: &str) -> Option<&Model> {
        self.models.iter().find(|m| m.id == id)
    }

    /// What `tier` starts with.
    ///
    /// # Panics
    ///
    /// Does not panic for a catalogue that was parsed: parsing requires a set for every class.
    #[must_use]
    pub fn set(&self, tier: Tier) -> ChosenSet<'_> {
        let set = self
            .sets
            .get(&tier)
            .expect("parsing requires a set for every size class");
        let look = |id: &str| {
            self.model(id)
                .expect("parsing requires every set to name existing models")
        };
        ChosenSet {
            speech_to_text: look(&set.speech_to_text),
            chat: look(&set.chat),
            reasoning: set.reasoning.as_deref().map(look),
            text_to_speech: look(&set.text_to_speech),
        }
    }
}

/// The text value of `key`, or a problem saying why not.
fn text_field(table: &Table, key: &str, who: &str, problems: &mut Vec<String>) -> Option<String> {
    match table.get(key) {
        None => {
            problems.push(format!("{who}: missing `{key}`"));
            None
        }
        Some(Value::String(text)) if text.trim().is_empty() => {
            problems.push(format!("{who}: `{key}` is empty"));
            None
        }
        Some(Value::String(text)) => Some(text.clone()),
        Some(other) => {
            problems.push(format!(
                "{who}: `{key}` must be text in quotes, found {}",
                other.type_str()
            ));
            None
        }
    }
}

fn valid_id(id: &str) -> bool {
    id.chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '_' | '-'))
}

/// The name a server serves a model under: letters, digits, `.`, `_` and `-`.
fn valid_served_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

fn valid_sha256(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn valid_url(url: &str) -> bool {
    url.starts_with("https://")
        && !url.chars().any(char::is_whitespace)
        && !file_name(url).is_empty()
}

/// A text field that must be one of `options`, named by `label`.
fn choice_field<T: Copy>(
    table: &Table,
    key: &str,
    who: &str,
    options: &[T],
    label: impl Fn(T) -> &'static str,
    problems: &mut Vec<String>,
) -> Option<T> {
    let text = text_field(table, key, who, problems)?;
    let found = options.iter().copied().find(|o| label(*o) == text);
    if found.is_none() {
        let allowed: Vec<_> = options.iter().map(|o| label(*o)).collect();
        problems.push(format!(
            "{who}: `{key}` is \"{text}\", but must be one of: {}",
            allowed.join(", ")
        ));
    }
    found
}

/// A text field that must pass `valid`, described by `expected` when it does not.
fn checked_field(
    table: &Table,
    key: &str,
    who: &str,
    valid: fn(&str) -> bool,
    expected: &str,
    problems: &mut Vec<String>,
) -> Option<String> {
    let text = text_field(table, key, who, problems)?;
    if valid(&text) {
        Some(text)
    } else {
        problems.push(format!(
            "{who}: `{key}` must be {expected}, found \"{text}\""
        ));
        None
    }
}

fn size_field(table: &Table, who: &str, problems: &mut Vec<String>) -> Option<u64> {
    match table.get("size_bytes") {
        None => {
            problems.push(format!("{who}: missing `size_bytes`"));
            None
        }
        Some(Value::Integer(n)) => {
            let size = u64::try_from(*n).ok().filter(|n| *n > 0);
            if size.is_none() {
                problems.push(format!(
                    "{who}: `size_bytes` must be more than 0, found {n}"
                ));
            }
            size
        }
        Some(other) => {
            problems.push(format!(
                "{who}: `size_bytes` must be a whole number, found {}",
                other.type_str()
            ));
            None
        }
    }
}

/// The optional link to the licence.
fn licence_url_field(table: &Table, who: &str, problems: &mut Vec<String>) -> Option<String> {
    match table.get("licence_url") {
        None => None,
        Some(Value::String(url)) if valid_url(url) => Some(url.clone()),
        Some(other) => {
            problems.push(format!(
                "{who}: `licence_url` must be an https:// address, found {other}"
            ));
            None
        }
    }
}

/// One `[[model]]` entry: `None` if anything about it is wrong (the problems are recorded).
fn parse_model(
    table: &Table,
    who: &str,
    ids: &mut BTreeSet<String>,
    urls: &mut BTreeSet<String>,
    problems: &mut Vec<String>,
) -> Option<Model> {
    let before = problems.len();
    for key in table.keys() {
        if !MODEL_FIELDS.contains(&key.as_str()) {
            problems.push(format!(
                "{who}: unknown field `{key}` (known: {})",
                MODEL_FIELDS.join(", ")
            ));
        }
    }
    let id = checked_field(
        table,
        "id",
        who,
        valid_id,
        "lowercase letters, digits, `.`, `_` and `-`",
        problems,
    );
    let name = text_field(table, "name", who, problems);
    let role = choice_field(table, "role", who, &Role::ALL, Role::label, problems);
    let class = choice_field(table, "class", who, &TIERS, Tier::label, problems);
    let url = checked_field(
        table,
        "url",
        who,
        valid_url,
        "an https:// address ending in a file name",
        problems,
    );
    let sha256 = checked_field(
        table,
        "sha256",
        who,
        valid_sha256,
        "64 lowercase hex digits",
        problems,
    );
    let size_bytes = size_field(table, who, problems);
    let licence = text_field(table, "licence", who, problems);
    let licence_url = licence_url_field(table, who, problems);
    let served_as = checked_field(
        table,
        "served_as",
        who,
        valid_served_name,
        "letters, digits, `.`, `_` and `-`",
        problems,
    );
    if let Some(id) = &id
        && !ids.insert(id.clone())
    {
        problems.push(format!("{who}: the id is used twice"));
    }
    if let Some(url) = &url
        && !urls.insert(url.clone())
    {
        problems.push(format!("{who}: the same `url` is listed twice"));
    }
    if problems.len() > before {
        return None;
    }
    Some(Model {
        id: id?,
        name: name?,
        role: role?,
        class: class?,
        url: url?,
        sha256: sha256?,
        size_bytes: size_bytes?,
        licence: licence?,
        licence_url,
        served_as: served_as?,
    })
}

/// The valid models, and the ids of the entries that had problems (so a set naming one of them
/// is not reported a second time as naming something that does not exist).
fn parse_models(document: &Table, problems: &mut Vec<String>) -> (Vec<Model>, BTreeSet<String>) {
    let entries = match document.get("model") {
        None => {
            problems.push("there are no `[[model]]` entries".to_string());
            return (Vec::new(), BTreeSet::new());
        }
        Some(Value::Array(entries)) => entries,
        Some(other) => {
            problems.push(format!(
                "`model` must be a list of `[[model]]` tables, found {}",
                other.type_str()
            ));
            return (Vec::new(), BTreeSet::new());
        }
    };
    let mut rejected = BTreeSet::new();
    let mut models = Vec::new();
    let (mut ids, mut urls) = (BTreeSet::new(), BTreeSet::new());
    for (index, entry) in entries.iter().enumerate() {
        let Value::Table(table) = entry else {
            problems.push(format!("model #{}: not a table", index + 1));
            continue;
        };
        let who = match table.get("id") {
            Some(Value::String(id)) => format!("model `{id}`"),
            _ => format!("model #{}", index + 1),
        };
        match parse_model(table, &who, &mut ids, &mut urls, problems) {
            Some(model) => models.push(model),
            None => {
                if let Some(Value::String(id)) = table.get("id") {
                    rejected.insert(id.clone());
                }
            }
        }
    }
    (models, rejected)
}

/// One class's set: the models for its jobs, or `None` if anything about it is wrong.
fn parse_set(
    tier: Tier,
    table: &Table,
    models: &[Model],
    rejected: &BTreeSet<String>,
    problems: &mut Vec<String>,
) -> Option<Set> {
    let who = format!("set `{}`", tier.label());
    let mut picked: BTreeMap<Job, &Model> = BTreeMap::new();
    for (key, value) in table {
        let Some(job) = Job::from_label(key) else {
            let allowed: Vec<_> = Job::ALL.iter().map(|j| j.label()).collect();
            problems.push(format!(
                "{who}: unknown job `{key}` (known: {})",
                allowed.join(", ")
            ));
            continue;
        };
        let Value::String(id) = value else {
            problems.push(format!(
                "{who}: `{key}` must be a model id in quotes, found {}",
                value.type_str()
            ));
            continue;
        };
        if rejected.contains(id) {
            continue; // already reported, with the reason
        }
        let Some(model) = models.iter().find(|m| &m.id == id) else {
            problems.push(format!(
                "{who}: `{key}` names \"{id}\", which is not in the catalogue"
            ));
            continue;
        };
        if model.role != job.role() {
            problems.push(format!(
                "{who}: `{key}` needs a {} model, but \"{id}\" is {}",
                job.role().label(),
                model.role.label()
            ));
        } else if model.class > tier {
            problems.push(format!(
                "{who}: \"{id}\" needs the {} class or more, so it does not belong in {}",
                model.class.label(),
                tier.label()
            ));
        } else {
            picked.insert(job, model);
        }
    }
    for job in Job::ALL {
        if job.required() && !table.contains_key(job.label()) {
            problems.push(format!("{who}: missing `{}`", job.label()));
        }
    }
    if let (Some(chat), Some(reasoning)) = (picked.get(&Job::Chat), picked.get(&Job::Reasoning))
        && chat.id == reasoning.id
    {
        problems.push(format!(
            "{who}: `chat` and `reasoning` are the same model (\"{}\"); leave `reasoning` out",
            chat.id
        ));
    }
    // A server serves one model per name, so two jobs of one set cannot share a name.
    let mut names = BTreeSet::new();
    for (job, model) in &picked {
        if !names.insert(model.served_as.as_str()) {
            problems.push(format!(
                "{who}: `{}` is served as \"{}\", which another job of the set already uses",
                job.label(),
                model.served_as
            ));
        }
    }
    let id_of = |job: Job| picked.get(&job).map(|m| m.id.clone());
    Some(Set {
        speech_to_text: id_of(Job::SpeechToText)?,
        chat: id_of(Job::Chat)?,
        reasoning: id_of(Job::Reasoning),
        text_to_speech: id_of(Job::TextToSpeech)?,
    })
}

fn parse_sets(
    document: &Table,
    models: &[Model],
    rejected: &BTreeSet<String>,
    problems: &mut Vec<String>,
) -> BTreeMap<Tier, Set> {
    let mut sets = BTreeMap::new();
    let tables = match document.get("sets") {
        None => {
            problems
                .push("there is no `[sets]` section with a set for each size class".to_string());
            return sets;
        }
        Some(Value::Table(tables)) => tables,
        Some(other) => {
            problems.push(format!(
                "`sets` must be a table, found {}",
                other.type_str()
            ));
            return sets;
        }
    };
    for name in tables.keys() {
        if !TIERS.iter().any(|t| t.label() == name) {
            problems.push(format!(
                "sets: `{name}` is not a size class (minimal, light, standard, power)"
            ));
        }
    }
    for tier in TIERS {
        let Some(Value::Table(table)) = tables.get(tier.label()) else {
            problems.push(format!(
                "set `{}`: missing; every size class needs a set",
                tier.label()
            ));
            continue;
        };
        if let Some(set) = parse_set(tier, table, models, rejected, problems) {
            sets.insert(tier, set);
        }
    }
    sets
}

#[cfg(test)]
mod tests {
    use super::*;

    fn builtin_text() -> String {
        BUILTIN.to_string()
    }

    /// Parse `text` and return the whole error message, which must exist.
    fn error_of(text: &str) -> String {
        match Catalogue::parse(text) {
            Ok(_) => panic!("expected an error, but the catalogue parsed"),
            Err(e) => e.to_string(),
        }
    }

    /// The shipped file with `old` replaced by `new` (which must be there).
    fn edited(old: &str, new: &str) -> String {
        let text = builtin_text();
        assert!(
            text.contains(old),
            "the test is out of date: {old:?} is not in models.toml"
        );
        text.replacen(old, new, 1)
    }

    /// Check that replacing `old` by `new` is rejected with a message containing `expected`.
    fn rejects(old: &str, new: &str, expected: &str) {
        let message = error_of(&edited(old, new));
        assert!(
            message.contains(expected),
            "expected {expected:?} in:\n{message}"
        );
    }

    /// The first `[[model]]` without the line that starts with `field =`.
    fn without_field(field: &str) -> String {
        let text = builtin_text();
        let start = 1 + text.find("\n[[model]]\n").unwrap();
        let end = start + text[start..].find("\n\n").unwrap();
        let block = &text[start..end];
        let kept: Vec<&str> = block
            .lines()
            .filter(|l| !l.starts_with(&format!("{field} =")))
            .collect();
        assert_eq!(
            kept.len() + 1,
            block.lines().count(),
            "{field} is not in the first model"
        );
        format!("{}{}{}", &text[..start], kept.join("\n"), &text[end..])
    }

    // ---- the shipped catalogue

    #[test]
    fn the_embedded_catalogue_is_valid() {
        let catalogue = Catalogue::parse(BUILTIN).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(catalogue.models().len(), 8);
        assert_eq!(Catalogue::builtin(), &catalogue);
    }

    /// The default sets of the project's hardware table.
    #[test]
    fn the_default_sets_follow_the_hardware_table() {
        let c = Catalogue::builtin();
        let ids = |tier| {
            let set = c.set(tier);
            (
                set.speech_to_text.id.as_str(),
                set.chat.id.as_str(),
                set.reasoning.map(|m| m.id.as_str()),
                set.text_to_speech.id.as_str(),
            )
        };
        let kokoro = "kokoro-v1_0";
        let turbo = "whisper-large-v3-turbo-q5_0";
        assert_eq!(
            ids(Tier::Minimal),
            ("whisper-small", "qwen3-4b", None, kokoro)
        );
        assert_eq!(ids(Tier::Light), (turbo, "qwen3-8b", None, kokoro));
        assert_eq!(ids(Tier::Standard), (turbo, "qwen3-14b", None, kokoro));
        assert_eq!(
            ids(Tier::Power),
            (turbo, "qwen3-coder-30b-a3b", Some("qwen3.8-27b"), kokoro)
        );
    }

    /// The ticket asks for a test that every entry is complete.
    #[test]
    fn every_entry_is_complete_and_pinned() {
        for m in Catalogue::builtin().models() {
            let who = &m.id;
            assert!(!m.name.trim().is_empty(), "{who}: name");
            assert!(m.size_bytes > 1_000_000, "{who}: a model is more than 1 MB");
            assert!(valid_sha256(&m.sha256), "{who}: sha256");
            assert!(!m.licence.trim().is_empty(), "{who}: licence");
            assert!(
                m.licence_url.as_deref().is_some_and(valid_url),
                "{who}: the licence must come with a link, it is shown before downloading"
            );
            assert!(!m.file_name().is_empty(), "{who}: file name");
            assert!(valid_served_name(&m.served_as), "{who}: served_as");
            // A commit, not a branch: see stack/MODEL_HOSTING.md.
            let pinned = m
                .url
                .split('/')
                .any(|part| part.len() == 40 && valid_sha1(part));
            assert!(
                pinned,
                "{who}: the url must name a commit, not a branch: {}",
                m.url
            );
        }
    }

    fn valid_sha1(text: &str) -> bool {
        text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    }

    #[test]
    fn every_model_is_in_some_set() {
        let c = Catalogue::builtin();
        let used: BTreeSet<&str> = TIERS
            .iter()
            .flat_map(|t| c.set(*t).models().into_iter().map(|m| m.id.as_str()))
            .collect();
        for m in c.models() {
            assert!(used.contains(m.id.as_str()), "{} is in no set", m.id);
        }
    }

    /// Downloads go into one folder under the file's name, so two models must not share a name.
    #[test]
    fn file_names_are_unique() {
        let mut seen = BTreeSet::new();
        for m in Catalogue::builtin().models() {
            assert!(
                seen.insert(m.file_name()),
                "{} reuses the file name {}",
                m.id,
                m.file_name()
            );
        }
    }

    #[test]
    fn bigger_classes_download_more() {
        let c = Catalogue::builtin();
        let sizes: Vec<u64> = TIERS.iter().map(|t| c.set(*t).size_bytes()).collect();
        assert!(sizes.windows(2).all(|w| w[0] < w[1]), "{sizes:?}");
    }

    #[test]
    fn a_set_counts_each_model_once() {
        let c = Catalogue::builtin();
        let power = c.set(Tier::Power);
        assert_eq!(power.jobs().len(), 4);
        assert_eq!(power.models().len(), 4);
        let by_hand: u64 = power.models().iter().map(|m| m.size_bytes).sum();
        assert_eq!(power.size_bytes(), by_hand);
        assert_eq!(
            c.set(Tier::Minimal).jobs().len(),
            3,
            "no reasoning model below power"
        );
    }

    #[test]
    fn models_can_be_found_by_id_and_expose_their_file_name() {
        let c = Catalogue::builtin();
        let m = c.model("qwen3-4b").unwrap();
        assert_eq!(m.file_name(), "Qwen3-4B-Q4_K_M.gguf");
        assert_eq!(m.role, Role::Chat);
        assert!(c.model("no-such-model").is_none());
        assert_eq!(file_name("https://h/a/b.bin?download=true#x"), "b.bin");
    }

    /// The stack's own catalogue files (used by the containers and the macOS script) describe
    /// some of the same files. They must agree with this catalogue, or one of them is stale.
    #[test]
    fn the_stack_files_agree_with_this_catalogue() {
        let stack = [
            include_str!("../stack/ci-models.toml"),
            include_str!("../stack/nightly-models.toml"),
            include_str!("../stack/profiles/minimal.toml"),
            include_str!("../stack/profiles/light.toml"),
            include_str!("../stack/profiles/standard.toml"),
            include_str!("../stack/profiles/power.toml"),
        ];
        let catalogue = Catalogue::builtin();
        let mut agreed = BTreeSet::new();
        for text in stack {
            let document: Table = text.parse().unwrap();
            for entry in document["model"].as_array().unwrap() {
                let url = entry["url"].as_str().unwrap();
                let Some(ours) = catalogue
                    .models()
                    .iter()
                    .find(|m| m.file_name() == file_name(url))
                else {
                    continue;
                };
                assert_eq!(
                    ours.url, url,
                    "{}: another address for the same file",
                    ours.id
                );
                assert_eq!(
                    ours.sha256,
                    entry["sha256"].as_str().unwrap(),
                    "{}",
                    ours.id
                );
                assert_eq!(
                    i64::try_from(ours.size_bytes).unwrap(),
                    entry["size_bytes"].as_integer().unwrap(),
                    "{}",
                    ours.id
                );
                agreed.insert(ours.id.clone());
            }
        }
        assert!(
            agreed.len() >= 6,
            "the cross-check compared too little: {agreed:?}"
        );
    }

    // ---- what the parser rejects

    #[test]
    fn every_required_field_is_reported_when_missing() {
        for field in [
            "id",
            "name",
            "role",
            "class",
            "url",
            "sha256",
            "size_bytes",
            "licence",
        ] {
            let message = error_of(&without_field(field));
            assert!(
                message.contains(&format!("missing `{field}`")),
                "{field}: {message}"
            );
        }
    }

    #[test]
    fn the_licence_link_is_optional_but_must_be_a_link() {
        Catalogue::parse(&without_field("licence_url")).unwrap();
        rejects(
            "licence_url = \"https://github.com/openai/whisper/blob/main/LICENSE\"",
            "licence_url = \"see the licence\"",
            "`licence_url` must be an https:// address",
        );
    }

    #[test]
    fn a_message_names_the_model_and_the_field() {
        let message = error_of(&without_field("sha256"));
        assert!(
            message.contains("model `whisper-small`: missing `sha256`"),
            "{message}"
        );
        let message = error_of(&without_field("id"));
        assert!(message.contains("model #1: missing `id`"), "{message}");
    }

    #[test]
    fn malformed_checksums_are_rejected() {
        let good = "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b";
        rejects(good, "1be3a9b2", "`sha256` must be 64 lowercase hex digits");
        rejects(
            good,
            &good.to_uppercase(),
            "`sha256` must be 64 lowercase hex digits",
        );
        rejects(
            good,
            &format!("{}zz", &good[2..]),
            "`sha256` must be 64 lowercase hex digits",
        );
    }

    #[test]
    fn unknown_roles_and_classes_list_the_choices() {
        rejects(
            "role = \"speech-to-text\"",
            "role = \"transcribe\"",
            "`role` is \"transcribe\", but must be one of: speech-to-text, chat, text-to-speech",
        );
        rejects(
            "class = \"minimal\"",
            "class = \"tiny\"",
            "`class` is \"tiny\", but must be one of: minimal, light, standard, power",
        );
    }

    #[test]
    fn sizes_must_be_positive_whole_numbers() {
        rejects(
            "size_bytes = 487601967",
            "size_bytes = \"big\"",
            "`size_bytes` must be a whole number, found string",
        );
        rejects(
            "size_bytes = 487601967",
            "size_bytes = 0",
            "`size_bytes` must be more than 0, found 0",
        );
        rejects(
            "size_bytes = 487601967",
            "size_bytes = -5",
            "`size_bytes` must be more than 0, found -5",
        );
        rejects(
            "size_bytes = 487601967",
            "size_bytes = 1.5",
            "`size_bytes` must be a whole number, found float",
        );
    }

    #[test]
    fn addresses_must_be_https_and_end_in_a_file_name() {
        let url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small.bin";
        rejects(
            url,
            &url.replace("https", "http"),
            "`url` must be an https:// address ending in a file name",
        );
        rejects(
            url,
            "https://huggingface.co/ggerganov/whisper.cpp/",
            "ending in a file name",
        );
        rejects(
            url,
            &url.replace("ggml-small", "ggml small"),
            "ending in a file name",
        );
        rejects(url, "ftp://host/file.bin", "ending in a file name");
    }

    #[test]
    fn other_fields_must_have_content_and_the_right_type() {
        rejects("licence = \"MIT\"", "licence = \"\"", "`licence` is empty");
        rejects(
            "name = \"Whisper small\"",
            "name = 7",
            "`name` must be text in quotes, found integer",
        );
        rejects(
            "id = \"whisper-small\"",
            "id = \"Whisper Small\"",
            "`id` must be lowercase letters, digits",
        );
    }

    #[test]
    fn unknown_fields_are_reported_so_typos_do_not_pass_silently() {
        rejects(
            "licence = \"MIT\"",
            "licence = \"MIT\"\nlicense = \"MIT\"",
            "unknown field `license`",
        );
        rejects("[meta]", "[metadata]", "unknown section `metadata`");
    }

    #[test]
    fn duplicates_are_rejected() {
        rejects(
            "id = \"qwen3-8b\"",
            "id = \"qwen3-4b\"",
            "the id is used twice",
        );
        let c = Catalogue::builtin();
        let (eight, fourteen) = (
            &c.model("qwen3-8b").unwrap().url,
            &c.model("qwen3-14b").unwrap().url,
        );
        rejects(eight, fourteen, "the same `url` is listed twice");
    }

    #[test]
    fn broken_files_give_a_clear_error() {
        assert!(error_of("this is [not toml").contains("not valid TOML"));
        let empty = error_of("");
        assert!(
            empty.contains("no `[[model]]` entries") && empty.contains("no `[sets]` section"),
            "{empty}"
        );
        assert!(error_of("model = 3\n").contains("`model` must be a list"));
    }

    // ---- what the sets must satisfy

    #[test]
    fn a_set_must_name_a_model_that_exists() {
        rejects(
            "chat = \"qwen3-4b\"",
            "chat = \"qwen3-4\"",
            "`chat` names \"qwen3-4\", which is not in the catalogue",
        );
    }

    #[test]
    fn a_job_needs_a_model_with_the_right_role() {
        rejects(
            "speech-to-text = \"whisper-small\"",
            "speech-to-text = \"qwen3-4b\"",
            "`speech-to-text` needs a speech-to-text model, but \"qwen3-4b\" is chat",
        );
    }

    #[test]
    fn a_model_too_big_for_a_class_cannot_be_in_its_set() {
        rejects(
            "chat = \"qwen3-4b\"",
            "chat = \"qwen3-8b\"",
            "set `minimal`: \"qwen3-8b\" needs the light class or more, so it does not belong in minimal",
        );
    }

    #[test]
    fn every_class_needs_a_complete_set() {
        let text = edited(
            "[sets.light]\nspeech-to-text = \"whisper-large-v3-turbo-q5_0\"\nchat = \"qwen3-8b\"\ntext-to-speech = \"kokoro-v1_0\"\n",
            "",
        );
        assert!(error_of(&text).contains("set `light`: missing; every size class needs a set"));
        let text = edited("chat = \"qwen3-4b\"\ntext-to-speech", "text-to-speech");
        assert!(error_of(&text).contains("set `minimal`: missing `chat`"));
        rejects(
            "[sets.power]",
            "[sets.extreme]\nchat = \"x\"\n[sets.power]",
            "`extreme` is not a size class",
        );
        rejects(
            "reasoning = \"qwen3.8-27b\"",
            "thinking = \"qwen3.8-27b\"",
            "unknown job `thinking`",
        );
        rejects(
            "chat = \"qwen3-8b\"",
            "chat = 8",
            "`chat` must be a model id in quotes, found integer",
        );
    }

    #[test]
    fn the_reasoning_model_must_differ_from_the_chat_model() {
        rejects(
            "reasoning = \"qwen3.8-27b\"",
            "reasoning = \"qwen3-coder-30b-a3b\"",
            "`chat` and `reasoning` are the same model",
        );
    }

    // ---- reporting

    #[test]
    fn all_problems_are_reported_together() {
        let text = builtin_text()
            .replacen("role = \"chat\"", "role = \"talk\"", 1)
            .replacen("size_bytes = 487601967", "size_bytes = 0", 1);
        let message = error_of(&text);
        // Two mistakes, two lines: the sets that name those models do not complain again.
        assert!(
            message.starts_with("models.toml has 2 problems:"),
            "{message}"
        );
        assert!(
            message.contains("\n  - model `qwen3-4b`: `role` is \"talk\""),
            "{message}"
        );
        assert!(
            message.contains("\n  - model `whisper-small`: `size_bytes` must be more than 0"),
            "{message}"
        );
    }

    #[test]
    fn a_single_problem_is_one_line() {
        let message = error_of(&without_field("name"));
        assert!(!message.contains('\n'), "{message}");
        assert!(
            message.starts_with("models.toml: model `whisper-small`: missing `name`"),
            "{message}"
        );
    }

    /// A model that is itself wrong is reported once, not again by every set that names it.
    #[test]
    fn a_broken_model_is_not_reported_again_by_the_sets_that_use_it() {
        let message = error_of(&edited("size_bytes = 487601967", "size_bytes = 0"));
        assert_eq!(message.matches("whisper-small").count(), 1, "{message}");
    }

    // ---- served_as

    #[test]
    fn the_model_names_clients_save_are_the_ones_the_client_already_knows() {
        let c = Catalogue::builtin();
        let served = |id: &str| c.model(id).unwrap().served_as.as_str();
        assert_eq!(
            served("qwen3-coder-30b-a3b"),
            "qwen3-coder-30b",
            "the client's default fast model"
        );
        assert_eq!(
            served("qwen3.8-27b"),
            "qwen-reasoner",
            "what the smart and think profiles ask for"
        );
        assert_eq!(
            served("whisper-small"),
            "whisper",
            "the client's default speech-to-text model"
        );
        assert_eq!(served("whisper-large-v3-turbo-q5_0"), "whisper");
        assert_eq!(served("kokoro-v1_0"), "kokoro");
    }

    /// The names promised here must be the names the stack's servers really use, or a model
    /// downloaded through the app would never be found by the server.
    #[test]
    fn served_names_match_the_stacks_llama_swap_configs() {
        let configs = [
            include_str!("../stack/profiles/minimal.yaml"),
            include_str!("../stack/profiles/light.yaml"),
            include_str!("../stack/profiles/standard.yaml"),
            include_str!("../stack/profiles/power.yaml"),
        ];
        let catalogue = Catalogue::builtin();
        let mut checked = BTreeSet::new();
        for yaml in configs {
            let models = yaml.split("\ngroups:").next().unwrap();
            let mut current = "";
            for line in models.lines() {
                if let Some(name) = line.strip_prefix("  ").and_then(|l| l.strip_suffix(':'))
                    && !name.contains(' ')
                    && !name.starts_with('#')
                {
                    current = name;
                }
                let Some(path) = line.trim().strip_prefix("-m /models/") else {
                    continue;
                };
                let Some(model) = catalogue.models().iter().find(|m| m.file_name() == path) else {
                    continue;
                };
                assert_eq!(
                    model.served_as, current,
                    "{} is served as `{current}` in the stack",
                    model.id
                );
                checked.insert(model.id.clone());
            }
        }
        assert!(
            checked.len() >= 6,
            "the cross-check compared too little: {checked:?}"
        );
    }

    #[test]
    fn served_as_is_required_and_must_be_a_plausible_name() {
        let message = error_of(&without_field("served_as"));
        assert!(
            message.contains("model `whisper-small`: missing `served_as`"),
            "{message}"
        );
        rejects(
            "served_as = \"qwen3-4b\"",
            "served_as = \"qwen 3\"",
            "`served_as` must be letters, digits",
        );
        rejects(
            "served_as = \"qwen3-4b\"",
            "served_as = \"\"",
            "`served_as` is empty",
        );
    }

    /// A server serves one model per name, so two jobs of a set cannot have the same one.
    #[test]
    fn two_jobs_of_a_set_cannot_share_a_served_name() {
        rejects(
            "served_as = \"qwen-reasoner\"",
            "served_as = \"qwen3-coder-30b\"",
            "is served as \"qwen3-coder-30b\", which another job of the set already uses",
        );
    }

    #[test]
    fn jobs_and_roles_have_stable_names() {
        assert_eq!(Job::Reasoning.label(), "reasoning");
        assert_eq!(Job::Reasoning.role(), Role::Chat);
        assert_eq!(Job::Chat.role(), Role::Chat);
        assert_eq!(Role::TextToSpeech.label(), "text-to-speech");
        assert!(Job::Chat.required() && !Job::Reasoning.required());
    }
}
