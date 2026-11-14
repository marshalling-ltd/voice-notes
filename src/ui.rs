//! Full-screen terminal interface built with ratatui.
//!
//! ```text
//! ╭ Voice Notes ───────────────────────────────────────────────╮
//! │ ● LISTENING   00:12   ● speaking                clean: off │
//! ╰────────────────────────────────────────────────────────────╯
//! ╭ Input ─────────────────────────────────────────────────────╮
//! │ ██████████████████┃████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
//! │ ▂▃▅▇▅▃▂▁▁▂▅▇█▇▅▃▂▁                                         │
//! ╰────────────────────────────────────────────────────────────╯
//! ╭ Transcript ────────────────────────────────────────────────╮
//! │ [00:03] Here it is, let's start again.                     │
//! │ [00:09] … transcribing                                     │
//! ╰────────────────────────────────────────────────────────────╯
//!  Enter start/stop · c clean · ↑↓ scroll · q quit
//! ```

use crate::audio::AudioRecorder;
use crate::chat::{Conversation, ReplyDelta};
use crate::config::Config;
use crate::endpoints::Endpoints;
use crate::pipeline::{Job, Outcome, Settings, spawn_transcriber};
use crate::setup::{self, Choice, Findings};
use crate::speech::{SentenceSplitter, Speaker};
use crate::transcribe::{TranscribeOptions, clean_transcript};
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph, Sparkline, Wrap};
use ratatui::{DefaultTerminal, Frame};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::task::JoinHandle;

const ACCENT: Color = Color::Cyan;
const GOOD: Color = Color::Green;
const WARN: Color = Color::Yellow;
const BAD: Color = Color::Red;
const DIM: Color = Color::DarkGray;

/// Shown whenever the Claude backend is in use: its replies come from Anthropic's servers.
const CLOUD_NOTICE: &str = "what you say is sent to Anthropic";

/// How many level samples the sparkline remembers.
const HISTORY: usize = 240;
/// Input level that fills the meter before the gate is known.
const DEFAULT_SCALE: f32 = 0.05;

/// Where the session is in its life cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Nothing recorded yet.
    Idle,
    /// Measuring background noise.
    Calibrating,
    /// Recording and transcribing as you speak.
    Listening,
    /// Recording stopped; waiting for the last transcriptions.
    Finishing,
    /// Waiting for the chat model to polish the transcript.
    Cleaning,
    /// Waiting for the assistant to answer (talk mode).
    Thinking,
    /// The reply (or the notes) is being read aloud.
    Speaking,
    /// Everything finished; ready for the next turn.
    Done,
}

/// What the app is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Dictate notes.
    Notes,
    /// Hold a spoken conversation with the assistant.
    Talk,
}

/// The assistant's answer to one turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reply {
    Pending,
    /// The answer so far, still being written.
    Partial(String),
    Text(String),
    Failed(String),
}

/// One question and answer in talk mode.
#[derive(Clone, Debug)]
pub struct Turn {
    pub user: String,
    pub reply: Reply,
    /// The model's reasoning so far, shown while it works (the `think` profile).
    pub thinking: String,
}

/// What a transcribed utterance looks like on screen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EntryState {
    Pending,
    Text(String),
    Dropped(String),
    Failed(String),
}

/// One utterance in the transcript pane.
#[derive(Clone, Debug)]
pub struct Entry {
    pub id: usize,
    pub at: Duration,
    pub state: EntryState,
}

/// A request from the keyboard that needs the recorder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    Start,
    Stop,
    /// Cut off speech that is playing.
    Skip,
    /// Read the notes (or the last reply) aloud.
    ReadAloud,
    /// Forget the conversation so far.
    NewConversation,
    /// Look around for models and services (the setup screen).
    OpenSetup,
    /// Save the recommended settings.
    SaveSetup,
    /// Close setup without saving, and stop asking at launch.
    SkipSetup,
    /// Switch to the named profile.
    SwitchProfile(String),
}

/// Turns a saved config into the settings the session runs with.
///
/// The second argument is a profile picked while running; `None` means the launch profile.
pub type Resolver = Arc<dyn Fn(&Config, Option<&str>) -> Result<Settings> + Send + Sync>;

/// The setup screen's content.
// One value lives in `App` at a time, so the size difference between variants is harmless.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SetupState {
    /// Still looking around.
    Probing,
    /// Findings are in; `config` is what saving would write, `recommended` is what setup chose
    /// by itself, and `models` are the model ids the server offers.
    Ready {
        lines: Vec<String>,
        config: Config,
        recommended: Config,
        models: Vec<String>,
    },
    /// Saved; shows the confirmation.
    Done(String),
}

/// The model chooser: one row per job, with the focused row highlighted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chooser {
    pub choices: Vec<Choice>,
    pub focus: usize,
}

/// Everything the screen shows.
// The bools are independent facts (speaking, clean, quit, speech on), not one state.
#[allow(clippy::struct_excessive_bools)]
pub struct App {
    pub phase: Phase,
    pub entries: Vec<Entry>,
    pub level: f32,
    pub gate: Option<f32>,
    pub speaking: bool,
    pub history: Vec<u64>,
    pub clean: bool,
    pub cleaned: Option<String>,
    pub notice: Option<String>,
    pub scroll: usize,
    pub elapsed: Duration,
    pub quit: bool,
    pub mode: Mode,
    pub turns: Vec<Turn>,
    pub backend_label: String,
    pub speak_enabled: bool,
    /// What you say is sent off this machine (the Claude backend); shown on screen.
    pub cloud: bool,
    /// The setup screen, when it is open.
    pub setup: Option<SetupState>,
    /// The model chooser, opened from the setup screen.
    pub chooser: Option<Chooser>,
    /// The profile in use, and every profile `p` can switch to.
    pub profile: String,
    pub profiles: Vec<String>,
    next_id: usize,
    started: Option<Instant>,
}

impl App {
    #[must_use]
    pub fn new(clean: bool) -> Self {
        Self {
            phase: Phase::Idle,
            entries: Vec::new(),
            level: 0.0,
            gate: None,
            speaking: false,
            history: Vec::new(),
            clean,
            cleaned: None,
            notice: None,
            scroll: 0,
            elapsed: Duration::ZERO,
            quit: false,
            mode: Mode::Notes,
            turns: Vec::new(),
            backend_label: String::new(),
            setup: None,
            chooser: None,
            profile: String::new(),
            profiles: Vec::new(),
            speak_enabled: true,
            cloud: false,
            next_id: 0,
            started: None,
        }
    }

    /// Configure talk mode: start in it if `talk`, show `backend_label`, and speak replies if
    /// `speak_enabled`.
    #[must_use]
    pub fn with_talk(mut self, talk: bool, backend_label: String, speak_enabled: bool) -> Self {
        if talk {
            self.mode = Mode::Talk;
        }
        self.backend_label = backend_label;
        self.speak_enabled = speak_enabled;
        self
    }

    /// Nothing is happening, so the user may start, switch mode or read aloud.
    #[must_use]
    pub fn idle(&self) -> bool {
        matches!(self.phase, Phase::Idle | Phase::Done)
    }

    /// Add the user's words as a new turn awaiting an answer.
    pub fn push_turn(&mut self, user: String) {
        self.turns.push(Turn {
            user,
            reply: Reply::Pending,
            thinking: String::new(),
        });
        self.scroll = 0;
    }

    /// Record the assistant's answer to the latest turn.
    pub fn set_reply(&mut self, reply: Reply) {
        if let Some(turn) = self.turns.last_mut() {
            turn.reply = reply;
        }
    }

    /// Mark whether what you say is sent off this machine, so the screen can say so.
    #[must_use]
    pub fn with_cloud(mut self, cloud: bool) -> Self {
        self.cloud = cloud;
        self
    }

    /// Show which profile is in use and which others can be switched to.
    #[must_use]
    pub fn with_profiles(mut self, profile: String, profiles: Vec<String>) -> Self {
        self.profile = profile;
        self.profiles = profiles;
        self
    }

    /// The profile after the current one, wrapping around; `None` if there is no choice.
    #[must_use]
    pub fn next_profile(&self) -> Option<String> {
        if self.profiles.len() < 2 {
            return None;
        }
        let at = self.profiles.iter().position(|p| *p == self.profile);
        let next = at.map_or(0, |i| (i + 1) % self.profiles.len());
        Some(self.profiles[next].clone())
    }

    /// Add streamed reasoning to the latest turn.
    pub fn append_thinking(&mut self, delta: &str) {
        if let Some(turn) = self.turns.last_mut() {
            turn.thinking.push_str(delta);
        }
    }

    /// Add streamed text to the latest turn's answer.
    pub fn append_reply(&mut self, delta: &str) {
        if let Some(turn) = self.turns.last_mut() {
            match &mut turn.reply {
                Reply::Partial(text) => text.push_str(delta),
                other => *other = Reply::Partial(delta.to_string()),
            }
        }
    }

    /// Text that `r` reads aloud: the notes, or the assistant's latest answer.
    #[must_use]
    pub fn readable(&self) -> Option<String> {
        match self.mode {
            Mode::Notes => Some(self.final_text()).filter(|t| !t.is_empty()),
            Mode::Talk => self.turns.iter().rev().find_map(|t| match &t.reply {
                Reply::Text(text) => Some(text.clone()),
                _ => None,
            }),
        }
    }

    /// True while the microphone is open.
    #[must_use]
    pub fn recording(&self) -> bool {
        matches!(self.phase, Phase::Calibrating | Phase::Listening)
    }

    /// Number of utterances still waiting for text.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.state == EntryState::Pending)
            .count()
    }

    /// Begin a fresh note, discarding the previous one.
    pub fn begin(&mut self) {
        self.entries.clear();
        self.cleaned = None;
        self.notice = None;
        self.scroll = 0;
        self.level = 0.0;
        self.gate = None;
        self.speaking = false;
        self.history.clear();
        self.next_id = 0;
        self.elapsed = Duration::ZERO;
        self.started = Some(Instant::now());
        self.phase = Phase::Calibrating;
    }

    /// Register a new utterance and return its id.
    pub fn queue(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(Entry {
            id,
            at: self.elapsed,
            state: EntryState::Pending,
        });
        id
    }

    /// Apply a finished transcription.
    pub fn on_outcome(&mut self, outcome: Outcome) {
        let id = outcome.id();
        let state = match outcome {
            Outcome::Text { text, .. } => EntryState::Text(text),
            Outcome::Dropped { text, .. } => EntryState::Dropped(text),
            Outcome::Failed { error, .. } => EntryState::Failed(error),
        };
        if let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) {
            entry.state = state;
        }
    }

    /// Update the live meter and the timer.
    pub fn on_tick(&mut self, level: f32, gate: Option<f32>, speaking: bool) {
        self.level = level;
        self.gate = gate;
        self.speaking = speaking;
        if self.phase == Phase::Calibrating && gate.is_some() {
            self.phase = Phase::Listening;
        }
        self.history.push(spark_value(meter_ratio(level, gate)));
        if self.history.len() > HISTORY {
            self.history.remove(0);
        }
        if let Some(start) = self.started {
            self.elapsed = start.elapsed();
        }
    }

    /// Recording has stopped; freeze the meter.
    pub fn stopped(&mut self) {
        self.phase = Phase::Finishing;
        self.speaking = false;
        self.level = 0.0;
    }

    /// Accepted text of every utterance, in order.
    #[must_use]
    pub fn transcript(&self) -> String {
        self.entries
            .iter()
            .filter_map(|e| match &e.state {
                EntryState::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// What to print when the program exits: the cleaned text if there is any.
    #[must_use]
    pub fn final_text(&self) -> String {
        self.cleaned.clone().unwrap_or_else(|| self.transcript())
    }

    /// Handle a key press; returns what the recorder should do, if anything.
    pub fn on_key(&mut self, key: KeyEvent) -> Option<Action> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        if self.setup.is_some() {
            return self.on_setup_key(key);
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => self.quit = true,
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            KeyCode::Char('u') if self.idle() => return Some(Action::OpenSetup),
            KeyCode::Char('p') if self.idle() => {
                return self.next_profile().map(Action::SwitchProfile);
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                return match self.phase {
                    Phase::Idle | Phase::Done => Some(Action::Start),
                    Phase::Calibrating | Phase::Listening => Some(Action::Stop),
                    Phase::Speaking => Some(Action::Skip),
                    Phase::Finishing | Phase::Cleaning | Phase::Thinking => None,
                };
            }
            KeyCode::Char('t') if self.idle() => {
                self.mode = match self.mode {
                    Mode::Notes => Mode::Talk,
                    Mode::Talk => Mode::Notes,
                };
                self.scroll = 0;
            }
            KeyCode::Char('r') if self.idle() => {
                return self.readable().map(|_| Action::ReadAloud);
            }
            KeyCode::Char('n') if self.idle() && self.mode == Mode::Talk => {
                self.turns.clear();
                self.scroll = 0;
                return Some(Action::NewConversation);
            }
            KeyCode::Char('c') => self.clean = !self.clean,
            KeyCode::Up | KeyCode::Char('k') => self.scroll_up(1),
            KeyCode::Down | KeyCode::Char('j') => self.scroll_down(1),
            KeyCode::PageUp => self.scroll_up(5),
            KeyCode::PageDown => self.scroll_down(5),
            _ => {}
        }
        None
    }

    /// Keys while the setup screen is open.
    fn on_setup_key(&mut self, key: KeyEvent) -> Option<Action> {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.quit = true;
            return None;
        }
        if key.code == KeyCode::Char('q') {
            self.quit = true;
            return None;
        }
        if self.chooser.is_some() {
            self.on_chooser_key(key);
            return None;
        }
        match &self.setup {
            None => None,
            Some(SetupState::Probing) => match key.code {
                KeyCode::Char('s') | KeyCode::Esc => Some(Action::SkipSetup),
                _ => None,
            },
            Some(SetupState::Done(_)) => {
                self.setup = None;
                None
            }
            Some(SetupState::Ready { config, .. }) => {
                let nothing_to_save = *config == Config::default();
                match key.code {
                    KeyCode::Enter | KeyCode::Char('y') if nothing_to_save => {
                        self.setup = None;
                        None
                    }
                    KeyCode::Enter | KeyCode::Char('y') => Some(Action::SaveSetup),
                    KeyCode::Char('s') | KeyCode::Esc => Some(Action::SkipSetup),
                    KeyCode::Char('c') => {
                        self.open_chooser();
                        None
                    }
                    _ => None,
                }
            }
        }
    }

    /// Open the model chooser from the setup screen, showing what is currently picked. Does
    /// nothing if the server offered no models to choose between.
    fn open_chooser(&mut self) {
        let Some(SetupState::Ready {
            config,
            recommended,
            models,
            ..
        }) = &self.setup
        else {
            return;
        };
        let mut choices = setup::choices(models, recommended);
        if choices.is_empty() {
            return;
        }
        setup::preselect(&mut choices, config);
        self.chooser = Some(Chooser { choices, focus: 0 });
    }

    /// Keys while the model chooser is open.
    fn on_chooser_key(&mut self, key: KeyEvent) {
        let Some(mut chooser) = self.chooser.take() else {
            return;
        };
        let last = chooser.choices.len().saturating_sub(1);
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => chooser.focus = chooser.focus.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => chooser.focus = (chooser.focus + 1).min(last),
            KeyCode::Right | KeyCode::Tab | KeyCode::Char('l' | ' ') => {
                if let Some(choice) = chooser.choices.get_mut(chooser.focus) {
                    choice.next();
                }
            }
            KeyCode::Left | KeyCode::Char('h') => {
                if let Some(choice) = chooser.choices.get_mut(chooser.focus) {
                    choice.previous();
                }
            }
            KeyCode::Enter => {
                if let Some(SetupState::Ready { lines, config, .. }) = &mut self.setup {
                    *config = setup::apply_choices(config, &chooser.choices);
                    refresh_save_block(lines, config);
                }
                return; // the chooser closes
            }
            KeyCode::Esc => return, // the chooser closes, nothing changes
            _ => {}
        }
        self.chooser = Some(chooser);
    }

    /// Update what the header shows after the settings changed.
    pub fn reconfigure(&mut self, backend_label: String, speak_enabled: bool, cloud: bool) {
        self.backend_label = backend_label;
        self.speak_enabled = speak_enabled;
        self.cloud = cloud;
    }

    fn scroll_up(&mut self, n: usize) {
        let items = match self.mode {
            Mode::Notes => self.entries.len(),
            Mode::Talk => self.turns.len(),
        };
        self.scroll = (self.scroll + n).min(items.saturating_sub(1));
    }

    fn scroll_down(&mut self, n: usize) {
        self.scroll = self.scroll.saturating_sub(n);
    }
}

/// Map an input level onto 0.0-1.0 of the meter; the gate sits a third of the way along.
fn meter_ratio(level: f32, gate: Option<f32>) -> f32 {
    (level / meter_scale(gate)).clamp(0.0, 1.0)
}

fn meter_scale(gate: Option<f32>) -> f32 {
    gate.map_or(DEFAULT_SCALE, |g| (g * 3.0).max(DEFAULT_SCALE))
}

/// Position of `ratio` along a bar `width` cells wide.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn scale(ratio: f32, width: usize) -> usize {
    (ratio.clamp(0.0, 1.0) * width as f32).round() as usize
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn spark_value(ratio: f32) -> u64 {
    (ratio.clamp(0.0, 1.0) * 100.0).round() as u64
}

fn clock(d: Duration) -> String {
    let s = d.as_secs();
    format!("{:02}:{:02}", s / 60, s % 60)
}

/// Render the whole screen.
pub fn draw(frame: &mut Frame, app: &App) {
    if let Some(setup) = &app.setup {
        draw_setup(frame, setup, app.chooser.as_ref());
        return;
    }
    let [header, meter, body, footer] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(5),
        Constraint::Min(5),
        Constraint::Length(1),
    ])
    .areas(frame.area());
    draw_header(frame, header, app);
    draw_meter(frame, meter, app);
    draw_body(frame, body, app);
    draw_footer(frame, footer, app);
}

fn panel(title: &str) -> Block<'_> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(DIM))
        .title(Span::styled(
            format!(" {title} "),
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        ))
}

fn status(app: &App) -> (&'static str, Color) {
    match app.phase {
        Phase::Idle => ("○ READY", DIM),
        Phase::Calibrating => ("◐ CALIBRATING", WARN),
        Phase::Listening => ("● LISTENING", GOOD),
        Phase::Finishing => ("◌ TRANSCRIBING", WARN),
        Phase::Cleaning => ("◌ CLEANING", WARN),
        Phase::Thinking => ("◌ THINKING", WARN),
        Phase::Speaking => ("♪ SPEAKING", ACCENT),
        Phase::Done => ("✔ DONE", ACCENT),
    }
}

fn draw_header(frame: &mut Frame, area: Rect, app: &App) {
    let block = panel("Voice Notes");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let (label, color) = status(app);
    let mut spans = vec![
        Span::styled(label, Style::new().fg(color).add_modifier(Modifier::BOLD)),
        Span::raw("   "),
        Span::styled(clock(app.elapsed), Style::new().fg(Color::White)),
    ];
    if app.speaking {
        spans.push(Span::styled(
            "   ● speaking",
            Style::new().fg(GOOD).add_modifier(Modifier::BOLD),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), inner);

    let right = match app.mode {
        Mode::Notes => {
            let clean = if app.clean { "on" } else { "off" };
            let clean_color = if app.clean { GOOD } else { DIM };
            vec![
                Span::styled("notes · clean: ", Style::new().fg(DIM)),
                Span::styled(clean, Style::new().fg(clean_color)),
            ]
        }
        Mode::Talk => {
            let mut spans = vec![Span::styled("talk · ", Style::new().fg(DIM))];
            if !app.profile.is_empty() {
                spans.push(Span::styled(
                    format!("{} · ", app.profile),
                    Style::new().fg(GOOD).add_modifier(Modifier::BOLD),
                ));
            }
            spans.push(Span::styled(
                app.backend_label.clone(),
                Style::new().fg(ACCENT),
            ));
            if !app.speak_enabled {
                spans.push(Span::styled(" · muted", Style::new().fg(DIM)));
            }
            spans
        }
    };
    frame.render_widget(
        Paragraph::new(Line::from(right)).alignment(Alignment::Right),
        inner,
    );
}

fn draw_meter(frame: &mut Frame, area: Rect, app: &App) {
    let block = panel("Input");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let [bar_area, spark_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).areas(inner);

    frame.render_widget(
        Paragraph::new(level_bar(app, bar_area.width as usize)),
        bar_area,
    );

    let keep = spark_area.width as usize;
    let start = app.history.len().saturating_sub(keep);
    frame.render_widget(
        Sparkline::default()
            .data(&app.history[start..])
            .max(100)
            .style(Style::new().fg(ACCENT)),
        spark_area,
    );
}

/// The level bar with the speech gate marked as `┃`.
fn level_bar(app: &App, width: usize) -> Line<'static> {
    let filled = scale(meter_ratio(app.level, app.gate), width);
    let gate_pos = app
        .gate
        .map(|g| scale(g / meter_scale(app.gate), width).min(width.saturating_sub(1)));
    let spans: Vec<Span<'static>> = (0..width)
        .map(|i| {
            if Some(i) == gate_pos {
                Span::styled("┃", Style::new().fg(WARN))
            } else if i < filled {
                let loud = gate_pos.is_some_and(|g| i > g);
                Span::styled("█", Style::new().fg(if loud { GOOD } else { ACCENT }))
            } else {
                Span::styled("░", Style::new().fg(DIM))
            }
        })
        .collect();
    Line::from(spans)
}

fn draw_body(frame: &mut Frame, area: Rect, app: &App) {
    if app.mode == Mode::Talk {
        draw_conversation(frame, area, app);
    } else if let Some(cleaned) = &app.cleaned {
        let [raw, polished] =
            Layout::vertical([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(area);
        draw_transcript(frame, raw, app);
        let block = panel("Cleaned");
        frame.render_widget(
            Paragraph::new(cleaned.as_str())
                .wrap(Wrap { trim: true })
                .block(block),
            polished,
        );
    } else {
        draw_transcript(frame, area, app);
    }
}

fn draw_transcript(frame: &mut Frame, area: Rect, app: &App) {
    let block = panel("Transcript");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.entries.is_empty() {
        let hint = match app.phase {
            Phase::Idle => "Press Enter and start speaking.",
            Phase::Calibrating => "Calibrating to background noise… stay quiet for a second.",
            Phase::Listening => "Listening… text appears after each pause.",
            _ => "Nothing was transcribed.",
        };
        frame.render_widget(
            Paragraph::new(Span::styled(hint, Style::new().fg(DIM))).alignment(Alignment::Center),
            inner,
        );
        return;
    }

    let lines = visible_lines(
        &app.entries,
        inner.width as usize,
        inner.height as usize,
        app.scroll,
    );
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

/// The text shown for one entry, without its timestamp.
fn entry_span(entry: &Entry) -> Span<'static> {
    match &entry.state {
        EntryState::Pending => Span::styled(
            "… transcribing",
            Style::new().fg(DIM).add_modifier(Modifier::ITALIC),
        ),
        EntryState::Text(t) => Span::raw(t.clone()),
        EntryState::Dropped(t) => Span::styled(format!("⊘ filtered: {t:?}"), Style::new().fg(DIM)),
        EntryState::Failed(e) => Span::styled(format!("✗ {e}"), Style::new().fg(BAD)),
    }
}

fn entry_line(entry: &Entry) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("[{}] ", clock(entry.at)), Style::new().fg(DIM)),
        entry_span(entry),
    ])
}

/// Pick the newest entries that fit in `height` rows, skipping `scroll` entries from the end.
fn visible_lines(
    entries: &[Entry],
    width: usize,
    height: usize,
    scroll: usize,
) -> Vec<Line<'static>> {
    let end = entries.len().saturating_sub(scroll);
    tail(
        entries[..end].iter().map(entry_line).collect(),
        width,
        height,
    )
}

/// Keep the newest `lines` that fit in `height` rows once wrapped to `width`.
fn tail(lines: Vec<Line<'static>>, width: usize, height: usize) -> Vec<Line<'static>> {
    let width = width.max(1);
    let mut used = 0;
    let mut picked: Vec<Line<'static>> = Vec::new();
    for line in lines.into_iter().rev() {
        let rows = line.width().div_ceil(width).max(1);
        if used + rows > height && !picked.is_empty() {
            break;
        }
        used += rows;
        picked.push(line);
    }
    picked.reverse();
    picked
}

/// What to show while the assistant has not started answering: the tail of its reasoning when
/// there is some, else a plain "thinking".
fn thinking_label(thinking: &str) -> String {
    const TAIL: usize = 160;
    let flat = thinking.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.is_empty() {
        return "… thinking".to_string();
    }
    let chars: Vec<char> = flat.chars().collect();
    let tail: String = chars[chars.len().saturating_sub(TAIL)..].iter().collect();
    if chars.len() > TAIL {
        format!("… thinking: …{tail}")
    } else {
        format!("… thinking: {tail}")
    }
}

/// The talk-mode lines: each turn's question and answer, then the question being spoken now.
fn conversation_lines(app: &App) -> Vec<Line<'static>> {
    let who = |name: &'static str, color: Color| {
        Span::styled(name, Style::new().fg(color).add_modifier(Modifier::BOLD))
    };
    let end = app.turns.len().saturating_sub(app.scroll);
    let mut lines = Vec::new();
    for turn in &app.turns[..end] {
        lines.push(Line::from(vec![
            who("you ", ACCENT),
            Span::raw(turn.user.clone()),
        ]));
        let answer = match &turn.reply {
            Reply::Pending => Span::styled(
                thinking_label(&turn.thinking),
                Style::new().fg(DIM).add_modifier(Modifier::ITALIC),
            ),
            Reply::Partial(t) => Span::raw(format!("{t}▌")),
            Reply::Text(t) => Span::raw(t.clone()),
            Reply::Failed(e) => Span::styled(format!("✗ {e}"), Style::new().fg(BAD)),
        };
        lines.push(Line::from(vec![who("ai  ", GOOD), answer]));
        lines.push(Line::default());
    }
    if app.scroll == 0
        && matches!(
            app.phase,
            Phase::Calibrating | Phase::Listening | Phase::Finishing
        )
    {
        let live = app.transcript();
        let words = if live.is_empty() {
            Span::styled("…", Style::new().fg(DIM))
        } else {
            Span::raw(live)
        };
        lines.push(Line::from(vec![who("you ", ACCENT), words]));
    }
    lines
}

fn draw_conversation(frame: &mut Frame, area: Rect, app: &App) {
    let mut block = panel("Conversation");
    if app.cloud {
        block = block.title(Span::styled(
            format!("· {CLOUD_NOTICE} "),
            Style::new().fg(WARN).add_modifier(Modifier::BOLD),
        ));
    }
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let lines = conversation_lines(app);
    if lines.is_empty() {
        let mut hint = vec![Line::from(Span::styled(
            "Press Enter, ask your question, then press Enter again.",
            Style::new().fg(DIM),
        ))];
        if app.cloud {
            hint.push(Line::from(Span::styled(
                format!("With this profile, {CLOUD_NOTICE}."),
                Style::new().fg(WARN),
            )));
        }
        frame.render_widget(Paragraph::new(hint).alignment(Alignment::Center), inner);
        return;
    }
    let lines = tail(lines, inner.width as usize, inner.height as usize);
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

/// One line of the setup report, coloured by its marker.
fn setup_line(text: &str) -> Line<'static> {
    let trimmed = text.trim_start();
    let color = if trimmed.starts_with("[ok]") {
        Some(GOOD)
    } else if trimmed.starts_with("[~~]") {
        Some(WARN)
    } else if trimmed.starts_with("[!!]") {
        Some(BAD)
    } else {
        None
    };
    let style = match color {
        Some(c) => Style::new().fg(c),
        None if !text.is_empty() && !text.starts_with(' ') => {
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)
        }
        None => Style::new(),
    };
    Line::from(Span::styled(text.to_string(), style))
}

/// The first-run / `u` setup screen.
fn draw_setup(frame: &mut Frame, setup: &SetupState, chooser: Option<&Chooser>) {
    if let Some(chooser) = chooser {
        draw_chooser(frame, chooser);
        return;
    }
    let [body, footer] =
        Layout::vertical([Constraint::Min(5), Constraint::Length(1)]).areas(frame.area());
    let block = panel("Voice Notes · Setup");
    let inner = block.inner(body);
    frame.render_widget(block, body);

    let (lines, keys): (Vec<Line<'static>>, Vec<(&'static str, &'static str)>) = match setup {
        SetupState::Probing => (
            vec![Line::from(Span::styled(
                "Looking around and timing your models… this can take a minute if they have to load first.",
                Style::new().fg(DIM),
            ))],
            vec![("s", " skip"), ("q", " quit")],
        ),
        SetupState::Ready {
            lines,
            config,
            models,
            ..
        } => {
            let mut keys = if *config == Config::default() {
                vec![
                    ("Enter", " close"),
                    ("s", " don't ask again"),
                    ("q", " quit"),
                ]
            } else {
                vec![("Enter", " save"), ("s", " skip"), ("q", " quit")]
            };
            if !setup::choices(models, config).is_empty() {
                keys.insert(1, ("c", " change models"));
            }
            (lines.iter().map(|l| setup_line(l)).collect(), keys)
        }
        SetupState::Done(message) => (
            vec![Line::from(Span::styled(
                message.clone(),
                Style::new().fg(GOOD),
            ))],
            vec![("any key", " continue")],
        ),
    };
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);

    let mut spans = vec![Span::raw(" ")];
    for (i, (key, what)) in keys.into_iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("  ·  ", Style::new().fg(DIM)));
        }
        spans.push(Span::styled(
            key,
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(what, Style::new().fg(DIM)));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), footer);
}

/// The "Will save" block that ends the setup report: the settings that saving would write.
fn save_block(config: &Config) -> Vec<String> {
    let Ok(toml) = config.render() else {
        return Vec::new();
    };
    if *config == Config::default() {
        return Vec::new();
    }
    let mut block = vec![String::new(), "Will save".to_string()];
    block.extend(toml.lines().map(|l| format!("  {l}")));
    block
}

/// Replace the "Will save" block in `lines` after the settings changed.
fn refresh_save_block(lines: &mut Vec<String>, config: &Config) {
    if let Some(at) = lines.iter().position(|l| l == "Will save") {
        // Also drop the blank line that precedes the block.
        lines.truncate(at.saturating_sub(1));
    }
    lines.extend(save_block(config));
}

/// Turn the findings into what the setup screen shows.
fn setup_state(found: &Findings) -> SetupState {
    let plan = setup::plan(found);
    let text = setup::render(found, &plan);
    // Drop the report's own title and the blank line after it.
    let mut lines: Vec<String> = text.lines().skip(2).map(ToString::to_string).collect();
    lines.extend(save_block(&plan.config));
    SetupState::Ready {
        lines,
        recommended: plan.config.clone(),
        config: plan.config,
        models: found.models.clone().unwrap_or_default(),
    }
}

/// The model chooser screen.
fn draw_chooser(frame: &mut Frame, chooser: &Chooser) {
    let [body, footer] =
        Layout::vertical([Constraint::Min(5), Constraint::Length(1)]).areas(frame.area());
    let block = panel("Voice Notes · Choose models");
    let inner = block.inner(body);
    frame.render_widget(block, body);

    let mut lines = vec![
        Line::from(Span::styled(
            "Pick the model for each job. The recommended one is already selected.",
            Style::new().fg(DIM),
        )),
        Line::from(Span::styled(
            "A saved choice is used straight away. Your server loads a model the first time it is used, so the first reply can be slow.",
            Style::new().fg(DIM),
        )),
        Line::default(),
    ];
    for (i, choice) in chooser.choices.iter().enumerate() {
        let focused = i == chooser.focus;
        let marker = if focused { "▶ " } else { "  " };
        let label = Style::new().fg(if focused { ACCENT } else { Color::White });
        let value = if focused {
            Style::new().fg(GOOD).add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(Color::White)
        };
        lines.push(Line::from(vec![
            Span::styled(marker, Style::new().fg(ACCENT)),
            Span::styled(format!("{:<22}", choice.field.label()), label),
            Span::styled("‹ ", Style::new().fg(DIM)),
            Span::styled(choice.display(), value),
            Span::styled(" ›", Style::new().fg(DIM)),
        ]));
    }
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);

    let keys = [
        ("↑↓", " row"),
        ("←→", " change"),
        ("Enter", " done"),
        ("Esc", " cancel"),
        ("q", " quit"),
    ];
    let mut spans = vec![Span::raw(" ")];
    for (i, (key, what)) in keys.into_iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled(" · ", Style::new().fg(DIM)));
        }
        spans.push(Span::styled(
            key,
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(what, Style::new().fg(DIM)));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), footer);
}

fn draw_footer(frame: &mut Frame, area: Rect, app: &App) {
    let key =
        |k: &'static str| Span::styled(k, Style::new().fg(ACCENT).add_modifier(Modifier::BOLD));
    let text = |t: &'static str| Span::styled(t, Style::new().fg(DIM));
    let action = match app.phase {
        Phase::Calibrating | Phase::Listening => " stop",
        Phase::Speaking => " skip",
        Phase::Finishing | Phase::Cleaning | Phase::Thinking => " wait",
        Phase::Idle | Phase::Done => " start",
    };
    let sep = || text(" · ");
    let mut spans = vec![Span::raw(" "), key("Enter"), text(action)];
    if app.idle() {
        spans.extend([
            sep(),
            key("u"),
            text(" setup"),
            sep(),
            key("p"),
            text(" profile"),
            sep(),
            key("t"),
            text(if app.mode == Mode::Talk {
                " notes"
            } else {
                " talk"
            }),
        ]);
        spans.extend([sep(), key("r"), text(" read")]);
    }
    if app.mode == Mode::Talk {
        spans.extend([sep(), key("n"), text(" new")]);
    } else {
        spans.extend([sep(), key("c"), text(" clean")]);
    }
    spans.extend([sep(), key("↑↓"), sep(), key("q"), text(" quit")]);
    if let Some(notice) = &app.notice {
        spans.push(Span::raw("   "));
        spans.push(Span::styled(notice.clone(), Style::new().fg(BAD)));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// Run the interface until the user quits; returns the text to print afterwards.
///
/// # Errors
///
/// Returns an error if the terminal cannot be drawn to or read from.
pub async fn run(settings: Settings, resolve: Resolver, config: Config) -> Result<String> {
    let mut terminal = ratatui::init();
    let result = event_loop(&mut terminal, settings, resolve, config).await;
    ratatui::restore();
    result
}

/// Everything the event loop drives besides the screen state.
struct Session {
    recorder: AudioRecorder,
    speaker: Speaker,
    client: reqwest::Client,
    jobs: UnboundedSender<Job>,
    outcomes: UnboundedReceiver<Outcome>,
    conversation: Arc<Mutex<Conversation>>,
    reply_task: Option<JoinHandle<Result<String>>>,
    cleaning: Option<JoinHandle<Result<String>>>,
    /// Where the services live, and the model that polishes transcripts.
    endpoints: Endpoints,
    clean_model: String,
    /// Pieces of the assistant's answer as they stream in.
    deltas_tx: UnboundedSender<ReplyDelta>,
    deltas_rx: UnboundedReceiver<ReplyDelta>,
    splitter: SentenceSplitter,
    /// Speech was skipped or failed; stay quiet for the rest of this answer.
    silenced: bool,
    /// Some of the current answer has arrived as streamed text.
    text_seen: bool,
    /// The saved config in use, for resolving profiles.
    config: Config,
    /// Looking around for models and services (setup screen).
    probe_task: Option<JoinHandle<Findings>>,
    /// Builds settings from a freshly saved config.
    resolve: Resolver,
}

impl Session {
    fn new(settings: &Settings, resolve: Resolver, config: Config) -> Self {
        let client = reqwest::Client::new();
        let (jobs, outcomes) =
            spawn_transcriber(client.clone(), Self::transcribe_options(settings));
        let (deltas_tx, deltas_rx) = tokio::sync::mpsc::unbounded_channel();
        Self {
            recorder: AudioRecorder::new(settings.min_threshold, false),
            speaker: Speaker::new(settings.speech.clone()),
            client,
            jobs,
            outcomes,
            conversation: Arc::new(Mutex::new(Self::conversation(settings))),
            reply_task: None,
            cleaning: None,
            endpoints: settings.endpoints.clone(),
            clean_model: settings.clean_model.clone(),
            deltas_tx,
            deltas_rx,
            splitter: SentenceSplitter::default(),
            silenced: false,
            text_seen: false,
            config,
            probe_task: None,
            resolve,
        }
    }

    /// Where and how speech is transcribed under `settings`.
    fn transcribe_options(settings: &Settings) -> TranscribeOptions {
        TranscribeOptions {
            language: settings.language.clone(),
            prompt: None,
            url: settings.endpoints.stt.clone(),
            model: settings.stt_model.clone(),
        }
    }

    /// A fresh conversation for the backend and prompt in `settings`.
    fn conversation(settings: &Settings) -> Conversation {
        Conversation::new(settings.backend.clone())
            .with_url(settings.endpoints.chat.clone())
            .with_prompt(settings.system_prompt.clone())
    }

    /// Apply transcriptions that finished since the last tick.
    fn collect_outcomes(&mut self, app: &mut App) {
        while let Ok(outcome) = self.outcomes.try_recv() {
            app.on_outcome(outcome);
        }
    }

    /// Queue finished utterances and refresh the meter while the microphone is open.
    fn poll_recording(&mut self, app: &mut App) {
        if !app.recording() {
            return;
        }
        match self.recorder.poll_segments() {
            Ok(segments) => {
                for wav in segments {
                    let id = app.queue();
                    let _ = self.jobs.send(Job { id, wav });
                }
                app.on_tick(
                    self.recorder.level(),
                    self.recorder.calibrated_threshold(),
                    self.recorder.is_speaking(),
                );
            }
            Err(e) => {
                app.notice = Some(format!("recording failed: {e:#}"));
                let _ = self.recorder.stop_recording();
                app.phase = Phase::Idle;
            }
        }
    }

    /// Read pending key presses and carry out what they ask for.
    async fn handle_keys(&mut self, app: &mut App) -> Result<()> {
        while event::poll(Duration::ZERO)? {
            let Event::Key(key) = event::read()? else {
                continue;
            };
            if let Some(action) = app.on_key(key) {
                self.act(app, action).await;
            }
        }
        Ok(())
    }

    async fn act(&mut self, app: &mut App, action: Action) {
        match action {
            Action::Start => match self.recorder.start_recording() {
                Ok(()) => app.begin(),
                Err(e) => app.notice = Some(format!("cannot start: {e:#}")),
            },
            Action::Stop => self.stop_recording(app),
            Action::Skip => {
                self.speaker.stop();
                self.silenced = true;
                app.phase = if self.reply_task.is_some() {
                    Phase::Thinking
                } else {
                    Phase::Done
                };
            }
            Action::ReadAloud => {
                if let Some(text) = app.readable() {
                    self.speak(app, &text);
                }
            }
            Action::NewConversation => self.conversation.lock().await.reset(),
            Action::OpenSetup => self.start_probe(app),
            Action::SaveSetup => self.save_setup(app),
            Action::SkipSetup => self.skip_setup(app),
            Action::SwitchProfile(name) => self.switch_profile(app, &name),
        }
    }

    /// Switch to another profile; the conversation starts afresh with the new model and prompt.
    fn switch_profile(&mut self, app: &mut App, name: &str) {
        match (self.resolve)(&self.config, Some(name)) {
            Ok(settings) => {
                self.apply(app, &settings);
                app.turns.clear();
                app.scroll = 0;
                app.notice = None;
            }
            Err(e) => app.notice = Some(format!("{e:#}")),
        }
    }

    /// Open the setup screen and look around in the background.
    fn start_probe(&mut self, app: &mut App) {
        app.setup = Some(SetupState::Probing);
        let client = self.client.clone();
        let endpoints = self.endpoints.clone();
        self.probe_task = Some(tokio::spawn(async move {
            setup::probe(&client, &endpoints, false).await
        }));
    }

    /// Save the recommended settings and start using them straight away.
    fn save_setup(&mut self, app: &mut App) {
        let Some(SetupState::Ready { config, .. }) = &app.setup else {
            return;
        };
        let saved = match Config::load() {
            Ok(saved) => saved,
            Err(e) => {
                app.notice = Some(format!("not saved: {e:#}"));
                app.setup = None;
                return;
            }
        };
        let merged = setup::overlay(saved, config);
        match merged.save() {
            Ok(path) => {
                self.config = merged;
                match (self.resolve)(&self.config, None) {
                    Ok(settings) => self.apply(app, &settings),
                    Err(e) => app.notice = Some(format!("{e:#}")),
                }
                app.setup = Some(SetupState::Done(format!(
                    "Saved to {}. These settings are in use now.",
                    path.display()
                )));
            }
            Err(e) => {
                app.notice = Some(format!("not saved: {e:#}"));
                app.setup = None;
            }
        }
    }

    /// Close setup; leave an empty config behind so launch stops asking.
    fn skip_setup(&mut self, app: &mut App) {
        if let Some(task) = self.probe_task.take() {
            task.abort();
        }
        if let Some(path) = Config::path()
            && !path.exists()
        {
            let _ = Config::default().save();
        }
        app.setup = None;
    }

    /// Switch the running session to new settings.
    fn apply(&mut self, app: &mut App, settings: &Settings) {
        self.conversation = Arc::new(Mutex::new(Self::conversation(settings)));
        self.speaker.stop();
        self.speaker = Speaker::new(settings.speech.clone());
        // A new speech-to-text model or address needs a new worker. The old one finishes what it
        // is doing and stops when its sender is dropped; this runs only while nothing is being
        // transcribed (setup and profile switching are available only when idle).
        let (jobs, outcomes) =
            spawn_transcriber(self.client.clone(), Self::transcribe_options(settings));
        self.jobs = jobs;
        self.outcomes = outcomes;
        self.endpoints = settings.endpoints.clone();
        self.clean_model.clone_from(&settings.clean_model);
        app.reconfigure(
            settings.backend.label(),
            settings.speak,
            settings.backend.is_remote(),
        );
        app.profile.clone_from(&settings.profile);
        app.profiles.clone_from(&settings.profiles);
    }

    /// Stop the recorder and send any trailing utterance to the transcriber.
    fn stop_recording(&mut self, app: &mut App) {
        match self.recorder.stop_recording() {
            Ok(Some(wav)) => {
                let id = app.queue();
                let _ = self.jobs.send(Job { id, wav });
            }
            Ok(None) => {}
            Err(e) => app.notice = Some(format!("stop failed: {e:#}")),
        }
        app.stopped();
    }

    /// Start reading `text` aloud, or report why that is not possible.
    fn speak(&mut self, app: &mut App, text: &str) {
        match self.speaker.speak(text) {
            Ok(()) if self.speaker.is_busy() => app.phase = Phase::Speaking,
            Ok(()) => app.phase = Phase::Done,
            Err(e) => {
                app.notice = Some(format!("{e:#}"));
                app.phase = Phase::Done;
            }
        }
    }

    /// Once every utterance is transcribed, ask the assistant (talk) or clean the notes.
    fn advance(&mut self, app: &mut App) {
        if app.phase != Phase::Finishing || app.pending() != 0 {
            return;
        }
        let text = app.transcript();
        let client = self.client.clone();
        if app.mode == Mode::Talk {
            if text.is_empty() {
                app.notice = Some("didn't catch anything, try again".to_string());
                app.phase = Phase::Done;
            } else {
                app.push_turn(text.clone());
                app.phase = Phase::Thinking;
                self.splitter = SentenceSplitter::default();
                self.silenced = false;
                self.text_seen = false;
                let conversation = Arc::clone(&self.conversation);
                let deltas = self.deltas_tx.clone();
                self.reply_task = Some(tokio::spawn(async move {
                    conversation
                        .lock()
                        .await
                        .reply(&client, &text, Some(&deltas))
                        .await
                }));
            }
        } else if app.clean && !text.is_empty() {
            app.phase = Phase::Cleaning;
            let (url, model) = (self.endpoints.chat.clone(), self.clean_model.clone());
            self.cleaning = Some(tokio::spawn(async move {
                clean_transcript(&client, &url, &model, &text).await
            }));
        } else {
            app.phase = Phase::Done;
        }
    }

    /// Pick up the cleaned text or the assistant's answer once their tasks finish.
    async fn finish_tasks(&mut self, app: &mut App) {
        if self
            .probe_task
            .as_ref()
            .is_some_and(JoinHandle::is_finished)
            && let Some(handle) = self.probe_task.take()
        {
            match handle.await {
                Ok(found) => app.setup = Some(setup_state(&found)),
                Err(e) => {
                    app.notice = Some(format!("setup check failed: {e}"));
                    app.setup = None;
                }
            }
        }

        if self.cleaning.as_ref().is_some_and(JoinHandle::is_finished)
            && let Some(handle) = self.cleaning.take()
        {
            match handle.await {
                Ok(Ok(text)) => app.cleaned = Some(text),
                Ok(Err(e)) => app.notice = Some(format!("cleaning failed: {e:#}")),
                Err(e) => app.notice = Some(format!("cleaning failed: {e}")),
            }
            app.phase = Phase::Done;
        }

        if self
            .reply_task
            .as_ref()
            .is_some_and(JoinHandle::is_finished)
            && let Some(handle) = self.reply_task.take()
        {
            // The task is finished, so every piece it produced is already in the channel.
            self.collect_deltas(app);
            match handle.await {
                Ok(Ok(reply)) => {
                    app.set_reply(Reply::Text(reply.clone()));
                    if app.speak_enabled && !self.silenced {
                        if let Some(rest) = self.splitter.finish() {
                            self.say(app, &rest);
                        } else if !self.text_seen {
                            // The answer never arrived as streamed text, so say it all now.
                            self.say(app, &reply);
                        }
                    }
                }
                Ok(Err(e)) => app.set_reply(Reply::Failed(format!("{e:#}"))),
                Err(e) => app.set_reply(Reply::Failed(e.to_string())),
            }
            app.phase = Phase::Thinking; // settled just below
        }

        if let Err(e) = self.speaker.pump() {
            app.notice = Some(format!("{e:#}"));
            self.silenced = true;
            self.speaker.stop();
        }
        if let Some(notice) = self.speaker.take_notice() {
            app.notice = Some(notice);
        }
        if self.reply_task.is_none() && matches!(app.phase, Phase::Thinking | Phase::Speaking) {
            app.phase = if self.speaker.is_busy() {
                Phase::Speaking
            } else {
                Phase::Done
            };
        }
    }

    /// Show streamed pieces of the answer and speak each sentence as soon as it is complete.
    fn collect_deltas(&mut self, app: &mut App) {
        while let Ok(delta) = self.deltas_rx.try_recv() {
            match delta {
                ReplyDelta::Thinking(reasoning) => app.append_thinking(&reasoning),
                ReplyDelta::Text(text) => {
                    self.text_seen = true;
                    app.append_reply(&text);
                    if app.speak_enabled && !self.silenced {
                        for sentence in self.splitter.push(&text) {
                            self.say(app, &sentence);
                        }
                    }
                }
            }
        }
    }

    /// Queue one sentence for speech; on failure report it and stay quiet.
    fn say(&mut self, app: &mut App, text: &str) {
        match self.speaker.enqueue(text) {
            Ok(()) => {
                if app.phase == Phase::Thinking && self.speaker.is_busy() {
                    app.phase = Phase::Speaking;
                }
            }
            Err(e) => {
                app.notice = Some(format!("{e:#}"));
                self.silenced = true;
                self.speaker.stop();
            }
        }
    }
}

async fn event_loop(
    terminal: &mut DefaultTerminal,
    settings: Settings,
    resolve: Resolver,
    config: Config,
) -> Result<String> {
    let mut session = Session::new(&settings, resolve, config);
    let mut app = App::new(settings.clean)
        .with_talk(settings.talk, settings.backend.label(), settings.speak)
        .with_cloud(settings.backend.is_remote())
        .with_profiles(settings.profile.clone(), settings.profiles.clone());
    // First launch: no saved config yet, so offer to set things up.
    if Config::path().is_some_and(|path| !path.exists()) {
        session.start_probe(&mut app);
    }
    let mut tick = tokio::time::interval(Duration::from_millis(50));

    while !app.quit {
        terminal.draw(|frame| draw(frame, &app))?;
        tick.tick().await;

        session.collect_outcomes(&mut app);
        session.poll_recording(&mut app);
        session.handle_keys(&mut app).await?;
        if app.quit && app.recording() {
            let _ = session.recorder.stop_recording();
        }
        session.collect_deltas(&mut app);
        session.advance(&mut app);
        session.finish_tasks(&mut app).await;
    }
    Ok(app.final_text())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    /// Render `app` on an 80x24 screen and return its text, one string per row.
    fn render(app: &App) -> Vec<String> {
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| draw(frame, app)).unwrap();
        let buffer = terminal.backend().buffer().clone();
        let width = buffer.area.width as usize;
        buffer
            .content()
            .chunks(width)
            .map(|row| row.iter().map(ratatui::buffer::Cell::symbol).collect())
            .collect()
    }

    fn screen(app: &App) -> String {
        render(app).join("\n")
    }

    /// Enter starts from idle, stops while recording, and does nothing while finishing.
    #[test]
    fn enter_toggles_recording() {
        let mut app = App::new(false);
        assert_eq!(app.on_key(press(KeyCode::Enter)), Some(Action::Start));
        app.begin();
        assert_eq!(app.on_key(press(KeyCode::Char(' '))), Some(Action::Stop));
        app.stopped();
        assert_eq!(app.on_key(press(KeyCode::Enter)), None);
        app.phase = Phase::Done;
        assert_eq!(app.on_key(press(KeyCode::Enter)), Some(Action::Start));
    }

    /// `q`, Esc and Ctrl-C quit; `c` toggles cleaning.
    #[test]
    fn quit_and_clean_keys() {
        let mut app = App::new(false);
        app.on_key(press(KeyCode::Char('c')));
        assert!(app.clean && !app.quit);
        app.on_key(press(KeyCode::Char('q')));
        assert!(app.quit);

        let mut app = App::new(false);
        app.on_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(app.quit);
    }

    /// Outcomes land on the matching entry, and only accepted text enters the transcript.
    #[test]
    fn outcomes_fill_entries_in_order() {
        let mut app = App::new(false);
        let (a, b, c) = (app.queue(), app.queue(), app.queue());
        assert_eq!(app.pending(), 3);
        app.on_outcome(Outcome::Text {
            id: a,
            text: "Hello there.".into(),
        });
        app.on_outcome(Outcome::Dropped {
            id: b,
            text: "you".into(),
        });
        app.on_outcome(Outcome::Text {
            id: c,
            text: "Second one.".into(),
        });
        assert_eq!(app.pending(), 0);
        assert_eq!(app.transcript(), "Hello there. Second one.");
        assert_eq!(app.final_text(), "Hello there. Second one.");
        app.cleaned = Some("Cleaned.".into());
        assert_eq!(app.final_text(), "Cleaned.");
    }

    /// The first tick that reports a gate moves calibration on to listening.
    #[test]
    fn calibration_finishes_when_gate_arrives() {
        let mut app = App::new(false);
        app.begin();
        app.on_tick(0.2, None, false);
        assert_eq!(app.phase, Phase::Calibrating);
        app.on_tick(0.2, Some(0.1), false);
        assert_eq!(app.phase, Phase::Listening);
    }

    /// Scrolling stays within the entries.
    #[test]
    fn scroll_is_clamped() {
        let mut app = App::new(false);
        for _ in 0..3 {
            app.queue();
        }
        for _ in 0..10 {
            app.on_key(press(KeyCode::Up));
        }
        assert_eq!(app.scroll, 2);
        for _ in 0..10 {
            app.on_key(press(KeyCode::Down));
        }
        assert_eq!(app.scroll, 0);
    }

    /// The idle screen shows the title, status, hint and key help.
    #[test]
    fn draws_idle_screen() {
        let text = screen(&App::new(false));
        for expected in [
            "Voice Notes",
            "READY",
            "clean: off",
            "Press Enter and start speaking",
            "Enter",
            "quit",
        ] {
            assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
        }
    }

    /// While listening the screen shows transcribed, pending and filtered entries.
    #[test]
    fn draws_transcript_entries() {
        let mut app = App::new(true);
        app.begin();
        app.on_tick(0.3, Some(0.1), true);
        let (a, b) = (app.queue(), app.queue());
        app.queue();
        app.on_outcome(Outcome::Text {
            id: a,
            text: "Here it is, let's start again.".into(),
        });
        app.on_outcome(Outcome::Dropped {
            id: b,
            text: "you".into(),
        });
        let text = screen(&app);
        for expected in [
            "LISTENING",
            "speaking",
            "clean: on",
            "Here it is, let's start again.",
            "filtered",
            "transcribing",
        ] {
            assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
        }
    }

    /// The level bar marks the gate and fills in proportion to the level.
    #[test]
    fn level_bar_shows_gate_and_fill() {
        let mut app = App::new(false);
        app.on_tick(0.3, Some(0.1), false); // scale 0.3: gate at 1/3, bar full
        let line: String = level_bar(&app, 30)
            .spans
            .iter()
            .map(|s| s.content.to_string())
            .collect();
        assert_eq!(line.chars().count(), 30);
        assert_eq!(line.chars().filter(|c| *c == '┃').count(), 1);
        assert_eq!(line.chars().filter(|c| *c == '░').count(), 0);

        app.on_tick(0.0, Some(0.1), false);
        let line: String = level_bar(&app, 30)
            .spans
            .iter()
            .map(|s| s.content.to_string())
            .collect();
        assert_eq!(line.chars().filter(|c| *c == '█').count(), 0);
    }

    /// Only the newest entries that fit are shown, and scrolling reveals older ones.
    #[test]
    fn visible_lines_follow_the_tail() {
        let entries: Vec<Entry> = (0..10)
            .map(|id| Entry {
                id,
                at: Duration::from_secs(id as u64),
                state: EntryState::Text(format!("line {id}")),
            })
            .collect();
        let tail = visible_lines(&entries, 40, 3, 0);
        assert_eq!(tail.len(), 3);
        assert!(tail[2].to_string().contains("line 9"));
        let older = visible_lines(&entries, 40, 3, 4);
        assert!(older[2].to_string().contains("line 5"));
    }

    /// The finished screen shows both the raw and the cleaned transcript.
    #[test]
    fn draws_cleaned_pane_when_done() {
        let mut app = App::new(true);
        let id = app.queue();
        app.on_outcome(Outcome::Text {
            id,
            text: "so um i think its slow".into(),
        });
        app.cleaned = Some("I think it's slow.".into());
        app.phase = Phase::Done;
        let text = screen(&app);
        assert!(
            text.contains("DONE")
                && text.contains("Cleaned")
                && text.contains("I think it's slow.")
        );
    }
}

/// Tests for talk mode.
#[cfg(test)]
mod talk_tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn talk_app() -> App {
        App::new(false).with_talk(true, "qwen-reasoner".to_string(), true)
    }

    fn screen(app: &App) -> String {
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| draw(frame, app)).unwrap();
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    /// `t` switches between notes and talk, but only while idle.
    #[test]
    fn t_toggles_mode_only_when_idle() {
        let mut app = App::new(false);
        app.on_key(press(KeyCode::Char('t')));
        assert_eq!(app.mode, Mode::Talk);
        app.on_key(press(KeyCode::Char('t')));
        assert_eq!(app.mode, Mode::Notes);
        app.phase = Phase::Listening;
        app.on_key(press(KeyCode::Char('t')));
        assert_eq!(app.mode, Mode::Notes);
    }

    /// Enter skips speech while the assistant is talking and waits during thinking.
    #[test]
    fn enter_skips_speech_and_waits_while_thinking() {
        let mut app = talk_app();
        app.phase = Phase::Thinking;
        assert_eq!(app.on_key(press(KeyCode::Enter)), None);
        app.phase = Phase::Speaking;
        assert_eq!(app.on_key(press(KeyCode::Enter)), Some(Action::Skip));
    }

    /// `r` reads the latest answer aloud, only when there is one.
    #[test]
    fn r_reads_the_last_reply() {
        let mut app = talk_app();
        assert_eq!(app.on_key(press(KeyCode::Char('r'))), None);
        app.push_turn("hello".into());
        assert_eq!(app.on_key(press(KeyCode::Char('r'))), None); // still pending
        app.set_reply(Reply::Text("Hi there.".into()));
        assert_eq!(app.readable().as_deref(), Some("Hi there."));
        assert_eq!(
            app.on_key(press(KeyCode::Char('r'))),
            Some(Action::ReadAloud)
        );
    }

    /// In notes mode `r` reads the transcript, preferring the cleaned text.
    #[test]
    fn r_reads_notes_in_notes_mode() {
        let mut app = App::new(false);
        assert_eq!(app.on_key(press(KeyCode::Char('r'))), None);
        let id = app.queue();
        app.on_outcome(Outcome::Text {
            id,
            text: "raw words".into(),
        });
        assert_eq!(app.readable().as_deref(), Some("raw words"));
        app.cleaned = Some("Clean words.".into());
        assert_eq!(app.readable().as_deref(), Some("Clean words."));
    }

    /// `n` clears the conversation and asks the backend to forget it.
    #[test]
    fn n_starts_a_new_conversation() {
        let mut app = talk_app();
        app.push_turn("one".into());
        assert_eq!(
            app.on_key(press(KeyCode::Char('n'))),
            Some(Action::NewConversation)
        );
        assert!(app.turns.is_empty());
        let mut notes = App::new(false);
        assert_eq!(notes.on_key(press(KeyCode::Char('n'))), None);
    }

    /// Failed answers are shown but never read aloud.
    #[test]
    fn failed_replies_are_not_readable() {
        let mut app = talk_app();
        app.push_turn("hello".into());
        app.set_reply(Reply::Failed("server down".into()));
        assert_eq!(app.readable(), None);
    }

    /// The conversation pane shows both sides, the backend name and talk key help.
    #[test]
    fn draws_conversation() {
        let mut app = talk_app();
        app.push_turn("what is two plus two".into());
        app.set_reply(Reply::Text("Four.".into()));
        app.push_turn("and times two".into());
        let text = screen(&app);
        for expected in [
            "Conversation",
            "talk · qwen-reasoner",
            "what is two plus two",
            "Four.",
            "and times two",
            "thinking",
            " new",
        ] {
            assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
        }
    }

    /// While you are speaking, your words appear live as the current question.
    #[test]
    fn shows_live_question_while_listening() {
        let mut app = talk_app();
        app.begin();
        app.on_tick(0.3, Some(0.1), true);
        let id = app.queue();
        app.on_outcome(Outcome::Text {
            id,
            text: "how do I list files".into(),
        });
        let text = screen(&app);
        assert!(text.contains("how do I list files") && text.contains("LISTENING"));
    }

    /// The idle talk screen explains what to do, and muted speech is shown.
    #[test]
    fn idle_talk_screen_and_muted_flag() {
        let app = App::new(false).with_talk(true, "claude".to_string(), false);
        let text = screen(&app);
        assert!(text.contains("ask your question") && text.contains("muted"));
    }

    /// Streamed pieces build up the latest answer, which is shown with a cursor until it is done
    /// and is not read aloud by `r` while incomplete.
    #[test]
    fn streamed_reply_shows_cursor_until_complete() {
        let mut app = talk_app();
        app.push_turn("tell me something".into());
        app.append_reply("Octopuses have ");
        app.append_reply("three hearts.");
        assert_eq!(
            app.turns[0].reply,
            Reply::Partial("Octopuses have three hearts.".into())
        );
        assert_eq!(app.readable(), None);
        assert!(screen(&app).contains("three hearts.▌"));

        app.set_reply(Reply::Text("Octopuses have three hearts.".into()));
        assert!(!screen(&app).contains('▌'));
        assert_eq!(
            app.readable().as_deref(),
            Some("Octopuses have three hearts.")
        );
    }

    /// Streamed text with no turn to attach to is ignored instead of panicking.
    #[test]
    fn append_without_a_turn_is_ignored() {
        let mut app = talk_app();
        app.append_reply("stray");
        assert!(app.turns.is_empty());
    }
}

/// Tests for the setup screen.
#[cfg(test)]
mod setup_tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn screen(app: &App) -> String {
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        terminal.draw(|frame| draw(frame, app)).unwrap();
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    fn found_everything() -> Findings {
        Findings {
            models: Some(vec!["whisper".into(), "qwen3-coder-30b".into()]),
            kokoro_url: Some("http://127.0.0.1:8880/v1/audio/speech".into()),
            system_voice: Some("espeak-ng"),
            player: Some("paplay"),
            claude: false,
            ..Findings::default()
        }
    }

    fn ready(found: &Findings) -> App {
        let mut app = App::new(false);
        app.setup = Some(setup_state(found));
        app
    }

    /// `u` opens setup only when nothing else is going on.
    #[test]
    fn u_opens_setup_when_idle() {
        let mut app = App::new(false);
        assert_eq!(
            app.on_key(press(KeyCode::Char('u'))),
            Some(Action::OpenSetup)
        );
        app.phase = Phase::Listening;
        assert_eq!(app.on_key(press(KeyCode::Char('u'))), None);
    }

    /// While looking around, Enter and other keys do nothing; `s` or Esc skips and `q` quits.
    #[test]
    fn probing_allows_skip_and_quit_only() {
        let mut app = App::new(false);
        app.setup = Some(SetupState::Probing);
        assert_eq!(app.on_key(press(KeyCode::Enter)), None);
        assert_eq!(app.on_key(press(KeyCode::Char('x'))), None);
        assert!(app.setup.is_some() && !app.quit);
        assert_eq!(
            app.on_key(press(KeyCode::Char('s'))),
            Some(Action::SkipSetup)
        );
        assert_eq!(app.on_key(press(KeyCode::Esc)), Some(Action::SkipSetup));
        app.on_key(press(KeyCode::Char('q')));
        assert!(app.quit);
    }

    /// With something to save, Enter or `y` saves, and `s` or Esc skips.
    #[test]
    fn ready_screen_keys() {
        let mut app = ready(&found_everything());
        assert_eq!(app.on_key(press(KeyCode::Enter)), Some(Action::SaveSetup));
        assert_eq!(
            app.on_key(press(KeyCode::Char('y'))),
            Some(Action::SaveSetup)
        );
        assert_eq!(
            app.on_key(press(KeyCode::Char('s'))),
            Some(Action::SkipSetup)
        );
        assert_eq!(app.on_key(press(KeyCode::Esc)), Some(Action::SkipSetup));
        assert!(
            app.setup.is_some(),
            "the screen stays until the action is carried out"
        );
        assert_eq!(app.on_key(press(KeyCode::Char('x'))), None);
    }

    /// When nothing was found there is nothing to save, so Enter just closes the screen.
    #[test]
    fn nothing_found_closes_on_enter() {
        let mut app = ready(&Findings::default());
        assert_eq!(app.on_key(press(KeyCode::Enter)), None);
        assert!(app.setup.is_none());
    }

    /// After saving, any key returns to the main screen.
    #[test]
    fn done_screen_closes_on_any_key() {
        let mut app = App::new(false);
        app.setup = Some(SetupState::Done("Saved.".into()));
        assert_eq!(app.on_key(press(KeyCode::Char('x'))), None);
        assert!(app.setup.is_none());
    }

    /// Findings become report lines plus the exact settings that would be saved.
    #[test]
    fn findings_become_a_report_and_a_config() {
        let SetupState::Ready { lines, config, .. } = setup_state(&found_everything()) else {
            panic!("expected a ready state");
        };
        let text = lines.join("\n");
        assert!(text.contains("[ok] Chat"), "{text}");
        assert!(text.contains("Will save") && text.contains("chat_model = \"qwen3-coder-30b\""));
        assert!(
            !text.contains("Voice Notes setup"),
            "the report's own title is dropped"
        );
        assert_eq!(config.chat_model.as_deref(), Some("qwen3-coder-30b"));

        let SetupState::Ready { lines, config, .. } = setup_state(&Findings::default()) else {
            panic!("expected a ready state");
        };
        assert_eq!(config, Config::default());
        let text = lines.join("\n");
        assert!(text.contains("managed install") && !text.contains("Will save"));
    }

    /// The setup screen replaces the main screen and shows the right keys for each state.
    #[test]
    fn draws_each_setup_state() {
        let mut app = App::new(false);
        app.setup = Some(SetupState::Probing);
        let text = screen(&app);
        assert!(text.contains("Setup") && text.contains("Looking around"));
        assert!(!text.contains("Input"), "the main screen is hidden");

        let text = screen(&ready(&found_everything()));
        for expected in ["Kokoro at", "Will save", " save", " skip", " quit"] {
            assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
        }

        let text = screen(&ready(&Findings::default()));
        assert!(text.contains("preview, not available yet") && text.contains("Qwen3-8B"));
        assert!(text.contains("don't ask again"));

        app.setup = Some(SetupState::Done("Saved to /x/config.toml.".into()));
        assert!(screen(&app).contains("Saved to /x/config.toml."));
    }

    /// Report markers are coloured so problems stand out.
    #[test]
    fn setup_lines_are_coloured_by_marker() {
        let colour = |text: &str| setup_line(text).spans[0].style.fg;
        assert_eq!(colour("  [ok] Chat  x"), Some(GOOD));
        assert_eq!(colour("  [~~] Voice x"), Some(WARN));
        assert_eq!(colour("  [!!] Server x"), Some(BAD));
        assert_eq!(colour("What I found"), Some(ACCENT));
        assert_eq!(colour("  plain"), None);
    }

    /// The idle screen advertises the setup key.
    #[test]
    fn idle_footer_mentions_setup() {
        assert!(screen(&App::new(false)).contains(" setup"));
    }

    /// New settings update the header's backend name and speech flag.
    #[test]
    fn reconfigure_updates_header_state() {
        let mut app = App::new(false).with_talk(true, "old".into(), true);
        app.reconfigure("qwen3-coder-30b".into(), false, false);
        assert_eq!(app.backend_label, "qwen3-coder-30b");
        assert!(!app.speak_enabled);
        assert!(screen(&app).contains("qwen3-coder-30b") && screen(&app).contains("muted"));
    }
}

/// Tests for profiles and the thinking display.
#[cfg(test)]
mod profile_tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn screen(app: &App) -> String {
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| draw(frame, app)).unwrap();
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    fn app_with_profiles(current: &str) -> App {
        App::new(false)
            .with_talk(true, "qwen-reasoner".into(), true)
            .with_profiles(
                current.into(),
                ["fast", "dev", "smart", "think"].map(String::from).to_vec(),
            )
    }

    /// `p` asks for the next profile and wraps around after the last one.
    #[test]
    fn p_cycles_through_profiles() {
        let mut app = app_with_profiles("fast");
        assert_eq!(
            app.on_key(press(KeyCode::Char('p'))),
            Some(Action::SwitchProfile("dev".into()))
        );
        let mut app = app_with_profiles("think");
        assert_eq!(
            app.on_key(press(KeyCode::Char('p'))),
            Some(Action::SwitchProfile("fast".into()))
        );
    }

    /// With one profile, or while busy, `p` does nothing.
    #[test]
    fn p_needs_a_choice_and_an_idle_app() {
        let mut one = App::new(false).with_profiles("fast".into(), vec!["fast".into()]);
        assert_eq!(one.on_key(press(KeyCode::Char('p'))), None);
        let mut none = App::new(false);
        assert_eq!(none.on_key(press(KeyCode::Char('p'))), None);
        let mut busy = app_with_profiles("fast");
        busy.phase = Phase::Thinking;
        assert_eq!(busy.on_key(press(KeyCode::Char('p'))), None);
        assert_eq!(busy.next_profile().as_deref(), Some("dev"));
    }

    /// An unknown current profile starts the cycle from the first one.
    #[test]
    fn next_profile_from_unknown() {
        let app = app_with_profiles("gone");
        assert_eq!(app.next_profile().as_deref(), Some("fast"));
    }

    /// Talk mode shows the profile and its model; `p` is offered only when there is a choice.
    #[test]
    fn header_and_footer_show_the_profile() {
        let text = screen(&app_with_profiles("smart"));
        assert!(text.contains("talk · smart · qwen-reasoner"), "{text}");
        assert!(text.contains(" profile"), "{text}");
        let single = App::new(false).with_talk(true, "m".into(), true);
        let text = screen(&single);
        assert!(
            text.contains("talk · m") && !text.contains("smart"),
            "{text}"
        );
    }

    /// Reasoning is shown, shortened to its tail, until the answer starts.
    #[test]
    fn thinking_label_shows_the_tail() {
        assert_eq!(thinking_label(""), "… thinking");
        assert_eq!(thinking_label("   \n "), "… thinking");
        assert_eq!(
            thinking_label("Add the numbers.\nCarry one."),
            "… thinking: Add the numbers. Carry one."
        );
        let long = "word ".repeat(100);
        let label = thinking_label(&long);
        assert!(
            label.starts_with("… thinking: …") && label.chars().count() < 190,
            "{label}"
        );
    }

    /// While a `think` answer is pending its reasoning is on screen; once the answer streams the
    /// reasoning is replaced by the text.
    #[test]
    fn reasoning_is_shown_until_the_answer_starts() {
        let mut app = app_with_profiles("think");
        app.push_turn("what is two plus two".into());
        app.append_thinking("Let me add ");
        app.append_thinking("two and two.");
        assert!(screen(&app).contains("thinking: Let me add two and two."));

        app.append_reply("Four.");
        let text = screen(&app);
        assert!(
            text.contains("Four.") && !text.contains("Let me add"),
            "{text}"
        );
        assert_eq!(app.turns[0].thinking, "Let me add two and two.");
    }

    /// Reasoning with no turn to attach to is ignored.
    #[test]
    fn thinking_without_a_turn_is_ignored() {
        let mut app = app_with_profiles("think");
        app.append_thinking("stray");
        assert!(app.turns.is_empty());
    }
}

/// Tests for the notice that the Claude backend sends what you say to Anthropic.
#[cfg(test)]
mod privacy_tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    const NOTICE: &str = "what you say is sent to Anthropic";

    fn screen(app: &App) -> String {
        let mut terminal = Terminal::new(TestBackend::new(100, 24)).unwrap();
        terminal.draw(|frame| draw(frame, app)).unwrap();
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    fn talk(cloud: bool) -> App {
        App::new(false)
            .with_talk(true, "claude".into(), true)
            .with_cloud(cloud)
    }

    /// With the Claude backend the conversation panel says so on its border and, before the
    /// first question, again in plain words.
    #[test]
    fn claude_shows_the_notice_twice_when_empty() {
        let text = screen(&talk(true));
        assert_eq!(text.matches(NOTICE).count(), 2, "{text}");
        assert!(
            text.contains(&format!("With this profile, {NOTICE}.")),
            "{text}"
        );
    }

    /// Once a conversation has started the border notice stays visible.
    #[test]
    fn notice_stays_during_a_conversation() {
        let mut app = talk(true);
        app.push_turn("hello".into());
        app.set_reply(Reply::Text("Hi there.".into()));
        let text = screen(&app);
        assert!(
            text.contains("Hi there.") && text.contains(NOTICE),
            "{text}"
        );
    }

    /// Local models keep everything on the machine, so nothing is said about Anthropic.
    #[test]
    fn local_models_show_no_notice() {
        let text = screen(&talk(false));
        assert!(!text.contains("Anthropic"), "{text}");
    }

    /// Notes mode never talks to the chat backend, so it shows no notice either.
    #[test]
    fn notes_mode_shows_no_notice() {
        let text = screen(&App::new(false).with_cloud(true));
        assert!(!text.contains("Anthropic"), "{text}");
    }

    /// Switching profile updates the notice: it appears for Claude and disappears for a local
    /// model.
    #[test]
    fn switching_backend_updates_the_notice() {
        let mut app = talk(false);
        assert!(!screen(&app).contains(NOTICE));
        app.reconfigure("claude".into(), true, true);
        assert!(screen(&app).contains(NOTICE));
        app.reconfigure("qwen3-coder-30b".into(), true, false);
        assert!(!screen(&app).contains(NOTICE));
    }
}

/// Tests for the model chooser opened from the setup screen.
#[cfg(test)]
mod chooser_tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn screen(app: &App) -> String {
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        terminal.draw(|frame| draw(frame, app)).unwrap();
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    /// Your server: a fast coder model, a reasoning model, Whisper, and models that are not chat.
    fn your_server() -> Findings {
        Findings {
            models: Some(
                [
                    "nomic-embed",
                    "qwen-reasoner",
                    "qwen25-vl-7b",
                    "qwen3-coder-30b",
                    "whisper",
                ]
                .map(String::from)
                .to_vec(),
            ),
            kokoro_url: Some("http://127.0.0.1:8880/v1/audio/speech".into()),
            system_voice: Some("espeak-ng"),
            player: Some("paplay"),
            ..Findings::default()
        }
    }

    fn app_with(found: &Findings) -> App {
        let mut app = App::new(false);
        app.setup = Some(setup_state(found));
        app
    }

    fn config_of(app: &App) -> Config {
        match &app.setup {
            Some(SetupState::Ready { config, .. }) => config.clone(),
            other => panic!("expected a ready setup screen, got {other:?}"),
        }
    }

    fn report_of(app: &App) -> String {
        match &app.setup {
            Some(SetupState::Ready { lines, .. }) => lines.join("\n"),
            other => panic!("expected a ready setup screen, got {other:?}"),
        }
    }

    /// `c` opens the chooser, which shows the three jobs with their recommended models.
    #[test]
    fn c_opens_the_chooser() {
        let mut app = app_with(&your_server());
        assert!(
            screen(&app).contains(" change models"),
            "the setup screen offers it"
        );
        assert_eq!(app.on_key(press(KeyCode::Char('c'))), None);
        assert!(app.chooser.is_some());
        let text = screen(&app);
        for expected in [
            "Choose models",
            "Chat (fast, dev)",
            "qwen3-coder-30b (recommended)",
            "Smart (smart, think)",
            "qwen-reasoner (recommended)",
            "Speech-to-text",
            "whisper (recommended)",
            "row",
            "change",
            "cancel",
        ] {
            assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
        }
        assert!(
            !text.contains("nomic-embed") && !text.contains("qwen25-vl"),
            "not chat models"
        );
    }

    /// Your example: swap the coder model for the reasoning model, then save it.
    #[test]
    fn swapping_coder_for_reasoner_and_saving() {
        let mut app = app_with(&your_server());
        assert_eq!(
            config_of(&app).chat_model.as_deref(),
            Some("qwen3-coder-30b")
        );
        app.on_key(press(KeyCode::Char('c')));
        app.on_key(press(KeyCode::Right)); // chat row: the next model
        assert!(
            screen(&app).contains("‹ qwen-reasoner ›"),
            "{}",
            screen(&app)
        );
        app.on_key(press(KeyCode::Enter));
        assert!(app.chooser.is_none(), "the chooser closes");

        assert_eq!(config_of(&app).chat_model.as_deref(), Some("qwen-reasoner"));
        let report = report_of(&app);
        assert!(
            report.contains("chat_model = \"qwen-reasoner\""),
            "{report}"
        );
        assert!(
            !report.contains("chat_model = \"qwen3-coder-30b\""),
            "{report}"
        );
        assert_eq!(
            report.matches("Will save").count(),
            1,
            "the block is replaced, not repeated"
        );
        // Saving is the usual Enter on the setup screen.
        assert_eq!(app.on_key(press(KeyCode::Enter)), Some(Action::SaveSetup));
    }

    /// Escape leaves everything as it was.
    #[test]
    fn escape_cancels() {
        let mut app = app_with(&your_server());
        let before = config_of(&app);
        app.on_key(press(KeyCode::Char('c')));
        app.on_key(press(KeyCode::Right));
        app.on_key(press(KeyCode::Esc));
        assert!(app.chooser.is_none() && app.setup.is_some());
        assert_eq!(config_of(&app), before);
    }

    /// Up and down move between jobs without going past the ends, and left goes back.
    #[test]
    fn focus_and_direction_keys() {
        let mut app = app_with(&your_server());
        app.on_key(press(KeyCode::Char('c')));
        let focus = |app: &App| app.chooser.as_ref().unwrap().focus;
        assert_eq!(focus(&app), 0);
        app.on_key(press(KeyCode::Up));
        assert_eq!(focus(&app), 0, "no going above the first row");
        for _ in 0..5 {
            app.on_key(press(KeyCode::Down));
        }
        assert_eq!(focus(&app), 2, "no going below the last row");
        app.on_key(press(KeyCode::Char('k')));
        assert_eq!(focus(&app), 1);

        // Left steps back to the recommendation after stepping forward.
        app.on_key(press(KeyCode::Right));
        assert_eq!(app.chooser.as_ref().unwrap().choices[1].selected, Some(0));
        app.on_key(press(KeyCode::Left));
        assert_eq!(app.chooser.as_ref().unwrap().choices[1].selected, None);
    }

    /// Reopening the chooser shows the change that was made, not the original recommendation.
    #[test]
    fn reopening_shows_the_current_pick() {
        let mut app = app_with(&your_server());
        app.on_key(press(KeyCode::Char('c')));
        app.on_key(press(KeyCode::Right));
        app.on_key(press(KeyCode::Enter));
        app.on_key(press(KeyCode::Char('c')));
        let chat = &app.chooser.as_ref().unwrap().choices[0];
        assert_eq!(chat.value().as_deref(), Some("qwen-reasoner"));
        assert_eq!(
            chat.display(),
            "qwen-reasoner",
            "not marked as the recommendation"
        );
    }

    /// With no models to choose between (the server did not answer), `c` does nothing and the
    /// setup screen does not offer it.
    #[test]
    fn nothing_to_choose_without_models() {
        let mut app = app_with(&Findings::default());
        assert!(!screen(&app).contains("change models"));
        app.on_key(press(KeyCode::Char('c')));
        assert!(app.chooser.is_none());
        let mut probing = App::new(false);
        probing.setup = Some(SetupState::Probing);
        probing.on_key(press(KeyCode::Char('c')));
        assert!(probing.chooser.is_none());
    }

    /// Quitting works from inside the chooser.
    #[test]
    fn quit_works_in_the_chooser() {
        let mut app = app_with(&your_server());
        app.on_key(press(KeyCode::Char('c')));
        app.on_key(press(KeyCode::Char('q')));
        assert!(app.quit);
    }

    /// A machine with a single chat model still gets rows, and picking a Whisper name by hand
    /// is saved.
    #[test]
    fn single_model_and_named_whisper() {
        let found = Findings {
            models: Some(
                ["whisper-small", "whisper-large-v3", "llama-3-8b"]
                    .map(String::from)
                    .to_vec(),
            ),
            ..your_server()
        };
        let mut app = app_with(&found);
        app.on_key(press(KeyCode::Char('c')));
        for _ in 0..2 {
            app.on_key(press(KeyCode::Down));
        }
        app.on_key(press(KeyCode::Right)); // speech-to-text: first option, whisper-large-v3
        app.on_key(press(KeyCode::Enter));
        assert_eq!(
            config_of(&app).stt_model.as_deref(),
            Some("whisper-large-v3")
        );
        assert_eq!(config_of(&app).chat_model.as_deref(), Some("llama-3-8b"));
    }
}

/// Tests for switching a running session to new settings.
#[cfg(test)]
mod session_tests {
    use super::*;
    use crate::chat::Backend;
    use crate::speech::SpeechConfig;
    use std::time::Duration;

    fn settings(stt_url: &str, stt_model: &str, clean_model: &str) -> Settings {
        Settings {
            clean: false,
            min_threshold: 0.01,
            language: Some("en".into()),
            verbose: false,
            endpoints: Endpoints {
                stt: stt_url.into(),
                ..Endpoints::default()
            },
            stt_model: stt_model.into(),
            clean_model: clean_model.into(),
            backend: Backend::Local {
                model: "m".into(),
                think: false,
            },
            system_prompt: None,
            profile: "fast".into(),
            profiles: vec!["fast".into()],
            speak: false,
            speech: SpeechConfig::system("en"),
            talk: false,
        }
    }

    fn unused_resolver() -> Resolver {
        Arc::new(|_, _| anyhow::bail!("not used in this test"))
    }

    /// After new settings are applied, speech is transcribed at the new address with the new
    /// model name, even though the session started with an address that goes nowhere.
    #[tokio::test]
    async fn applying_settings_rebuilds_the_transcriber() {
        let (base, request) =
            crate::test_support::capture_once("application/json", r#"{"text":"hello there"}"#)
                .await;
        let old = settings("http://127.0.0.1:1/old-stt", "old-whisper", "old-clean");
        let mut session = Session::new(&old, unused_resolver(), Config::default());
        let mut app = App::new(false);

        let new = settings(&format!("{base}/new-stt"), "new-whisper", "new-clean");
        session.apply(&mut app, &new);

        session
            .jobs
            .send(Job {
                id: 0,
                wav: b"RIFFfakewav".to_vec(),
            })
            .unwrap();
        let outcome = tokio::time::timeout(Duration::from_secs(5), session.outcomes.recv())
            .await
            .expect("the new transcriber should answer")
            .expect("the outcome channel should stay open");
        assert_eq!(
            outcome,
            Outcome::Text {
                id: 0,
                text: "hello there".into()
            }
        );
        let sent = request.await.unwrap();
        assert!(sent.starts_with("POST /new-stt "), "{sent}");
        assert!(
            sent.contains("new-whisper") && !sent.contains("old-whisper"),
            "{sent}"
        );
    }

    /// The other settings that the session keeps for later use are refreshed too.
    #[tokio::test]
    async fn applying_settings_refreshes_endpoints_and_clean_model() {
        let old = settings("http://127.0.0.1:1/a", "w", "old-clean");
        let mut session = Session::new(&old, unused_resolver(), Config::default());
        let mut app = App::new(false);
        let new = settings("http://127.0.0.1:1/b", "w", "new-clean");
        session.apply(&mut app, &new);
        assert_eq!(session.endpoints.stt, "http://127.0.0.1:1/b");
        assert_eq!(session.clean_model, "new-clean");
        assert_eq!(app.profile, "fast");
    }
}
