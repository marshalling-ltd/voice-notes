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
use crate::pipeline::{Job, Outcome, Settings, spawn_transcriber};
use crate::transcribe::{TranscribeOptions, clean_transcript};
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph, Sparkline, Wrap};
use ratatui::{DefaultTerminal, Frame};
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;

const ACCENT: Color = Color::Cyan;
const GOOD: Color = Color::Green;
const WARN: Color = Color::Yellow;
const BAD: Color = Color::Red;
const DIM: Color = Color::DarkGray;

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
    /// Everything finished.
    Done,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Start,
    Stop,
}

/// Everything the screen shows.
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
            next_id: 0,
            started: None,
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
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => self.quit = true,
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            KeyCode::Enter | KeyCode::Char(' ') => {
                return match self.phase {
                    Phase::Idle | Phase::Done => Some(Action::Start),
                    Phase::Calibrating | Phase::Listening => Some(Action::Stop),
                    Phase::Finishing | Phase::Cleaning => None,
                };
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

    fn scroll_up(&mut self, n: usize) {
        self.scroll = (self.scroll + n).min(self.entries.len().saturating_sub(1));
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

    let clean = if app.clean { "on" } else { "off" };
    let clean_color = if app.clean { GOOD } else { DIM };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("clean: ", Style::new().fg(DIM)),
            Span::styled(clean, Style::new().fg(clean_color)),
        ]))
        .alignment(Alignment::Right),
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
    if let Some(cleaned) = &app.cleaned {
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
    let width = width.max(1);
    let end = entries.len().saturating_sub(scroll);
    let mut used = 0;
    let mut picked: Vec<Line<'static>> = Vec::new();
    for entry in entries[..end].iter().rev() {
        let line = entry_line(entry);
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

fn draw_footer(frame: &mut Frame, area: Rect, app: &App) {
    let key =
        |k: &'static str| Span::styled(k, Style::new().fg(ACCENT).add_modifier(Modifier::BOLD));
    let text = |t: &'static str| Span::styled(t, Style::new().fg(DIM));
    let action = if app.recording() { " stop" } else { " start" };
    let mut spans = vec![
        Span::raw(" "),
        key("Enter"),
        text(action),
        text("  ·  "),
        key("c"),
        text(" clean"),
        text("  ·  "),
        key("↑↓"),
        text(" scroll"),
        text("  ·  "),
        key("q"),
        text(" quit"),
    ];
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
pub async fn run(settings: Settings) -> Result<String> {
    let mut terminal = ratatui::init();
    let result = event_loop(&mut terminal, settings).await;
    ratatui::restore();
    result
}

async fn event_loop(terminal: &mut DefaultTerminal, settings: Settings) -> Result<String> {
    let mut recorder = AudioRecorder::new(settings.min_threshold, false);
    let client = reqwest::Client::new();
    let opts = TranscribeOptions {
        language: settings.language.clone(),
        prompt: None,
    };
    let (tx, mut rx) = spawn_transcriber(client.clone(), opts);
    let mut app = App::new(settings.clean);
    let mut cleaning: Option<JoinHandle<Result<String>>> = None;
    let mut tick = tokio::time::interval(Duration::from_millis(50));

    while !app.quit {
        terminal.draw(|frame| draw(frame, &app))?;
        tick.tick().await;

        while let Ok(outcome) = rx.try_recv() {
            app.on_outcome(outcome);
        }

        if app.recording() {
            match recorder.poll_segments() {
                Ok(segments) => {
                    for wav in segments {
                        let id = app.queue();
                        let _ = tx.send(Job { id, wav });
                    }
                    app.on_tick(
                        recorder.level(),
                        recorder.calibrated_threshold(),
                        recorder.is_speaking(),
                    );
                }
                Err(e) => {
                    app.notice = Some(format!("recording failed: {e:#}"));
                    let _ = recorder.stop_recording();
                    app.phase = Phase::Idle;
                }
            }
        }

        while event::poll(Duration::ZERO)? {
            let Event::Key(key) = event::read()? else {
                continue;
            };
            match app.on_key(key) {
                Some(Action::Start) => match recorder.start_recording() {
                    Ok(()) => app.begin(),
                    Err(e) => app.notice = Some(format!("cannot start: {e:#}")),
                },
                Some(Action::Stop) => {
                    stop_and_queue(&mut recorder, &mut app, &tx);
                }
                None => {}
            }
        }

        if app.quit && app.recording() {
            let _ = recorder.stop_recording();
        }

        if app.phase == Phase::Finishing && app.pending() == 0 {
            let text = app.transcript();
            if app.clean && !text.is_empty() {
                app.phase = Phase::Cleaning;
                let client = client.clone();
                cleaning = Some(tokio::spawn(async move {
                    clean_transcript(&client, &text).await
                }));
            } else {
                app.phase = Phase::Done;
            }
        }

        if cleaning.as_ref().is_some_and(JoinHandle::is_finished)
            && let Some(handle) = cleaning.take()
        {
            match handle.await {
                Ok(Ok(text)) => app.cleaned = Some(text),
                Ok(Err(e)) => app.notice = Some(format!("cleaning failed: {e:#}")),
                Err(e) => app.notice = Some(format!("cleaning failed: {e}")),
            }
            app.phase = Phase::Done;
        }
    }
    Ok(app.final_text())
}

/// Stop the recorder and send any trailing utterance to the transcriber.
fn stop_and_queue(
    recorder: &mut AudioRecorder,
    app: &mut App,
    tx: &tokio::sync::mpsc::UnboundedSender<Job>,
) {
    match recorder.stop_recording() {
        Ok(Some(wav)) => {
            let id = app.queue();
            let _ = tx.send(Job { id, wav });
        }
        Ok(None) => {}
        Err(e) => app.notice = Some(format!("stop failed: {e:#}")),
    }
    app.stopped();
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
