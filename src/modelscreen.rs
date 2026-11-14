//! The state of the models screen: which models a size class starts with, which of them are on
//! disk, and what the keys do. The screen itself is drawn by `ui.rs`; downloading is done by
//! [`crate::download`]. Nothing here draws or touches the network, so it is tested without a
//! terminal.

use crate::catalogue::{Catalogue, Job, Model};
use crate::download::Progress;
use crate::hardware::Tier;
use crate::setup::DownloadPick;
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind};
use std::path::{Path, PathBuf};

/// The size classes in order, smallest first.
const TIERS: [Tier; 4] = [Tier::Minimal, Tier::Light, Tier::Standard, Tier::Power];

/// Whether a model is on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstallState {
    /// Not downloaded.
    Missing,
    /// An earlier download was interrupted; `have` bytes are kept and will be continued.
    Partial { have: u64 },
    /// The file is there with exactly the right size. (It is not read again to check its sha256,
    /// which would take a while for a large model; the download checked it.)
    Installed,
}

/// Look at the models folder: is `model` there?
#[must_use]
pub fn install_state(model: &Model, dir: &Path) -> InstallState {
    let name = model.file_name();
    let size_of = |path: PathBuf| std::fs::metadata(path).ok().map(|m| m.len());
    if size_of(dir.join(name)) == Some(model.size_bytes) {
        return InstallState::Installed;
    }
    match size_of(dir.join(format!("{name}.part"))) {
        Some(have) if have > 0 && have < model.size_bytes => InstallState::Partial { have },
        _ => InstallState::Missing,
    }
}

/// One line of the screen: a job, the model for it, and whether it is on disk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelRow {
    pub job: Job,
    pub model: Model,
    pub state: InstallState,
}

/// The download that is running, as far as the screen knows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActiveDownload {
    pub model_id: String,
    /// `None` until the first report arrives.
    pub progress: Option<Progress>,
    /// How many more models are waiting behind this one.
    pub waiting: usize,
}

/// A model about to be downloaded, waiting for the user to accept its licence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingModel {
    pub model: Model,
    /// How many bytes this download has to fetch (less than the size if part is already kept).
    pub remaining_bytes: u64,
}

/// What the screen asks the program to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelsAction {
    /// Download these models, one after the other, by catalogue id.
    Download(Vec<String>),
    /// Stop the running download (and drop the ones waiting). What arrived is kept.
    Cancel,
}

/// The models screen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelsScreen {
    /// What was found out about this machine, one line.
    pub hardware: String,
    /// The class that fits this machine, or `None` if its memory could not be found out.
    pub detected: Option<Tier>,
    /// The class whose models are listed.
    pub viewing: Tier,
    pub rows: Vec<ModelRow>,
    /// The highlighted row.
    pub focus: usize,
    pub active: Option<ActiveDownload>,
    /// The models the user asked to download. While this is set the screen shows their licences,
    /// and nothing is downloaded until the user accepts.
    pub pending: Option<Vec<PendingModel>>,
    /// The choices from the setup screen that these downloads are for. Once they are installed
    /// and the server serves them, they are saved and used. Empty when the screen was opened by
    /// itself.
    pub picks: Vec<DownloadPick>,
    /// What happened after the downloads: what is now in use, or what is still to be done. Shown
    /// at the bottom of the screen.
    pub message: Option<String>,
    dir: PathBuf,
}

impl ModelsScreen {
    /// The screen for a machine of class `detected` (`None` if unknown: the smallest class is
    /// shown then, since its models run anywhere), with models looked for in `dir`.
    #[must_use]
    pub fn new(hardware: String, detected: Option<Tier>, dir: PathBuf) -> Self {
        let mut screen = Self {
            hardware,
            detected,
            viewing: detected.unwrap_or(Tier::Minimal),
            rows: Vec::new(),
            focus: 0,
            active: None,
            pending: None,
            picks: Vec::new(),
            message: None,
            dir,
        };
        screen.refresh();
        screen
    }

    /// The screen opened by the setup screen for models the user picked there: it starts on the
    /// licence question for `pending`, and remembers `picks` to use them once they are in place.
    #[must_use]
    pub fn for_picks(
        hardware: String,
        detected: Option<Tier>,
        dir: PathBuf,
        pending: Vec<PendingModel>,
        picks: Vec<DownloadPick>,
    ) -> Self {
        let mut screen = Self::new(hardware, detected, dir);
        screen.pending = Some(pending);
        screen.picks = picks;
        screen
    }

    /// The folder models are looked for in and downloaded to.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Rebuild the rows for the viewed class and look at the disk again.
    pub fn refresh(&mut self) {
        let set = Catalogue::builtin().set(self.viewing);
        self.rows = set
            .jobs()
            .into_iter()
            .map(|(job, model)| ModelRow {
                job,
                state: install_state(model, &self.dir),
                model: model.clone(),
            })
            .collect();
        self.focus = self.focus.min(self.rows.len().saturating_sub(1));
    }

    /// How many bytes the missing models of the viewed class still have to download, counting
    /// what is already kept of partial ones.
    #[must_use]
    pub fn bytes_to_download(&self) -> u64 {
        self.rows
            .iter()
            .map(|r| match r.state {
                InstallState::Missing => r.model.size_bytes,
                InstallState::Partial { have } => r.model.size_bytes - have,
                InstallState::Installed => 0,
            })
            .sum()
    }

    /// Show the next (`1`) or previous (`-1`) class, staying within the four.
    pub fn change_class(&mut self, step: i8) {
        let index = TIERS.iter().position(|t| *t == self.viewing).unwrap_or(0);
        let moved = if step < 0 {
            index.saturating_sub(1)
        } else {
            (index + 1).min(TIERS.len() - 1)
        };
        self.viewing = TIERS[moved];
        self.refresh();
    }

    /// Whether a download is running.
    #[must_use]
    pub fn busy(&self) -> bool {
        self.active.is_some()
    }

    /// Whether the licences of the models to download are being shown for the user to accept.
    #[must_use]
    pub fn confirming(&self) -> bool {
        self.pending.is_some()
    }

    /// Handle a key. Quitting, and closing the screen when no question is open, are left to the
    /// caller.
    pub fn on_key(&mut self, key: KeyEvent) -> Option<ModelsAction> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        if self.pending.is_some() {
            return self.on_confirm_key(key);
        }
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.focus = self.focus.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                self.focus = (self.focus + 1).min(self.rows.len().saturating_sub(1));
            }
            KeyCode::Left | KeyCode::Char('h') => self.change_class(-1),
            KeyCode::Right | KeyCode::Char('l') => self.change_class(1),
            KeyCode::Enter => self.ask_focused(),
            KeyCode::Char('a') => self.ask_missing(),
            KeyCode::Char('x') => return self.active.is_some().then_some(ModelsAction::Cancel),
            _ => {}
        }
        None
    }

    /// Ask to download the highlighted model, unless it is there already or another download is
    /// running. The licence is shown first; nothing is downloaded yet.
    fn ask_focused(&mut self) {
        if self.busy() {
            return;
        }
        self.ask(&[self.focus]);
    }

    /// Ask to download everything of the viewed class that is not there yet.
    fn ask_missing(&mut self) {
        if self.busy() {
            return;
        }
        let wanted: Vec<usize> = (0..self.rows.len()).collect();
        self.ask(&wanted);
    }

    /// Show the licences of these rows (those not installed yet), if there are any.
    fn ask(&mut self, wanted: &[usize]) {
        let pending: Vec<PendingModel> = wanted
            .iter()
            .filter_map(|&i| self.rows.get(i))
            .filter(|row| row.state != InstallState::Installed)
            .map(|row| PendingModel {
                model: row.model.clone(),
                remaining_bytes: match row.state {
                    InstallState::Partial { have } => row.model.size_bytes - have,
                    _ => row.model.size_bytes,
                },
            })
            .collect();
        self.pending = (!pending.is_empty()).then_some(pending);
    }

    /// Keys while the licences are shown: accept, or decline. Nothing else does anything, so
    /// the list behind the question cannot change under it.
    fn on_confirm_key(&mut self, key: KeyEvent) -> Option<ModelsAction> {
        match key.code {
            KeyCode::Enter | KeyCode::Char('y' | 'Y') => {
                let ids = self
                    .pending
                    .take()?
                    .into_iter()
                    .map(|p| p.model.id)
                    .collect();
                Some(ModelsAction::Download(ids))
            }
            KeyCode::Esc | KeyCode::Char('n' | 'N') => {
                self.pending = None;
                None
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::download::Phase;
    use crossterm::event::KeyModifiers;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    /// A models folder for one test, removed afterwards.
    struct Folder(PathBuf);

    impl Folder {
        fn new(label: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "voice-notes-modelscreen-{}-{label}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn put(&self, name: &str, bytes: u64) {
            let file = std::fs::File::create(self.0.join(name)).unwrap();
            file.set_len(bytes).unwrap();
        }
    }

    impl Drop for Folder {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn model(id: &str) -> Model {
        Catalogue::builtin().model(id).unwrap().clone()
    }

    fn screen(folder: &Folder, tier: Tier) -> ModelsScreen {
        ModelsScreen::new("test machine".into(), Some(tier), folder.0.clone())
    }

    // ---- what is installed

    #[test]
    fn a_model_is_installed_only_if_the_file_has_exactly_the_right_size() {
        let folder = Folder::new("installed");
        let small = model("whisper-small");
        assert_eq!(install_state(&small, &folder.0), InstallState::Missing);
        folder.put(small.file_name(), small.size_bytes - 1);
        assert_eq!(
            install_state(&small, &folder.0),
            InstallState::Missing,
            "too short is not installed"
        );
        folder.put(small.file_name(), small.size_bytes);
        assert_eq!(install_state(&small, &folder.0), InstallState::Installed);
    }

    #[test]
    fn an_interrupted_download_shows_how_much_is_kept() {
        let folder = Folder::new("partial");
        let small = model("whisper-small");
        folder.put(&format!("{}.part", small.file_name()), 1000);
        assert_eq!(
            install_state(&small, &folder.0),
            InstallState::Partial { have: 1000 }
        );
        folder.put(&format!("{}.part", small.file_name()), 0);
        assert_eq!(
            install_state(&small, &folder.0),
            InstallState::Missing,
            "an empty .part is nothing"
        );
        folder.put(&format!("{}.part", small.file_name()), small.size_bytes + 5);
        assert_eq!(
            install_state(&small, &folder.0),
            InstallState::Missing,
            "too long to continue"
        );
    }

    #[test]
    fn a_missing_folder_means_nothing_is_installed() {
        let nowhere = std::env::temp_dir().join("voice-notes-modelscreen-does-not-exist");
        assert_eq!(
            install_state(&model("qwen3-4b"), &nowhere),
            InstallState::Missing
        );
    }

    // ---- the rows

    #[test]
    fn the_rows_are_the_classes_set_in_job_order() {
        let folder = Folder::new("rows");
        let power = screen(&folder, Tier::Power);
        let jobs: Vec<Job> = power.rows.iter().map(|r| r.job).collect();
        assert_eq!(
            jobs,
            [
                Job::SpeechToText,
                Job::Chat,
                Job::Reasoning,
                Job::TextToSpeech
            ]
        );
        assert_eq!(power.rows[1].model.id, "qwen3-coder-30b-a3b");
        assert_eq!(
            screen(&folder, Tier::Minimal).rows.len(),
            3,
            "no reasoning model below power"
        );
    }

    #[test]
    fn the_rows_reflect_what_is_on_disk_and_refresh_notices_changes() {
        let folder = Folder::new("refresh");
        let mut s = screen(&folder, Tier::Minimal);
        assert!(s.rows.iter().all(|r| r.state == InstallState::Missing));
        let small = model("whisper-small");
        folder.put(small.file_name(), small.size_bytes);
        s.refresh();
        assert_eq!(s.rows[0].state, InstallState::Installed);
        assert_eq!(s.rows[1].state, InstallState::Missing);
    }

    #[test]
    fn the_amount_still_to_download_counts_kept_data_once() {
        let folder = Folder::new("bytes");
        let mut s = screen(&folder, Tier::Minimal);
        let all: u64 = s.rows.iter().map(|r| r.model.size_bytes).sum();
        assert_eq!(s.bytes_to_download(), all);
        let small = model("whisper-small");
        folder.put(small.file_name(), small.size_bytes);
        folder.put(
            &format!("{}.part", model("qwen3-4b").file_name()),
            1_000_000,
        );
        s.refresh();
        let expected = all - small.size_bytes - 1_000_000;
        assert_eq!(s.bytes_to_download(), expected);
    }

    // ---- choosing the class

    #[test]
    fn the_screen_starts_on_the_detected_class() {
        let folder = Folder::new("start");
        assert_eq!(screen(&folder, Tier::Standard).viewing, Tier::Standard);
        let unknown = ModelsScreen::new("?".into(), None, folder.0.clone());
        assert_eq!(unknown.viewing, Tier::Minimal, "the smallest runs anywhere");
        assert_eq!(unknown.detected, None);
    }

    #[test]
    fn left_and_right_change_the_class_and_stop_at_the_ends() {
        let folder = Folder::new("class");
        let mut s = screen(&folder, Tier::Light);
        s.on_key(press(KeyCode::Right));
        assert_eq!(s.viewing, Tier::Standard);
        s.on_key(press(KeyCode::Right));
        s.on_key(press(KeyCode::Right));
        assert_eq!(s.viewing, Tier::Power, "no wrapping round");
        assert_eq!(
            s.detected,
            Some(Tier::Light),
            "what fits this machine does not change"
        );
        for _ in 0..5 {
            s.on_key(press(KeyCode::Left));
        }
        assert_eq!(s.viewing, Tier::Minimal);
    }

    #[test]
    fn changing_class_keeps_the_focus_on_a_row_that_exists() {
        let folder = Folder::new("focus");
        let mut s = screen(&folder, Tier::Power);
        for _ in 0..3 {
            s.on_key(press(KeyCode::Down));
        }
        assert_eq!(s.focus, 3);
        s.on_key(press(KeyCode::Left));
        s.on_key(press(KeyCode::Left));
        assert!(s.focus < s.rows.len(), "{} of {}", s.focus, s.rows.len());
    }

    #[test]
    fn up_and_down_stop_at_the_ends() {
        let folder = Folder::new("rows-keys");
        let mut s = screen(&folder, Tier::Minimal);
        s.on_key(press(KeyCode::Up));
        assert_eq!(s.focus, 0);
        for _ in 0..10 {
            s.on_key(press(KeyCode::Down));
        }
        assert_eq!(s.focus, 2);
    }

    // ---- asking, and the licences

    fn ids(pending: &[PendingModel]) -> Vec<&str> {
        pending.iter().map(|p| p.model.id.as_str()).collect()
    }

    #[test]
    fn enter_asks_about_the_highlighted_model_but_downloads_nothing() {
        let folder = Folder::new("enter");
        let mut s = screen(&folder, Tier::Minimal);
        s.on_key(press(KeyCode::Down));
        assert_eq!(
            s.on_key(press(KeyCode::Enter)),
            None,
            "nothing is downloaded yet"
        );
        assert!(s.confirming());
        assert_eq!(ids(s.pending.as_ref().unwrap()), ["qwen3-4b"]);
    }

    #[test]
    fn accepting_starts_the_download() {
        let folder = Folder::new("accept");
        for accept in [KeyCode::Enter, KeyCode::Char('y'), KeyCode::Char('Y')] {
            let mut s = screen(&folder, Tier::Minimal);
            s.on_key(press(KeyCode::Down));
            s.on_key(press(KeyCode::Enter));
            assert_eq!(
                s.on_key(press(accept)),
                Some(ModelsAction::Download(vec!["qwen3-4b".into()])),
                "{accept:?}"
            );
            assert!(!s.confirming(), "the question is closed once answered");
        }
    }

    #[test]
    fn declining_downloads_nothing_and_leaves_the_list_as_it_was() {
        let folder = Folder::new("decline");
        for decline in [KeyCode::Esc, KeyCode::Char('n'), KeyCode::Char('N')] {
            let mut s = screen(&folder, Tier::Minimal);
            s.on_key(press(KeyCode::Down));
            s.on_key(press(KeyCode::Enter));
            assert_eq!(s.on_key(press(decline)), None, "{decline:?}");
            assert!(!s.confirming());
            assert_eq!((s.focus, s.viewing), (1, Tier::Minimal));
            // And it can be asked again.
            s.on_key(press(KeyCode::Enter));
            assert!(s.confirming(), "declining once does not block asking again");
        }
    }

    /// While the licences are on screen no other key may change the list or start anything.
    #[test]
    fn other_keys_do_nothing_while_the_licences_are_shown() {
        let folder = Folder::new("frozen");
        let mut s = screen(&folder, Tier::Light);
        s.on_key(press(KeyCode::Char('a')));
        let before = s.clone();
        for code in [
            KeyCode::Down,
            KeyCode::Up,
            KeyCode::Left,
            KeyCode::Right,
            KeyCode::Char('a'),
            KeyCode::Char('x'),
            KeyCode::Char('z'),
            KeyCode::Tab,
            KeyCode::Char('j'),
            KeyCode::Char('h'),
        ] {
            assert_eq!(s.on_key(press(code)), None, "{code:?}");
        }
        assert_eq!(s, before, "nothing changed");
    }

    #[test]
    fn key_releases_do_not_answer_the_question() {
        let folder = Folder::new("release");
        let mut s = screen(&folder, Tier::Minimal);
        s.on_key(press(KeyCode::Enter));
        let release =
            KeyEvent::new_with_kind(KeyCode::Enter, KeyModifiers::NONE, KeyEventKind::Release);
        assert_eq!(s.on_key(release), None);
        assert!(s.confirming(), "a release is not an acceptance");
    }

    #[test]
    fn enter_on_a_partial_download_asks_and_counts_only_what_is_left() {
        let folder = Folder::new("enter-partial");
        let small = model("whisper-small");
        folder.put(&format!("{}.part", small.file_name()), 4096);
        let mut s = screen(&folder, Tier::Minimal);
        s.on_key(press(KeyCode::Enter));
        let pending = s.pending.as_ref().unwrap();
        assert_eq!(ids(pending), ["whisper-small"]);
        assert_eq!(pending[0].remaining_bytes, small.size_bytes - 4096);
        assert_eq!(
            s.on_key(press(KeyCode::Enter)),
            Some(ModelsAction::Download(vec!["whisper-small".into()]))
        );
    }

    #[test]
    fn a_asks_about_everything_missing_in_order_and_nothing_else() {
        let folder = Folder::new("all");
        let small = model("whisper-small");
        folder.put(small.file_name(), small.size_bytes);
        let mut s = screen(&folder, Tier::Minimal);
        assert_eq!(s.on_key(press(KeyCode::Char('a'))), None);
        assert_eq!(
            ids(s.pending.as_ref().unwrap()),
            ["qwen3-4b", "kokoro-v1_0"],
            "the installed one is not asked about"
        );
        assert_eq!(
            s.on_key(press(KeyCode::Enter)),
            Some(ModelsAction::Download(vec![
                "qwen3-4b".into(),
                "kokoro-v1_0".into()
            ]))
        );
    }

    #[test]
    fn every_pending_model_carries_its_licence() {
        let folder = Folder::new("licences");
        let mut s = screen(&folder, Tier::Power);
        s.on_key(press(KeyCode::Char('a')));
        let pending = s.pending.as_ref().unwrap();
        assert_eq!(pending.len(), 4);
        for p in pending {
            assert!(!p.model.licence.is_empty(), "{}", p.model.id);
            assert!(
                p.model
                    .licence_url
                    .as_deref()
                    .is_some_and(|u| u.starts_with("https://")),
                "{}",
                p.model.id
            );
        }
    }

    #[test]
    fn nothing_is_asked_about_a_model_that_is_already_there() {
        let folder = Folder::new("enter-installed");
        let small = model("whisper-small");
        folder.put(small.file_name(), small.size_bytes);
        let mut s = screen(&folder, Tier::Minimal);
        assert_eq!(s.on_key(press(KeyCode::Enter)), None);
        assert!(
            !s.confirming(),
            "there is nothing to download, so no licence to accept"
        );
    }

    #[test]
    fn a_has_nothing_to_ask_when_everything_is_installed() {
        let folder = Folder::new("all-done");
        for row in &screen(&folder, Tier::Minimal).rows {
            folder.put(row.model.file_name(), row.model.size_bytes);
        }
        let mut s = screen(&folder, Tier::Minimal);
        assert_eq!(s.on_key(press(KeyCode::Char('a'))), None);
        assert!(!s.confirming());
        assert_eq!(s.bytes_to_download(), 0);
    }

    #[test]
    fn nothing_is_asked_while_a_download_is_running() {
        let folder = Folder::new("busy");
        let mut s = screen(&folder, Tier::Minimal);
        s.active = Some(ActiveDownload {
            model_id: "whisper-small".into(),
            progress: None,
            waiting: 0,
        });
        assert!(s.busy());
        assert_eq!(s.on_key(press(KeyCode::Enter)), None);
        assert_eq!(s.on_key(press(KeyCode::Char('a'))), None);
        assert!(!s.confirming());
    }

    #[test]
    fn x_cancels_only_while_something_is_downloading() {
        let folder = Folder::new("cancel");
        let mut s = screen(&folder, Tier::Minimal);
        assert_eq!(s.on_key(press(KeyCode::Char('x'))), None);
        s.active = Some(ActiveDownload {
            model_id: "whisper-small".into(),
            progress: Some(Progress {
                phase: Phase::Downloading,
                done: 5,
                total: 10,
            }),
            waiting: 2,
        });
        assert_eq!(
            s.on_key(press(KeyCode::Char('x'))),
            Some(ModelsAction::Cancel)
        );
    }

    #[test]
    fn key_releases_and_other_keys_do_nothing() {
        let folder = Folder::new("other");
        let mut s = screen(&folder, Tier::Minimal);
        let release =
            KeyEvent::new_with_kind(KeyCode::Enter, KeyModifiers::NONE, KeyEventKind::Release);
        assert_eq!(s.on_key(release), None);
        for code in [
            KeyCode::Char('z'),
            KeyCode::Tab,
            KeyCode::F(3),
            KeyCode::Esc,
            KeyCode::Char('q'),
        ] {
            assert_eq!(s.on_key(press(code)), None, "{code:?}");
        }
        assert_eq!((s.focus, s.viewing), (0, Tier::Minimal));
    }
}
