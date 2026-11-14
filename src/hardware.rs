//! What this machine can run: memory, GPU and disk, and the model size that suits it.

use std::process::Command;

const GIB: u64 = 1 << 30;

/// The GPU, as far as running models is concerned.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Gpu {
    /// No usable GPU found: models run on the CPU.
    #[default]
    None,
    /// A card with its own memory (NVIDIA, or a discrete AMD card).
    Dedicated { name: String, vram_bytes: u64 },
    /// A GPU that shares system memory (Apple Silicon, AMD APUs); `bytes` is how much of it the
    /// GPU can use for models.
    Shared { name: String, bytes: u64 },
}

/// What was found on this machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hardware {
    pub os: &'static str,
    pub arch: &'static str,
    pub cores: usize,
    pub ram_bytes: Option<u64>,
    pub gpu: Gpu,
    /// Free space in the home folder, where models are usually stored.
    pub disk_free_bytes: Option<u64>,
}

/// How big the models can be on this machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// Under 8 GiB for models: small models only.
    Minimal,
    /// 8 to 16 GiB.
    Light,
    /// 16 to 24 GiB.
    Standard,
    /// 24 GiB or more: room for a large fast model, and a reasoning model.
    Power,
}

impl Tier {
    /// The tier for `bytes` of memory available to models.
    #[must_use]
    pub fn for_memory(bytes: u64) -> Self {
        match bytes / GIB {
            0..=7 => Self::Minimal,
            8..=15 => Self::Light,
            16..=23 => Self::Standard,
            _ => Self::Power,
        }
    }

    /// A short name for the report.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Light => "light",
            Self::Standard => "standard",
            Self::Power => "power",
        }
    }
}

impl Hardware {
    /// Memory that models can use at a reasonable speed: a card's VRAM, the shared pool for
    /// Apple Silicon and AMD APUs, or half of the RAM when running on the CPU alone.
    #[must_use]
    pub fn model_memory(&self) -> Option<u64> {
        match &self.gpu {
            Gpu::Dedicated { vram_bytes, .. } => Some(*vram_bytes),
            Gpu::Shared { bytes, .. } => Some(*bytes),
            Gpu::None => self.ram_bytes.map(|ram| ram / 2),
        }
    }

    /// The size class that suits this machine, if its memory is known.
    #[must_use]
    pub fn tier(&self) -> Option<Tier> {
        self.model_memory().map(Tier::for_memory)
    }

    /// True when there is no GPU, so replies will be noticeably slower.
    #[must_use]
    pub fn cpu_only(&self) -> bool {
        self.gpu == Gpu::None
    }

    /// One line describing the machine.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut parts = vec![
            format!("{} {}", self.os, self.arch),
            format!("{} cores", self.cores),
        ];
        if let Some(ram) = self.ram_bytes {
            parts.push(format!("{} RAM", gib(ram)));
        }
        parts.push(match &self.gpu {
            Gpu::None => "no GPU found".to_string(),
            Gpu::Dedicated { name, vram_bytes } => format!("{name} ({} VRAM)", gib(*vram_bytes)),
            Gpu::Shared { name, bytes } => {
                format!("{name} (shares memory, {} usable)", gib(*bytes))
            }
        });
        if let Some(free) = self.disk_free_bytes {
            parts.push(format!("{} free", gib(free)));
        }
        parts.join(" · ")
    }
}

/// A size in GiB for people, e.g. `80 GB`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn gib(bytes: u64) -> String {
    let value = bytes as f64 / GIB as f64;
    if value >= 10.0 {
        format!("{value:.0} GB")
    } else {
        format!("{value:.1} GB")
    }
}

/// `MemTotal` from the text of `/proc/meminfo`, in bytes.
#[must_use]
pub fn parse_meminfo(text: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    kib.checked_mul(1024)
}

/// The largest GPU in `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`
/// output (memory in MiB), as `(name, vram_bytes)`.
#[must_use]
pub fn parse_nvidia_smi(output: &str) -> Option<(String, u64)> {
    output
        .lines()
        .filter_map(|line| {
            let (name, mib) = line.rsplit_once(',')?;
            let mib: u64 = mib.trim().parse().ok()?;
            Some((name.trim().to_string(), mib * 1024 * 1024))
        })
        .max_by_key(|(_, bytes)| *bytes)
}

/// Free bytes from `df -Pk PATH` output.
#[must_use]
pub fn parse_df(output: &str) -> Option<u64> {
    let line = output.lines().nth(1)?;
    let kib: u64 = line.split_whitespace().nth(3)?.parse().ok()?;
    kib.checked_mul(1024)
}

/// The name of an AMD display adapter from `lspci` output, e.g. `Strix [Radeon 880M / 890M]`.
#[must_use]
pub fn parse_lspci_amd(output: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let lower = line.to_lowercase();
        let is_display = lower.contains("vga") || lower.contains("3d") || lower.contains("display");
        if !is_display || !line.contains("AMD/ATI") {
            return None;
        }
        let after = line.split_once("[AMD/ATI]")?.1.trim();
        let name = after.split(" (rev").next()?.trim();
        (!name.is_empty()).then(|| name.to_string())
    })
}

/// Classify an AMD GPU from its dedicated (`vram`) and shared (`gtt`) memory sizes in bytes.
///
/// An APU has a small carve-out of dedicated memory but a large shared pool, which is what
/// models can really use; a discrete card is measured by its own VRAM.
#[must_use]
pub fn classify_amd(name: &str, vram: u64, gtt: u64, ram: Option<u64>) -> Gpu {
    if vram < 4 * GIB && gtt >= vram.saturating_mul(4) && gtt >= 4 * GIB {
        // Never count more than most of the machine's RAM.
        let cap = ram.map_or(u64::MAX, |r| r / 10 * 8);
        Gpu::Shared {
            name: name.to_string(),
            bytes: gtt.min(cap),
        }
    } else {
        Gpu::Dedicated {
            name: name.to_string(),
            vram_bytes: vram,
        }
    }
}

/// Free bytes on the disk that holds `path` (a folder that exists), from `df`. `None` if it
/// cannot be found out.
#[must_use]
pub fn free_space(path: &std::path::Path) -> Option<u64> {
    run("df", &["-Pk", &path.to_string_lossy()]).and_then(|out| parse_df(&out))
}

fn run(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).to_string())
}

fn read_number(path: &std::path::Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// The AMD GPU's memory from the kernel's `amdgpu` files, as `(vram, gtt)` in bytes.
fn amd_sysfs() -> Option<(u64, u64)> {
    let cards = std::fs::read_dir("/sys/class/drm").ok()?;
    cards
        .filter_map(Result::ok)
        .filter_map(|card| {
            let device = card.path().join("device");
            Some((
                read_number(&device.join("mem_info_vram_total"))?,
                read_number(&device.join("mem_info_gtt_total"))?,
            ))
        })
        .max_by_key(|(vram, gtt)| (*gtt).max(*vram))
}

fn detect_gpu(ram: Option<u64>) -> Gpu {
    if cfg!(target_os = "macos") {
        if std::env::consts::ARCH != "aarch64" {
            return Gpu::None;
        }
        let name = run("sysctl", &["-n", "machdep.cpu.brand_string"])
            .map_or_else(|| "Apple Silicon".to_string(), |s| s.trim().to_string());
        // The GPU may use roughly two thirds to three quarters of unified memory.
        return Gpu::Shared {
            name,
            bytes: ram.map_or(0, |r| r / 10 * 7),
        };
    }
    if let Some((name, vram_bytes)) = run(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
    )
    .and_then(|out| parse_nvidia_smi(&out))
    {
        return Gpu::Dedicated { name, vram_bytes };
    }
    if let Some((vram, gtt)) = amd_sysfs() {
        let name = run("lspci", &[])
            .and_then(|out| parse_lspci_amd(&out))
            .unwrap_or_else(|| "AMD GPU".to_string());
        return classify_amd(&name, vram, gtt, ram);
    }
    Gpu::None
}

fn detect_ram() -> Option<u64> {
    if cfg!(target_os = "macos") {
        run("sysctl", &["-n", "hw.memsize"])?.trim().parse().ok()
    } else {
        parse_meminfo(&std::fs::read_to_string("/proc/meminfo").ok()?)
    }
}

/// Look at this machine.
#[must_use]
pub fn detect() -> Hardware {
    let ram_bytes = detect_ram();
    let disk_free_bytes = std::env::var("HOME")
        .ok()
        .and_then(|home| run("df", &["-Pk", &home]))
        .and_then(|out| parse_df(&out));
    Hardware {
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        cores: std::thread::available_parallelism().map_or(1, std::num::NonZero::get),
        ram_bytes,
        gpu: detect_gpu(ram_bytes),
        disk_free_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tier boundaries are in GiB of memory available to models.
    #[test]
    fn tiers_by_memory() {
        let tier = |gib: u64| Tier::for_memory(gib * GIB);
        assert_eq!(tier(0), Tier::Minimal);
        assert_eq!(tier(7), Tier::Minimal);
        assert_eq!(tier(8), Tier::Light);
        assert_eq!(tier(15), Tier::Light);
        assert_eq!(tier(16), Tier::Standard);
        assert_eq!(tier(23), Tier::Standard);
        assert_eq!(tier(24), Tier::Power);
        assert_eq!(tier(80), Tier::Power);
        assert!(Tier::Minimal < Tier::Power);
    }

    /// Your machine, read from the real values on it: 0.5 GiB dedicated and an 80 GiB shared
    /// pool on a 125 GiB machine is an APU, usable at the full shared size.
    #[test]
    fn amd_apu_uses_its_shared_pool() {
        let gpu = classify_amd("Strix", 536_870_912, 85_899_345_920, Some(134_127_000_000));
        assert_eq!(
            gpu,
            Gpu::Shared {
                name: "Strix".into(),
                bytes: 85_899_345_920
            }
        );
    }

    /// The shared pool is never counted above 80% of the RAM.
    #[test]
    fn apu_pool_is_capped_by_ram() {
        let gpu = classify_amd("Small", GIB / 2, 64 * GIB, Some(16 * GIB));
        let Gpu::Shared { bytes, .. } = gpu else {
            panic!("expected shared")
        };
        assert!(bytes <= 16 * GIB / 10 * 8);
    }

    /// A discrete AMD card is measured by its own VRAM.
    #[test]
    fn discrete_amd_uses_vram() {
        let gpu = classify_amd("RX", 24 * GIB, 32 * GIB, Some(64 * GIB));
        assert_eq!(
            gpu,
            Gpu::Dedicated {
                name: "RX".into(),
                vram_bytes: 24 * GIB
            }
        );
    }

    /// `/proc/meminfo` is read in KiB.
    #[test]
    fn parses_meminfo() {
        let text = "MemTotal:       130984308 kB\nMemFree:        1000 kB\n";
        assert_eq!(parse_meminfo(text), Some(130_984_308 * 1024));
        assert_eq!(parse_meminfo("nothing here"), None);
        assert_eq!(parse_meminfo("MemTotal: lots kB"), None);
    }

    /// With several NVIDIA cards the largest one counts; bad lines are skipped.
    #[test]
    fn parses_nvidia_smi() {
        let out = "NVIDIA GeForce RTX 3060, 12288\nNVIDIA GeForce RTX 4090, 24564\n";
        assert_eq!(
            parse_nvidia_smi(out),
            Some(("NVIDIA GeForce RTX 4090".to_string(), 24_564 * 1024 * 1024))
        );
        assert_eq!(parse_nvidia_smi(""), None);
        assert_eq!(parse_nvidia_smi("No devices were found"), None);
        assert_eq!(parse_nvidia_smi("name, N/A"), None);
    }

    /// Free space comes from the "Available" column of `df -Pk`.
    /// The free space of a folder that exists is found with the real `df` on this machine.
    #[test]
    #[cfg(unix)]
    fn free_space_of_a_real_folder_is_known() {
        let free = free_space(&std::env::temp_dir());
        assert!(free.is_some_and(|bytes| bytes > 0), "{free:?}");
        assert_eq!(
            free_space(std::path::Path::new("/definitely/not/a/folder")),
            None
        );
    }

    #[test]
    fn parses_df() {
        let out = "Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/x 3795284464 39487576 3562932224 2% /\n";
        assert_eq!(parse_df(out), Some(3_562_932_224 * 1024));
        assert_eq!(parse_df("Filesystem only"), None);
    }

    /// The AMD adapter's name is taken from the `lspci` line.
    #[test]
    fn parses_lspci() {
        let out = "00:00.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Root Complex\n66:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Strix [Radeon 880M / 890M] (rev c1)\n";
        assert_eq!(
            parse_lspci_amd(out).as_deref(),
            Some("Strix [Radeon 880M / 890M]")
        );
        assert_eq!(
            parse_lspci_amd("01:00.0 VGA compatible controller: NVIDIA Corporation GA102"),
            None
        );
    }

    fn machine(gpu: Gpu, ram_gib: Option<u64>) -> Hardware {
        Hardware {
            os: "linux",
            arch: "x86_64",
            cores: 8,
            ram_bytes: ram_gib.map(|g| g * GIB),
            gpu,
            disk_free_bytes: Some(500 * GIB),
        }
    }

    /// Model memory is VRAM for a card, the shared pool for APUs and Macs, half the RAM on CPU.
    #[test]
    fn model_memory_by_gpu_kind() {
        let card = Gpu::Dedicated {
            name: "RTX".into(),
            vram_bytes: 12 * GIB,
        };
        assert_eq!(machine(card, Some(64)).model_memory(), Some(12 * GIB));
        let shared = Gpu::Shared {
            name: "M2".into(),
            bytes: 22 * GIB,
        };
        assert_eq!(machine(shared, Some(32)).tier(), Some(Tier::Standard));
        let cpu = machine(Gpu::None, Some(16));
        assert_eq!(cpu.model_memory(), Some(8 * GIB));
        assert!(cpu.cpu_only());
        assert_eq!(cpu.tier(), Some(Tier::Light));
        assert_eq!(machine(Gpu::None, None).tier(), None);
    }

    /// The one-line summary mentions each part that was found.
    #[test]
    fn summary_describes_the_machine() {
        let hw = machine(
            Gpu::Shared {
                name: "Strix".into(),
                bytes: 80 * GIB,
            },
            Some(125),
        );
        let text = hw.summary();
        for expected in [
            "linux x86_64",
            "8 cores",
            "125 GB RAM",
            "Strix",
            "80 GB usable",
            "500 GB free",
        ] {
            assert!(text.contains(expected), "missing {expected:?} in {text}");
        }
        assert!(
            machine(Gpu::None, Some(8))
                .summary()
                .contains("no GPU found")
        );
    }

    /// Sizes read naturally.
    #[test]
    fn formats_gib() {
        assert_eq!(gib(0), "0.0 GB");
        assert_eq!(gib(5 * GIB), "5.0 GB");
        assert_eq!(gib(24 * GIB), "24 GB");
        assert_eq!(gib(125 * GIB), "125 GB");
    }

    /// Detection works on whatever machine the tests run on, and never panics.
    #[test]
    fn detect_runs_here() {
        let hw = detect();
        assert!(hw.cores >= 1);
        assert!(!hw.os.is_empty() && !hw.arch.is_empty());
    }
}
