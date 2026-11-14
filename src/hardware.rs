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
    ///
    /// Memory is counted 2% generously: graphics cards report a little less than their nominal
    /// size (a "24 GB" card shows 24 564 MiB, a "16 GB" one 16 376 MiB), and such a card belongs
    /// in the class its box says, not the one below.
    #[must_use]
    pub fn for_memory(bytes: u64) -> Self {
        match bytes.saturating_add(bytes / 50) / GIB {
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

    /// Which rule gave [`model_memory`](Self::model_memory), in words.
    #[must_use]
    pub fn memory_rule(&self) -> &'static str {
        match &self.gpu {
            Gpu::Dedicated { .. } => "the graphics card's own memory",
            Gpu::Shared { .. } if self.os == "macos" => {
                "70% of the unified memory, which the GPU can use"
            }
            Gpu::Shared { .. } => "the memory the GPU shares with the system",
            Gpu::None => "half of the RAM (no GPU found, so models run on the CPU)",
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

/// The memory size in bytes from `sysctl -n hw.memsize` output, which is one number, e.g.
/// `8589934592` on an 8 GB Mac.
#[must_use]
pub fn parse_sysctl_memsize(output: &str) -> Option<u64> {
    output.trim().parse().ok().filter(|bytes| *bytes > 0)
}

/// The chip name from `sysctl -n machdep.cpu.brand_string` output, e.g. `Apple M2 Pro`.
#[must_use]
pub fn parse_sysctl_brand(output: &str) -> Option<String> {
    let name = output.trim();
    (!name.is_empty()).then(|| name.to_string())
}

/// How much of an Apple Silicon Mac's unified memory models can use: the GPU may wire roughly
/// two thirds to three quarters of it, so 70%.
#[must_use]
pub fn apple_gpu_pool(ram_bytes: u64) -> u64 {
    ram_bytes / 10 * 7
}

/// The GPU of an Apple Silicon Mac with `ram_bytes` of unified memory and the chip name from
/// [`parse_sysctl_brand`]. Without the memory size nothing can be said about the pool, so the
/// machine counts as having no GPU rather than as having a pool of zero.
#[must_use]
pub fn apple_silicon_gpu(brand: Option<&str>, ram_bytes: Option<u64>) -> Gpu {
    // Run from an Intel build under Rosetta, the brand string is not the chip's name.
    let name = brand
        .filter(|b| b.starts_with("Apple"))
        .unwrap_or("Apple Silicon");
    match ram_bytes {
        Some(ram) => Gpu::Shared {
            name: name.to_string(),
            bytes: apple_gpu_pool(ram),
        },
        None => Gpu::None,
    }
}

/// Whether `sysctl -n hw.optional.arm64` says this Mac has an Apple Silicon chip: it prints `1`
/// there, even to an Intel program running under Rosetta, and nothing (or an error) on an Intel
/// Mac.
#[must_use]
pub fn parse_sysctl_arm64(output: &str) -> bool {
    output.trim() == "1"
}

/// A graphics adapter from `system_profiler SPDisplaysDataType`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacGpu {
    pub name: String,
    /// `PCIe` for a card, `Built-In` for graphics that are part of the chip.
    pub bus: String,
    /// The `VRAM (Total)` of a card with memory of its own. Integrated graphics print
    /// `VRAM (Dynamic, Max)` instead, which is a share of system memory and is not counted.
    pub vram_bytes: Option<u64>,
}

/// A memory size as `system_profiler` prints it: `4 GB`, `8 GB`, `1536 MB`.
fn parse_mac_size(text: &str) -> Option<u64> {
    let (number, unit) = text.trim().split_once(' ')?;
    let number: u64 = number.parse().ok()?;
    match unit.trim() {
        "GB" => number.checked_mul(GIB),
        "MB" => number.checked_mul(1 << 20),
        _ => None,
    }
}

/// The graphics adapters in `system_profiler SPDisplaysDataType` output. Each adapter starts at
/// its `Chipset Model:` line; the lines after it describe it.
#[must_use]
pub fn parse_system_profiler_gpus(output: &str) -> Vec<MacGpu> {
    let mut gpus: Vec<MacGpu> = Vec::new();
    for line in output.lines() {
        let Some((key, value)) = line.trim().split_once(':') else {
            continue;
        };
        let value = value.trim();
        if key == "Chipset Model" {
            gpus.push(MacGpu {
                name: value.to_string(),
                bus: String::new(),
                vram_bytes: None,
            });
            continue;
        }
        let Some(gpu) = gpus.last_mut() else {
            continue;
        };
        match key {
            "Bus" => gpu.bus = value.to_string(),
            "VRAM (Total)" => gpu.vram_bytes = parse_mac_size(value),
            _ => {}
        }
    }
    gpus
}

/// The GPU of an Intel Mac: its biggest card with memory of its own, or none. The graphics that
/// are part of an Intel chip share system memory like the CPU does, so they add nothing.
#[must_use]
pub fn intel_mac_gpu(system_profiler: Option<&str>) -> Gpu {
    system_profiler
        .map_or_else(Vec::new, parse_system_profiler_gpus)
        .into_iter()
        .filter_map(|g| Some((g.name, g.vram_bytes.filter(|v| *v > 0)?)))
        .max_by_key(|(_, vram)| *vram)
        .map_or(Gpu::None, |(name, vram_bytes)| Gpu::Dedicated {
            name,
            vram_bytes,
        })
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

/// What was found out about this machine, for `voice-notes hardware`: the machine, the memory
/// models can use and why, and the class. With `probes` (see [`probes`]) also the raw output the
/// detection read, so that a wrong result can be reported with the text it came from.
#[must_use]
pub fn report(hardware: &Hardware, probes: Option<&[(String, String)]>) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, "This machine: {}", hardware.summary());
    if let (Some(memory), Some(tier)) = (hardware.model_memory(), hardware.tier()) {
        let _ = writeln!(
            out,
            "Memory for models: {} ({})",
            gib(memory),
            hardware.memory_rule()
        );
        let _ = writeln!(out, "Size class: {}", tier.label());
    } else {
        let _ = writeln!(
            out,
            "Memory for models: unknown (the memory size could not be read)"
        );
        let _ = writeln!(out, "Size class: unknown");
    }
    if let Some(probes) = probes {
        let _ = writeln!(out, "\nRaw output the detection read:");
        for (command, output) in probes {
            let _ = writeln!(out, "\n$ {command}");
            let text = output.trim_end();
            let _ = writeln!(
                out,
                "{}",
                if text.is_empty() { "(no output)" } else { text }
            );
        }
    }
    out
}

/// What a probe printed, in words: its output, or that there was none, or that the command could
/// not be run. Neither case is left as a blank, which would look like a missing piece of the
/// report. (A machine without a GPU runs `lspci` fine and gets no display adapters from it.)
fn spell_out(output: Option<String>) -> String {
    match output {
        None => "(not available: the command is missing or failed)".to_string(),
        Some(text) if text.trim().is_empty() => "(no output)".to_string(),
        Some(text) => text,
    }
}

/// The commands and files the detection reads on this system, with what each printed (or why it
/// printed nothing), for [`report`].
#[must_use]
pub fn probes() -> Vec<(String, String)> {
    let mut probes = Vec::new();
    let mut add = |what: &str, output: Option<String>| {
        probes.push((what.to_string(), spell_out(output)));
    };
    if cfg!(target_os = "macos") {
        add("sysctl -n hw.memsize", run("sysctl", &["-n", "hw.memsize"]));
        add(
            "sysctl -n machdep.cpu.brand_string",
            run("sysctl", &["-n", "machdep.cpu.brand_string"]),
        );
        add(
            "sysctl -n hw.optional.arm64",
            run("sysctl", &["-n", "hw.optional.arm64"]),
        );
        add(
            "system_profiler SPDisplaysDataType",
            run("system_profiler", &["SPDisplaysDataType"]),
        );
    } else {
        add(
            "head -n 3 /proc/meminfo",
            std::fs::read_to_string("/proc/meminfo")
                .ok()
                .map(|t| t.lines().take(3).collect::<Vec<_>>().join("\n")),
        );
        add(
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits",
            run(
                "nvidia-smi",
                &[
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
            ),
        );
        add(
            "lspci (display adapters)",
            run("lspci", &[]).map(|t| {
                t.lines()
                    .filter(|l| {
                        let l = l.to_lowercase();
                        l.contains("vga") || l.contains("3d") || l.contains("display")
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }),
        );
        add(
            "amdgpu memory (vram, gtt) in bytes",
            amd_sysfs().map(|(vram, gtt)| format!("vram {vram}\ngtt {gtt}")),
        );
    }
    let home = std::env::var("HOME").unwrap_or_default();
    add("df -Pk $HOME", run("df", &["-Pk", &home]));
    probes
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

/// Whether this is a Mac with an Apple Silicon chip, asked of the hardware, so that an Intel
/// build running under Rosetta is still recognised.
fn is_apple_silicon() -> bool {
    run("sysctl", &["-n", "hw.optional.arm64"]).is_some_and(|out| parse_sysctl_arm64(&out))
}

fn detect_gpu(ram: Option<u64>) -> Gpu {
    if cfg!(target_os = "macos") {
        if is_apple_silicon() {
            let brand = run("sysctl", &["-n", "machdep.cpu.brand_string"])
                .and_then(|out| parse_sysctl_brand(&out));
            return apple_silicon_gpu(brand.as_deref(), ram);
        }
        // An Intel Mac: a discrete card counts by its own memory, Intel graphics not at all.
        return intel_mac_gpu(run("system_profiler", &["SPDisplaysDataType"]).as_deref());
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
        parse_sysctl_memsize(&run("sysctl", &["-n", "hw.memsize"])?)
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
    let gpu = detect_gpu(ram_bytes);
    // An Intel build under Rosetta is running on an Apple Silicon chip: say so.
    let arch = if cfg!(target_os = "macos") && matches!(&gpu, Gpu::Shared { .. }) {
        "aarch64"
    } else {
        std::env::consts::ARCH
    };
    Hardware {
        os: std::env::consts::OS,
        arch,
        cores: std::thread::available_parallelism().map_or(1, std::num::NonZero::get),
        ram_bytes,
        gpu,
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

    // ---- macOS: parsers, with sample outputs

    /// `sysctl -n hw.memsize` prints one number. The 8 GB value is from a real M1 Mac mini's
    /// `sysctl -a` (cpufun.substack.com/p/more-m1-fun-hardware-information); the others are the
    /// same format for the usual memory sizes.
    #[test]
    fn parses_sysctl_memsize() {
        assert_eq!(parse_sysctl_memsize("8589934592\n"), Some(8 * GIB));
        assert_eq!(parse_sysctl_memsize("17179869184"), Some(16 * GIB));
        assert_eq!(parse_sysctl_memsize("  34359738368  \n"), Some(32 * GIB));
        assert_eq!(
            parse_sysctl_memsize("206158430208\n"),
            Some(192 * GIB),
            "a Mac Studio Ultra"
        );
    }

    #[test]
    fn sysctl_memsize_rejects_anything_else() {
        for bad in [
            "",
            "\n",
            "hw.memsize: 8589934592",
            "8 GB",
            "-1",
            "1.5",
            "0",
            "99999999999999999999999",
        ] {
            assert_eq!(parse_sysctl_memsize(bad), None, "{bad:?}");
        }
    }

    /// `sysctl -n machdep.cpu.brand_string` prints the chip name ("Apple M1" is from a real
    /// machine, macports-dev mailing list, April 2021).
    #[test]
    fn parses_the_chip_name() {
        assert_eq!(
            parse_sysctl_brand("Apple M1\n").as_deref(),
            Some("Apple M1")
        );
        assert_eq!(
            parse_sysctl_brand("Apple M2 Pro").as_deref(),
            Some("Apple M2 Pro")
        );
        assert_eq!(
            parse_sysctl_brand("Apple M3 Max\n").as_deref(),
            Some("Apple M3 Max")
        );
        assert_eq!(parse_sysctl_brand("  \n"), None);
        assert_eq!(parse_sysctl_brand(""), None);
    }

    #[test]
    fn an_apple_silicon_mac_shares_70_percent_of_its_memory_with_the_gpu() {
        assert_eq!(apple_gpu_pool(8 * GIB), 8 * GIB / 10 * 7);
        assert_eq!(
            apple_silicon_gpu(Some("Apple M2 Pro"), Some(32 * GIB)),
            Gpu::Shared {
                name: "Apple M2 Pro".into(),
                bytes: 32 * GIB / 10 * 7
            }
        );
        assert_eq!(
            apple_silicon_gpu(None, Some(16 * GIB)),
            Gpu::Shared {
                name: "Apple Silicon".into(),
                bytes: 16 * GIB / 10 * 7
            },
            "the chip name is only for display"
        );
    }

    /// Without the memory size there is nothing to say about the pool, so it is not guessed.
    #[test]
    fn an_apple_silicon_mac_of_unknown_memory_is_not_given_a_pool_of_zero() {
        let gpu = apple_silicon_gpu(Some("Apple M1"), None);
        assert_eq!(gpu, Gpu::None);
        let unknown = Hardware {
            ram_bytes: None,
            gpu,
            ..machine(Gpu::None, Some(1))
        };
        assert_eq!(unknown.model_memory(), None);
        assert_eq!(unknown.tier(), None, "unknown, not minimal");
    }

    // ---- `voice-notes hardware`

    #[test]
    fn the_report_names_the_memory_the_rule_and_the_class() {
        let text = report(&mac("Apple M2 Pro", 32), None);
        assert!(text.contains("This machine: macos aarch64"), "{text}");
        assert!(
            text.contains(
                "Memory for models: 22 GB (70% of the unified memory, which the GPU can use)"
            ),
            "{text}"
        );
        assert!(text.contains("Size class: standard"), "{text}");
        assert!(
            !text.contains("Raw output"),
            "the raw output is only shown when asked for"
        );
    }

    #[test]
    fn every_kind_of_machine_says_which_rule_applied() {
        let rule = |h: &Hardware| h.memory_rule();
        let card = from_nvidia("NVIDIA GeForce RTX 4090, 24564\n", 64);
        assert!(rule(&card).contains("graphics card's own memory"));
        assert!(rule(&machine(Gpu::None, Some(16))).contains("half of the RAM"));
        let apu = machine(
            classify_amd("Strix", 512 << 20, 80 * GIB, Some(125 * GIB)),
            Some(125),
        );
        assert!(rule(&apu).contains("shares with the system"));
        assert!(rule(&mac("Apple M1", 8)).contains("70% of the unified memory"));
    }

    #[test]
    fn a_machine_whose_memory_is_unknown_says_so_instead_of_guessing() {
        let unknown = Hardware {
            ram_bytes: None,
            ..machine(Gpu::None, Some(1))
        };
        let text = report(&unknown, None);
        assert!(text.contains("Memory for models: unknown"), "{text}");
        assert!(text.contains("Size class: unknown"), "{text}");
    }

    /// The bug CI found: a runner without a GPU gets an empty answer from `lspci`, which must not
    /// be left blank.
    #[test]
    fn an_empty_answer_is_spelled_out_like_a_failed_command() {
        assert_eq!(spell_out(Some(String::new())), "(no output)");
        assert_eq!(spell_out(Some("  \n\n".into())), "(no output)");
        assert_eq!(
            spell_out(None),
            "(not available: the command is missing or failed)"
        );
        assert_eq!(
            spell_out(Some("17179869184\n".into())),
            "17179869184\n",
            "real output is kept as it is"
        );
    }

    #[test]
    fn the_raw_section_shows_each_commands_output_or_that_there_was_none() {
        let probes = vec![
            (
                "sysctl -n hw.memsize".to_string(),
                "17179869184\n".to_string(),
            ),
            (
                "system_profiler SPDisplaysDataType".to_string(),
                String::new(),
            ),
            (
                "nvidia-smi".to_string(),
                "(not available: the command is missing or failed)".to_string(),
            ),
        ];
        let text = report(&mac("Apple M1", 16), Some(&probes));
        assert!(text.contains("Raw output the detection read:"), "{text}");
        assert!(
            text.contains("$ sysctl -n hw.memsize\n17179869184\n"),
            "{text}"
        );
        assert!(
            text.contains("$ system_profiler SPDisplaysDataType\n(no output)"),
            "{text}"
        );
        assert!(text.contains("(not available"), "{text}");
    }

    /// The probes are the very commands detection runs, so what is reported is what was read.
    #[test]
    fn the_probes_cover_what_this_system_reads_and_never_fail() {
        let probes = probes();
        let commands: Vec<&str> = probes.iter().map(|(c, _)| c.as_str()).collect();
        assert!(
            commands.iter().any(|c| c.starts_with("df ")),
            "{commands:?}"
        );
        if cfg!(target_os = "macos") {
            for expected in [
                "sysctl -n hw.memsize",
                "sysctl -n hw.optional.arm64",
                "system_profiler SPDisplaysDataType",
            ] {
                assert!(commands.contains(&expected), "{expected} in {commands:?}");
            }
        } else {
            assert!(
                commands.iter().any(|c| c.contains("/proc/meminfo")),
                "{commands:?}"
            );
        }
        assert!(
            probes.iter().all(|(_, output)| !output.is_empty()),
            "an empty answer is spelled out"
        );
    }

    // ---- Intel Macs and Rosetta

    /// `hw.optional.arm64` is `1` on Apple Silicon, also for an Intel program under Rosetta, and
    /// absent on an Intel Mac (the command then fails and prints nothing).
    #[test]
    fn recognises_apple_silicon_from_the_hardware_flag() {
        assert!(parse_sysctl_arm64("1\n"));
        assert!(parse_sysctl_arm64("1"));
        for not_apple_silicon in [
            "",
            "\n",
            "0",
            "0\n",
            "unknown oid",
            "sysctl: unknown oid 'hw.optional.arm64'",
        ] {
            assert!(
                !parse_sysctl_arm64(not_apple_silicon),
                "{not_apple_silicon:?}"
            );
        }
    }

    #[test]
    fn under_rosetta_the_chip_name_is_not_the_brand_string() {
        // An Intel-looking brand string must not be shown as the name of an Apple Silicon chip.
        let gpu = apple_silicon_gpu(
            Some("Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz"),
            Some(16 * GIB),
        );
        assert_eq!(
            gpu,
            Gpu::Shared {
                name: "Apple Silicon".into(),
                bytes: 16 * GIB / 10 * 7
            }
        );
        let native = apple_silicon_gpu(Some("Apple M1 Pro"), Some(16 * GIB));
        assert!(matches!(native, Gpu::Shared { name, .. } if name == "Apple M1 Pro"));
    }

    /// An Intel Mac laptop with a discrete card. The lines for the two adapters (their names,
    /// `Type`, `Bus` and `VRAM`) are quoted from a real capture on discussions.apple.com
    /// (thread 251471750); the other property lines are typical of the format and are only here
    /// to be skipped.
    const MACBOOK_PRO_WITH_RADEON: &str = "\
Graphics/Displays:

    Intel HD Graphics 630:

      Chipset Model: Intel HD Graphics 630
      Type: GPU
      Bus: Built-In
      VRAM (Dynamic, Max): 1536 MB
      Vendor: Intel
      Metal: Supported

    Radeon Pro 560:

      Chipset Model: Radeon Pro 560
      Type: GPU
      Bus: PCIe
      PCIe Lane Width: x8
      VRAM (Total): 4 GB
      Vendor: AMD (0x1002)
      Metal: Supported
      Displays:
        Color LCD:
          Display Type: Built-In Retina LCD
          Resolution: 2880 x 1800 Retina
";

    #[test]
    fn parses_the_adapters_of_an_intel_macbook_pro() {
        let gpus = parse_system_profiler_gpus(MACBOOK_PRO_WITH_RADEON);
        assert_eq!(
            gpus,
            [
                MacGpu {
                    name: "Intel HD Graphics 630".into(),
                    bus: "Built-In".into(),
                    vram_bytes: None
                },
                MacGpu {
                    name: "Radeon Pro 560".into(),
                    bus: "PCIe".into(),
                    vram_bytes: Some(4 * GIB)
                },
            ],
            "the integrated one has only a dynamic share, which is not its own memory"
        );
    }

    #[test]
    fn an_intel_mac_with_a_card_counts_the_cards_own_memory() {
        let gpu = intel_mac_gpu(Some(MACBOOK_PRO_WITH_RADEON));
        assert_eq!(
            gpu,
            Gpu::Dedicated {
                name: "Radeon Pro 560".into(),
                vram_bytes: 4 * GIB
            }
        );
        let machine = Hardware {
            os: "macos",
            ..machine(gpu, Some(16))
        };
        assert_eq!(
            machine.model_memory(),
            Some(4 * GIB),
            "its own memory, not half of the 16 GB"
        );
        assert_eq!(machine.tier(), Some(Tier::Minimal));
    }

    #[test]
    fn a_bigger_card_makes_a_bigger_class() {
        let eight = MACBOOK_PRO_WITH_RADEON
            .replace("Radeon Pro 560", "AMD Radeon Pro 5500M")
            .replace("VRAM (Total): 4 GB", "VRAM (Total): 8 GB");
        assert_eq!(
            intel_mac_gpu(Some(&eight)),
            Gpu::Dedicated {
                name: "AMD Radeon Pro 5500M".into(),
                vram_bytes: 8 * GIB
            }
        );
        assert_eq!(
            Hardware {
                os: "macos",
                ..machine(intel_mac_gpu(Some(&eight)), Some(32))
            }
            .tier(),
            Some(Tier::Light)
        );
    }

    #[test]
    fn with_two_cards_the_one_with_more_memory_counts() {
        let two = format!(
            "{MACBOOK_PRO_WITH_RADEON}\n    Radeon Pro Vega 56:\n\n      Chipset Model: Radeon Pro Vega 56\n      Bus: PCIe\n      VRAM (Total): 8 GB\n"
        );
        assert_eq!(
            intel_mac_gpu(Some(&two)),
            Gpu::Dedicated {
                name: "Radeon Pro Vega 56".into(),
                vram_bytes: 8 * GIB
            }
        );
    }

    /// Only Intel graphics (a thin laptop, a small desktop): it shares system memory, so the machine
    /// counts as having no GPU, and half of its RAM is what models can use.
    #[test]
    fn an_intel_mac_with_only_intel_graphics_has_no_gpu_to_count() {
        let only_intel = "Graphics/Displays:\n\n    Intel UHD Graphics 630:\n\n      Chipset Model: Intel UHD Graphics 630\n      Type: GPU\n      Bus: Built-In\n      VRAM (Dynamic, Max): 1536 MB\n";
        assert_eq!(intel_mac_gpu(Some(only_intel)), Gpu::None);
        let machine = Hardware {
            os: "macos",
            ..machine(Gpu::None, Some(16))
        };
        assert_eq!(machine.model_memory(), Some(8 * GIB));
    }

    /// A card that does not report its memory cannot be counted; the machine falls back to half
    /// of its RAM rather than guessing.
    #[test]
    fn a_card_without_a_memory_line_is_not_guessed_at() {
        let silent = "    Radeon Pro 5500M:\n\n      Chipset Model: AMD Radeon Pro 5500M\n      Type: GPU\n      Bus: PCIe\n      Metal: Supported\n";
        assert_eq!(intel_mac_gpu(Some(silent)), Gpu::None);
    }

    #[test]
    fn unusable_system_profiler_output_gives_no_gpu_and_never_a_panic() {
        for text in [
            "",
            "\n\n",
            "garbage",
            "Bus: PCIe\nVRAM (Total): 8 GB\n",
            "Chipset Model: X\nVRAM (Total): lots\n",
            "Chipset Model: X\nVRAM (Total): 0 GB\n",
        ] {
            assert_eq!(intel_mac_gpu(Some(text)), Gpu::None, "{text:?}");
        }
        assert_eq!(
            intel_mac_gpu(None),
            Gpu::None,
            "the command is missing or failed"
        );
    }

    #[test]
    fn mac_memory_sizes_are_read_in_gb_and_mb() {
        assert_eq!(parse_mac_size("4 GB"), Some(4 * GIB));
        assert_eq!(parse_mac_size("16 GB"), Some(16 * GIB));
        assert_eq!(parse_mac_size("1536 MB"), Some(1536 << 20));
        for bad in ["", "GB", "4", "4 TB", "four GB", "-4 GB", "4GB"] {
            assert_eq!(parse_mac_size(bad), None, "{bad:?}");
        }
    }

    // ---- machines from every platform, and the class each one gets

    fn from_nvidia(sample: &str, ram_gib: u64) -> Hardware {
        let (name, vram_bytes) = parse_nvidia_smi(sample).unwrap();
        machine(Gpu::Dedicated { name, vram_bytes }, Some(ram_gib))
    }

    fn mac(chip: &str, ram_gib: u64) -> Hardware {
        Hardware {
            os: "macos",
            arch: "aarch64",
            ..machine(
                apple_silicon_gpu(Some(chip), Some(ram_gib * GIB)),
                Some(ram_gib),
            )
        }
    }

    /// The project's table: under 8 GB minimal, 8 to 16 light, 16 to 24 standard, 24 or more power
    /// (of the memory models can use). The memory sizes of the graphics cards are the values
    /// `nvidia-smi` commonly reports; they were not captured on this machine.
    #[test]
    fn sample_machines_get_the_class_the_table_gives_them() {
        let samples: Vec<(&str, Hardware, Tier)> = vec![
            ("Mac mini M1, 8 GB", mac("Apple M1", 8), Tier::Minimal),
            ("MacBook Air M2, 16 GB", mac("Apple M2", 16), Tier::Light),
            (
                "MacBook Pro M2 Pro, 32 GB",
                mac("Apple M2 Pro", 32),
                Tier::Standard,
            ),
            (
                "MacBook Pro M3 Max, 64 GB",
                mac("Apple M3 Max", 64),
                Tier::Power,
            ),
            (
                "Mac Studio M2 Ultra, 192 GB",
                mac("Apple M2 Ultra", 192),
                Tier::Power,
            ),
            (
                "Intel Mac, 16 GB, no usable GPU",
                Hardware {
                    os: "macos",
                    ..machine(Gpu::None, Some(16))
                },
                Tier::Light,
            ),
            (
                "Linux laptop, CPU only, 8 GB",
                machine(Gpu::None, Some(8)),
                Tier::Minimal,
            ),
            (
                "Linux laptop, CPU only, 32 GB",
                machine(Gpu::None, Some(32)),
                Tier::Standard,
            ),
            (
                "Linux server, CPU only, 64 GB",
                machine(Gpu::None, Some(64)),
                Tier::Power,
            ),
            (
                "RTX 3060, 12 GB",
                from_nvidia("NVIDIA GeForce RTX 3060, 12288\n", 32),
                Tier::Light,
            ),
            (
                "RTX 4060, 8 GB",
                from_nvidia("NVIDIA GeForce RTX 4060, 8188\n", 32),
                Tier::Light,
            ),
            (
                "RTX 4080, 16 GB",
                from_nvidia("NVIDIA GeForce RTX 4080, 16376\n", 32),
                Tier::Standard,
            ),
            (
                "RTX 3090, 24 GB",
                from_nvidia("NVIDIA GeForce RTX 3090, 24576\n", 64),
                Tier::Power,
            ),
            (
                "RTX 4090, 24 GB",
                from_nvidia("NVIDIA GeForce RTX 4090, 24564\n", 64),
                Tier::Power,
            ),
            (
                "two cards: the bigger one counts",
                from_nvidia(
                    "NVIDIA GeForce GTX 1650, 4096\nNVIDIA GeForce RTX 4090, 24564\n",
                    64,
                ),
                Tier::Power,
            ),
            (
                "AMD Strix APU with an 80 GB shared pool (this machine)",
                machine(
                    classify_amd("Strix", 536_870_912, 85_899_345_920, Some(134_127_000_000)),
                    Some(125),
                ),
                Tier::Power,
            ),
            (
                "AMD RX 7900 XTX, 24 GB of its own",
                machine(
                    classify_amd("Navi 31", 25_753_026_560, 33_554_432_000, Some(64 * GIB)),
                    Some(64),
                ),
                Tier::Power,
            ),
        ];
        for (label, hardware, expected) in samples {
            assert_eq!(
                hardware.tier(),
                Some(expected),
                "{label}: {:?} for models",
                hardware.model_memory()
            );
        }
    }

    /// A card with a GPU uses its own memory, however much RAM the machine has.
    #[test]
    fn a_small_card_in_a_big_machine_is_still_a_small_machine() {
        let hardware = from_nvidia("NVIDIA GeForce GTX 1650, 4096\n", 128);
        assert_eq!(hardware.tier(), Some(Tier::Minimal));
    }

    // ---- the card-size tolerance

    /// Cards report a little under their nominal size; each still gets the class on its box.
    #[test]
    fn cards_reporting_slightly_under_their_nominal_size_get_the_class_on_the_box() {
        let mib = |n: u64| Tier::for_memory(n * 1024 * 1024);
        assert_eq!(mib(8188), Tier::Light, "RTX 4060, 8 GB");
        assert_eq!(mib(16376), Tier::Standard, "RTX 4080, 16 GB");
        assert_eq!(mib(24564), Tier::Power, "RTX 4090, 24 GB");
        assert_eq!(mib(24576), Tier::Power, "RTX 3090, exactly 24 GiB");
    }

    /// The tolerance is 2%, not a rounding of everything: real in-between sizes keep their class.
    #[test]
    fn the_tolerance_does_not_promote_machines_that_are_clearly_below_a_boundary() {
        let mib = |n: u64| Tier::for_memory(n << 20);
        assert_eq!(mib(7987), Tier::Minimal, "7.8 GiB");
        assert_eq!(
            mib(15_872),
            Tier::Light,
            "15.5 GiB: an 8 GB machine's half of 31 GiB stays light"
        );
        assert_eq!(mib(23_552), Tier::Standard, "23 GiB");
        assert_eq!(
            mib(8090),
            Tier::Light,
            "7.9 GiB: within 2% of 8 GB counts as 8 GB"
        );
        // A 16 GB laptop reports about 15.3 GiB of RAM; half of it is 7.65 GiB: not light.
        assert_eq!(Tier::for_memory(15_300_000_000 / 2), Tier::Minimal);
    }

    #[test]
    fn the_tolerance_cannot_overflow() {
        assert_eq!(Tier::for_memory(u64::MAX), Tier::Power);
        assert_eq!(Tier::for_memory(0), Tier::Minimal);
    }

    // ---- on a real Mac (runs in CI on macos-latest)

    /// The only test that runs the real commands: it checks the parsers against a real Mac's
    /// output, on an Apple Silicon runner and on an Intel one.
    #[test]
    #[cfg(target_os = "macos")]
    fn detect_runs_on_a_mac() {
        let hardware = detect();
        assert_eq!(hardware.os, "macos");
        let ram = hardware
            .ram_bytes
            .expect("sysctl hw.memsize must be readable and parse");
        assert!(ram >= 2 * GIB, "{ram}");
        match &hardware.gpu {
            Gpu::Shared { name, bytes } => {
                assert_eq!(
                    hardware.arch, "aarch64",
                    "a shared pool means Apple Silicon"
                );
                assert!(
                    name.starts_with("Apple"),
                    "the chip name from sysctl: {name}"
                );
                assert_eq!(*bytes, apple_gpu_pool(ram));
            }
            Gpu::Dedicated { vram_bytes, .. } => {
                assert_eq!(
                    hardware.arch, "x86_64",
                    "a card with its own memory: an Intel Mac"
                );
                assert!(*vram_bytes > 0);
            }
            Gpu::None => assert_eq!(
                hardware.arch, "x86_64",
                "Apple Silicon always has a pool: {hardware:?}"
            ),
        }
        assert!(hardware.tier().is_some(), "{hardware:?}");
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
