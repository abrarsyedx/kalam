use std::path::Path;
use std::process::Command;

/// Bundled Whisper models: (file, minimum plausible size). They are too large
/// for git, so the first build downloads them.
const MODELS: [(&str, u64); 2] = [
    // Accurate (default): Whisper Large-v3 Turbo, multilingual, ~574 MB.
    ("ggml-large-v3-turbo-q5_0.bin", 500_000_000),
    // Fast: Whisper Small, English, ~190 MB.
    ("ggml-small.en-q5_1.bin", 150_000_000),
];
const MODEL_DIR: &str = "resources/models";
/// Khronos/LunarG redistributable Vulkan loader (MIT / Apache-2.0), shipped next
/// to kalam.exe so Kalam starts even on PCs without a graphics driver.
const VULKAN_RUNTIME_URL: &str = "https://sdk.lunarg.com/sdk/download/1.4.363.0/windows/vulkan-runtime-components.zip";
const VULKAN_DIR: &str = "resources/vulkan";

/// sherpa-onnx (speaker diarization). Must match the exact version pinned in
/// Cargo.toml. On Windows Kalam uses its DLL build (see Cargo.toml).
const SHERPA_VERSION: &str = "1.13.6";
const SHERPA_DIR: &str = "resources/sherpa";
const SHERPA_DLLS: [&str; 3] = ["sherpa-onnx-c-api.dll", "onnxruntime.dll", "onnxruntime_providers_shared.dll"];

/// Speaker voiceprint model (NVIDIA TitaNet-small via sherpa-onnx, CC-BY-4.0).
const SPEAKER_DIR: &str = "resources/speakers";
const VOICEPRINT_FILE: &str = "titanet_small.onnx";
const VOICEPRINT_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx";
const VOICEPRINT_MIN_BYTES: u64 = 30_000_000;
const MODEL_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

fn main() {
    for (file, min_bytes) in MODELS {
        ensure_model(file, min_bytes);
    }
    ensure_speaker_models();
    if std::env::var("CARGO_CFG_WINDOWS").is_ok() {
        ensure_vulkan_loader();
        ensure_sherpa_dlls();
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        link_clang_runtime();
    }
    let mut attrs = tauri_build::Attributes::new();
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        // Tauri needs Common Controls v6 on Windows. tauri-build embeds that
        // manifest only into the app executable, so the test binaries crash at
        // load (STATUS_ENTRYPOINT_NOT_FOUND). Let the linker embed the same
        // dependency into every binary instead.
        attrs = attrs.windows_attributes(tauri_build::WindowsAttributes::new_without_app_manifest());
        println!("cargo:rustc-link-arg=/MANIFEST:EMBED");
        println!(
            "cargo:rustc-link-arg=/MANIFESTDEPENDENCY:type='win32' name='Microsoft.Windows.Common-Controls' \
             version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'"
        );
    }
    tauri_build::try_build(attrs).expect("failed to run tauri-build")
}

/// Download a model once if it is missing. curl ships with Windows 10+,
/// macOS and every Linux distro.
/// macOS: whisper.cpp's Metal code checks the macOS version at run time
/// (`@available`), which calls `__isPlatformVersionAtLeast` from clang's runtime
/// library. rustc links with `-nodefaultlibs`, so that library is left out
/// unless it is named explicitly.
fn link_clang_runtime() {
    let resource_dir = Command::new("clang")
        .arg("--print-resource-dir")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    let Some(dir) = resource_dir else {
        println!("cargo:warning=clang not found; linking may fail with \"___isPlatformVersionAtLeast\" undefined");
        return;
    };
    // A universal archive: covers both Apple Silicon and Intel builds.
    let lib = Path::new(&dir).join("lib/darwin/libclang_rt.osx.a");
    if lib.is_file() {
        println!("cargo:rustc-link-arg={}", lib.display());
    } else {
        println!("cargo:warning={} not found; linking may fail with \"___isPlatformVersionAtLeast\" undefined", lib.display());
    }
}

fn ensure_model(file: &str, min_bytes: u64) {
    let path_str = format!("{MODEL_DIR}/{file}");
    println!("cargo:rerun-if-changed={path_str}");
    let path = Path::new(&path_str);
    let ok = |p: &Path| p.metadata().map(|m| m.len() >= min_bytes).unwrap_or(false);
    if ok(path) {
        return;
    }
    std::fs::create_dir_all(MODEL_DIR).expect("create resources/models");
    let url = format!("{MODEL_URL}/{file}");
    println!("cargo:warning=Downloading Whisper model {file} to {MODEL_DIR} ...");
    let tmp = format!("{path_str}.part");
    let status = Command::new("curl").args(["-L", "--fail", "--retry", "3", "-o", &tmp, &url]).status();
    match status {
        Ok(s) if s.success() && ok(Path::new(&tmp)) => {
            std::fs::rename(&tmp, path).expect("move downloaded model into place");
        }
        _ => {
            let _ = std::fs::remove_file(&tmp);
            panic!(
                "\n\nCould not download the Whisper model automatically.\n\
                 Download it manually from:\n  {url}\n\
                 and save it as:\n  src-tauri/{path_str}\n\
                 (or run scripts/fetch-model.ps1 / scripts/fetch-model.sh)\n\n"
            );
        }
    }
}

/// Fetch vulkan-1.dll (+ license) from LunarG's runtime components once.
fn ensure_vulkan_loader() {
    let dll = Path::new(VULKAN_DIR).join("vulkan-1.dll");
    println!("cargo:rerun-if-changed={}", dll.display());
    if dll.is_file() {
        return;
    }
    std::fs::create_dir_all(VULKAN_DIR).expect("create resources/vulkan");
    let zip = Path::new(VULKAN_DIR).join("runtime.zip.part");
    let unpack = Path::new(VULKAN_DIR).join("unpacked.part");
    println!("cargo:warning=Downloading the Vulkan runtime (vulkan-1.dll) ...");
    let ok = Command::new("curl")
        .args(["-L", "--fail", "--retry", "3", "-o"])
        .arg(&zip)
        .arg(VULKAN_RUNTIME_URL)
        .status()
        .is_ok_and(|s| s.success())
        && std::fs::create_dir_all(&unpack).is_ok()
        // bsdtar ships with Windows 10+ and reads zip files.
        && Command::new("tar").arg("-xf").arg(&zip).arg("-C").arg(&unpack).status().is_ok_and(|s| s.success());
    let found = ok.then(|| find_file(&unpack, "x64", "vulkan-1.dll")).flatten();
    match found {
        Some(src) => {
            std::fs::copy(&src, &dll).expect("copy vulkan-1.dll");
            if let Some(license) = std::fs::read_dir(src.parent().and_then(Path::parent).unwrap())
                .ok()
                .and_then(|mut d| d.find_map(|e| e.ok().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "txt"))))
            {
                let _ = std::fs::copy(license, Path::new(VULKAN_DIR).join("VulkanRT-License.txt"));
            }
            let _ = std::fs::remove_file(&zip);
            let _ = std::fs::remove_dir_all(&unpack);
        }
        None => panic!(
            "

Could not download the Vulkan runtime automatically.
             Download {VULKAN_RUNTIME_URL},
             unzip it and copy x64/vulkan-1.dll to src-tauri/{VULKAN_DIR}/vulkan-1.dll

"
        ),
    }
}

/// Find `<root>/**/<parent>/<name>`.
fn find_file(root: &Path, parent: &str, name: &str) -> Option<std::path::PathBuf> {
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_file(&path, parent, name) {
                return Some(found);
            }
        } else if path.file_name().is_some_and(|n| n == name)
            && path.parent().and_then(Path::file_name).is_some_and(|p| p == parent)
        {
            return Some(path);
        }
    }
    None
}

fn curl(url: &str, dest: &Path) -> bool {
    Command::new("curl")
        .args(["-L", "--fail", "--retry", "3", "-o"])
        .arg(dest)
        .arg(url)
        .status()
        .is_ok_and(|s| s.success())
}

fn untar(archive: &Path, into: &Path) -> bool {
    // bsdtar ships with Windows 10+ and macOS and reads .tar.bz2 and .zip.
    std::fs::create_dir_all(into).is_ok()
        && Command::new("tar").arg("-xf").arg(archive).arg("-C").arg(into).status().is_ok_and(|s| s.success())
}

/// The voiceprint model used to tell speakers apart.
fn ensure_speaker_models() {
    let model = Path::new(SPEAKER_DIR).join(VOICEPRINT_FILE);
    println!("cargo:rerun-if-changed={}", model.display());
    let ok = |p: &Path| p.metadata().map(|m| m.len() >= VOICEPRINT_MIN_BYTES).unwrap_or(false);
    if ok(&model) {
        return;
    }
    std::fs::create_dir_all(SPEAKER_DIR).expect("create resources/speakers");
    println!("cargo:warning=Downloading the speaker voiceprint model ...");
    let part = model.with_extension("part");
    if !(curl(VOICEPRINT_URL, &part) && ok(&part)) {
        let _ = std::fs::remove_file(&part);
        panic!("

Could not download {VOICEPRINT_URL}
Save it as src-tauri/{SPEAKER_DIR}/{VOICEPRINT_FILE}

");
    }
    std::fs::rename(&part, &model).expect("move voiceprint model into place");
}

/// Windows: fetch the sherpa-onnx DLLs once and put them next to every binary
/// this build produces (app, tests). Windows looks in an exe's own folder
/// first; without that it would find the older onnxruntime.dll that Windows 11
/// ships in System32 and crash.
fn ensure_sherpa_dlls() {
    let have_all = |dir: &Path| SHERPA_DLLS.iter().all(|d| dir.join(d).is_file());
    let dir = Path::new(SHERPA_DIR);
    if !have_all(dir) {
        println!("cargo:warning=Downloading the sherpa-onnx {SHERPA_VERSION} DLLs ...");
        std::fs::create_dir_all(dir).expect("create resources/sherpa");
        let name = format!("sherpa-onnx-v{SHERPA_VERSION}-win-x64-shared-MT-Release-lib");
        let url = format!("https://github.com/k2-fsa/sherpa-onnx/releases/download/v{SHERPA_VERSION}/{name}.tar.bz2");
        let archive = dir.join("dlls.tar.bz2.part");
        let unpack = dir.join("dlls.part");
        if !(curl(&url, &archive) && untar(&archive, &unpack)) {
            panic!("\n\nCould not download {url}\nUnpack it and copy lib/*.dll to src-tauri/{SHERPA_DIR}/\n\n");
        }
        for dll in SHERPA_DLLS {
            let src = find_file(&unpack, "lib", dll).unwrap_or_else(|| panic!("{dll} missing from {url}"));
            std::fs::copy(src, dir.join(dll)).expect("copy sherpa-onnx DLL");
        }
        let _ = std::fs::remove_file(&archive);
        let _ = std::fs::remove_dir_all(&unpack);
    }
    // OUT_DIR is <target>/<profile>/build/kalam-<hash>/out.
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    if let Some(profile_dir) = out.ancestors().nth(3) {
        for dest in [profile_dir.to_path_buf(), profile_dir.join("deps")] {
            let _ = std::fs::create_dir_all(&dest);
            for dll in SHERPA_DLLS {
                let _ = std::fs::copy(dir.join(dll), dest.join(dll));
            }
        }
    }
}
