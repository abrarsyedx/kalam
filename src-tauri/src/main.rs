// Prevents an extra console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // `kalam --gpu-probe <model>`: the isolated graphics-card test (see gpu.rs).
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() == Some(kalam_lib::gpu::PROBE_ARG) {
        let model = args.next().unwrap_or_default();
        std::process::exit(kalam_lib::gpu::probe_main(std::path::Path::new(&model)));
    }
    kalam_lib::run()
}
