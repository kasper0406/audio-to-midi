use std::env;

fn main() {
    if env::var("CARGO_FEATURE_PYTHON").is_ok() {
        let rustflags = "-C target-feature=+avx -C opt-level=3 -C target-cpu=native";
        env::set_var("RUSTFLAGS", rustflags);
        println!("cargo:rustc-env=RUSTFLAGS={}", rustflags);
    }

    // Ensure the build script is rerun if the feature flag changes
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PYTHON");
}
