# quick sanity check, useful to run before committing
lint: fmt check clippy

# format src and fuzzer
fmt:
    cargo fmt
    cargo fmt --manifest-path fuzz/Cargo.toml

# type-check src and the fuzzer
#
# Type-checking with and without features helps catch,
# for example, an import statement that is `#[cfg(_)]`'d
# to a different feature than the code that relies on it.
check:
    cargo check
    cargo check --all-features
    cargo check --manifest-path fuzz/Cargo.toml

# run clippy on src and fuzzer
clippy:
    cargo clippy
    cargo clippy --manifest-path fuzz/Cargo.toml

# test default features
test:
    cargo test

# run benchmarks (without SIMD)
bench:
    cargo bench --features bench

# run benchmarks (with SIMD)
bench_simd:
    cargo bench --features bench,simd

# WIP/temporary
bench_fancy:
    cd benches; python3 plot.py

# fuzz the library
fuzz:
    cargo fuzz run fuzz

# fuzz the library, but stick to BVH's with at most 3 shapes
# that do not undergo mutations
fuzz-max-three-shapes:
    MAX_SHAPES=3 cargo fuzz run fuzz

# fuzz the library, but stick to BVH's with at most 5 shapes
# that do not undergo mutations
fuzz-max-five-shapes:
    MAX_SHAPES=5 cargo fuzz run fuzz

# find the current MSRV, if it is greater than or equal to
# the rust-version in Cargo.toml
find-msrv:
    cargo msrv find --ignore-lockfile

# verify the rust-version in Cargo.toml is compatible
verify-msrv:
    cargo msrv verify --ignore-lockfile 