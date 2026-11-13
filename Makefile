.PHONY: check format test clean

check:
	cargo check

format:
	cargo fmt

clippy:
	cargo clippy --all-targets --all-features -- -D warnings

fix:
	cargo fix --allow-dirty --allow-staged

test:
	cargo test

build:
	cargo build

clean:
	cargo clean

lint: clippy format
