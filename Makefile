.PHONY: code check nextest test nextest-cov test-cov clean upgrade bump-patch requirements

# --- Developer targets (require extra tools; see `make requirements`) ---

code:
	@echo "voice-notes: code check\n"
	@rustup update
	@rustup run nightly cargo update
	@rustup run nightly cargo upgrade
	@rustup run nightly cargo machete
	@rustup run nightly cargo audit
	@rustup run nightly cargo deny --log-level error check
	@typos
	@rustup run nightly cargo fmt
	@rustup run nightly cargo fix --allow-dirty --allow-no-vcs --allow-staged
	@rustup run nightly cargo clippy --fix --allow-dirty --allow-staged --all-targets --all-features -- -W clippy::pedantic
	@rustup run nightly cargo clippy --all-targets --all-features -- -W clippy::pedantic
	@rustup run nightly cargo test --doc

# Mandatory before every push; CI runs the same checks on pull requests.
check: code nextest

# --- Test targets ---

nextest:
	@echo "voice-notes: test (nextest)\n"
	@rustup run nightly cargo nextest run

test:
	@echo "voice-notes: test\n"
	@rustup run nightly cargo test

# --- Code coverage ---

nextest-cov:
	@echo "voice-notes: code coverage (nextest)\n"
	@rustup run nightly cargo llvm-cov --open nextest

test-cov:
	@echo "voice-notes: code coverage\n"
	@rustup run nightly cargo llvm-cov --open

# --- Misc ---

clean:
	cargo clean

upgrade:
	@cargo upgrade --verbose

bump-patch:
	@bash ./scripts/bump_patch_version.sh

requirements:
	@echo "voice-notes: requirements\n"
	@rustup update
	@rustup install nightly
	@rustup component add rustfmt clippy llvm-tools-preview --toolchain nightly
	@cargo install cargo-audit
	@cargo install cargo-deny
	@cargo install cargo-edit
	@cargo install cargo-llvm-cov
	@cargo install cargo-machete
	@cargo install cargo-nextest --locked
	@cargo install typos-cli
	@echo "System packages (Debian/Ubuntu): sudo apt-get install -y libasound2-dev pkg-config build-essential"
