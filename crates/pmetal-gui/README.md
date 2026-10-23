# pmetal-gui

Desktop GUI application for PMetal built with Tauri + Svelte + TailwindCSS.

## Overview

A desktop application for visual model management, training, serving and evaluation on Apple
Silicon. Every `pmetal` command has a form in the GUI, drawn from the same job specs in
`pmetal-core` that the CLI, the TUI, the MCP server and the Python bindings are built from, so a
field, its default and its range are the command line's own.

## Pages

| Page | Description |
|------|-------------|
| **Dashboard** | Models on this Mac, what is running, the device |
| **Jobs** | Every job since the app was opened, with progress, output and the result |
| **Train** | Fine-tune (LoRA, QLoRA, full), GRPO, RLKD, preference tuning, distillation, embedding training, pretraining |
| **Run** | Chat, the OpenAI-compatible server, DFlash decoding |
| **Models**, **Datasets** | What is on this Mac, search and download from the Hub, fit estimates |
| **Tools** | Merge, fuse an adapter, quantize, write an Ollama Modelfile, pack experts |
| **Measure** | Perplexity, training throughput, a real workload |
| **Advanced** | MTP and draft training, diffusion LoRA, tokenize, vocode, decide |
| **Settings** | Theme, Hugging Face token (Keychain), model folders |

A job page is `/run/<job>` (`/run/train`, `/run/serve`, ...), one component for all of them.

The app is not published to crates.io (`publish = false`); it ships as the `PMetal.app` bundle,
built from this crate's own Cargo workspace.

## Requirements

- macOS 14 or later on Apple Silicon
- To build: Rust 1.95 or later, [bun](https://bun.sh), and the Tauri prerequisites

## Tech Stack

- **Tauri 2**: native desktop framework (Rust backend)
- **Svelte 5**: reactive UI framework
- **TailwindCSS**: utility-first styling
- **Vite**: build tooling
- **bun**: JavaScript runtime, package manager and test runner

## Development

```bash
cd crates/pmetal-gui

# Install frontend dependencies
bun install

# Start development server with hot reload
bun tauri dev

# Checks
bun run check      # svelte-check, strict TypeScript
bun test           # form helpers, chart, sidebar, component renders
cargo test --manifest-path src-tauri/Cargo.toml

# Build production binary
bun tauri build
```

`just check-gui lint-gui test-gui` runs the same checks the release does.

## Architecture

```text
pmetal-gui/
├── src/                    # Svelte frontend
│   ├── routes/             # Dashboard, Jobs, Models, Datasets, Chat (/inference), Settings, /run/<job>
│   ├── lib/                # api.ts (the command contract), stores, forms, nav, components
│   │   └── specs.generated.ts  # field names of every job, generated from the specs
│   └── app.html
├── src-tauri/              # Rust backend (its own Cargo workspace)
│   ├── src/
│   │   ├── lib.rs          # Builder, start-up, orderly exit
│   │   ├── commands.rs     # Tauri commands, thin wrappers
│   │   ├── state.rs        # AppState: settings, token, models, jobs, downloads
│   │   ├── jobs/           # kinds (spec to LaunchPlan), JobHub over the supervisor
│   │   ├── models.rs       # Hub cache + trained + added folders
│   │   ├── adapters.rs     # LoRA adapters trained on this Mac
│   │   ├── datasets.rs     # a look inside a dataset file before a job uses it
│   │   ├── hub.rs          # search, downloads
│   │   ├── system.rs       # the chip, memory, and whether a model fits
│   │   ├── secrets.rs      # the Hugging Face token, in the Keychain
│   │   ├── settings.rs     # the settings file
│   │   ├── paths.rs        # which locations a job may write to
│   │   ├── logging.rs      # the app's log file, and what a crash leaves behind
│   │   └── inference.rs    # chat: `pmetal infer --messages-file`, run by a supervisor of its own
│   ├── build.rs            # declares the commands, so capabilities can name them
│   ├── capabilities/       # what the window may call
│   └── tauri.conf.json     # CSP and window
└── package.json
```

### Jobs

Every job is a `pmetal` process run by `pmetal-supervisor` (process groups, bounded and redacted
output, progress read from the job's own event stream, SIGTERM then SIGKILL on stop). `list_job_forms`
sends the spec descriptors; `start_job` takes the text entered for fields, builds the spec with
`spec_from_text`, runs its `normalize` and turns it into the supervisor's `LaunchPlan`, the same
plan the terminal UI, the MCP server and the Python bindings start their jobs from (the command,
the event log or metrics file it asks for, where it writes). A field left at its default is not
passed, so a command's own default applies. The app decides only where a job writes
(a new folder under `~/pmetal-output` unless you choose one), refuses locations that are system
folders or that would overwrite a model the job reads, and tells the window what changed at most
five times a second. Quitting asks first if jobs run, then stops them; a training run saves its
adapter on the way.

The `pmetal` binary is found by `pmetal_supervisor::locate_pmetal_binary`: `PMETAL_BIN`, else the
one beside the app (the release bundle carries it as a sidecar), else `~/.cargo/bin`,
`/opt/homebrew/bin`, `/usr/local/bin` or `PATH`. `just build-gui-bundle` reproduces the release
bundle locally. The Models and Datasets pages read the Hub cache and search the Hub through
`pmetal-hub` (`pmetal_hub::local`, `search_models`, `search_datasets`), as the other surfaces do.

### Chat

A reply is `pmetal infer --messages-file`, run by the supervisor like every other job: the
conversation (the system message first, then the turns) is written to a private file that goes when
the reply ends, and the command answers it. What it prints is read as it arrives: the reasoning of a
thinking model streams apart from the answer, and the statistics come from the command's own
footer. Stop sends the command the signal that ends a generation after the token in flight, and the
text so far is kept. Nothing in the app loads a model or samples a token, so a chat answers exactly
as `pmetal infer` does with the same settings. Chats run under a supervisor of their own, so a reply
is not listed as a job, takes none of the jobs' slots and is not stopped with them.

### Security

- The Hugging Face token lives in the macOS Keychain. It is never written to the settings file,
  never sent to the window after it is entered, never logged, and given to jobs through their
  environment (`HF_TOKEN`), not their arguments. A token an earlier version wrote to the settings
  file is moved to the Keychain at first start.
- Each command is listed in `build.rs`, and the window's capability allows it by name. The window
  has no file system, shell, opener or updater access; folders are chosen through backend commands.
  Tests (`src-tauri/src/tests.rs`) tie the capability file, the handler list, the CSP and the
  updater state to the code.
- The CSP is `default-src 'none'`, scripts from the app only, connections to the app's IPC only.
- Paths are checked in the backend: outputs cannot climb out with `..` or name a system folder,
  and deleting a model is limited to the folders the app manages.

### Updates

There is no in-app updater yet. It needs a signing key: generate one with
`bunx tauri signer generate`, put the private key in the release workflow's secrets, then add
`tauri-plugin-updater`, `plugins.updater` (`pubkey`, `https` endpoints) and
`bundle.createUpdaterArtifacts` to the configuration, grant `updater:default` to the window, and
restore a screen for it. The test `the_updater_is_off_everywhere_or_on_with_a_key` fails if the
plugin is added without a key.

## License

MIT OR Apache-2.0
