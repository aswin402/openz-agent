//! Channel implementations and orchestration for messaging platform integrations.
//!
//! **Channels trimmed 2026-06-02**: only `discord` + `telegram` + `whatsapp-web/cloud` +
//! `acp-server` (inbound ACP) are compiled by default. The 25 other channels
//! (slack, signal, matrix, nostr, imessage, email, lark, wechat, qq, bluesky,
//! twitter, reddit, notion, linq, wati, nextcloud, mochat, wecom, clawdtalk,
//! webhook, voice-call, voice-wake, etc.) are no longer wired. Re-add by
//! reintroducing the `#[cfg(feature = "...")]` block + the corresponding file
//! in `src/` + the feature flag in `Cargo.toml` and the workspace `Cargo.toml`.

#![allow(
    clippy::to_string_in_format_args,
    clippy::useless_format,
    clippy::explicit_auto_deref
)]

pub mod allowlist;
pub mod orchestrator;
pub mod util;

// Always-compiled channels and utilities (no feature gate)
#[cfg(feature = "channel-acp-server")]
pub mod acp_channel;
pub mod cli;
pub mod link_enricher;
pub mod transcription;
pub mod tts;

// Feature-gated channels (trimmed 2026-06-02)
#[cfg(feature = "channel-discord")]
pub mod discord;
#[cfg(feature = "channel-telegram")]
pub mod telegram;
#[cfg(any(feature = "channel-whatsapp-cloud", feature = "whatsapp-web"))]
pub mod whatsapp;
#[cfg(feature = "whatsapp-web")]
pub mod whatsapp_storage;
#[cfg(feature = "whatsapp-web")]
pub mod whatsapp_web;
