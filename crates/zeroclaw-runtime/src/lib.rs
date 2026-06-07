#![allow(
    clippy::to_string_in_format_args,
    clippy::useless_format,
    clippy::manual_inspect,
    clippy::items_after_test_module,
    clippy::manual_strip,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::needless_borrows_for_generic_args,
    clippy::print_with_newline,
    clippy::unnecessary_get_then_check,
    clippy::disallowed_macros,
    clippy::single_char_add_str,
    clippy::field_reassign_with_default,
    clippy::single_match
)]
//! Agent runtime — orchestration, security, observability, cron, SOP, skills, hardware, and more.

pub mod cli_input;
pub mod identity;
pub mod migration;
pub mod util;

pub mod agent;
pub mod approval;
pub mod browse;
pub mod cost;
pub mod cron;
pub mod daemon;
pub mod doctor;
pub mod hands;
pub mod health;
pub mod heartbeat;
pub mod hooks;
pub mod i18n;
pub mod integrations;
pub mod nodes;
pub mod observability;
pub mod onboard;
pub mod orchestrate;
pub mod peers;
pub mod planning;
pub mod platform;
pub mod process_stats;
pub mod rag;
pub mod reflection;
pub mod routines;
pub mod security;
pub mod service;
pub mod skillforge;
pub mod skills;
pub mod sop;
pub mod subagent;
pub mod tools;
pub mod trust;
pub mod tunnel;
pub mod verifiable_intent;
