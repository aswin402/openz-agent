# OpenZ (openz) Cinematic CLI UI/UX Specification

This document defines the visual standards, interactive protocols, and system architecture for the next-generation **OpenZ** command-line interface. It establishes a design language focused on minimalism, calm intelligence, and native shell integration, moving away from heavy ncurses-style TUI dashboards.

---

## 1. Core Design Philosophy

The interface is built to feel like an **intelligent autonomous presence living quietly inside the developer's shell**. The design focuses on whitespace, subtle transitions, and native scrollback.

*   **Breathable Whitespace**: Spacing is used in place of borders; indentation is used instead of boxes.
*   **Zero alternate-screen takeover**: All interaction is streamed directly into the standard terminal buffer, preserving user shell history.
*   **Calm Visuals**: Avoids hacker neon palettes, heavy unicode frames, and cluttered information grids.

### Color Palette & Theme
All output is rendered using ANSI colors map matching the following dark cinematic workspace values:

| Element | Color Value / ANSI Equivalent | Visual Purpose |
| :--- | :--- | :--- |
| **Background** | Near-black (`#09090B`) | Terminal default background configuration |
| **Primary Text** | Soft white (`#E4E4E7`) | Agent responses and user messages |
| **Muted Metadata** | Slate gray (`#71717A`) | Timestamps, durations, and tool parameter details |
| **Accent** | Indigo / Violet (`#8B5CF6`) | Prompts, cursor highlights, and command keywords |
| **Success** | Muted forest green | Successful tool executions and process confirmations |
| **Errors** | Muted rust red | Failed tool actions and system warning messages |

---

## 2. Layout Structure

The user interface is composed of only three layers, interacting dynamically in a single native terminal flow.

```
+------------------------------------------------------------------+
| Stream Layer (Conversational Flow)                               |
|                                                                  |
|   user message...                                                |
|                                                                  |
|   ● read  Cargo.toml                                             |
|     0.2s                                                         |
|                                                                  |
|   agent response text streams here chunk-by-chunk...             |
|                                                                  |
+------------------------------------------------------------------+
| Event Layer (Transient Timeline Activity)                        |
|                                                                  |
|   ● thinking · gemini-2.5-pro · 14k tok                          |
+------------------------------------------------------------------+
| Input Layer (Dynamic Prompt Interface)                           |
|                                                                  |
|   > /mod[els] (ghost-text autocomplete suggestion)               |
+------------------------------------------------------------------+
```

### 1. Stream Layer
The permanent record of the interaction, containing:
*   User query lines.
*   Agent conversational responses.
*   Timeline markers showing tool executions and outcomes.

### 2. Event Layer
Transient activity indicators showing what the agent is currently doing. This layer includes reasoning states and pending tool executions. These indicators automatically clean up or collapse when streaming begins or completes.

### 3. Minimal HUD
A tiny, inline status line containing active context (e.g., `● thinking · gemini · 14k tok`). It is only visible during active operations and remains invisible when idle.

---

## 3. Interaction Flow & Core States

### A. Elegant Startup
Silent and clean. No raw logs, diagnostic dumps, or startup tables.

```
openz

loading workspace...
connected 28 tools

gemini-2.5-pro · ~/openz

> 
```

### B. Session Picker UX
An editorial list layout without borders or boxes. Muted metadata is rendered below the session name, with a soft highlight cursor denoting selection.

```
OpenZ

recent sessions

❯ refactor websocket layer
  2h ago · gemini-2.5-pro

  build autonomous agent
  yesterday · claude-3-5-sonnet

  new session
```
*   **Controls**: `Up/Down` to select, `Enter` to resume, `n` to start a new session.

### C. Prompt & Multiline Modes
*   **Single-line prompt**: A clean prompt glyph followed by the cursor.
    ```
    > 
    ```
*   **Multiline prompt**: Used when inputting complex instructions. Utilizes lightweight bracket indicators instead of bounding boxes.
    ```
    ╭─
    │ explain this architecture
    │ and optimize performance
    ╰─
    ```
*   **Autocomplete**: Rendered inline as muted gray ghost-text without popup menus.
    ```
    > /models
    ```
    *(Where `els` is drawn in slate gray when the user has typed `/mod`)*

---

## 4. Response & Tool Execution Rendering

### Response Styling
*   **No heavy frames**: Code blocks are syntax-highlighted, softly indented by two spaces, and formatted with clean line breaks instead of border boxes.
*   **Clean Typography**: Bulleted lists use simple characters (`●`, `-`) and headers are styled with bold weight and color accents.

### Tool Execution Timeline
Instead of drawing status cards, tool actions are output as discrete timeline events.

*   **Successful Tool Run**:
    ```
    ● read  Cargo.toml
      0.2s

    ● search  async rust channels
      3 results

    ● edit  src/main.rs
      +18 -4

    ● shell  cargo test
      12 passed
    ```
*   **Failed Tool Run**:
    ```
    ✕ shell  cargo build
      unresolved import tokio::sync
    ```

---

## 5. Interactive Commands (/ Commands)

Interactive commands handle diagnostics and settings locally.

| Command | Arguments | Behavior | Visual Style |
| :--- | :--- | :--- | :--- |
| `/help` | None | Displays a clean list of available commands. | Indented list formatted with subtle column layouts. |
| `/clear` \| `/new` | None | Resets active conversation history and session tokens. | Simple prompt: `Clear session memory? [y/n]` |
| `/model` | `<p>/<m>` | Swings the active model to the specified model path. | Muted indicator: `model -> gemini-2.5-pro` |
| `/models` | None | Select active model profile using an editorial picker. | Reuses the Session Picker layout style. |
| `/mcp` | None | Lists connected MCP servers and their available tools. | Minimal list showing name and count. |
| `/skills` | None | Displays loaded skill workflows. | Muted name listing. |
| `/status` | None | Runs vital connectivity checks and shows metrics. | Simple checkmarks list. |
| `/logs` | `[limit]` | Shows the last few lines of the tracing logs. | Grayed text block. |
| `/think:<level>`| `<depth>` | Set reasoning depth (off, low, medium, high, max). | Status confirmation line. |
| `/quit` \| `/exit` | None | Exits the interactive session and returns to shell. | Short exit summary (tokens, cost). |

---

## 6. Implementation & Folder Structure

To support this design, terminal logic will be structured inside a dedicated CLI system module.

```
crates/zeroclaw-cli/
├── Cargo.toml                  # Dependencies: crossterm, syntect, terminal-light
└── src/
    ├── lib.rs                  # Crate interface
    ├── app.rs                  # Event-loop & raw-mode input coordinator
    ├── session.rs              # Session picker logic
    ├── theme.rs                # Styling constants and color definitions
    ├── renderer.rs             # Markdown parsing, syntax highlighting, and text streaming
    └── widgets/
        ├── mod.rs              # Component library
        ├── hud.rs              # Transient HUD status bar drawing
        ├── timeline.rs         # Tool events timeline printing
        └── autocomplete.rs     # Ghost-text completion logic
```

### Technical Implementation Guidelines
1.  **Crossterm Raw Mode**: Used to process keys (Tab, Backspace, Enter, Esc) interactively on the prompt line.
2.  **No Full Redraws**: Render incrementally by writing directly to standard output. Use `\x1B[K` (clear line) and `\r` (carriage return) only for the line currently being edited.
3.  **Active Line Clearing**: When a transient event finishes (e.g., transition from "thinking" to "streaming"), print `\r\x1B[K` to clear the thinking line, and begin printing the streamed content at the cursor's current line.
