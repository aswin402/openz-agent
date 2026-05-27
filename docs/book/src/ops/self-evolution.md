# Dynamic Skill & Prompt Self-Evolution

ZeroClaw includes autonomous self-improvement capabilities under its `Skillforge` engine. The engine continuously optimizes the agent's prompts and extracts recurring command sequences into native tools/skills.

---

## 1. Dynamic Skill Generation (Skillforge)

When dynamic evolution is enabled, ZeroClaw analyzes historical trace logs at the end of each session to identify recurring successful command sequences (e.g. build-test-debug loops) and synthesizes them into parameterized reusable skill templates.

### How It Works

1. **Trace Analysis (`crates/zeroclaw-runtime/src/skillforge/local_evolution.rs`)**:
   - Parses the JSONL trace log (`runtime-trace.jsonl`) sequentially.
   - Groups shell execution spans by `trace_id`.
   - Filters out failed commands, extracting only sequences of successful operations.
2. **LLM Synthesis**:
   - Sends sequence traces to the primary model provider.
   - Instructs the model to synthesize the sequential steps into a generalized, parameterized template with configuration knobs.
3. **Packaging**:
   - Creates a structured ZeroClaw skill directory under `shared/skills/synthesized/<skill_name>/`.
   - Generates `SKILL.toml` (metadata, parameters, usage guidelines) and `SKILL.md` (instructions and scripts).
4. **Integration**:
   - Automatically registers the synthesized skill inside the agent's environment, making it immediately available as a tool in subsequent user sessions.

### Configuration

Skill creation is activated in `config.toml` under the `skills` section:

```toml
[skills.skill_creation]
enabled = true
```

---

## 2. Genetic Prompt Optimizer

The genetic optimizer continuously mutates and refines the agent's identity system instructions (`IDENTITY.md`) using simulated turn-by-turn evaluations.

### Optimization Loop

1. **Gene Pool Generation (`crates/zeroclaw-runtime/src/skillforge/prompt_optimizer.rs`)**:
   - Mutates components of the active `IDENTITY.md` system prompt.
   - Conducts crossover (recombination) between high-performing prompt variants.
2. **Evaluations**:
   - Simulates typical user request triggers.
   - Executes candidate prompt configurations inside test runs.
3. **Fitness Evaluation**:
   - Scores each candidate prompt using a multi-dimensional fitness function:
     - **Success / Correctness (70%)**: Correct tool selection and output validation.
     - **Turn Efficiency (15%)**: Minimizing steps taken to reach outcomes.
     - **Token Economy (15%)**: Minimizing prompt/completion token consumption.
4. **Promotion**:
   - Overwrites `IDENTITY.md` with the highest-fitness prompt candidate.

### Manual Trigger

You can also trigger optimization directly using the following CLI command:

```bash
openz prompt-optimize
```
