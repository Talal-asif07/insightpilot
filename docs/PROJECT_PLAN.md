# InsightPilot: Project Plan

This is the plan written before development. It shows how the project was scoped, which risks were anticipated and how each one is handled in the code.

## 1. Problem statement

Non-technical users can't query their own data, and analysts lose time to repetitive ad-hoc questions. InsightPilot lets a user upload a CSV or Excel file and ask questions in plain English. An agent profiles the data, writes and runs pandas code, makes charts and explains the results like an analyst, showing its code, flagging data-quality issues and stating limitations.

## 2. Analyst workflow the agent must follow

| Step | Behaviour | Rule |
|---|---|---|
| Clarify | Restate the question, or ask about it if it's ambiguous | Never assume silently |
| Profile | Check schema, types, missing values and duplicates | Profile before analysing |
| Clean | Fix only what the question needs | Report every change |
| Analyse | Run pandas code | Every number comes from executed code |
| Visualise | Choose a chart that fits the question | Labelled axes and titles |
| Interpret | Use plain English | Association, not causation; flag small samples |
| Caveat | State limitations and data issues | Honest uncertainty |

## 3. Scope

- **MVP:** CSV/Excel upload, automatic profiling, 4 tools, code viewer, session memory, downloads, evaluation suite.
- **Version 2:** multi-file joins, SQL connector, PDF report export, forecasting.
- **Out of scope:** user accounts, persistence, real-time data, fine-tuning.

## 4. Risk register and where each risk is handled

### Data

| Risk | Mitigation | Implemented in |
|---|---|---|
| Non-UTF-8 encoding | UTF-8 → cp1252 → latin-1 fallback | `data_loader.decode_bytes` |
| `;` / tab / `\|` delimiters, quoted commas | Consistency-based delimiter detection | `data_loader.detect_delimiter` |
| Multi-sheet Excel | Sheet selector | `data_loader._read_excel`, `app.py` |
| Numbers stored as text (`£1,200`) | Flagged in the profile; the agent cleans and reports | `analysis.numeric_text_ratio` |
| DD/MM vs MM/DD dates | Detect the format; when ambiguous, assume UK order and warn | `data_loader.parse_date_columns` |
| Missing values, duplicates, outliers | Quality report with severity and suggested fixes | `analysis.quality_report` |
| Very large files | 50 MB limit; sampling above 500k rows, with a note | `data_loader`, `config` |
| Empty or header-only files | Clear error messages | `data_loader._finalise` |
| Awkward column names | snake_case with a mapping to the originals | `data_loader.normalise_columns` |
| Personal data sent to the API | Only the profile, sample rows and results are sent; privacy notice in the UI | `prompts`, `app.py` |

### LLM behaviour

| Risk | Mitigation | Implemented in |
|---|---|---|
| Invented numbers | Prompt rule; numeric evaluation against ground truth | `prompts.py`, `evals/` |
| Nonexistent columns or code errors | Errors returned to the model so it can self-correct | `tools._run_python`, `agent.ask` |
| Infinite tool loops | Iteration cap (10) | `agent.ask` |
| Huge outputs | Truncated to 3,000 characters with guidance | `agent.truncate` |
| Long conversations | History trimmed at question boundaries | `agent.trimmed_history` |
| Causal over-claiming | Prompt rule; evaluation case | `prompts.py`, `evals/` |
| Ambiguous or off-topic questions | Prompt rules; evaluation cases | `prompts.py`, `evals/` |

### Security

| Risk | Mitigation | Implemented in |
|---|---|---|
| Arbitrary code execution | AST checks, restricted built-ins, subprocess, timeout | `sandbox.py`, `_sandbox_worker.py` |
| Prompt injection in cells | Data labelled as untrusted; evaluation case | `prompts.build_system`, `evals/` |
| API key leakage | `.env`/secrets, `.gitignore`, `.env.example` | repo config |

### API and operations

| Risk | Mitigation | Implemented in |
|---|---|---|
| 429 / 5xx / 529 errors | Exponential backoff with retries | `agent._call_api` |
| Bad key or wrong model | Friendly messages, no retry, history rolled back | `agent._friendly_error` |
| Responses cut off | Detected and flagged to the user | `agent.ask` (`max_tokens`) |
| Cost | Selectable model, token usage shown, prompt caching | `app.py`, `prompts.build_system` |
| Dependency drift | Minimum versions, CI on 3 Python versions | `requirements.txt`, CI |

## 5. Evaluation strategy

A golden set of 20 questions on a seeded synthetic dataset with planted issues. Numeric answers are graded against independently computed ground truth; behaviour is graded by an LLM judge against rubrics. The evaluation also tracks tool calls, tool error rate, self-correction, latency and token usage.

## 6. Phases

| Phase | Deliverable | Acceptance criteria |
|---|---|---|
| 1 | Loader, dataset, sandbox | Loader edge cases pass; the sandbox blocks dangerous code |
| 2 | Tools, agent, prompt | Correct answers to sample questions in the CLI |
| 3 | Streamlit UI | Upload, chat, charts, steps and downloads work |
| 4 | Hardening | Every risk above has a handling path and a test where feasible |
| 5 | Evaluation | Results table generated |
| 6 | Release | README, CI green, deployed demo |
