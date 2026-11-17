"""InsightPilot - Streamlit web app. Run: streamlit run app.py"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src import config
from src.agent import AgentAPIError, DataAnalystAgent, create_client
from src.analysis import profile_dataframe, quality_report
from src.data_loader import DataLoadError, LoadResult, load_file
from src.tools import TOOL_LABELS, ToolExecutor

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_retail.csv"
EXAMPLE_QUESTIONS = [
    "Is this data ready for analysis? What should I fix first?",
    "How has monthly revenue changed over time?",
    "Which product category has the highest return rate?",
    "Compare revenue by region after cleaning the region names.",
]

st.set_page_config(page_title="InsightPilot · AI Data Analyst", page_icon="📊", layout="wide")


# --- helpers -------------------------------------------------------------------------------
def get_env_api_key() -> str | None:
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:  # no secrets.toml
        key = None
    return key or os.getenv("ANTHROPIC_API_KEY")


def escape_md(text: str) -> str:
    """Streamlit renders $...$ as LaTeX; escape dollar signs in model output."""
    return text.replace("$", r"\$")


def init_state() -> None:
    defaults = {"loaded": None, "file_key": None, "file_name": None, "executor": None, "agent": None, "messages": []}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def set_dataset(result: LoadResult, name: str, key: str) -> None:
    st.session_state.loaded = result
    st.session_state.file_name = name
    st.session_state.file_key = key
    st.session_state.executor = ToolExecutor(result.df)
    st.session_state.agent = None
    st.session_state.messages = []


def get_agent(api_key: str, model: str) -> DataAnalystAgent:
    agent: DataAnalystAgent | None = st.session_state.agent
    if agent is None or getattr(agent, "_api_key", None) != api_key:
        client = create_client(api_key)
        history = agent.history if agent else []
        agent = DataAnalystAgent(client, st.session_state.executor, model=model)
        agent.history = history
        agent._api_key = api_key
        st.session_state.agent = agent
    agent.model = model
    return agent


def render_steps(steps, key_prefix: str) -> None:
    if not steps:
        return
    with st.expander(f"🔍 Show analysis steps ({len(steps)})"):
        for i, step in enumerate(steps, 1):
            label = TOOL_LABELS.get(step["tool"], step["tool"])
            icon = "❌" if step["is_error"] else "✅"
            purpose = step["input"].get("purpose") or step["input"].get("title") or ""
            st.markdown(f"**{icon} Step {i}: {label}**" + (f" — {purpose}" if purpose else ""))
            if step["tool"] == "run_python":
                st.code(step["input"].get("code", ""), language="python")
            elif step["tool"] == "create_chart":
                st.json(step["input"], expanded=False)
            st.code(step["output"][:3000], language="text")


def render_assistant(msg: dict, index: int) -> None:
    st.markdown(escape_md(msg["text"]))
    for j, chart in enumerate(msg.get("charts", [])):
        st.image(chart["png"], width="stretch")
        st.download_button("⬇️ Download chart", chart["png"], file_name=f"chart_{index}_{j}.png",
                           mime="image/png", key=f"dl_chart_{index}_{j}")
    render_steps(msg.get("steps", []), f"m{index}")
    usage = msg.get("usage") or {}
    if usage:
        st.caption(f"{len(msg.get('steps', []))} steps · {msg.get('elapsed', 0)}s · "
                   f"{usage.get('input_tokens', 0):,} input / {usage.get('output_tokens', 0):,} output tokens")


# --- sidebar -------------------------------------------------------------------------------
init_state()
with st.sidebar:
    st.title("📊 InsightPilot")
    st.caption("Ask questions about your data in plain English. An AI analyst writes and runs the code, "
               "makes charts and explains the results.")

    env_key = get_env_api_key()
    if env_key:
        api_key = env_key
        st.success("API key loaded from environment", icon="🔑")
    else:
        api_key = st.text_input("Anthropic API key", type="password",
                                help="Used only for this session and never stored. Get one at console.anthropic.com")
    model = st.selectbox("Model", config.AVAILABLE_MODELS,
                         index=config.AVAILABLE_MODELS.index(config.DEFAULT_MODEL)
                         if config.DEFAULT_MODEL in config.AVAILABLE_MODELS else 0)

    st.divider()
    uploaded = st.file_uploader("Upload a dataset", type=["csv", "tsv", "txt", "xlsx", "xls"],
                                help=f"Up to {config.MAX_FILE_MB} MB")
    use_sample = st.button("Use sample UK retail dataset", width="stretch")

    sheet = None
    loaded: LoadResult | None = st.session_state.loaded
    if uploaded is not None and loaded is not None and len(loaded.sheet_names) > 1 \
            and st.session_state.file_name == uploaded.name:
        sheet = st.selectbox("Excel sheet", loaded.sheet_names)

    try:
        if use_sample:
            set_dataset(load_file(SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name), SAMPLE_PATH.name, "sample")
        elif uploaded is not None:
            key = f"{uploaded.name}:{uploaded.size}:{sheet}"
            if key != st.session_state.get("last_upload_key"):  # only (re)load when the upload changes
                st.session_state.last_upload_key = key
                set_dataset(load_file(uploaded.getvalue(), uploaded.name, sheet), uploaded.name, key)
    except DataLoadError as exc:
        st.error(str(exc))

    st.info("🔒 The AI sees a summary of your columns, a few sample rows and the results of the code it runs. "
            "Code runs on this server. Avoid uploading personal or confidential data.", icon="ℹ️")

    if st.session_state.executor is not None:
        st.divider()
        executor: ToolExecutor = st.session_state.executor
        col1, col2 = st.columns(2)
        if col1.button("Clear chat", width="stretch"):
            st.session_state.messages = []
            if st.session_state.agent:
                st.session_state.agent.reset()
            st.rerun()
        if col2.button("Reset data", width="stretch", disabled=not executor.is_modified,
                       help="Undo all cleaning steps and restore the original upload"):
            executor.reset()
            st.rerun()
        st.download_button("⬇️ Download current data (CSV)", executor.df.to_csv(index=False).encode("utf-8"),
                           file_name="insightpilot_data.csv", mime="text/csv", width="stretch")
        if st.session_state.agent:
            total = st.session_state.agent.total_usage
            st.caption(f"Session tokens: {total['input_tokens']:,} in · {total['output_tokens']:,} out")


# --- main area -----------------------------------------------------------------------------
executor: ToolExecutor | None = st.session_state.executor
if executor is None:
    st.header("Your AI data analyst")
    st.markdown(
        "Upload a CSV or Excel file (or try the sample dataset) and ask questions such as "
        "*“Which region grew fastest last year?”* or *“Is this data ready for analysis?”*.\n\n"
        "**How it works**\n"
        "1. The file is profiled automatically: column types, missing values, duplicates and quality flags.\n"
        "2. You ask a question. The agent plans the analysis, writes pandas code and runs it in a guarded process.\n"
        "3. You get a direct answer, a chart where it helps, and the caveats a careful analyst would mention.\n"
        "4. Every step is shown, including the code, so the results can be checked."
    )
    st.stop()

loaded = st.session_state.loaded
df = executor.df
profile = profile_dataframe(df)
st.subheader(f"📄 {st.session_state.file_name}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{profile['n_rows']:,}")
c2.metric("Columns", profile["n_cols"])
c3.metric("Missing cells", f"{profile['missing_cells_pct']}%")
c4.metric("Duplicate rows", f"{profile['duplicate_rows']:,}")

for note in loaded.notes:
    st.caption(f"ℹ️ {note}")
if executor.change_log:
    st.caption("🧹 Cleaning applied this session: " + " · ".join(executor.change_log))

with st.expander("Preview data"):
    st.dataframe(df.head(200), width="stretch")
with st.expander("Automated data-quality check"):
    issues = quality_report(df)
    if not issues:
        st.success("No issues detected by the automated checks.")
    for issue in issues:
        icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}[issue["severity"]]
        st.markdown(f"{icon} **{issue['column']} — {issue['issue']}.** {issue['detail']}  \n"
                    f"<small>Suggested fix: {issue['suggestion']}</small>", unsafe_allow_html=True)

st.divider()
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(escape_md(msg["text"]))
        else:
            render_assistant(msg, i)

prompt = st.chat_input("Ask a question about your data…")
if not st.session_state.messages:
    st.caption("Try one of these:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, question in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(question, width="stretch"):
            prompt = question

if prompt:
    if not api_key:
        st.warning("Add your Anthropic API key in the sidebar to start chatting.")
        st.stop()
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(escape_md(prompt))
    with st.chat_message("assistant"):
        try:
            agent = get_agent(api_key, model)
        except AgentAPIError as exc:
            st.error(str(exc))
            st.session_state.messages.pop()
            st.stop()
        with st.status("Thinking…", expanded=False) as status:
            def on_event(kind: str, payload: dict) -> None:
                if kind == "tool_start":
                    label = TOOL_LABELS.get(payload["name"], payload["name"])
                    detail = payload["input"].get("purpose") or payload["input"].get("title") or ""
                    status.update(label=f"{label}…")
                    status.write(f"• {label}" + (f": {detail}" if detail else ""))
                elif kind == "retry":
                    status.write(f"⏳ {payload['error']} Retrying in {payload['wait']}s…")

            agent.on_event = on_event
            changes_before = len(executor.change_log)
            response = agent.ask(prompt)
            failed = response.stop_reason == "api_error"
            status.update(label="Something went wrong" if failed else f"Done in {response.elapsed_seconds}s",
                          state="error" if failed else "complete")

        msg = {
            "role": "assistant",
            "text": response.text,
            "charts": [{"png": c.png, "title": c.title} for c in response.charts],
            "steps": [{"tool": s.tool, "input": s.input, "output": s.output, "is_error": s.is_error}
                      for s in response.steps],
            "usage": response.usage,
            "elapsed": response.elapsed_seconds,
        }
        render_assistant(msg, len(st.session_state.messages))
        st.session_state.messages.append(msg)
    if len(executor.change_log) != changes_before:
        st.rerun()  # the data was cleaned: refresh the metrics and quality panel
