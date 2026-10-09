import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full", layout_file="layouts/app.grid.json")


@app.cell
def _():
    # Make natcap_agents importable regardless of where marimo is launched from.
    import pathlib
    import sys

    _root = next(
        (p for p in (pathlib.Path.cwd(), *pathlib.Path.cwd().parents) if (p / "natcap_agents").is_dir()),
        None,
    )
    if _root and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    import marimo as mo

    return (mo,)


@app.cell
def _():
    # One call authenticates Earth Engine + the models (Gemini or Ollama) from
    # .env, and builds the crew (orchestrator + geospatial model tools + researcher).
    # The energy meter starts first so the session total covers setup too.
    from natcap_agents import energy, results
    from natcap_agents.agents import build_crew
    from natcap_agents.auth import bootstrap

    energy_meter = energy.session_meter()
    setup_error = None
    settings = None
    crew = None
    try:
        with energy_meter.measure("Setup (auth + build crew)"):
            settings = bootstrap()
            crew = build_crew(settings)
    except Exception as e:  # noqa: BLE001
        setup_error = f"{type(e).__name__}: {e}"
    return crew, energy_meter, results, settings, setup_error


@app.cell
def _(mo, settings, setup_error):
    if setup_error is not None:
        _status = mo.callout(
            mo.md(
                f"**Setup needed** — {setup_error}\n\n"
                "Copy `.env.example` to `.env` and fill in your GCP project, Earth "
                "Engine service account, and model auth, then rerun this notebook."
            ),
            kind="danger",
        )
    else:
        _status = mo.callout(
            mo.md(
                f"**Project:** `{settings.project_id}` · **backend:** `{settings.llm_backend}`"
                f" · **models:** `{settings.orchestrator_model}` / `{settings.worker_model}`"
            ),
            kind="success",
        )
    mo.vstack([mo.md("## NatCap geospatial agent crew"), _status])
    return


@app.cell
def _(mo):
    prompt = mo.ui.text_area(
        placeholder="e.g. How much forest was lost in <region> since 2015?",
        rows=3,
        full_width=True,
        label="Prompt",
    )
    run_button = mo.ui.run_button(label="Run crew", kind="success")
    mo.vstack([prompt, run_button])
    return prompt, run_button


@app.cell
def _(crew, energy_meter, mo, prompt, results, run_button, setup_error):
    import anywidget

    # The rolling "thinking" log: streams each planning/action/tool-call step as
    # the crew produces it (mo.output.append), so this cell's output IS the live
    # sidebar. The map/stats cells below react to `board` once this cell finishes.

    class _FollowLog(anywidget.AnyWidget):
        # Invisible; keeps the log's scroll area (its grid cell) pinned to the
        # newest step as steps stream in. Scrolling up to read pauses it, and
        # scrolling back to the bottom resumes. A widget because marimo won't
        # run inline <script>s in HTML output.
        _esm = """
        function render({ el }) {
          // Nearest scrollable ancestor, stepping out of the widget's shadow root.
          let box = el;
          while ((box = box.parentElement ?? box.getRootNode().host)) {
            if (box === document.body) return;
            if (/auto|scroll/.test(getComputedStyle(box).overflowY)) break;
          }
          if (!box) return;
          let pinned = true;
          const onScroll = () => {
            pinned = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
          };
          const follow = () => {
            if (pinned) box.scrollTop = box.scrollHeight;
          };
          const observer = new MutationObserver(follow);
          observer.observe(box, { childList: true, subtree: true, characterData: true });
          box.addEventListener("scroll", onScroll);
          follow();
          return () => {
            observer.disconnect();
            box.removeEventListener("scroll", onScroll);
          };
        }
        export default { render };
        """

    def _fmt(text, limit=400):
        text = "" if text is None else str(text)
        return text if len(text) <= limit else text[:limit] + "…"

    def _describe_step(step):
        kind = type(step).__name__
        if kind == "TaskStep":
            return f"**Task**\n\n{_fmt(step.task)}"
        if kind == "PlanningStep":
            return f"**Plan**\n\n{_fmt(step.plan, 800)}"
        if kind == "ActionStep":
            lines = [f"**Step {step.step_number}**"]
            if step.model_output:
                lines.append(_fmt(step.model_output, 500))
            for tc in step.tool_calls or []:
                lines.append(f"→ `{tc.name}({tc.arguments})`")
            if step.observations:
                lines.append(f"```\n{_fmt(step.observations, 400)}\n```")
            if step.error:
                lines.append(f"⚠️ {_fmt(step.error)}")
            return "\n\n".join(lines)
        if kind == "FinalAnswerStep":
            return f"**Final answer**\n\n{_fmt(step.output, 800)}"
        return _fmt(step)

    answer = None

    if setup_error is not None:
        mo.output.replace(mo.md("_Fix the setup error above, then rerun._"))
        board = results.current()
    elif not run_button.value:
        mo.output.replace(mo.md("_Enter a prompt and click **Run crew** to start._"))
        board = results.current()
    elif not prompt.value.strip():
        mo.output.replace(mo.md("_Type a prompt first._"))
        board = results.current()
    else:
        results.reset()
        mo.output.append(mo.ui.anywidget(_FollowLog()))
        mo.output.append(mo.md(f"### Running\n\n{prompt.value}"))
        mo.output.append(mo.md("---"))
        with energy_meter.measure(prompt.value):
            for step in crew.run(prompt.value, stream=True):
                mo.output.append(mo.md(_describe_step(step)))
                mo.output.append(mo.md("---"))
                if type(step).__name__ == "FinalAnswerStep":
                    answer = step.output
        board = results.current()
    return (board,)


@app.cell
def _(board, mo):
    import branca.colormap as _cm
    import folium

    _BASEMAP_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    _BASEMAP_ATTR = "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"

    _m = folium.Map(location=[20, 0], zoom_start=2, tiles=_BASEMAP_URL, attr=_BASEMAP_ATTR)
    for _layer in board.layers:
        folium.TileLayer(
            tiles=_layer.tile_url, attr=_layer.attribution, name=_layer.name, overlay=True, control=True,
        ).add_to(_m)
    if board.layers:
        folium.LayerControl().add_to(_m)

    # One legend per distinct vis (skip duplicates — e.g. the same model run
    # twice for a year-over-year comparison shares one legend, not two).
    _seen_legends = set()
    for _layer in board.layers:
        _legend = _layer.legend
        if _legend is None:
            continue
        _key = (_legend.label, _legend.min, _legend.max, tuple(_legend.palette))
        if _key in _seen_legends:
            continue
        _seen_legends.add(_key)
        _caption = f"{_legend.label} ({_legend.unit})" if _legend.unit else _legend.label
        _cm.LinearColormap(
            colors=_legend.palette, vmin=_legend.min, vmax=_legend.max, caption=_caption,
        ).add_to(_m)

    mo.Html(f'<div style="height:100%;width:100%;overflow:hidden">{_m._repr_html_()}</div>')
    return


@app.cell
def _(board, mo):
    # board.rows is the flat one-metric-per-row table every model tool feeds;
    # board.tables holds any additional named summary tables a tool chooses to
    # publish (results.Board.add_table) — both render here, stacked, in a
    # scrollable area so neither is cut off regardless of row count.
    _blocks = []
    if board.rows:
        _blocks.append(mo.ui.table(board.rows, label="Stats", selection=None))
    else:
        _blocks.append(mo.md("_No stats yet — run a prompt that calls a model/compute tool._"))

    for _name, _rows in board.tables.items():
        _blocks.append(mo.md(f"**{_name}**"))
        _blocks.append(mo.ui.table(_rows, selection=None))

    mo.vstack(_blocks)
    return


@app.cell
def _(board, energy_meter, mo):
    # Electricity used this session, refreshed after each run (`board` is only
    # referenced so this cell reruns when the run cell finishes). It's the
    # chip's CPU + GPU + Neural Engine + DRAM energy; "above idle" subtracts the
    # average draw between runs — with a local backend like Ollama, that's
    # mostly inference.
    board

    def _wh(value):
        return None if value is None else round(value, 3)

    if not energy_meter.available:
        _energy = mo.callout(
            mo.md(
                "**Energy tracking is off** — this machine doesn't expose Apple Silicon's "
                "energy counters, so runs aren't measured and no energy use is shown."
            ),
            kind="warn",
        )
    else:
        _idle_w = energy_meter.idle_w
        _rows = [
            {
                "run": _run.label if len(_run.label) <= 60 else _run.label[:60] + "…",
                "seconds": round(_run.seconds, 1),
                "energy (Wh)": _wh(_run.energy_wh),
                "avg power (W)": None if _run.avg_w is None else round(_run.avg_w, 1),
                "above idle (Wh)": _wh(_run.above_idle_wh(_idle_w)),
            }
            for _run in energy_meter.runs
        ]
        _idle = f"{_idle_w:.1f} W" if _idle_w is not None else "not measured yet"
        _energy = mo.vstack([
            mo.md(
                f"**Energy** · session: **{energy_meter.session_wh:.3f} Wh** over "
                f"{energy_meter.elapsed_s / 60:.1f} min · idle baseline: {_idle}"
            ),
            mo.ui.table(_rows, selection=None),
        ])
    _energy
    return


if __name__ == "__main__":
    app.run()
