# geoforgood2026

## Agent sandbox

A **smolagents** crew running on Gemini via Google's `google-genai` SDK
(Vertex API key, Vertex service account, or the Gemini API) or on local
models via **Ollama**, backed by **Google Earth Engine** for compute. Forked from
[natcap/ee-agent-sandbox](../ee-agent-sandbox)'s infra.

### The crew

- **Orchestrator** — evaluates the prompt, picks whichever geospatial model
  tool(s) fit (auto-loaded from `tools/models/`), calls them (or writes ad hoc
  `ee` code directly), and reports the result.
- **Researcher** — sub-agent for a fact Earth Engine can't supply (an official
  date/status, a disputed statistic) via web search + page fetch.

A model tool computes over Earth Engine and publishes its result to a shared
**results board** (`natcap_agents/results.py`) — a map layer and/or a
stats-table row — as a side effect, rather than through the final-answer text.
The marimo app reads that board to draw the map and table. `tools/models/`
holds the project's own models (`example_forest_loss.py` is a working
template — swap it for the real 5); tools a Tool Smith agent forges land in
`tools/generated/` instead (see `forge.py`) and are auto-loaded the same way.

**Regions stay server-side.** A place name resolves via `resolve_region()`
(`natcap_agents/regions.py`) to a short `region_id` — the orchestrator never
pulls a boundary's coordinates back into its own context (a country/park
polygon can be thousands of points, which would bloat every later turn). Every
tool that takes a `region` accepts that `region_id`, or a plain
`'west,south,east,north'` bounding box — nothing bulkier.

**Dataset search has exactly two sources**: `search_full_catalog` (Google's
official EE Data Catalog) and `search_community_catalog`
([gee-community-catalog.org](https://gee-community-catalog.org/about/), mostly
`projects/sat-io/...` assets). `researcher`'s web search is for external facts
only, never for finding datasets.

### App

```bash
source .venv/bin/activate
marimo run notebooks/app.py       # read-only app view
# or: marimo edit notebooks/app.py   # to edit cells / the layout
```

Type a prompt, click **Run crew**, and watch: a rolling "thinking" log in the
side column streams each plan/step/tool-call as the crew works (staying
scrolled to the newest entry unless you scroll up to read), while the
central map and the stats table below it fill in once the run's model tool(s)
report their layers/stats. The dashboard arrangement lives in
`notebooks/layouts/app.grid.json`.

### Energy use

The **Energy** panel under the stats table shows the electricity this session
has used: one row per run (plus setup), and a session total, refreshed after
each run. It reads Apple Silicon's cumulative energy counters for the CPU,
GPU, Neural Engine and DRAM (what `powermetrics` reports, but via IOReport,
so no sudo), which makes each run's figure exact rather than sampled. Other
apps running at the same time are included, so **above idle** subtracts the
average draw measured between runs; with a local backend such as Ollama that
difference is mostly inference. Not counted: the display, SSD, Wi-Fi, charger
losses, and anything on Google's servers (Gemini, Earth Engine). On Intel
Macs and Linux there are no counters to read: `measure()` becomes a no-op and
the panel shows a warning instead. Outside the app:

```python
from natcap_agents import energy
meter = energy.session_meter()
with meter.measure("forest loss") as run:
    crew.run("How much forest was lost in <region> since 2015?")
print(f"{run.energy_wh:.2f} Wh, session {meter.session_wh:.2f} Wh")
```

### Setup

```bash
./setup.sh
$EDITOR .env                     # project id, EE service account, model backend/API key
source .venv/bin/activate
marimo run notebooks/app.py
```

You need a GCP project with the **Vertex AI** and **Earth Engine** APIs
enabled and the project **registered for Earth Engine**. Whoever authenticates
— a service account or your own user account — needs
`roles/earthengine.writer` + `roles/serviceusage.serviceUsageConsumer` (add
`roles/aiplatform.user` only if the models use that same identity).

### Earth Engine auth (`.env`)

Two credentials work, and `init_earth_engine()` picks between them by looking
at what is actually on disk:

- `GOOGLE_APPLICATION_CREDENTIALS` names a key file **that exists** → that
  service-account key (paired with `EE_SERVICE_ACCOUNT`).
- otherwise → **Application Default Credentials**. On a laptop, run once:
  ```bash
  gcloud auth application-default login --project $GCP_PROJECT_ID
  ```
  On a GCE VM / Cloud Run / Colab there is nothing to run — the attached
  service account already *is* the ADC, so leave both `.env` lines blank.

A key path that points at a missing file does not raise: the variable is
dropped (a stale one makes google-auth fail instead of falling back to ADC)
and ADC is used, with the path mentioned if initialization then fails.
`ee.Authenticate()` is never called — it opens a browser and would hang a
headless run.

### Model auth (`.env`)

The models use, in order:

- `LLM_BACKEND=ollama` → a local **Ollama** server; no key or service account
  for the models (see below).
- `VERTEX_API_KEY` set → Vertex **Express mode** with that key (no service
  account needed for the models).
- else `LLM_BACKEND=gemini` + `GEMINI_API_KEY` → Gemini Developer API.
- else → Vertex via the same credential Earth Engine uses (service-account key
  if present, otherwise ADC).

### Local models with Ollama (`.env`)

Set `LLM_BACKEND=ollama` and name the models by their Ollama tags:

```bash
LLM_BACKEND=ollama
ORCHESTRATOR_MODEL=qwen2.5-coder:14b
WORKER_MODEL=                     # blank: the researcher reuses the orchestrator's model
```

`ORCHESTRATOR_MODEL` is required (there is no sensible default for what you
have pulled), and a blank `WORKER_MODEL` reuses it, so only one model sits in
memory. Earth Engine still needs its Google credential as above; only the
models move off Google.

- `OLLAMA_HOST` — the server, default `http://localhost:11434` (same syntax as
  the ollama CLI).
- `OLLAMA_NUM_CTX` — the context window, default `32768`. It's sent with every
  request because Ollama's own default (often 4096) is smaller than the
  orchestrator's prompt, and Ollama cuts an over-long prompt off silently.
- `OLLAMA_THINK` — `true`/`false` for models that think (qwen3, deepseek-r1,
  gemma4, ...), `low`/`medium`/`high` for gpt-oss, or blank for the model's
  default. Thinking is slower and uses more energy.

The model is checked when the crew is built, so a server that isn't running,
a tag that isn't pulled (`ollama pull <tag>`), or `OLLAMA_THINK` on a model
that can't think fails at setup with the fix in the message. Ollama would also
apply stop sequences to a model's thinking and end the reply before the answer
starts, so for models that may think, `OllamaModel` matches them against the
streamed answer itself. Each reply is capped at 8192 tokens, thinking included:
Ollama's default is no cap, and a thinking model can loop without ever
reaching an answer. Sampling (temperature etc.) is left at each model's own
defaults, the values its publisher tuned it with.

### Layout

```
natcap_agents/
  config.py       # .env -> Settings
  auth.py         # bootstrap(): init EE (SA key or ADC) + build the models
  models.py       # VertexAIServerModel (google-genai; Vertex key / SA / Gemini API), OllamaModel
  safety.py       # authorized imports for the code-executing orchestrator
  results.py      # shared board: model tools publish layers/stats here
  energy.py       # session_meter(): electricity used, per run and per session
  regions.py      # resolve_region() + the region_id registry (no bulky coords)
  project_tools.py # auto-loads tools/models/ (the project's own models)
  forge.py        # auto-loads tools/generated/ (Tool Smith output, if added)
  tools/
    catalog.py           # search_full_catalog, verify_asset (official EE catalog)
    community_catalog.py # search_community_catalog (gee-community-catalog.org)
    compute.py    # compute_region_stats, get_map_tiles — publish to the board
    web.py        # researcher's web-search + page-fetch tools (facts, not datasets)
    models/       # the project's own geospatial model tools (5, eventually)
    generated/    # tool_smith output, if you add one
  agents.py       # build_crew(), build_specialists()
notebooks/
  app.py                     # the marimo dashboard (prompt -> map/stats/thinking)
  layouts/app.grid.json      # its grid arrangement (map/table/sidebar positions)
secrets/          # git-ignored; sa-key.json here
```
