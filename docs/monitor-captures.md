# Monitor TUI & SQLite Task Round Captures

[`monitor.sh`](../monitor.sh) (powered by [`scripts/monitor.py`](../scripts/monitor.py)) provides a live terminal dashboard for inspecting `llama-server` slots, token generation throughput, context utilization, and prompt/response streams.

It also includes an **optional SQLite round recorder** (`RoundRecorder`) that persists per-task prefill metrics, decode metrics, sampler parameters, input prompts, and generated responses for offline analysis.

---

## 1. Quick Start & Usage

By default, SQLite recording is **inactive** (`REC OFF`) so normal monitoring creates no files on disk.

### Start with recording disabled (default)
```bash
./monitor.sh
# or specify a custom server URL:
./monitor.sh http://127.0.0.1:8080
```

### Start with recording enabled from launch
```bash
./monitor.sh --record
# short flag:
./monitor.sh -r
# custom server URL + custom output directory:
./monitor.sh http://127.0.0.1:8080 --record --captures-dir ./captures
```

### Toggle recording live inside the TUI
Press `R` at any time (in either the **Overview** or **Drill-Down** view) to start or stop recording:
- **When turned ON**: Initializes a capture session and begins recording active and subsequent task rounds.
- **When turned OFF**: Flushes any in-progress rounds to disk and pauses recording.

---

## 2. TUI Controls

### Overview Mode
| Key | Action |
| :--- | :--- |
| `↑` / `↓` or `k` / `j` | Select slot |
| `Enter` or `d` or `0-7` | Open **Drill-Down** view for the selected slot |
| `R` | Toggle SQLite round recording (`REC ON` / `REC OFF`) |
| `Space` | Pause / resume UI refresh |
| `Q` | Quit monitor |

### Drill-Down Mode
| Key | Action |
| :--- | :--- |
| `↑` / `↓` or `k` / `j` | Scroll active text section up / down by 1 line |
| `PgUp` / `PgDn` or `u` / `d` | Scroll active text section up / down by 1 page |
| `Home` / `End` or `g` / `G` | Jump to oldest line (`Home`) or return to live bottom follow (`End`) |
| `Tab` or `s` | Switch active focus between `LIVE GENERATED TOKENS STREAM` and `RECEIVED PROMPT INPUT` |
| `f` | Toggle full-screen expansion for the active text section |
| `←` / `→` or `h` / `l` | Switch to previous / next slot |
| `R` | Toggle SQLite round recording (`REC ON` / `REC OFF`) |
| `Esc` / `Backspace` / `b` / `q` | Return to Overview mode |

> [!NOTE]
> When viewing a section at the bottom (`↓ LIVE`), new streaming tokens auto-scroll the viewport. Scrolling up (`↑ SCROLLED`) locks your position so you can read older text undisturbed while new tokens continue streaming below.

---

## 3. Capture Directory & File Layout

All captures are stored under `captures/` (which is listed in [`.gitignore`](../.gitignore)):

```text
captures/
└── <model_slug>_<YYYY-MM-DD_HH-MM-SS>/
    └── rounds.db
```

Example:
```text
captures/k2-horizon-7b_2026-10-16_11-55-00/rounds.db
```

- **WAL Mode**: The database is opened with `PRAGMA journal_mode=WAL;`, allowing you to run read queries with `sqlite3`, Python scripts, or GUI tools (DBeaver, DataGrip, SQLite Browser) **while `monitor.sh` is still actively recording**.
- **Incremental Upserts**: Active rounds are updated continuously during `prefill` and `decoding` phases and finalized as `completed` when the slot finishes processing or switches to a new `id_task`.

---

## 4. Database Schema

### Table: `capture_sessions`
Tracks each recording session.

| Column | Type | Description |
| :--- | :--- | :--- |
| `session_id` | `INTEGER PRIMARY KEY` | Auto-incremented session identifier |
| `model_name` | `TEXT` | Loaded model name reported by `/props` or `/v1/models` |
| `server_url` | `TEXT` | Target `llama-server` base URL |
| `started_at` | `TEXT` | ISO-8601 timestamp when recording started |
| `ended_at` | `TEXT` | ISO-8601 timestamp when recording stopped / exited |

### Table: `task_rounds`
Stores one row per `(session_id, slot_id, task_id)` round.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `INTEGER PRIMARY KEY` | Auto-incremented row ID |
| `session_id` | `INTEGER` | Foreign key to `capture_sessions.session_id` |
| `slot_id` | `INTEGER` | Server slot ID (`0..N-1`) |
| `task_id` | `INTEGER` | Server task ID (`id_task`) |
| `model_name` | `TEXT` | Loaded model name during the round |
| `status` | `TEXT` | Round lifecycle state: `prefill`, `decoding`, `completed`, or `interrupted` |
| `started_at` | `TEXT` | ISO-8601 timestamp when the round was first observed |
| `prefill_done_at` | `TEXT` | ISO-8601 timestamp when prefill completed and token decoding began |
| `finished_at` | `TEXT` | ISO-8601 timestamp when the round finished |
| `n_ctx` | `INTEGER` | Slot context capacity (`n_ctx`) |
| `n_prompt_tokens` | `INTEGER` | Total tokens in the input prompt |
| `n_cache_tokens` | `INTEGER` | Prompt tokens reused from KV cache (`n_past`) |
| `n_prompt_processed` | `INTEGER` | Prompt tokens processed during prefill |
| `prefill_duration_ms` | `REAL` | Prefill duration in milliseconds |
| `prefill_tps` | `REAL` | Prefill ingestion speed in tokens/second |
| `n_decoded_tokens` | `INTEGER` | Total generated tokens (`n_decoded`) |
| `n_remain_tokens` | `INTEGER` | Remaining token budget at last observation (`n_remain`) |
| `decode_duration_ms` | `REAL` | Token generation phase duration in milliseconds |
| `decode_avg_tps` | `REAL` | Average generation speed in tokens/second |
| `decode_peak_tps` | `REAL` | Peak observed generation speed in tokens/second |
| `total_duration_ms` | `REAL` | End-to-end round duration (`prefill + decode`) in milliseconds |
| `prompt_text` | `TEXT` | Full input prompt text |
| `response_text` | `TEXT` | Full generated output text |
| `params_json` | `TEXT` | JSON-serialized sampler parameters (`temperature`, `top_p`, `top_k`, `repeat_penalty`, etc.) |

---

## 5. Offline Analysis Recipes (`sqlite3`)

Set `DB` to the capture database you want to inspect (for example, the most recent capture):
```bash
DB="$(ls -t captures/*/rounds.db | head -n 1)"
echo "Inspecting: $DB"
```

### 5.1 Overview of all recorded rounds
```bash
sqlite3 -header -column "$DB" "
SELECT
    id,
    slot_id AS slot,
    task_id AS task,
    status,
    n_prompt_tokens AS prompt_tok,
    n_cache_tokens AS cache_tok,
    ROUND(prefill_duration_ms / 1000.0, 2) AS prefill_s,
    ROUND(prefill_tps, 1) AS prefill_tps,
    n_decoded_tokens AS gen_tok,
    ROUND(decode_duration_ms / 1000.0, 2) AS decode_s,
    ROUND(decode_avg_tps, 1) AS avg_tps,
    ROUND(decode_peak_tps, 1) AS peak_tps
FROM task_rounds
ORDER BY id DESC;
"
```

### 5.2 Aggregate throughput & KV cache efficiency summary
```bash
sqlite3 -header -column "$DB" "
SELECT
    model_name,
    COUNT(*) AS total_rounds,
    SUM(n_prompt_tokens) AS total_prompt_tok,
    ROUND(AVG(CASE WHEN n_prompt_tokens > 0 THEN (100.0 * n_cache_tokens / n_prompt_tokens) ELSE 0 END), 1) AS avg_cache_hit_pct,
    ROUND(AVG(prefill_tps), 1) AS avg_prefill_tps,
    SUM(n_decoded_tokens) AS total_gen_tok,
    ROUND(AVG(decode_avg_tps), 1) AS mean_decode_tps,
    ROUND(MAX(decode_peak_tps), 1) AS max_peak_tps
FROM task_rounds
WHERE status = 'completed'
GROUP BY model_name;
"
```

### 5.3 Inspect full prompt and response for a specific round
Replace `1` with the target `id` or `task_id`:
```bash
sqlite3 -line "$DB" "
SELECT
    id,
    slot_id,
    task_id,
    started_at,
    finished_at,
    n_prompt_tokens,
    n_decoded_tokens,
    prompt_text,
    response_text
FROM task_rounds
WHERE id = 1;
"
```

### 5.4 Find slowest prefill or decode rounds
```bash
sqlite3 -header -column "$DB" "
SELECT
    id,
    task_id,
    n_prompt_tokens,
     n_cache_tokens,
    ROUND(prefill_duration_ms / 1000.0, 2) AS prefill_s,
    n_decoded_tokens,
    ROUND(decode_duration_ms / 1000.0, 2) AS decode_s,
    ROUND(decode_avg_tps, 1) AS decode_tps
FROM task_rounds
ORDER BY total_duration_ms DESC
LIMIT 10;
"
```

### 5.5 Extract sampler parameters using SQLite JSON functions
```bash
sqlite3 -header -column "$DB" "
SELECT
    id,
    task_id,
    json_extract(params_json, '$.temperature') AS temp,
    json_extract(params_json, '$.top_p') AS top_p,
    json_extract(params_json, '$.top_k') AS top_k,
    json_extract(params_json, '$.repeat_penalty') AS rep_pen,
    ROUND(decode_avg_tps, 1) AS decode_tps
FROM task_rounds;
"
```

### 5.6 Export rounds to CSV or JSONL (for datasets / fine-tuning / evaluation)
**Export metrics to CSV:**
```bash
sqlite3 -header -csv "$DB" "
SELECT
    id, slot_id, task_id, model_name, status,
    started_at, prefill_done_at, finished_at,
    n_ctx, n_prompt_tokens, n_cache_tokens,
    prefill_duration_ms, prefill_tps,
    n_decoded_tokens, decode_duration_ms, decode_avg_tps, decode_peak_tps,
    total_duration_ms
FROM task_rounds;
" > rounds_metrics.csv
```

**Export prompt + response pairs to JSON:**
```bash
sqlite3 -json "$DB" "
SELECT
    id, task_id, model_name,
    n_prompt_tokens, n_decoded_tokens,
    decode_avg_tps, prompt_text, response_text
FROM task_rounds
WHERE status = 'completed';
" > rounds_conversations.json
```

---

## 6. Python Analysis Snippet

You can also load all captures across multiple models or dates directly in Python:

```python
import glob
import sqlite3

for db_file in sorted(glob.glob("captures/*/rounds.db")):
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT model_name, COUNT(*) AS rounds,
                   AVG(prefill_tps) AS prefill_tps,
                   AVG(decode_avg_tps) AS decode_tps
            FROM task_rounds
            GROUP BY model_name
            """
        ).fetchall()
        for r in rows:
            print(
                f"{db_file}: model={r['model_name']} | "
                f"rounds={r['rounds']} | "
                f"prefill={r['prefill_tps'] or 0:.1f} t/s | "
                f"decode={r['decode_tps'] or 0:.1f} t/s"
            )
```
