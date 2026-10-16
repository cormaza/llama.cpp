#!/usr/bin/env python3
import argparse
import curses
from datetime import datetime
import json
import os
import re
import sqlite3
import textwrap
import time
import urllib.request
import urllib.error
import sys

SERVER_URL = 'http://127.0.0.1:8080'
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CAPTURES_DIR = os.path.join(REPO_ROOT, 'captures')

_cached_model_name = None
_last_model_fetch = 0.0

def sanitize_model_slug(model_name):
    """Create a filesystem-safe folder prefix from the loaded model name."""
    if not model_name or model_name in ('Detecting...', 'Unknown Model', 'llama.cpp Server'):
        return 'unknown_model'
    primary = model_name.split('(')[0].strip() or model_name
    if primary.lower().endswith('.gguf'):
        primary = primary[:-5]
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', primary).strip('_')
    return slug or 'unknown_model'


class RoundRecorder:
    """Optional SQLite recorder for llama-server slot rounds (prefill, decode, prompt, and response)."""

    def __init__(self, base_dir=DEFAULT_CAPTURES_DIR, enabled=False):
        self.base_dir = base_dir
        self.enabled = enabled
        self.conn = None
        self.db_path = None
        self.rel_db_path = None
        self.session_id = None
        self.session_model = None
        self.active_rounds = {}      # slot_id -> dict tracking active round state
        self.recorded_keys = set()   # (slot_id, task_id) recorded in current session
        self.rounds_count = 0

    def _ensure_db(self, model_name, server_url):
        if self.conn is not None:
            return
        slug = sanitize_model_slug(model_name)
        ts_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        folder_name = f'{slug}_{ts_str}'
        folder_path = os.path.join(self.base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        self.db_path = os.path.join(folder_path, 'rounds.db')
        try:
            self.rel_db_path = os.path.relpath(self.db_path, REPO_ROOT)
        except ValueError:
            self.rel_db_path = self.db_path

        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute('PRAGMA journal_mode=WAL;')
        self.conn.execute('PRAGMA synchronous=NORMAL;')
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS capture_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                server_url TEXT,
                started_at TEXT,
                ended_at TEXT
            );

            CREATE TABLE IF NOT EXISTS task_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                slot_id INTEGER,
                task_id INTEGER,
                model_name TEXT,
                status TEXT,
                started_at TEXT,
                prefill_done_at TEXT,
                finished_at TEXT,
                n_ctx INTEGER,
                n_prompt_tokens INTEGER,
                n_cache_tokens INTEGER,
                n_prompt_processed INTEGER,
                prefill_duration_ms REAL,
                prefill_tps REAL,
                n_decoded_tokens INTEGER,
                n_remain_tokens INTEGER,
                decode_duration_ms REAL,
                decode_avg_tps REAL,
                decode_peak_tps REAL,
                total_duration_ms REAL,
                prompt_text TEXT,
                response_text TEXT,
                params_json TEXT,
                UNIQUE(session_id, slot_id, task_id)
            );
            """
        )
        now_iso = datetime.now().isoformat(timespec='milliseconds')
        cur = self.conn.execute(
            'INSERT INTO capture_sessions (model_name, server_url, started_at) VALUES (?, ?, ?)',
            (model_name, server_url, now_iso),
        )
        self.session_id = cur.lastrowid
        self.session_model = model_name
        self.conn.commit()

    def toggle(self, model_name=None, server_url=SERVER_URL):
        self.enabled = not self.enabled
        if self.enabled and model_name and model_name not in ('Detecting...',):
            self._ensure_db(model_name, server_url)
        elif not self.enabled:
            self.flush_active()
        return self.enabled

    def _upsert_round(self, r):
        if self.conn is None:
            return
        self.conn.execute(
            """
            INSERT INTO task_rounds (
                session_id, slot_id, task_id, model_name, status,
                started_at, prefill_done_at, finished_at,
                n_ctx, n_prompt_tokens, n_cache_tokens, n_prompt_processed,
                prefill_duration_ms, prefill_tps,
                n_decoded_tokens, n_remain_tokens,
                decode_duration_ms, decode_avg_tps, decode_peak_tps,
                total_duration_ms, prompt_text, response_text, params_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, slot_id, task_id) DO UPDATE SET
                model_name = excluded.model_name,
                status = excluded.status,
                prefill_done_at = COALESCE(excluded.prefill_done_at, task_rounds.prefill_done_at),
                finished_at = excluded.finished_at,
                n_ctx = excluded.n_ctx,
                n_prompt_tokens = excluded.n_prompt_tokens,
                n_cache_tokens = excluded.n_cache_tokens,
                n_prompt_processed = MAX(task_rounds.n_prompt_processed, excluded.n_prompt_processed),
                prefill_duration_ms = excluded.prefill_duration_ms,
                prefill_tps = excluded.prefill_tps,
                n_decoded_tokens = MAX(task_rounds.n_decoded_tokens, excluded.n_decoded_tokens),
                n_remain_tokens = excluded.n_remain_tokens,
                decode_duration_ms = excluded.decode_duration_ms,
                decode_avg_tps = excluded.decode_avg_tps,
                decode_peak_tps = MAX(task_rounds.decode_peak_tps, excluded.decode_peak_tps),
                total_duration_ms = excluded.total_duration_ms,
                prompt_text = CASE WHEN length(excluded.prompt_text) >= length(task_rounds.prompt_text)
                                   THEN excluded.prompt_text ELSE task_rounds.prompt_text END,
                response_text = CASE WHEN length(excluded.response_text) >= length(task_rounds.response_text)
                                     THEN excluded.response_text ELSE task_rounds.response_text END,
                params_json = excluded.params_json
            """,
            (
                self.session_id,
                r['slot_id'],
                r['task_id'],
                r['model_name'],
                r['status'],
                r['started_at'],
                r['prefill_done_at'],
                r['finished_at'],
                r['n_ctx'],
                r['n_prompt_tokens'],
                r['n_cache_tokens'],
                r['n_prompt_processed'],
                round(r['prefill_duration_ms'], 2),
                round(r['prefill_tps'], 2),
                r['n_decoded_tokens'],
                r['n_remain_tokens'],
                round(r['decode_duration_ms'], 2),
                round(r['decode_avg_tps'], 2),
                round(r['decode_peak_tps'], 2),
                round(r['total_duration_ms'], 2),
                r['prompt_text'],
                r['response_text'],
                r['params_json'],
            ),
        )
        key = (r['slot_id'], r['task_id'])
        if key not in self.recorded_keys:
            self.recorded_keys.add(key)
            self.rounds_count = len(self.recorded_keys)

    def update(self, slots_cache, slot_speeds, model_name, server_url=SERVER_URL):
        if not self.enabled or not slots_cache:
            return
        if self.conn is None:
            self._ensure_db(model_name, server_url)

        now = time.time()
        now_iso = datetime.now().isoformat(timespec='milliseconds')

        for s in slots_cache:
            sid = s.get('id', 0)
            is_proc = s.get('is_processing', False)
            raw_task_id = s.get('id_task')
            task_id = int(raw_task_id) if isinstance(raw_task_id, int) or (isinstance(raw_task_id, str) and raw_task_id.isdigit()) else -1

            nt = s.get('next_token', [{}])
            nt_info = nt[0] if isinstance(nt, list) and len(nt) > 0 else {}
            has_next = nt_info.get('has_next_token', False)
            n_decoded = nt_info.get('n_decoded', 0)
            n_remain = nt_info.get('n_remain', -1)
            n_prompt = s.get('n_prompt_tokens', 0)
            n_proc = s.get('n_prompt_tokens_processed', 0)
            n_cache = s.get('n_prompt_tokens_cache', 0)
            n_ctx = s.get('n_ctx', 0)
            prompt_text = s.get('prompt', '') or ''
            response_text = s.get('generated', '') or ''
            params = s.get('params', {})
            cur_speed = slot_speeds.get(sid, 0.0)

            prev_round = self.active_rounds.get(sid)

            # Finalize previous active round if slot became idle or switched to a different task_id
            if prev_round is not None and (not is_proc or prev_round['task_id'] != task_id):
                prev_round['status'] = 'completed'
                prev_round['finished_at'] = now_iso
                prev_round['total_duration_ms'] = max(0.0, (now - prev_round['start_ts']) * 1000.0)
                if prev_round['task_id'] == task_id:
                    prev_round['n_decoded_tokens'] = max(prev_round['n_decoded_tokens'], n_decoded)
                    prev_round['n_remain_tokens'] = n_remain
                    if len(prompt_text) >= len(prev_round['prompt_text']):
                        prev_round['prompt_text'] = prompt_text
                    if len(response_text) >= len(prev_round['response_text']):
                        prev_round['response_text'] = response_text
                self._upsert_round(prev_round)
                del self.active_rounds[sid]
                prev_round = None

            if is_proc:
                if prev_round is None:
                    already_decoding = bool(has_next and n_decoded > 0)
                    prev_round = {
                        'slot_id': sid,
                        'task_id': task_id,
                        'model_name': model_name,
                        'status': 'decoding' if already_decoding else 'prefill',
                        'start_ts': now,
                        'started_at': now_iso,
                        'prefill_done_ts': now if already_decoding else None,
                        'prefill_done_at': now_iso if already_decoding else None,
                        'finished_at': now_iso,
                        'initial_decoded': n_decoded if already_decoding else 0,
                        'n_ctx': n_ctx,
                        'n_prompt_tokens': n_prompt,
                        'n_cache_tokens': n_cache,
                        'n_prompt_processed': n_proc,
                        'prefill_duration_ms': 0.0,
                        'prefill_tps': 0.0,
                        'n_decoded_tokens': n_decoded,
                        'n_remain_tokens': n_remain,
                        'decode_duration_ms': 0.0,
                        'decode_avg_tps': cur_speed if already_decoding else 0.0,
                        'decode_peak_tps': cur_speed if already_decoding else 0.0,
                        'total_duration_ms': 0.0,
                        'prompt_text': prompt_text,
                        'response_text': response_text,
                        'params_json': json.dumps(params, ensure_ascii=False),
                    }
                    self.active_rounds[sid] = prev_round
                else:
                    prev_round['model_name'] = model_name
                    prev_round['finished_at'] = now_iso
                    prev_round['n_ctx'] = n_ctx
                    prev_round['n_prompt_tokens'] = max(prev_round['n_prompt_tokens'], n_prompt)
                    prev_round['n_cache_tokens'] = max(prev_round['n_cache_tokens'], n_cache)
                    prev_round['n_prompt_processed'] = max(prev_round['n_prompt_processed'], n_proc)
                    prev_round['n_decoded_tokens'] = max(prev_round['n_decoded_tokens'], n_decoded)
                    prev_round['n_remain_tokens'] = n_remain
                    prev_round['total_duration_ms'] = max(0.0, (now - prev_round['start_ts']) * 1000.0)
                    if len(prompt_text) >= len(prev_round['prompt_text']):
                        prev_round['prompt_text'] = prompt_text
                    if len(response_text) >= len(prev_round['response_text']):
                        prev_round['response_text'] = response_text

                    if has_next:
                        prev_round['status'] = 'decoding'
                        if prev_round['prefill_done_ts'] is None:
                            prev_round['prefill_done_ts'] = now
                            prev_round['prefill_done_at'] = now_iso
                            prev_round['prefill_duration_ms'] = max(0.0, (now - prev_round['start_ts']) * 1000.0)
                            eff_proc = prev_round['n_prompt_processed'] or max(0, n_prompt - n_cache)
                            if prev_round['prefill_duration_ms'] > 0:
                                prev_round['prefill_tps'] = eff_proc / (prev_round['prefill_duration_ms'] / 1000.0)

                        dec_ms = max(0.0, (now - prev_round['prefill_done_ts']) * 1000.0)
                        prev_round['decode_duration_ms'] = dec_ms
                        dec_tokens = max(0, n_decoded - prev_round['initial_decoded'])
                        if dec_ms > 100 and dec_tokens > 0:
                            prev_round['decode_avg_tps'] = dec_tokens / (dec_ms / 1000.0)
                        elif cur_speed > 0:
                            prev_round['decode_avg_tps'] = cur_speed
                        prev_round['decode_peak_tps'] = max(prev_round['decode_peak_tps'], cur_speed)
                    else:
                        prev_round['status'] = 'prefill'
                        pf_ms = max(0.0, (now - prev_round['start_ts']) * 1000.0)
                        prev_round['prefill_duration_ms'] = pf_ms
                        eff_proc = prev_round['n_prompt_processed'] or max(0, n_prompt - n_cache)
                        if pf_ms > 0 and eff_proc > 0:
                            prev_round['prefill_tps'] = eff_proc / (pf_ms / 1000.0)

                self._upsert_round(prev_round)

            elif (n_prompt > 0 or response_text) and (sid, task_id) not in self.recorded_keys:
                # Snapshot already-completed slot state if recording was toggled ON after completion
                snap_round = {
                    'slot_id': sid,
                    'task_id': task_id,
                    'model_name': model_name,
                    'status': 'completed',
                    'started_at': now_iso,
                    'prefill_done_at': now_iso,
                    'finished_at': now_iso,
                    'n_ctx': n_ctx,
                    'n_prompt_tokens': n_prompt,
                    'n_cache_tokens': n_cache,
                    'n_prompt_processed': n_proc or max(0, n_prompt - n_cache),
                    'prefill_duration_ms': 0.0,
                    'prefill_tps': 0.0,
                    'n_decoded_tokens': n_decoded,
                    'n_remain_tokens': n_remain,
                    'decode_duration_ms': 0.0,
                    'decode_avg_tps': 0.0,
                    'decode_peak_tps': 0.0,
                    'total_duration_ms': 0.0,
                    'prompt_text': prompt_text,
                    'response_text': response_text,
                    'params_json': json.dumps(params, ensure_ascii=False),
                }
                self._upsert_round(snap_round)

        self.conn.commit()

    def flush_active(self):
        if self.conn is None:
            return
        now_iso = datetime.now().isoformat(timespec='milliseconds')
        for r in list(self.active_rounds.values()):
            r['finished_at'] = now_iso
            self._upsert_round(r)
        if self.session_id is not None:
            self.conn.execute(
                'UPDATE capture_sessions SET ended_at = ? WHERE session_id = ?',
                (now_iso, self.session_id),
            )
        self.conn.commit()

    def close(self):
        if self.conn is not None:
            try:
                self.flush_active()
                self.conn.close()
            except Exception:
                pass
            self.conn = None


def fetch_model_name(url, force=False):
    global _cached_model_name, _last_model_fetch
    now = time.time()
    if not force and _cached_model_name and (now - _last_model_fetch) < 5.0:
        return _cached_model_name

    try:
        req_p = urllib.request.Request(f'{url}/props', headers={'User-Agent': 'llama-monitor/1.0'})
        with urllib.request.urlopen(req_p, timeout=0.8) as resp_p:
            props = json.loads(resp_p.read().decode())
            alias = (props.get('model_alias') or '').strip()
            mpath = (props.get('model_path') or '').strip()
            mfile = os.path.basename(mpath) if mpath else ''
            nested = (props.get('model', {}).get('name') or '').strip() if isinstance(props.get('model'), dict) else ''

            if alias and mfile and alias.lower() != mfile.lower():
                _cached_model_name = f'{alias} ({mfile})'
            elif mfile:
                _cached_model_name = mfile
            elif alias:
                _cached_model_name = alias
            elif nested:
                _cached_model_name = nested
            if _cached_model_name:
                _last_model_fetch = now
                return _cached_model_name
    except Exception:
        pass

    try:
        req_m = urllib.request.Request(f'{url}/v1/models', headers={'User-Agent': 'llama-monitor/1.0'})
        with urllib.request.urlopen(req_m, timeout=0.8) as resp_m:
            mdata = json.loads(resp_m.read().decode())
            data_list = mdata.get('data', [])
            if isinstance(data_list, list) and data_list:
                mid = (data_list[0].get('id') or '').strip()
                if mid:
                    _cached_model_name = os.path.basename(mid)
                    _last_model_fetch = now
                    return _cached_model_name
            models_list = mdata.get('models', [])
            if isinstance(models_list, list) and models_list:
                mname = (models_list[0].get('name') or models_list[0].get('model') or '').strip()
                if mname:
                    _cached_model_name = os.path.basename(mname)
                    _last_model_fetch = now
                    return _cached_model_name
    except Exception:
        pass

    return _cached_model_name or 'Unknown Model'

def fetch_data(url):
    try:
        req = urllib.request.Request(f'{url}/slots', headers={'User-Agent': 'llama-monitor/1.0'})
        with urllib.request.urlopen(req, timeout=1.2) as resp:
            slots = json.loads(resp.read().decode())

        model_name = fetch_model_name(url)
        return slots, model_name
    except Exception as e:
        return None, str(e)

def draw_bar(used, total, width=20):
    if total <= 0:
        return '[' + ' ' * width + ']'
    ratio = min(1.0, max(0.0, used / total))
    filled = int(round(ratio * width))
    bar = '█' * filled + '░' * (width - filled)
    return f'[{bar}] {ratio*100:4.1f}%'

def safe_addstr(win, y, x, text, attr=0):
    max_y, max_x = win.getmaxyx()
    if 0 <= y < max_y and 0 <= x < max_x:
        try:
            win.addstr(y, x, text[:max_x - x - 1], attr)
        except curses.error:
            pass

def wrap_text_lines(text, width):
    """Wrap text into display lines fitting within `width`, preserving newlines and indentation."""
    if not text:
        return []
    width = max(10, width)
    wrapped = []
    for raw_line in text.replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        line = raw_line.expandtabs(4)
        if len(line) <= width:
            wrapped.append(line)
        else:
            leading_len = len(line) - len(line.lstrip(' '))
            leading = line[:leading_len] if leading_len < (width // 2) else ''
            rest = line[len(leading):]
            chunks = textwrap.wrap(
                rest,
                width=width,
                initial_indent=leading,
                replace_whitespace=False,
                drop_whitespace=True,
                break_long_words=True,
                break_on_hyphens=False,
            )
            if chunks:
                wrapped.extend(chunks)
            else:
                wrapped.append(line[:width])
    return wrapped

def update_scroll_offset(current_top, total_lines, vis_h, key):
    """Update scroll top index based on key press. Returns None when anchored at the bottom (auto-follow)."""
    max_top = max(0, total_lines - vis_h)
    if max_top <= 0:
        return None

    eff_top = max_top if current_top is None else min(max(0, current_top), max_top)
    page_step = max(1, vis_h - 1)

    if key in (curses.KEY_UP, ord('k'), ord('K')):
        new_top = max(0, eff_top - 1)
    elif key in (curses.KEY_DOWN, ord('j'), ord('J')):
        new_top = min(max_top, eff_top + 1)
    elif key in (curses.KEY_PPAGE, ord('u'), ord('U')):
        new_top = max(0, eff_top - page_step)
    elif key in (curses.KEY_NPAGE, ord('d'), ord('D')):
        new_top = min(max_top, eff_top + page_step)
    elif key in (curses.KEY_HOME, ord('g')):
        new_top = 0
    elif key in (curses.KEY_END, ord('G')):
        new_top = max_top
    else:
        return current_top

    return None if new_top >= max_top else new_top

def main(stdscr, record_default=False, captures_dir=DEFAULT_CAPTURES_DIR):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(500) # 500ms refresh

    # Setup colors
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)     # Header/Titles
    curses.init_pair(2, curses.COLOR_GREEN, -1)    # Generating / Active
    curses.init_pair(3, curses.COLOR_YELLOW, -1)   # Prefill / Busy
    curses.init_pair(4, curses.COLOR_WHITE, -1)    # Normal text
    curses.init_pair(5, curses.COLOR_RED, -1)      # Warning/Error
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # Accent / Highlight
    curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_CYAN) # Inverted bar

    prev_decoded = {}
    prev_time = time.time()
    slot_speeds = {}
    paused = False
    view_mode = 'overview'  # 'overview' or 'drill'
    selected_slot = 0
    active_section = 0      # 0 = Generated Stream, 1 = Received Prompt
    full_section = False    # True = Expand active section to full height
    scroll_tops = {}        # (slot_id, section_idx) -> int top_line or None (auto-follow bottom)
    slot_tasks = {}         # slot_id -> last seen task_id (to reset scroll on new task)
    slots_cache = []
    current_model = 'Detecting...'
    recorder = RoundRecorder(base_dir=captures_dir, enabled=record_default)

    SCROLL_KEYS = (
        curses.KEY_UP, ord('k'), ord('K'),
        curses.KEY_DOWN, ord('j'), ord('J'),
        curses.KEY_PPAGE, ord('u'), ord('U'),
        curses.KEY_NPAGE, ord('d'), ord('D'),
        curses.KEY_HOME, ord('g'),
        curses.KEY_END, ord('G'),
    )

    try:
        while True:
            key = stdscr.getch()
            
            # Navigation & Controls
            if key in (ord('q'), ord('Q')):
                if view_mode == 'drill':
                    view_mode = 'overview'
                else:
                    break
            elif key in (27, curses.KEY_BACKSPACE, 127, ord('b'), ord('B')): # ESC or Backspace
                view_mode = 'overview'
            elif key == ord(' '):
                paused = not paused
            elif key in (ord('r'), ord('R')):
                recorder.toggle(current_model, SERVER_URL)
            elif ord('0') <= key <= ord('7'):
                selected_slot = key - ord('0')
                view_mode = 'drill'
            elif view_mode == 'overview':
                if key in (curses.KEY_UP, ord('k'), ord('K')):
                    selected_slot = max(0, selected_slot - 1)
                elif key in (curses.KEY_DOWN, ord('j'), ord('J')):
                    if slots_cache:
                        selected_slot = min(len(slots_cache) - 1, selected_slot + 1)
                elif key in (curses.KEY_LEFT, ord('h'), ord('H')):
                    if slots_cache:
                        selected_slot = (selected_slot - 1) % len(slots_cache)
                elif key in (curses.KEY_RIGHT, ord('l'), ord('L'), ord('\t')):
                    if slots_cache:
                        selected_slot = (selected_slot + 1) % len(slots_cache)
                elif key in (10, 13, ord('d'), ord('D'), curses.KEY_ENTER): # Enter or d
                    view_mode = 'drill'
            elif view_mode == 'drill':
                if key in (curses.KEY_LEFT, ord('h'), ord('H')):
                    if slots_cache:
                        selected_slot = (selected_slot - 1) % len(slots_cache)
                elif key in (curses.KEY_RIGHT, ord('l'), ord('L')):
                    if slots_cache:
                        selected_slot = (selected_slot + 1) % len(slots_cache)
                elif key in (ord('\t'), ord('s'), ord('S')):
                    active_section = 1 - active_section
                elif key in (ord('f'), ord('F')):
                    full_section = not full_section

            now = time.time()
            dt = now - prev_time

            if not paused or not slots_cache:
                data, model_info = fetch_data(SERVER_URL)
                if data is not None:
                    slots_cache = data
                    current_model = model_info
            else:
                model_info = '(PAUSED)'

            stdscr.erase()
            max_y, max_x = stdscr.getmaxyx()

            if not slots_cache:
                title = '  llama.cpp Server Task & Slot Monitor (ROCm / HIP)  '
                safe_addstr(stdscr, 0, max(0, (max_x - len(title)) // 2), title, curses.color_pair(1) | curses.A_BOLD)
                safe_addstr(stdscr, 3, 2, f'Waiting for server at {SERVER_URL}...', curses.color_pair(5))
                safe_addstr(stdscr, 4, 2, f'Status: {model_info}', curses.color_pair(5))
                safe_addstr(stdscr, max_y - 1, 2, '  [Q] Quit  |  Refresh: 0.5s  ', curses.color_pair(1) | curses.A_DIM)
                stdscr.refresh()
                time.sleep(0.5)
                continue

            # Keep selected_slot in bounds
            if selected_slot >= len(slots_cache):
                selected_slot = len(slots_cache) - 1

            # Calculate speeds & totals
            total_speed = 0.0
            active_slots = 0
            total_gen_tokens = 0
            total_ctx_used = 0
            total_ctx_capacity = 0

            for s in slots_cache:
                sid = s.get('id', 0)
                is_proc = s.get('is_processing', False)
                nt = s.get('next_token', [{}])
                nt_info = nt[0] if isinstance(nt, list) and len(nt) > 0 else {}
                n_decoded = nt_info.get('n_decoded', 0)
                n_prompt = s.get('n_prompt_tokens', 0)
                n_ctx = s.get('n_ctx', 65536)

                total_gen_tokens += n_decoded
                total_ctx_used += (n_prompt + n_decoded)
                total_ctx_capacity += n_ctx

                if is_proc:
                    active_slots += 1

                prev = prev_decoded.get(sid, n_decoded)
                if dt >= 0.4:
                    delta = max(0, n_decoded - prev)
                    if delta > 3000:  # task changed, reset spike
                        delta = 0
                    speed = delta / dt
                    slot_speeds[sid] = speed
                    prev_decoded[sid] = n_decoded
                else:
                    speed = slot_speeds.get(sid, 0.0)

                total_speed += speed

            if dt >= 0.4:
                prev_time = now

            if not paused:
                recorder.update(slots_cache, slot_speeds, current_model, SERVER_URL)

            pause_badge = ' [PAUSED]' if paused else ''
            if recorder.enabled:
                db_label = recorder.rel_db_path or 'initializing...'
                rec_status_full = f'● REC ON ({recorder.rounds_count} rounds -> {db_label})'
                rec_status_short = f'● REC ({recorder.rounds_count} rounds)'
                rec_color = curses.color_pair(5) | curses.A_BOLD
            else:
                rec_status_full = '○ REC OFF (Press R to record SQLite)'
                rec_status_short = '○ REC OFF [R]'
                rec_color = curses.color_pair(4) | curses.A_DIM

            # ======================================================================
            # VIEW 1: OVERVIEW TABLE MODE
            # ======================================================================
            if view_mode == 'overview':
                title = f'  llama.cpp Server Tasks & Slots Overview (ROCm / HIP){pause_badge}  '
                safe_addstr(stdscr, 0, max(0, (max_x - len(title)) // 2), title, curses.color_pair(1) | curses.A_BOLD)
                model_line = f' Loaded Model: {current_model}  |  '
                safe_addstr(stdscr, 1, 2, model_line, curses.color_pair(2) | curses.A_BOLD)
                safe_addstr(stdscr, 1, 2 + len(model_line), rec_status_full, rec_color)

                summary_line = (
                    f' Target: {SERVER_URL}  |  Active Slots: {active_slots}/{len(slots_cache)}  |  '
                    f'Total GPU Speed: {total_speed:5.1f} t/s  |  Total Decoded: {total_gen_tokens:,}'
                )
                safe_addstr(stdscr, 2, 2, summary_line, curses.color_pair(1))

                ctx_bar = draw_bar(total_ctx_used, total_ctx_capacity, width=28)
                safe_addstr(stdscr, 3, 2, f' Global Context Pool: {ctx_bar} ({total_ctx_used:,} / {total_ctx_capacity:,} tokens)', curses.color_pair(4))

                # Table Header
                h_y = 5
                hdr = f'  {"Slot":<6} {"Task ID":<9} {"State":<12} {"Tokens Gen":<12} {"Speed":<10} {"Prompt":<9} {"Cache Hit":<11} {"Context Usage (%)":<28}'
                safe_addstr(stdscr, h_y, 2, hdr.ljust(max_x - 4)[:max_x - 4], curses.A_REVERSE | curses.color_pair(1))

                # Rows
                row_y = h_y + 1
                for idx, s in enumerate(slots_cache):
                    if row_y >= max_y - 3:
                        break

                    sid = s.get('id', 0)
                    is_proc = s.get('is_processing', False)
                    task_id = str(s.get('id_task', '-'))
                    nt = s.get('next_token', [{}])
                    nt_info = nt[0] if isinstance(nt, list) and len(nt) > 0 else {}
                    has_next = nt_info.get('has_next_token', False)
                    n_decoded = nt_info.get('n_decoded', 0)
                    n_prompt = s.get('n_prompt_tokens', 0)
                    n_cache = s.get('n_prompt_tokens_cache', 0)
                    n_ctx = s.get('n_ctx', 65536)
                    cur_ctx = n_prompt + n_decoded
                    speed = slot_speeds.get(sid, 0.0)

                    if not is_proc:
                        state_str = 'IDLE'
                        c_pair = curses.color_pair(4)
                    elif has_next:
                        state_str = 'DECODING'
                        c_pair = curses.color_pair(2) | curses.A_BOLD
                    else:
                        state_str = 'PREFILL'
                        c_pair = curses.color_pair(3) | curses.A_BOLD

                    bar_str = draw_bar(cur_ctx, n_ctx, width=15) + f' {cur_ctx:,}/{n_ctx//1024}k'
                    speed_str = f'{speed:5.1f} t/s' if is_proc and has_next else '-'

                    cursor = '▶ ' if idx == selected_slot else '  '
                    row = f'{cursor}{sid:<6} {task_id:<9} {state_str:<12} {n_decoded:<12,} {speed_str:<10} {n_prompt:<9,} {n_cache:<11,} {bar_str}'
                    
                    if idx == selected_slot:
                        safe_addstr(stdscr, row_y, 2, row[:max_x - 4], curses.color_pair(6) | curses.A_BOLD | curses.A_STANDOUT)
                    else:
                        safe_addstr(stdscr, row_y, 2, row[:max_x - 4], c_pair)
                    row_y += 1

                footer = '  [↑/↓] Select Slot  |  [Enter/0-7] Drill-Down  |  [R] Record SQLite  |  [Space] Pause  |  [Q] Quit  '
                safe_addstr(stdscr, max_y - 1, 2, footer, curses.color_pair(1) | curses.A_DIM)

            # ======================================================================
            # VIEW 2: DRILL-DOWN TASK INSPECTOR MODE
            # ======================================================================
            else:
                slot_obj = slots_cache[selected_slot]
                sid = slot_obj.get('id', selected_slot)
                task_id = slot_obj.get('id_task', 'N/A')
                is_proc = slot_obj.get('is_processing', False)
                nt = slot_obj.get('next_token', [{}])
                nt_info = nt[0] if isinstance(nt, list) and len(nt) > 0 else {}
                has_next = nt_info.get('has_next_token', False)
                n_decoded = nt_info.get('n_decoded', 0)
                n_prompt = slot_obj.get('n_prompt_tokens', 0)
                n_proc = slot_obj.get('n_prompt_tokens_processed', 0)
                n_cache = slot_obj.get('n_prompt_tokens_cache', 0)
                n_remain = nt_info.get('n_remain', -1)
                n_ctx = slot_obj.get('n_ctx', 65536)
                cur_ctx = n_prompt + n_decoded
                speed = slot_speeds.get(sid, 0.0)
                params = slot_obj.get('params', {})

                # Reset scroll positions if slot switched to a new task
                if slot_tasks.get(sid) != task_id:
                    slot_tasks[sid] = task_id
                    scroll_tops[(sid, 0)] = None
                    scroll_tops[(sid, 1)] = None

                if not is_proc:
                    state_badge = '● IDLE (Waiting for request)'
                    b_color = curses.color_pair(4)
                elif has_next:
                    state_badge = f'● DECODING (Streaming Tokens @ {speed:4.1f} t/s)'
                    b_color = curses.color_pair(2) | curses.A_BOLD
                else:
                    state_badge = f'● PREFILL (Ingesting Prompt: {n_proc}/{n_prompt} tokens)'
                    b_color = curses.color_pair(3) | curses.A_BOLD

                # Drill Header
                title = f'  === SLOT [{sid}] DRILL-DOWN: TASK #{task_id}{pause_badge} ===  '
                safe_addstr(stdscr, 0, max(0, (max_x - len(title)) // 2), title, curses.color_pair(6) | curses.A_BOLD)
                drill_info_line = f'State: {state_badge}  |  Model: {current_model}  |  '
                safe_addstr(stdscr, 1, 2, drill_info_line, b_color)
                safe_addstr(stdscr, 1, 2 + len(drill_info_line), rec_status_short, rec_color)

                # Metrics Box
                safe_addstr(stdscr, 2, 2, f'Context Usage : {draw_bar(cur_ctx, n_ctx, 25)} ({cur_ctx:,} / {n_ctx:,} tokens, max {n_ctx//1024}k)', curses.color_pair(1))
                safe_addstr(stdscr, 3, 2, f'Tokens Gen    : {n_decoded:,} tokens  |  Remaining: {n_remain}  |  Speed: {speed:5.1f} t/s', curses.color_pair(4))
                safe_addstr(stdscr, 4, 2, f'Prompt Tokens : {n_prompt:,} (Cache Hit: {n_cache:,} tok, {(n_cache/max(1,n_prompt))*100:.1f}%)', curses.color_pair(4))

                # Sampler Parameters
                temp = params.get('temperature', 'N/A')
                rep_p = params.get('repeat_penalty', 'N/A')
                dry_m = params.get('dry_multiplier', 'N/A')
                top_p = params.get('top_p', 'N/A')
                top_k = params.get('top_k', 'N/A')
                safe_addstr(stdscr, 5, 2, f'Samplers      : temp={temp} | top_p={top_p} | top_k={top_k} | rep_penalty={rep_p} | dry={dry_m}', curses.color_pair(1) | curses.A_DIM)

                safe_addstr(stdscr, 6, 2, '─' * max(1, max_x - 4), curses.color_pair(1) | curses.A_DIM)

                # Wrap Live Text Generation & Prompt Inspection
                content_width = max(10, max_x - 8)
                prompt_text = slot_obj.get('prompt', '')
                generated_text = slot_obj.get('generated', '')
                gen_lines = wrap_text_lines(generated_text, content_width)
                prompt_lines = wrap_text_lines(prompt_text, content_width)

                # Calculate section heights (rows 7 .. max_y - 2)
                avail_h = max(6, max_y - 8)
                if full_section:
                    gen_vis_h = max(1, avail_h - 1)
                    prompt_vis_h = max(1, avail_h - 1)
                else:
                    gen_box_h = max(3, int(avail_h * 0.58))
                    gen_vis_h = max(1, gen_box_h - 1)
                    p_start_y = 7 + gen_box_h
                    prompt_vis_h = max(1, (max_y - 1) - (p_start_y + 1))

                sec_lines_map = {0: gen_lines, 1: prompt_lines}
                sec_vis_map = {0: gen_vis_h, 1: prompt_vis_h}

                # Handle scroll keys in Drill-Down mode
                if key in SCROLL_KEYS:
                    cur_lines = sec_lines_map[active_section]
                    cur_vis = sec_vis_map[active_section]
                    other_sec = 1 - active_section
                    other_lines = sec_lines_map[other_sec]
                    other_vis = sec_vis_map[other_sec]

                    # Auto-focus the scrollable section if the active one fits completely in split view
                    if not full_section and len(cur_lines) <= cur_vis and len(other_lines) > other_vis:
                        active_section = other_sec
                        cur_lines = other_lines
                        cur_vis = other_vis

                    scroll_tops[(sid, active_section)] = update_scroll_offset(
                        scroll_tops.get((sid, active_section)),
                        len(cur_lines),
                        cur_vis,
                        key,
                    )

                def draw_section(sec_idx, start_y, vis_h, header_label, header_color, lines, empty_msg, empty_color):
                    is_active = (active_section == sec_idx)
                    total_l = len(lines)
                    max_top = max(0, total_l - vis_h)
                    raw_top = scroll_tops.get((sid, sec_idx))
                    top = max_top if raw_top is None else min(max(0, raw_top), max_top)
                    if raw_top is not None and top >= max_top:
                        scroll_tops[(sid, sec_idx)] = None
                        raw_top = None

                    if total_l > 0:
                        start_ln = top + 1
                        end_ln = min(total_l, top + vis_h)
                        if max_top == 0:
                            scroll_info = f'[Lines 1-{total_l}/{total_l}]'
                        elif raw_top is None:
                            scroll_info = f'[Lines {start_ln}-{end_ln}/{total_l} | ↓ LIVE]'
                        else:
                            below = total_l - end_ln
                            scroll_info = f'[Lines {start_ln}-{end_ln}/{total_l} | ↑ SCROLLED -{below}]'
                    else:
                        scroll_info = '[Empty]'

                    focus_marker = '▶ ' if is_active else '  '
                    mode_tag = ' [ACTIVE - FULL]' if (is_active and full_section) else (' [ACTIVE]' if is_active else '')
                    hdr_text = f'{focus_marker}{header_label} {scroll_info}{mode_tag}'
                    hdr_attr = header_color | (curses.A_STANDOUT if is_active else 0)
                    safe_addstr(stdscr, start_y, 2, hdr_text.ljust(max(1, max_x - 4))[:max(1, max_x - 4)], hdr_attr)

                    if lines:
                        visible = lines[top : top + vis_h]
                        for i, line_str in enumerate(visible):
                            safe_addstr(stdscr, start_y + 1 + i, 4, line_str, curses.color_pair(4))

                        # Draw scrollbar on the right if content overflows
                        if total_l > vis_h and max_x > 10:
                            sb_x = max_x - 3
                            thumb_pos = int(round((top / max(1, max_top)) * max(0, vis_h - 1)))
                            for i in range(vis_h):
                                if i == 0 and top > 0:
                                    ch = '▲'
                                elif i == vis_h - 1 and top < max_top:
                                    ch = '▼'
                                elif i == thumb_pos:
                                    ch = '█'
                                else:
                                    ch = '│'
                                safe_addstr(stdscr, start_y + 1 + i, sb_x, ch, curses.color_pair(1) | (curses.A_BOLD if i == thumb_pos else curses.A_DIM))
                    else:
                        safe_addstr(stdscr, start_y + 1, 4, empty_msg, empty_color)

                # Empty state messages
                if is_proc and has_next:
                    gen_empty_msg = f'Generating tokens... ({n_decoded:,} tokens decoded so far)'
                    gen_empty_col = curses.color_pair(2)
                elif is_proc:
                    gen_empty_msg = 'Currently in prefill stage (ingesting prompt)...'
                    gen_empty_col = curses.color_pair(3)
                else:
                    gen_empty_msg = 'No active task on this slot.'
                    gen_empty_col = curses.color_pair(4) | curses.A_DIM

                if is_proc:
                    prompt_empty_msg = f'Prompt size: {n_prompt:,} tokens ({n_cache:,} cached)'
                    prompt_empty_col = curses.color_pair(4)
                else:
                    prompt_empty_msg = 'No prompt received.'
                    prompt_empty_col = curses.color_pair(4) | curses.A_DIM

                if full_section:
                    if active_section == 0:
                        draw_section(
                            0, 7, gen_vis_h,
                            f'▼ LIVE GENERATED TOKENS STREAM ({n_decoded:,} tok):',
                            curses.color_pair(2) | curses.A_BOLD,
                            gen_lines, gen_empty_msg, gen_empty_col,
                        )
                    else:
                        draw_section(
                            1, 7, prompt_vis_h,
                            f'▼ RECEIVED PROMPT INPUT ({n_prompt:,} tok):',
                            curses.color_pair(3) | curses.A_BOLD,
                            prompt_lines, prompt_empty_msg, prompt_empty_col,
                        )
                else:
                    draw_section(
                        0, 7, gen_vis_h,
                        f'▼ LIVE GENERATED TOKENS STREAM ({n_decoded:,} tok):',
                        curses.color_pair(2) | curses.A_BOLD,
                        gen_lines, gen_empty_msg, gen_empty_col,
                    )
                    draw_section(
                        1, p_start_y, prompt_vis_h,
                        f'▼ RECEIVED PROMPT INPUT ({n_prompt:,} tok):',
                        curses.color_pair(3) | curses.A_BOLD,
                        prompt_lines, prompt_empty_msg, prompt_empty_col,
                    )

                # Drill Footer
                footer = '  [↑/↓/PgUp/PgDn/Home/End] Scroll  |  [Tab] Section  |  [F] Full  |  [←/→] Slot  |  [R] Record  |  [Space] Pause  |  [Esc/Q] Back  '
                safe_addstr(stdscr, max_y - 1, 2, footer, curses.color_pair(6) | curses.A_DIM)

            stdscr.refresh()
    finally:
        recorder.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Live TUI Monitor for llama.cpp Server Tasks & Slots')
    parser.add_argument('url', nargs='?', default=SERVER_URL, help=f'llama-server base URL (default: {SERVER_URL})')
    parser.add_argument('-r', '--record', action='store_true', default=False, help='Enable SQLite round recording at startup (default: disabled)')
    parser.add_argument('--captures-dir', default=DEFAULT_CAPTURES_DIR, help='Base directory for SQLite capture folders')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    SERVER_URL = args.url.rstrip('/')
    try:
        curses.wrapper(main, record_default=args.record, captures_dir=args.captures_dir)
    except KeyboardInterrupt:
        pass
