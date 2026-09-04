#!/usr/bin/env python3
import curses
import json
import time
import urllib.request
import urllib.error
import sys

SERVER_URL = 'http://127.0.0.1:8080'

def fetch_data(url):
    try:
        req = urllib.request.Request(f'{url}/slots', headers={'User-Agent': 'llama-monitor/1.0'})
        with urllib.request.urlopen(req, timeout=1.2) as resp:
            slots = json.loads(resp.read().decode())
        
        model_name = 'llama.cpp Server'
        try:
            req_p = urllib.request.Request(f'{url}/props', headers={'User-Agent': 'llama-monitor/1.0'})
            with urllib.request.urlopen(req_p, timeout=0.8) as resp_p:
                props = json.loads(resp_p.read().decode())
                model_name = props.get('model', {}).get('name', 'llama.cpp Server')
        except Exception:
            pass

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

def main(stdscr):
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
    slots_cache = []

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
        elif key in (curses.KEY_UP, ord('k'), ord('K')):
            selected_slot = max(0, selected_slot - 1)
        elif key in (curses.KEY_DOWN, ord('j'), ord('J')):
            if slots_cache:
                selected_slot = min(len(slots_cache) - 1, selected_slot + 1)
        elif key in (curses.KEY_LEFT, ord('h'), ord('H')):
            if slots_cache:
                selected_slot = (selected_slot - 1) % len(slots_cache)
        elif key in (curses.KEY_RIGHT, ord('l'), ord('L'), ord('	')):
            if slots_cache:
                selected_slot = (selected_slot + 1) % len(slots_cache)
        elif key in (10, 13, ord('d'), ord('D'), curses.KEY_ENTER): # Enter or d
            view_mode = 'drill'
        elif ord('0') <= key <= ord('7'):
            selected_slot = key - ord('0')
            view_mode = 'drill'

        now = time.time()
        dt = now - prev_time

        if not paused or not slots_cache:
            data, model_info = fetch_data(SERVER_URL)
            if data is not None:
                slots_cache = data
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

        # ======================================================================
        # VIEW 1: OVERVIEW TABLE MODE
        # ======================================================================
        if view_mode == 'overview':
            title = '  llama.cpp Server Tasks & Slots Overview (ROCm / HIP)  '
            safe_addstr(stdscr, 0, max(0, (max_x - len(title)) // 2), title, curses.color_pair(1) | curses.A_BOLD)

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

            footer = '  [↑/↓] Select Slot  |  [Enter/0-7] Drill-Down (Inspect Task)  |  [Space] Pause  |  [Q] Quit  '
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
            title = f'  === SLOT [{sid}] DRILL-DOWN: TASK #{task_id} ===  '
            safe_addstr(stdscr, 0, max(0, (max_x - len(title)) // 2), title, curses.color_pair(6) | curses.A_BOLD)
            safe_addstr(stdscr, 1, 2, f'State: {state_badge}', b_color)

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

            safe_addstr(stdscr, 6, 2, '─' * (max_x - 4), curses.color_pair(1) | curses.A_DIM)

            # Live Text Generation & Prompt Inspection
            prompt_text = slot_obj.get('prompt', '')
            generated_text = slot_obj.get('generated', '')

            # Split remaining height between Generated Text and Prompt
            rem_h = max_y - 9
            gen_h = max(4, int(rem_h * 0.65))
            prompt_h = max(3, rem_h - gen_h)

            # Window 1: Live Generated Stream
            safe_addstr(stdscr, 7, 2, f'▼ LIVE GENERATED TOKENS STREAM ({n_decoded:,} tokens generated):', curses.color_pair(2) | curses.A_BOLD)
            if generated_text:
                lines = generated_text.splitlines()
                # Take the tail of the generated text to show live stream
                display_lines = lines[-(gen_h - 2):] if len(lines) > (gen_h - 2) else lines
                for i, l in enumerate(display_lines):
                    safe_addstr(stdscr, 8 + i, 4, l[:max_x - 6], curses.color_pair(4))
            else:
                if is_proc and has_next:
                    safe_addstr(stdscr, 8, 4, f'Generating tokens... ({n_decoded:,} tokens decoded so far)', curses.color_pair(2))
                elif is_proc:
                    safe_addstr(stdscr, 8, 4, 'Currently in prefill stage (ingesting prompt)...', curses.color_pair(3))
                else:
                    safe_addstr(stdscr, 8, 4, 'No active task on this slot.', curses.color_pair(4) | curses.A_DIM)

            # Window 2: Received Prompt
            p_start_y = 8 + gen_h
            safe_addstr(stdscr, p_start_y, 2, f'▼ RECEIVED PROMPT INPUT ({n_prompt:,} tokens):', curses.color_pair(3) | curses.A_BOLD)
            if prompt_text:
                plines = prompt_text.splitlines()
                p_display = plines[-(prompt_h - 2):] if len(plines) > (prompt_h - 2) else plines
                for j, pl in enumerate(p_display):
                    safe_addstr(stdscr, p_start_y + 1 + j, 4, pl[:max_x - 6], curses.color_pair(4))
            else:
                if is_proc:
                    safe_addstr(stdscr, p_start_y + 1, 4, f'Prompt size: {n_prompt:,} tokens ({n_cache:,} cached)', curses.color_pair(4))
                else:
                    safe_addstr(stdscr, p_start_y + 1, 4, 'No prompt received.', curses.color_pair(4) | curses.A_DIM)

            # Drill Footer
            footer = '  [Esc/Backspace/Q] Back to List  |  [←/→/Tab] Prev/Next Slot  |  [Space] Pause  '
            safe_addstr(stdscr, max_y - 1, 2, footer, curses.color_pair(6) | curses.A_DIM)

        stdscr.refresh()

if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
