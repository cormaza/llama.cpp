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
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            slots = json.loads(resp.read().decode())
        
        # Also try props
        model_name = 'Unknown'
        try:
            req_p = urllib.request.Request(f'{url}/props', headers={'User-Agent': 'llama-monitor/1.0'})
            with urllib.request.urlopen(req_p, timeout=1.0) as resp_p:
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

def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(1000)

    # Setup colors
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)     # Header/Titles
    curses.init_pair(2, curses.COLOR_GREEN, -1)    # Generating / Active
    curses.init_pair(3, curses.COLOR_YELLOW, -1)   # Prefill / Busy
    curses.init_pair(4, curses.COLOR_WHITE, -1)    # Normal text
    curses.init_pair(5, curses.COLOR_RED, -1)      # Warning/Error
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # Speculative / MTP

    prev_decoded = {}
    prev_time = time.time()
    slot_speeds = {}
    paused = False

    while True:
        key = stdscr.getch()
        if key in (ord('q'), ord('Q'), 27): # q or ESC
            break
        elif key == ord(' '):
            paused = not paused

        now = time.time()
        dt = now - prev_time

        if not paused:
            slots, model_info = fetch_data(SERVER_URL)
        else:
            model_info = '(PAUSED)'

        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()

        # Header
        title = '  llama.cpp Server Task & Slot Monitor (ROCm / HIP)  '
        stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
        stdscr.addstr(0, max(0, (max_x - len(title)) // 2), title)
        stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

        if not slots:
            stdscr.attron(curses.color_pair(5))
            stdscr.addstr(3, 2, f'Waiting for server at {SERVER_URL}...')
            stdscr.addstr(4, 2, f'Status: {model_info}')
            stdscr.attroff(curses.color_pair(5))
            stdscr.refresh()
            time.sleep(1)
            continue

        # Calculate speeds
        total_speed = 0.0
        active_slots = 0
        total_gen_tokens = 0
        total_ctx_used = 0
        total_ctx_capacity = 0

        for s in slots:
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

            # Speed calculation
            prev = prev_decoded.get(sid, n_decoded)
            if dt > 0.3:
                delta = max(0, n_decoded - prev)
                # If task changed, don't produce negative or spike
                if delta > 2000:
                    delta = 0
                speed = delta / dt
                slot_speeds[sid] = speed
                prev_decoded[sid] = n_decoded
            else:
                speed = slot_speeds.get(sid, 0.0)

            total_speed += speed

        if dt > 0.3:
            prev_time = now

        # Summary Bar
        summary_line = (
            f' Target: {SERVER_URL}  |  Active Slots: {active_slots}/{len(slots)}  |  '
            f'Total GPU Speed: {total_speed:5.1f} t/s  |  Total Tokens Gen: {total_gen_tokens:,}'
        )
        stdscr.addstr(2, 2, summary_line, curses.color_pair(1))
        
        ctx_bar = draw_bar(total_ctx_used, total_ctx_capacity, width=30)
        stdscr.addstr(3, 2, f' Global Context Pool: {ctx_bar} ({total_ctx_used:,} / {total_ctx_capacity:,} tokens)', curses.color_pair(4))

        # Table Header
        h_y = 5
        hdr = f'{"Slot":<6} {"Task ID":<9} {"State":<12} {"Tokens Gen":<12} {"Speed":<10} {"Prompt":<9} {"Cache Hit":<11} {"Context Usage (%)":<30}'
        stdscr.attron(curses.A_REVERSE | curses.color_pair(1))
        stdscr.addstr(h_y, 2, hdr.ljust(max_x - 4)[:max_x - 4])
        stdscr.attroff(curses.A_REVERSE | curses.color_pair(1))

        # Table Rows
        row_y = h_y + 1
        for s in slots:
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

            # Determine State & Color
            if not is_proc:
                state_str = 'IDLE'
                c_pair = curses.color_pair(4)
            elif has_next:
                state_str = 'DECODING'
                c_pair = curses.color_pair(2) | curses.A_BOLD
            else:
                state_str = 'PREFILL'
                c_pair = curses.color_pair(3) | curses.A_BOLD

            bar_str = draw_bar(cur_ctx, n_ctx, width=16) + f' {cur_ctx:,}/{n_ctx//1024}k'
            speed_str = f'{speed:5.1f} t/s' if is_proc and has_next else '-'

            row = f'{sid:<6} {task_id:<9} {state_str:<12} {n_decoded:<12,} {speed_str:<10} {n_prompt:<9,} {n_cache:<11,} {bar_str}'
            stdscr.addstr(row_y, 2, row[:max_x - 4], c_pair)
            row_y += 1

        # Footer
        footer = '  [Q] Quit  |  [Space] Pause/Resume  |  Refresh: 1.0s  '
        stdscr.addstr(max_y - 1, 2, footer, curses.color_pair(1) | curses.A_DIM)

        stdscr.refresh()

if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
