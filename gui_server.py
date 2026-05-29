import http.server
import socketserver
import json
import os
import signal
import subprocess
import threading
import sys
import urllib.parse

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

optimizer_process = None
process_lock = threading.Lock()
log_buffer = []
log_lock = threading.Lock()

def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_text(path):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def save_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def run_optimizer():
    global optimizer_process
    with process_lock:
        if optimizer_process is not None and optimizer_process.poll() is None:
            return False
        with log_lock:
            log_buffer.clear()
            log_buffer.append("[system] starting layout optimizer...\n")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        optimizer_process = subprocess.Popen(
            [sys.executable, "PT.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            cwd=DIRECTORY,
            env=env,
            start_new_session=True  # own process group so killpg reaches all workers
        )

    def reader_thread():
        global optimizer_process
        for line in iter(optimizer_process.stdout.readline, ''):
            with log_lock:
                log_buffer.append(line)
        optimizer_process.stdout.close()
        optimizer_process.wait()
        with log_lock:
            log_buffer.append(f"\n[system] optimizer finished with exit code {optimizer_process.returncode}.\n")

    threading.Thread(target=reader_thread, daemon=True).start()
    return True

def stop_optimizer():
    global optimizer_process
    with process_lock:
        if optimizer_process is not None and optimizer_process.poll() is None:
            try:
                # Kill the entire process group (PT.py + all pool worker processes)
                # so the inherited stdout pipe is closed and reader_thread unblocks.
                pgid = os.getpgid(optimizer_process.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                optimizer_process.terminate()  # fallback
            with log_lock:
                log_buffer.append("\n[system] optimizer terminated by user.\n")
            return True
        return False

class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

class GUIHandler(http.server.BaseHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS for local testing ease
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/config":
            config_data = {
                "fixed_keys": load_json("config/fixed_keys.json"),
                "fixed_shift_keys": load_json("config/fixed_shift_keys.json"),
                "fixed_alt_keys": load_json("config/fixed_alt_keys.json"),
                "fixed_shift_alt_keys": load_json("config/fixed_shift_alt_keys.json"),
                "available_keys": load_text("config/available_keys.txt"),
                "available_shift_keys": load_text("config/available_shift_keys.txt"),
                "available_alt_keys": load_text("config/available_alt_keys.txt"),
                "available_shift_alt_keys": load_text("config/available_shift_alt_keys.txt"),
                "keystrokes": load_json("config/keystrokes.json"),
                "parameters": load_json("config/parameters.json"),
                "home_keys": load_json("config/home_keys.json"),
                "layout": load_text("config/layout.txt"),
                "target_metrics": load_json("config/target_metrics.json"),
                "finger_natural_positions": load_json("config/finger_natural_positions.json"),
                "max_finger_distances": load_json("config/max_finger_distances.json"),
                "assigned_fingers": load_json("config/assigned_fingers.json")
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(config_data, ensure_ascii=False).encode("utf-8"))
            return

        elif path == "/api/status":
            running = False
            with process_lock:
                running = (optimizer_process is not None and optimizer_process.poll() is None)
            
            # Check if output/best_layouts.json exists
            elites = load_json("output/best_layouts.json")
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"running": running, "elites": elites}).encode("utf-8"))
            return

        elif path == "/api/logs":
            # Simple polling of logs with a start line offset query param
            query = urllib.parse.parse_qs(parsed_url.query)
            offset = int(query.get("offset", [0])[0])
            
            with log_lock:
                current_logs = log_buffer[offset:]
                next_offset = len(log_buffer)
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({
                "logs": "".join(current_logs),
                "next_offset": next_offset
            }, ensure_ascii=False).encode("utf-8"))
            return

        # Serve static files from 'gui/' and 'output/'
        local_path = path.lstrip("/")
        if not local_path or local_path == "index.html":
            local_path = "gui/index.html"
        elif not local_path.startswith("gui/") and not local_path.startswith("output/"):
            # Check if it exists in gui folder
            if os.path.exists(os.path.join(DIRECTORY, "gui", local_path)):
                local_path = os.path.join("gui", local_path)

        full_path = os.path.join(DIRECTORY, local_path)
        if os.path.exists(full_path) and not os.path.isdir(full_path):
            self.send_response(200)
            if local_path.endswith(".html"):
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif local_path.endswith(".css"):
                self.send_header("Content-Type", "text/css; charset=utf-8")
            elif local_path.endswith(".js"):
                self.send_header("Content-Type", "application/javascript; charset=utf-8")
            elif local_path.endswith(".svg"):
                self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
            elif local_path.endswith(".json"):
                self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            with open(full_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"File not found")

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')

        if path == "/api/config":
            try:
                data = json.loads(post_data)
                
                # Save each file if provided
                if "fixed_keys" in data: save_json("config/fixed_keys.json", data["fixed_keys"])
                if "fixed_shift_keys" in data: save_json("config/fixed_shift_keys.json", data["fixed_shift_keys"])
                if "fixed_alt_keys" in data: save_json("config/fixed_alt_keys.json", data["fixed_alt_keys"])
                if "fixed_shift_alt_keys" in data: save_json("config/fixed_shift_alt_keys.json", data["fixed_shift_alt_keys"])
                if "available_keys" in data: save_text("config/available_keys.txt", data["available_keys"])
                if "available_shift_keys" in data: save_text("config/available_shift_keys.txt", data["available_shift_keys"])
                if "available_alt_keys" in data: save_text("config/available_alt_keys.txt", data["available_alt_keys"])
                if "available_shift_alt_keys" in data: save_text("config/available_shift_alt_keys.txt", data["available_shift_alt_keys"])
                if "keystrokes" in data: save_json("config/keystrokes.json", data["keystrokes"])
                if "parameters" in data: save_json("config/parameters.json", data["parameters"])
                if "home_keys" in data: save_json("config/home_keys.json", data["home_keys"])
                if "layout" in data: save_text("config/layout.txt", data["layout"])
                if "target_metrics" in data: save_json("config/target_metrics.json", data["target_metrics"])
                if "finger_natural_positions" in data: save_json("config/finger_natural_positions.json", data["finger_natural_positions"])
                if "max_finger_distances" in data: save_json("config/max_finger_distances.json", data["max_finger_distances"])
                if "assigned_fingers" in data: save_json("config/assigned_fingers.json", data["assigned_fingers"])

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
            return

        elif path == "/api/run":
            success = run_optimizer()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": success}).encode("utf-8"))
            return

        elif path == "/api/stop":
            success = stop_optimizer()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": success}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

def main():
    print(f"Starting server on http://localhost:{PORT}")
    server = ThreadingHTTPServer(("localhost", PORT), GUIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.shutdown()

if __name__ == "__main__":
    main()
