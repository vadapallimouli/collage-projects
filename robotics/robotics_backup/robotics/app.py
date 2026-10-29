"""Tkinter dashboard for configurable Panda motion planning.

Place in the robotics project root. Requires tests/test_final_end_to_end.py
with the dashboard-config patch supplied in this bundle.
"""
from __future__ import annotations
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

ENVIRONMENTS = {
    "Tabletop": ("tabletop", 171, "Previously tested baseline"),
    "Cubby": ("cubby", 0, "Validate this environment before relying on results"),
    "Merged Cubby": ("merged_cubby", 0, "Validate this environment before relying on results"),
    "Dresser": ("dresser", 0, "Validate this environment before relying on results"),
}
DEFAULT_OBSTACLES = [
    {"x": .4, "y": 0.0, "z": .25, "w": .16, "d": .16, "h": .5},
    {"x": .55, "y": .2, "z": .2, "w": .12, "d": .12, "h": .4},
    {"x": .3, "y": -.2, "z": .15, "w": .2, "d": .15, "h": .3},
    {"x": .4, "y": 0.0, "z": .2, "w": .15, "d": .15, "h": .3},
]
BG, PANEL, ALT = "#f3f5fb", "#ffffff", "#eef1f8"
TEXT, MUTED, ACCENT = "#18233f", "#687895", "#6547ff"
GREEN, AMBER, RED = "#12805c", "#b56a00", "#c43d55"

class MotionPlannerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Intelligent Robot Motion Planner")
        self.geometry("1420x900")
        self.minsize(1080, 720)
        self.configure(bg=BG)
        self.root = Path(__file__).resolve().parent
        self.backend = self.root / "tests" / "test_final_end_to_end.py"
        self.config_path = self.root / "dashboard_run_config.json"
        self.log_queue = queue.Queue()
        self.process = None
        self.started = None
        self.obstacle_vars = []
        self.environment = tk.StringVar(value="Tabletop")
        self.problem = tk.StringVar(value="171")
        self.candidates = tk.StringVar(value="10")
        self.start_xyz = [tk.StringVar(value=v) for v in ("0.35", "-0.2", "0.45")]
        self.goal_xyz = [tk.StringVar(value=v) for v in ("0.45", "0.2", "0.45")]
        self.shape = tk.StringVar(value="Cylinder")
        self.obstacle_count = tk.IntVar(value=4)
        self.status = tk.StringVar(value="Ready to configure a planning run")
        self.elapsed = tk.StringVar(value="Elapsed: 00:00")
        self.clock = tk.StringVar(value="")
        self.note = tk.StringVar(value=ENVIRONMENTS["Tabletop"][2])
        self._styles(); self._build(); self._clock(); self.after(100, self._poll)

    def _styles(self):
        s = ttk.Style(self)
        try: s.theme_use("clam")
        except tk.TclError: pass
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        s.configure("TButton", padding=(9, 7), font=("Segoe UI", 10))
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", padding=(18, 10), font=("Segoe UI", 10, "bold"))
        s.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", ACCENT)])
        s.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=TEXT)
        s.configure("TSpinbox", fieldbackground=PANEL, background=PANEL, foreground=TEXT)
        s.configure("Run.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 11))

    def _build(self):
        header = tk.Frame(self, bg=BG, padx=24, pady=16); header.pack(fill="x")
        tk.Label(header, text="INTELLIGENT MOTION PLANNER", bg=BG, fg=TEXT,
                 font=("Segoe UI", 22, "bold")).pack(side="left")
        tk.Label(header, textvariable=self.clock, bg=PANEL, fg=MUTED,
                 font=("Consolas", 10), padx=12, pady=9).pack(side="right")
        strip = tk.Frame(self, bg=ALT, padx=16, pady=10); strip.pack(fill="x", padx=24, pady=(0, 12))
        tk.Label(strip, text="●", bg=ALT, fg=GREEN).pack(side="left")
        tk.Label(strip, textvariable=self.status, bg=ALT, fg=TEXT).pack(side="left", padx=8)
        tk.Label(strip, textvariable=self.elapsed, bg=ALT, fg=MUTED, font=("Consolas", 10)).pack(side="right")

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=24, pady=(0, 12))
        self.setup_tab = tk.Frame(self.tabs, bg=BG)
        self.results_tab = tk.Frame(self.tabs, bg=BG)
        self.tabs.add(self.setup_tab, text="  1  ·  PLAN & CONFIGURE  ")
        self.tabs.add(self.results_tab, text="  2  ·  RESULTS & EXPLANATION  ")
        self._build_setup_tab()
        self._build_results_tab()
        tk.Label(self, text="ML estimates risk. PyBullet collision checks remain the physical safety gate.",
                 bg=BG, fg=MUTED).pack(anchor="w", padx=24, pady=(0, 8))

    def _build_setup_tab(self):
        body = tk.Frame(self.setup_tab, bg=BG); body.pack(fill="both", expand=True, padx=4, pady=4)
        left = tk.Frame(body, bg=PANEL, width=510); left.pack(side="left", fill="y", padx=(0, 12)); left.pack_propagate(False)
        canvas = tk.Canvas(left, bg=PANEL, highlightthickness=0, width=490)
        sb = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set); sb.pack(side="right", fill="y"); canvas.pack(side="left", fill="both", expand=True)
        form = tk.Frame(canvas, bg=PANEL, padx=18, pady=14); win = canvas.create_window((0, 0), window=form, anchor="nw")
        form.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        self._section(form, "PLANNING CONFIGURATION")
        self._label(form, "Dataset environment").pack(anchor="w", pady=(0, 4))
        combo = ttk.Combobox(form, textvariable=self.environment, values=list(ENVIRONMENTS), state="readonly")
        combo.pack(fill="x", pady=(0, 5)); combo.bind("<<ComboboxSelected>>", self._env_changed)
        tk.Label(form, textvariable=self.note, bg=PANEL, fg=AMBER, wraplength=430, justify="left").pack(anchor="w", pady=(0, 8))
        row = tk.Frame(form, bg=PANEL); row.pack(fill="x")
        self._label(row, "Problem index").pack(side="left")
        ttk.Entry(row, textvariable=self.problem, width=8).pack(side="left", padx=(8, 22))
        self._label(row, "Candidate paths").pack(side="left")
        ttk.Spinbox(row, from_=1, to=30, textvariable=self.candidates, width=6).pack(side="left", padx=8)
        self._section(form, "END-EFFECTOR POSITIONS (METRES)")
        self._xyz_row(form, "Start XYZ", self.start_xyz); self._xyz_row(form, "Goal XYZ", self.goal_xyz)
        tk.Label(form, text="XYZ targets are converted to Panda joint configurations using inverse kinematics.",
                 bg=PANEL, fg=MUTED, wraplength=430, justify="left").pack(anchor="w", pady=(2, 10))
        self._section(form, "OBSTACLE CONFIGURATION")
        sh = tk.Frame(form, bg=PANEL); sh.pack(fill="x", pady=(0, 8))
        self._label(sh, "Number of obstacles").pack(side="left")
        ttk.Spinbox(sh, from_=0, to=20, textvariable=self.obstacle_count, width=5, command=self._render_obstacles).pack(side="left", padx=8)
        ttk.Button(sh, text="Apply count", command=self._render_obstacles).pack(side="left", padx=(4, 0))
        tk.Label(form, text="Set each obstacle's shape and dimensions independently. Cylinder uses radius + height; cuboid uses width + depth + height. Position is the centre.", bg=PANEL, fg=MUTED, wraplength=430, justify="left").pack(anchor="w", pady=(0, 8))
        self.obstacle_frame = tk.Frame(form, bg=PANEL); self.obstacle_frame.pack(fill="x")
        self._render_obstacles()
        actions = tk.Frame(form, bg=PANEL); actions.pack(fill="x", pady=(14, 6))
        self.run_btn = ttk.Button(actions, text="▶  START PLANNING", style="Run.TButton", command=self.start); self.run_btn.pack(fill="x", pady=(0, 7))
        self.stop_btn = ttk.Button(actions, text="■  STOP RUN", command=self.stop, state="disabled"); self.stop_btn.pack(fill="x", pady=(0, 7))
        ttk.Button(actions, text="Clear run results", command=self.clear).pack(fill="x")

        # Backend output is deliberately hidden by default, but remains available for debugging.
        right = tk.Frame(body, bg=PANEL, padx=16, pady=14); right.pack(side="left", fill="both", expand=True)
        tk.Label(right, text="RUN STATUS", bg=PANEL, fg=TEXT, font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(right, text="Configure the scene on the left, then start planning. Detailed terminal output is hidden unless you need it.", bg=PANEL, fg=MUTED, wraplength=620, justify="left").pack(anchor="w", pady=(5, 12))
        self.status_card = tk.Frame(right, bg=ALT, padx=16, pady=16); self.status_card.pack(fill="x", pady=(0, 12))
        tk.Label(self.status_card, text="CURRENT RUN", bg=ALT, fg=ACCENT, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.run_summary = tk.StringVar(value="No run yet. Select an environment and configure your start, goal and obstacles.")
        tk.Label(self.status_card, textvariable=self.run_summary, bg=ALT, fg=TEXT, wraplength=620, justify="left", font=("Segoe UI", 11)).pack(anchor="w", pady=(7, 0))
        self.log_toggle = ttk.Button(right, text="Show technical log", command=self._toggle_log); self.log_toggle.pack(anchor="w", pady=(2, 8))
        self.log_frame = tk.Frame(right, bg=PANEL)
        lf = tk.Frame(self.log_frame, bg="#f7f8fc"); lf.pack(fill="both", expand=True)
        self.log = tk.Text(lf, bg="#f7f8fc", fg="#27334d", insertbackground=TEXT, wrap="word", relief="flat", padx=10, pady=10, font=("Consolas", 9), state="disabled", height=18)
        scroll = ttk.Scrollbar(lf, orient="vertical", command=self.log.yview); self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True); scroll.pack(side="right", fill="y")
        for tag, color in (("pass", GREEN), ("error", RED), ("warn", AMBER), ("system", ACCENT)): self.log.tag_configure(tag, foreground=color)

    def _build_results_tab(self):
        outer = tk.Frame(self.results_tab, bg=BG); outer.pack(fill="both", expand=True, padx=8, pady=8)
        tk.Label(outer, text="PATH SELECTION & RECOVERABILITY", bg=BG, fg=TEXT, font=("Segoe UI", 17, "bold")).pack(anchor="w")
        tk.Label(outer, text="The explanation below is populated from the backend's actual run output; it does not invent reasons when the model does not provide them.", bg=BG, fg=MUTED, wraplength=1100, justify="left").pack(anchor="w", pady=(4, 14))
        cards = tk.Frame(outer, bg=BG); cards.pack(fill="x", pady=(0, 12))
        self.metric_vars = {}
        for key, title in (("candidates", "Candidates"), ("feasible", "Physically feasible"), ("selected", "Selected path"), ("safe", "Final verification")):
            card = tk.Frame(cards, bg=PANEL, padx=16, pady=14, highlightthickness=1, highlightbackground="#dfe4f0")
            card.pack(side="left", fill="both", expand=True, padx=(0, 10))
            tk.Label(card, text=title.upper(), bg=PANEL, fg=MUTED, font=("Segoe UI", 9, "bold")).pack(anchor="w")
            var = tk.StringVar(value="—"); self.metric_vars[key] = var
            tk.Label(card, textvariable=var, bg=PANEL, fg=ACCENT, font=("Segoe UI", 20, "bold")).pack(anchor="w", pady=(6, 0))
        sections = tk.Frame(outer, bg=BG); sections.pack(fill="both", expand=True)
        left = tk.Frame(sections, bg=PANEL, padx=16, pady=14, highlightthickness=1, highlightbackground="#dfe4f0"); left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right = tk.Frame(sections, bg=PANEL, padx=16, pady=14, highlightthickness=1, highlightbackground="#dfe4f0"); right.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="WHY THIS PATH WAS CHOSEN", bg=PANEL, fg=TEXT, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.explanation = tk.Text(left, height=18, wrap="word", bg=PANEL, fg=TEXT, relief="flat", font=("Segoe UI", 10), padx=2, pady=8, state="disabled")
        self.explanation.pack(fill="both", expand=True)
        tk.Label(right, text="RECOVERABILITY ANALYSIS", bg=PANEL, fg=TEXT, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.recovery = tk.Text(right, height=18, wrap="word", bg=PANEL, fg=TEXT, relief="flat", font=("Segoe UI", 10), padx=2, pady=8, state="disabled")
        self.recovery.pack(fill="both", expand=True)
        self._set_text(self.explanation, "Run the planner to see the selected candidate, its ML success/failure estimates, available feature contributions, and physical validation result.")
        self._set_text(self.recovery, "Recoverability details will appear here after the backend reports them. A fallback is only meaningful if it also passes physical validation.")

    @staticmethod
    def _set_text(widget, text):
        widget.configure(state="normal"); widget.delete("1.0", "end"); widget.insert("end", text); widget.configure(state="disabled")

    def _toggle_log(self):
        if self.log_frame.winfo_manager():
            self.log_frame.pack_forget(); self.log_toggle.configure(text="Show technical log")
        else:
            self.log_frame.pack(fill="both", expand=True); self.log_toggle.configure(text="Hide technical log")

    @staticmethod
    def _section(parent,text):
        tk.Label(parent,text=text,bg=PANEL,fg=ACCENT,font=("Segoe UI",10,"bold")).pack(anchor="w",pady=(10,8))
    @staticmethod
    def _label(parent, text):
        """Create and return a label; caller decides where to pack it."""
        label = tk.Label(
            parent,
            text=text,
            bg=PANEL,
            fg="#b8c9e6",
            font=("Segoe UI", 9, "bold"),
        )
        return label

    def _xyz_row(self, parent, label, vars_):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=(0, 8))
        tk.Label(row, text=label, bg=PANEL, fg=TEXT, width=11, anchor="w").pack(side="left")
        for axis, var in zip("XYZ", vars_):
            tk.Label(row, text=axis, bg=PANEL, fg=MUTED).pack(side="left", padx=(4, 2))
            ttk.Entry(row, textvariable=var, width=8).pack(side="left", padx=(0, 5))

    def _render_obstacles(self):
        """Render each obstacle with its own independent shape selector."""
        for widget in self.obstacle_frame.winfo_children():
            widget.destroy()
        self.obstacle_vars = []
        try:
            count = max(0, min(20, int(self.obstacle_count.get())))
        except (ValueError, tk.TclError):
            count = 4
            self.obstacle_count.set(count)

        for i in range(count):
            defaults = DEFAULT_OBSTACLES[i] if i < len(DEFAULT_OBSTACLES) else {
                "x": .4, "y": 0.0, "z": .2, "w": .15, "d": .15, "h": .3
            }
            card = tk.Frame(self.obstacle_frame, bg=ALT, padx=8, pady=7)
            card.pack(fill="x", pady=4)
            header = tk.Frame(card, bg=ALT)
            header.grid(row=0, column=0, columnspan=6, sticky="ew", pady=(0, 5))
            tk.Label(header, text=f"OBSTACLE {i+1}", bg=ALT, fg=TEXT,
                     font=("Segoe UI", 9, "bold")).pack(side="left")
            shape_var = tk.StringVar(value="Cylinder" if i == 0 else "Cuboid")
            shape_combo = ttk.Combobox(header, textvariable=shape_var,
                                       values=("Cylinder", "Cuboid"), state="readonly", width=11)
            shape_combo.pack(side="right")
            fields_frame = tk.Frame(card, bg=ALT)
            fields_frame.grid(row=1, column=0, columnspan=6, sticky="ew")
            rowvars = {"type": shape_var, "x": tk.StringVar(value=f"{defaults['x']:g}"),
                       "y": tk.StringVar(value=f"{defaults['y']:g}"),
                       "z": tk.StringVar(value=f"{defaults['z']:g}")}
            rowvars["width"] = tk.StringVar(value=f"{defaults['w']:g}")
            rowvars["depth"] = tk.StringVar(value=f"{defaults['d']:g}")
            rowvars["radius"] = tk.StringVar(value=f"{defaults['w']/2:g}")
            rowvars["height"] = tk.StringVar(value=f"{defaults['h']:g}")

            def draw_fields(frame=fields_frame, vars_for_row=rowvars):
                for child in frame.winfo_children():
                    child.destroy()
                keys = ["x", "y", "z", "radius", "height"] if vars_for_row["type"].get() == "Cylinder" else ["x", "y", "z", "width", "depth", "height"]
                for col, key in enumerate(keys):
                    tk.Label(frame, text=key, bg=ALT, fg=MUTED,
                             font=("Segoe UI", 8)).grid(row=0, column=col, sticky="w", padx=2)
                    ttk.Entry(frame, textvariable=vars_for_row[key], width=8).grid(row=1, column=col, sticky="ew", padx=2, pady=(0, 3))
            shape_combo.bind("<<ComboboxSelected>>", lambda _event, draw=draw_fields: draw())
            draw_fields()
            self.obstacle_vars.append(rowvars)

    def _env_changed(self,_=None):
        _key,idx,note=ENVIRONMENTS[self.environment.get()]; self.problem.set(str(idx)); self.note.set(note)

    def _config(self):
        def xyz(vs,label):
            try: vals=[float(v.get().strip()) for v in vs]
            except ValueError: raise ValueError(f"{label} XYZ must contain valid numbers.")
            if any(not (-2 <= x <= 2) for x in vals): raise ValueError(f"{label} coordinates must be between -2 m and 2 m.")
            return vals
        try:
            problem=int(self.problem.get()); candidates=int(self.candidates.get()); count=int(self.obstacle_count.get())
        except ValueError: raise ValueError("Problem index, candidate count, and obstacle count must be integers.")
        if problem<0 or not 1<=candidates<=30 or not 0<=count<=20: raise ValueError("Use problem index ≥ 0, candidates 1–30, and obstacles 0–20.")
        obs=[]
        for i, values in enumerate(self.obstacle_vars, 1):
            # The shape selector contains text (Cylinder/Cuboid), so only
            # convert the numeric entry fields here.
            shape = values["type"].get().strip()
            numeric_keys = ["x", "y", "z", "radius", "height"] if shape == "Cylinder" else ["x", "y", "z", "width", "depth", "height"]
            try:
                d = {key: float(values[key].get().strip()) for key in numeric_keys}
            except (ValueError, KeyError, tk.TclError) as exc:
                raise ValueError(
                    f"Obstacle {i} has an invalid or empty numeric field. "
                    "Enter numbers in the XYZ and dimension boxes."
                ) from exc
            if shape not in ("Cylinder", "Cuboid"):
                raise ValueError(f"Obstacle {i}: choose Cylinder or Cuboid.")
            if any(not (-2 <= d[k] <= 2) for k in ("x", "y", "z")):
                raise ValueError(f"Obstacle {i} position must be between -2 m and 2 m.")
            dimkeys = ("radius", "height") if shape == "Cylinder" else ("width", "depth", "height")
            if any(d[k] <= 0 or d[k] > 2 for k in dimkeys):
                raise ValueError(f"Obstacle {i} dimensions must be greater than 0 and no more than 2 m.")
            if shape == "Cylinder":
                obs.append({"type": "Cylinder", "x": d["x"], "y": d["y"], "z": d["z"], "radius": d["radius"], "h": d["height"]})
            else:
                obs.append({"type": "Cuboid", "x": d["x"], "y": d["y"], "z": d["z"], "w": d["width"], "d": d["depth"], "h": d["height"]})
        return {"start_xyz": xyz(self.start_xyz, "Start"), "goal_xyz": xyz(self.goal_xyz, "Goal"), "obstacles": obs}

    def _append(self, line, tag=None):
        line = line.rstrip()
        self.log.configure(state="normal"); self.log.insert("end", line + "\n", tag or ""); self.log.see("end"); self.log.configure(state="disabled")
        self._consume_result_line(line)

    def _consume_result_line(self, line):
        import re
        patterns = {
            "candidates": r"Candidates generated\s*:\s*(\d+)",
            "feasible": r"Physically feasible\s*:\s*(\d+)",
            "selected": r"Selected candidate\s*:\s*(\d+)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, line, re.I)
            if match: self.metric_vars[key].set(match.group(1))
        if "Final trajectory safe:" in line:
            self.metric_vars["safe"].set(line.split(":", 1)[-1].strip())
        if "Collision-free:" in line:
            self.metric_vars["safe"].set("YES" if line.split(":", 1)[-1].strip().lower() == "true" else "NO")
        if "WHY THIS TRAJECTORY WAS CHOSEN" in line:
            self._append_explanation = True
        if "[13] Analyzing recoverability" in line:
            self._append_recovery = True
        if not hasattr(self, "_append_explanation"): self._append_explanation = False
        if not hasattr(self, "_append_recovery"): self._append_recovery = False
        if self._append_explanation and (line.startswith("      ") or "WHY THIS TRAJECTORY" in line):
            existing = self.explanation.get("1.0", "end-1c")
            if existing.startswith("Run the planner") or existing.startswith("Selected path"):
                existing = ""
            self._set_text(self.explanation, (existing + "\n" + line.strip()).strip())
        if self._append_recovery and ("recover" in line.lower() or "candidate" in line.lower() or "feasible" in line.lower() or "[PASS]" in line):
            existing = self.recovery.get("1.0", "end-1c")
            if existing.startswith("Recoverability details"):
                existing = ""
            self._set_text(self.recovery, (existing + "\n" + line.strip()).strip())
    def start(self):
        if self.process is not None and self.process.poll() is None: return
        if not self.backend.exists():
            messagebox.showerror("Backend not found",f"Expected updated backend at:\n{self.backend}\n\nCopy the bundled test_final_end_to_end.py into tests/."); return
        try:
            config=self._config()
            config["obstacle_count"]=len(config["obstacles"])
            self.config_path.write_text(json.dumps(config,indent=2),encoding="utf-8")
            candidates=int(self.candidates.get()); problem=int(self.problem.get())
        except Exception as exc: messagebox.showerror("Invalid configuration",str(exc)); return
        envkey=ENVIRONMENTS[self.environment.get()][0]
        command=[sys.executable,"-u",str(self.backend),"--environment",envkey,"--problem-index",str(problem),"--num-candidates",str(candidates),"--config-json",str(self.config_path)]
        for key in self.metric_vars: self.metric_vars[key].set("—")
        self._append_explanation = False; self._append_recovery = False
        self._set_text(self.explanation, "Waiting for ML predictions and final selection details…")
        self._set_text(self.recovery, "Waiting for recoverability analysis…")
        self.run_summary.set(f"Environment: {self.environment.get()} | Problem: {problem} | Candidates requested: {candidates} | Obstacles: {len(config['obstacles'])}. Running full pipeline…")
        self.tabs.select(self.setup_tab)
        self._append("="*70,"system"); self._append(f"Launch: {datetime.now().astimezone().isoformat(timespec='seconds')}","system"); self._append("Start XYZ: "+str(config["start_xyz"]),"system"); self._append("Goal XYZ: "+str(config["goal_xyz"]),"system"); self._append("Obstacles: " + ", ".join(f"{n} {shape}(s)" for shape in ("Cylinder", "Cuboid") if (n := sum(1 for item in config["obstacles"] if item["type"] == shape))) if config["obstacles"] else "Obstacles: 0", "system"); self._append("Command: "+subprocess.list2cmdline(command),"system")
        self.status.set("Starting pipeline…"); self.run_btn.configure(state="disabled"); self.stop_btn.configure(state="normal"); self.started=time.monotonic()
        try:
            self.process=subprocess.Popen(command,cwd=str(self.root),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1)
        except Exception as exc:
            self.process=None; self.run_btn.configure(state="normal"); self.stop_btn.configure(state="disabled"); self.status.set("Could not start backend"); messagebox.showerror("Launch failed",str(exc)); return
        threading.Thread(target=self._reader,args=(self.process,),daemon=True).start(); self.status.set("Pipeline running")
    def _reader(self,proc):
        if proc.stdout:
            for line in proc.stdout: self.log_queue.put(("line",line.rstrip("\r\n")))
        self.log_queue.put(("exit",proc.wait()))
    def _poll(self):
        try:
            while True:
                kind,value=self.log_queue.get_nowait()
                if kind=="line":
                    lo=value.lower(); tag="pass" if "[pass]" in lo or "completed" in lo else "error" if any(x in lo for x in ("traceback","exception","error","failed")) else "warn" if "warning" in lo or "collision:" in lo else None
                    self._append(value,tag)
                else:
                    self.process=None; self.started=None; self.run_btn.configure(state="normal"); self.stop_btn.configure(state="disabled")
                    self.status.set("Pipeline completed" if value==0 else f"Pipeline failed (exit {value})")
                    self.run_summary.set((self.run_summary.get().split(" | Running full pipeline")[0] + (" | Completed — review Results & Explanation." if value == 0 else f" | Failed (exit {value}); show technical log for details.")))
                    self._append(f"Backend process exited with code {value}.","pass" if value==0 else "error")
                    if value == 0: self.tabs.select(self.results_tab)
        except queue.Empty: pass
        self.after(100,self._poll)
    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate(); self.status.set("Stop requested"); self._append("Stop requested from dashboard.","warn")
    def clear(self):
        if self.process is not None and self.process.poll() is None: return
        self.log.configure(state="normal"); self.log.delete("1.0","end"); self.log.configure(state="disabled")
        for key in self.metric_vars: self.metric_vars[key].set("—")
        self._set_text(self.explanation, "Run the planner to see the selected candidate and its ML explanation.")
        self._set_text(self.recovery, "Recoverability details will appear here after a run.")
        self.run_summary.set("No run yet. Select an environment and configure your start, goal and obstacles.")
    def _clock(self):
        self.clock.set(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"))
        if self.started is not None: s=max(0,int(time.monotonic()-self.started)); self.elapsed.set(f"Elapsed: {s//60:02d}:{s%60:02d}")
        self.after(1000,self._clock)

if __name__=="__main__": MotionPlannerApp().mainloop()
