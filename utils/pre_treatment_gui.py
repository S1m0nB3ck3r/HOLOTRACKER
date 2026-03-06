# -*- coding: utf-8 -*-
"""
pre_treatment_gui.py
--------------------
Tkinter GUI for the pre_treatment.py module.
Allows the user to configure and run the image cleaning pipeline
(mean image computation + subtraction / division + normalization).
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Add the utils folder to the path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pre_treatment


# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------

TITLE        = "Pre-Treatment – Mean image cleaning"
PAD          = 10
ENTRY_WIDTH  = 55
LABEL_WIDTH  = 18

BG_MAIN      = "#1e1e2e"   # main background (dark)
BG_FRAME     = "#2a2a3e"   # frame background
BG_ENTRY     = "#12121e"   # text field background
FG_TEXT      = "#cdd6f4"   # main text
FG_LABEL     = "#89b4fa"   # accent labels
FG_SUCCESS   = "#a6e3a1"   # success green
FG_ERROR     = "#f38ba8"   # error red
FG_WARNING   = "#fab387"   # warning orange
BTN_RUN_BG   = "#89b4fa"
BTN_RUN_FG   = "#1e1e2e"
BTN_BROWSE_BG = "#585b70"
BTN_BROWSE_FG = "#cdd6f4"


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class PreTreatmentGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(TITLE)
        self.resizable(True, True)
        self.configure(bg=BG_MAIN)
        self.minsize(650, 520)

        self._build_ui()
        self._center_window()

    # ------------------------------------------------------------------ #
    #  UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        """Build all widgets."""
        # ---- Header ----
        header = tk.Label(
            self,
            text="🔬  Pre-Treatment",
            font=("Segoe UI", 16, "bold"),
            bg=BG_MAIN, fg=FG_LABEL,
        )
        header.pack(pady=(PAD * 2, PAD // 2))

        subtitle = tk.Label(
            self,
            text="Compute the mean image then clean each individual image.",
            font=("Segoe UI", 9),
            bg=BG_MAIN, fg=FG_TEXT,
        )
        subtitle.pack(pady=(0, PAD))

        # ---- Parameters frame ----
        params_frame = tk.LabelFrame(
            self,
            text="  Parameters  ",
            font=("Segoe UI", 10, "bold"),
            bg=BG_FRAME, fg=FG_LABEL,
            bd=2, relief="groove",
            padx=PAD, pady=PAD,
        )
        params_frame.pack(fill="x", padx=PAD * 2, pady=(0, PAD))

        # Source directory
        self.var_dir_in = tk.StringVar()
        self._add_directory_row(params_frame, 0, "Source directory:", self.var_dir_in, self._browse_dir_in)

        # Output directory
        self.var_dir_out = tk.StringVar()
        self._add_directory_row(params_frame, 1, "Output directory:", self.var_dir_out, self._browse_dir_out)

        # Image type
        self.var_image_type = tk.StringVar(value="tif")
        self._add_combobox_row(
            params_frame, 2,
            "Image type:", self.var_image_type,
            ["tif", "tiff", "png", "bmp", "jpg", "jpeg"],
        )

        # Mean type
        self.var_mean_type = tk.StringVar(value="arithmetic")
        self._add_radiobutton_row(
            params_frame, 3,
            "Mean type:",
            self.var_mean_type,
            [("Arithmetic", "arithmetic"), ("Log (geometric)", "log")],
        )

        # Cleaning type
        self.var_clean_type = tk.StringVar(value="soustr")
        self._add_radiobutton_row(
            params_frame, 4,
            "Cleaning:",
            self.var_clean_type,
            [("Subtraction  (I − mean)", "soustr"), ("Division  (I / mean)", "div")],
        )

        # ---- Progress bar ----
        prog_frame = tk.Frame(self, bg=BG_MAIN)
        prog_frame.pack(fill="x", padx=PAD * 2, pady=(0, PAD // 2))

        tk.Label(
            prog_frame, text="Progress:",
            font=("Segoe UI", 9, "bold"),
            bg=BG_MAIN, fg=FG_LABEL,
        ).pack(anchor="w")

        self.progress_var = tk.DoubleVar(value=0)
        self.progressbar = ttk.Progressbar(
            prog_frame,
            variable=self.progress_var,
            maximum=100,
            length=600,
            mode="determinate",
        )
        self.progressbar.pack(fill="x", pady=(2, 0))

        self.lbl_progress = tk.Label(
            prog_frame, text="Idle…",
            font=("Segoe UI", 9, "italic"),
            bg=BG_MAIN, fg=FG_TEXT, anchor="w",
        )
        self.lbl_progress.pack(fill="x")

        # ---- Log console ----
        log_frame = tk.LabelFrame(
            self,
            text="  Log  ",
            font=("Segoe UI", 10, "bold"),
            bg=BG_FRAME, fg=FG_LABEL,
            bd=2, relief="groove",
            padx=PAD, pady=4,
        )
        log_frame.pack(fill="both", expand=True, padx=PAD * 2, pady=(0, PAD))

        self.log_text = tk.Text(
            log_frame,
            height=8,
            bg=BG_ENTRY, fg=FG_TEXT,
            font=("Consolas", 9),
            relief="flat", bd=0,
            state="disabled",
            wrap="word",
        )
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview, bg=BG_FRAME)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

        # Color tags for the log
        self.log_text.tag_config("info",    foreground=FG_TEXT)
        self.log_text.tag_config("success", foreground=FG_SUCCESS)
        self.log_text.tag_config("error",   foreground=FG_ERROR)
        self.log_text.tag_config("warning", foreground=FG_WARNING)

        # ---- Bottom buttons ----
        btn_frame = tk.Frame(self, bg=BG_MAIN)
        btn_frame.pack(fill="x", padx=PAD * 2, pady=(0, PAD * 2))

        self.btn_run = tk.Button(
            btn_frame,
            text="▶  Run",
            font=("Segoe UI", 11, "bold"),
            bg=BTN_RUN_BG, fg=BTN_RUN_FG,
            activebackground="#74c7ec", activeforeground=BTN_RUN_FG,
            relief="flat", padx=16, pady=6,
            cursor="hand2",
            command=self._on_run,
        )
        self.btn_run.pack(side="left", padx=(0, PAD))

        self.btn_clear = tk.Button(
            btn_frame,
            text="🗑  Clear log",
            font=("Segoe UI", 10),
            bg=BTN_BROWSE_BG, fg=BTN_BROWSE_FG,
            activebackground="#7f849c", activeforeground=BTN_BROWSE_FG,
            relief="flat", padx=12, pady=6,
            cursor="hand2",
            command=self._clear_log,
        )
        self.btn_clear.pack(side="left")

        self.btn_quit = tk.Button(
            btn_frame,
            text="✕  Quit",
            font=("Segoe UI", 10),
            bg=BTN_BROWSE_BG, fg=BTN_BROWSE_FG,
            activebackground="#7f849c", activeforeground=BTN_BROWSE_FG,
            relief="flat", padx=12, pady=6,
            cursor="hand2",
            command=self.destroy,
        )
        self.btn_quit.pack(side="right")

    # ------------------------------------------------------------------ #
    #  Form row helpers
    # ------------------------------------------------------------------ #

    def _add_directory_row(self, parent, row, label_text, var, browse_cmd):
        """Row: [label] [text entry] [Browse button]"""
        tk.Label(
            parent, text=label_text,
            font=("Segoe UI", 9, "bold"),
            width=LABEL_WIDTH, anchor="w",
            bg=BG_FRAME, fg=FG_LABEL,
        ).grid(row=row, column=0, sticky="w", pady=4)

        entry = tk.Entry(
            parent, textvariable=var,
            width=ENTRY_WIDTH,
            bg=BG_ENTRY, fg=FG_TEXT,
            insertbackground=FG_TEXT,
            relief="flat", bd=4,
            font=("Consolas", 9),
        )
        entry.grid(row=row, column=1, sticky="ew", padx=(PAD // 2, PAD // 2), pady=4)

        tk.Button(
            parent,
            text="📁 Browse",
            font=("Segoe UI", 9),
            bg=BTN_BROWSE_BG, fg=BTN_BROWSE_FG,
            activebackground="#7f849c", activeforeground=BTN_BROWSE_FG,
            relief="flat", padx=6, pady=2,
            cursor="hand2",
            command=browse_cmd,
        ).grid(row=row, column=2, sticky="w", padx=(0, PAD // 2), pady=4)

        parent.columnconfigure(1, weight=1)

    def _add_combobox_row(self, parent, row, label_text, var, values):
        """Row: [label] [combobox]"""
        tk.Label(
            parent, text=label_text,
            font=("Segoe UI", 9, "bold"),
            width=LABEL_WIDTH, anchor="w",
            bg=BG_FRAME, fg=FG_LABEL,
        ).grid(row=row, column=0, sticky="w", pady=4)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TCombobox",
                        fieldbackground=BG_ENTRY,
                        background=BG_ENTRY,
                        foreground=FG_TEXT,
                        selectbackground=BG_ENTRY,
                        selectforeground=FG_TEXT,
                        arrowcolor=FG_TEXT)

        cb = ttk.Combobox(
            parent, textvariable=var,
            values=values, width=12,
            state="readonly",
            style="Dark.TCombobox",
            font=("Consolas", 9),
        )
        cb.grid(row=row, column=1, sticky="w", padx=(PAD // 2, 0), pady=4)

    def _add_radiobutton_row(self, parent, row, label_text, var, options):
        """Row: [label] [radio 1] [radio 2] …"""
        tk.Label(
            parent, text=label_text,
            font=("Segoe UI", 9, "bold"),
            width=LABEL_WIDTH, anchor="w",
            bg=BG_FRAME, fg=FG_LABEL,
        ).grid(row=row, column=0, sticky="w", pady=4)

        rb_frame = tk.Frame(parent, bg=BG_FRAME)
        rb_frame.grid(row=row, column=1, columnspan=2, sticky="w", padx=(PAD // 2, 0), pady=4)

        for text, value in options:
            tk.Radiobutton(
                rb_frame,
                text=text, variable=var, value=value,
                font=("Segoe UI", 9),
                bg=BG_FRAME, fg=FG_TEXT,
                activebackground=BG_FRAME, activeforeground=FG_LABEL,
                selectcolor=BG_ENTRY,
                relief="flat",
            ).pack(side="left", padx=(0, PAD * 2))

    # ------------------------------------------------------------------ #
    #  Button actions
    # ------------------------------------------------------------------ #

    def _browse_dir_in(self):
        path = filedialog.askdirectory(title="Choose source directory")
        if path:
            self.var_dir_in.set(path)
            # Automatically suggest an output directory
            if not self.var_dir_out.get():
                self.var_dir_out.set(os.path.join(path, "cleaned"))

    def _browse_dir_out(self):
        path = filedialog.askdirectory(title="Choose output directory")
        if path:
            self.var_dir_out.set(path)

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        self.progress_var.set(0)
        self.lbl_progress.config(text="Idle…", fg=FG_TEXT)

    def _on_run(self):
        """Validate parameters and start processing in a separate thread."""
        dir_in    = self.var_dir_in.get().strip()
        dir_out   = self.var_dir_out.get().strip()
        img_type  = self.var_image_type.get().strip()
        mean_type = self.var_mean_type.get()
        clean_type = self.var_clean_type.get()

        # --- Basic validation ---
        errors = []
        if not dir_in:
            errors.append("Source directory is empty.")
        elif not os.path.isdir(dir_in):
            errors.append(f"Source directory not found:\n  {dir_in}")
        if not dir_out:
            errors.append("Output directory is empty.")
        if not img_type:
            errors.append("Image type is empty.")

        if errors:
            messagebox.showerror("Invalid parameters", "\n\n".join(errors))
            return

        # --- Launch in a thread to keep the GUI responsive ---
        self.btn_run.config(state="disabled")
        self.progress_var.set(0)
        self._log(f"=== Processing started ===", "info")
        self._log(f"  Source     : {dir_in}", "info")
        self._log(f"  Output     : {dir_out}", "info")
        self._log(f"  Image type : {img_type}", "info")
        self._log(f"  Mean type  : {mean_type}", "info")
        self._log(f"  Cleaning   : {clean_type}", "info")

        thread = threading.Thread(
            target=self._run_worker,
            args=(dir_in, dir_out, img_type, mean_type, clean_type),
            daemon=True,
        )
        thread.start()

    # ------------------------------------------------------------------ #
    #  Worker (separate thread)
    # ------------------------------------------------------------------ #

    def _run_worker(self, dir_in, dir_out, img_type, mean_type, clean_type):
        """Run the pipeline and forward results to the GUI via self.after."""
        try:
            self._run_pipeline(dir_in, dir_out, img_type, mean_type, clean_type)
            self.after(0, self._on_success)
        except Exception as exc:
            self.after(0, self._on_error, str(exc))

    def _run_pipeline(self, dir_in, dir_out, img_type, mean_type, clean_type):
        """Execute the full pipeline with progress updates."""
        import numpy as np

        os.makedirs(dir_out, exist_ok=True)
        image_paths = pre_treatment.list_images(dir_in, img_type)
        n = len(image_paths)
        self.after(0, self._log, f"{n} images found.", "info")

        # ---- Mean image + per-pixel maps (single pass) ----
        self.after(0, self._update_progress, 0, f"Computing mean image ({mean_type})…")
        mean_image, min_map, max_map = self._compute_mean_progress(image_paths, mean_type, n)

        mean_out = os.path.join(dir_out, "mean_image.tif")
        pre_treatment.save_image_tiff(mean_image, mean_out)
        self.after(0, self._log, f"Mean image saved: {mean_out}", "success")

        # ---- Normalization bounds (analytical, no extra pass) ----
        global_min, global_max = pre_treatment.compute_normalization_bounds(
            min_map, max_map, mean_image, clean_type
        )
        self.after(0, self._log,
                   f"Normalization bounds: min={global_min:.6f}  max={global_max:.6f}",
                   "info")

        # ---- Cleaning + normalization [0, 1] ----
        self.after(0, self._log, f"Cleaning + normalizing images ({clean_type})…", "info")
        for i, path in enumerate(image_paths):
            img = pre_treatment.read_image_float32(path)
            result = pre_treatment.clean_and_normalize(
                img, mean_image, clean_type, global_min, global_max
            )

            basename = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(dir_out, f"{basename}_cleaned.tif")
            pre_treatment.save_image_tiff(result, out_path)

            pct = 50 + 50 * (i + 1) / n
            self.after(0, self._update_progress, pct,
                       f"Cleaning: {i + 1}/{n} ({pct:.0f} %)")

    def _compute_mean_progress(self, image_paths, mean_type, n):
        """
        Compute the mean image with progress bar updates (0 → 50 %).
        Simultaneously accumulates per-pixel min_map and max_map to allow
        the exact analytical computation of normalization bounds.
        Returns (mean, min_map, max_map).
        """
        import numpy as np

        mean_type = mean_type.lower()
        first = pre_treatment.read_image_float32(image_paths[0]).astype(np.float64)
        accumulator = np.zeros_like(first, dtype=np.float64)
        min_map = np.full_like(first, fill_value=np.inf,  dtype=np.float64)
        max_map = np.full_like(first, fill_value=-np.inf, dtype=np.float64)

        for i, path in enumerate(image_paths):
            img = pre_treatment.read_image_float32(path).astype(np.float64)

            # Per-pixel maps on raw images
            np.minimum(min_map, img, out=min_map)
            np.maximum(max_map, img, out=max_map)

            if mean_type == "log":
                eps = np.finfo(np.float32).tiny
                np.clip(img, eps, None, out=img)
                accumulator += np.log(img)
            else:
                accumulator += img

            pct = 50 * (i + 1) / n
            self.after(0, self._update_progress, pct,
                       f"Mean: {i + 1}/{n} ({pct:.0f} %)")

        mean = accumulator / n
        if mean_type == "log":
            mean = np.exp(mean)
        return (
            mean.astype(np.float32),
            min_map.astype(np.float32),
            max_map.astype(np.float32),
        )

    # ------------------------------------------------------------------ #
    #  GUI callbacks (called via self.after from the worker thread)
    # ------------------------------------------------------------------ #

    def _on_success(self):
        self._update_progress(100, "Processing complete ✓")
        self._log("=== Processing completed successfully ===", "success")
        self.lbl_progress.config(fg=FG_SUCCESS)
        self.btn_run.config(state="normal")
        messagebox.showinfo("Success", "Processing completed successfully.")

    def _on_error(self, message):
        self._update_progress(0, "Error!")
        self._log(f"ERROR: {message}", "error")
        self.lbl_progress.config(fg=FG_ERROR)
        self.btn_run.config(state="normal")
        messagebox.showerror("Error", message)

    def _log(self, message: str, tag: str = "info"):
        """Append a line to the log area."""
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n", tag)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _update_progress(self, value: float, text: str = ""):
        self.progress_var.set(value)
        if text:
            self.lbl_progress.config(text=text, fg=FG_TEXT)

    # ------------------------------------------------------------------ #
    #  Window centering
    # ------------------------------------------------------------------ #

    def _center_window(self):
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.geometry(f"+{x}+{y}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = PreTreatmentGUI()
    app.mainloop()
