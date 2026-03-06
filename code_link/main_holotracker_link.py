"""
Minimal GUI for "code_link" app (renamed).
Creates the left tabbed panel (PATH / ACTIONS) and a right-side 3D matplotlib axes
matching the layout in the provided screenshots. This file only provides the GUI
and stub callbacks. Run with: python main_holotracker_link.py
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import concurrent.futures
import queue
import pandas as pd
import math
import json
from datetime import datetime
from . import processor
try:
    import ttkbootstrap as ttkb
    from ttkbootstrap import Style
    # expose a module-level `ttk` alias so code that uses `ttk.Frame` works
    ttk = ttkb
    TTKBOOTSTRAP_AVAILABLE = True
except Exception:
    import tkinter.ttk as ttk
    Style = None
    TTKBOOTSTRAP_AVAILABLE = False

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from matplotlib import cm
from matplotlib import colors as mcolors
try:
    import trackpy as tp
    TRACKPY_AVAILABLE = True
except Exception:
    tp = None
    TRACKPY_AVAILABLE = False


class CodeLinkGUI:
    def __init__(self, master):
        self.master = master
        master.title("HOLOTRACKER_LINK")
        master.geometry("1200x800")

        # Use ttkbootstrap style if available
        if TTKBOOTSTRAP_AVAILABLE:
            self.style = Style(theme='superhero')
            self.frame_style = ttkb.Frame
            self.Button = ttkb.Button
            self.Label = ttkb.Label
            self.Entry = ttkb.Entry
            self.Spinbox = ttkb.Spinbox
            self.Notebook = ttkb.Notebook
        else:
            self.style = None
            self.frame_style = ttk.Frame
            self.Button = ttk.Button
            self.Label = ttk.Label
            self.Entry = ttk.Entry
            self.Spinbox = ttk.Spinbox
            self.Notebook = ttk.Notebook

        # Particle status text variable (initialize before building layout)
        self.particle_status_var = tk.StringVar(value='')
        # Slider display variables
        self.feature_index_var = tk.StringVar(value='')
        self.trajectory_index_var = tk.StringVar(value='')
        # Status text variable (moved from layout to be available earlier)
        self.status_var = tk.StringVar(value='')

        # Build UI layout
        self._create_layout()

        # Load persisted params (if any)
        self._params_path = os.path.join(os.path.dirname(__file__), 'last_params_link.json')
        try:
            self._load_params()
        except Exception:
            # ignore load errors and proceed with defaults
            pass

        # Thread pool for background tasks
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._futures = []
        # Data storage
        self.loaded_df = None
        self.trajectories_df = None

        # queue for thread-safe callbacks (not strictly necessary but useful)
        self._cb_queue = queue.Queue()
        # Periodically check callback queue
        self.master.after(100, self._process_cb_queue)

        # Ensure we save params on normal close
        try:
            self.master.protocol('WM_DELETE_WINDOW', self._exit)
        except Exception:
            pass

    def _create_layout(self):
        # Main panes: left controls, right plot
        self.left_frame = self.frame_style(self.master)
        # give the left panel a fixed-ish width to match proportions
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        self.right_frame = ttk.Frame(self.master)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Single stacked left panel: PATH controls followed by ACTIONS controls
        self.path_tab = ttk.Frame(self.left_frame)
        self.path_tab.pack(fill=tk.X)
        self._build_path_tab(self.path_tab)

        # Separator
        ttk.Separator(self.left_frame, orient='horizontal').pack(fill=tk.X, pady=(4,8))

        self.actions_tab = ttk.Frame(self.left_frame)
        self.actions_tab.pack(fill=tk.BOTH, expand=False)
        self._build_actions_tab(self.actions_tab)

        # status label will be created inside the actions tab (after bottom buttons)

        # Build right plot area (matplotlib 3D)
        self._build_plot_area(self.right_frame)

    def _build_path_tab(self, parent):
        # use slightly smaller vertical padding to save vertical space
        pad = {'padx': 6, 'pady': 4}

    # feature localisation file path
        ttk.Label(parent, text='feature localisation file path').pack(anchor='w', **pad)
        # single-line entry to match screenshot
        self.feature_path_text = self.Entry(parent, width=60)
        self.feature_path_text.pack(fill=tk.X, **pad)

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, **pad)
        # use styled button if available
        try:
            self.Button(btn_frame, text='Browse...', bootstyle='info', command=self._browse_feature_file).pack(side=tk.LEFT)
        except Exception:
            ttk.Button(btn_frame, text='Browse...', command=self._browse_feature_file).pack(side=tk.LEFT)

    def _build_actions_tab(self, parent):
        # use slightly smaller vertical padding to compact the controls
        pad = {'padx': 6, 'pady': 4}

        # Top large Open localisation File button
        self.open_loc_btn = self.Button(parent, text='Open localisation File', bootstyle='success', command=self._open_localisation_file)
        self.open_loc_btn.pack(fill=tk.X, **pad)

        # Counters area: keep a small left placeholder
        counters_frame = ttk.Frame(parent)
        counters_frame.pack(fill=tk.X, **pad)

    # small placeholder on the left; do not expand vertically so we don't create
    # a large empty area under the Open button
        left_c = ttk.Frame(counters_frame)
        left_c.pack(side=tk.LEFT)

        # Feature slider: place full-width under the Open button (so it matches trajectory slider)
        feature_frame = ttk.Frame(parent)
        feature_frame.pack(fill=tk.X, **pad)
        ttk.Label(feature_frame, text='Feature hologram').pack(anchor='w')
        self.feature_holo_scale = tk.Scale(feature_frame, from_=1, to=1, orient='horizontal', command=self._on_hologram_scale)
        self.feature_holo_scale.set(1)
        self.feature_holo_scale.pack(fill=tk.X, padx=6)
        # show current feature index under slider
        ttk.Label(feature_frame, textvariable=self.feature_index_var).pack(anchor='w', padx=6, pady=(2,0))
        ttk.Label(feature_frame, text='Total holograms').pack(anchor='w')
        self.total_holo_entry = ttk.Entry(feature_frame, width=8)
        self.total_holo_entry.insert(0, '0')
        # align the total counter under the slider (left-aligned like the label)
        self.total_holo_entry.pack(anchor='w', padx=6)

    # Trajectory controls will be placed below link parameters to match layout

        # LINK button (disabled)
        self.link_btn = self.Button(parent, text='LINK', bootstyle='secondary', command=self._link_action)
        # disabled until a localisation file is loaded
        self.link_btn.state(['disabled'])
        self.link_btn.pack(fill=tk.X, **pad)

        # Link parameters frame
        params_frame = ttk.LabelFrame(parent, text='Link parameters')
        params_frame.pack(fill=tk.X, **pad)

        # memory and min length
        mem_frame = ttk.Frame(params_frame)
        mem_frame.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        ttk.Label(mem_frame, text='memory').grid(row=0, column=0, sticky='w')
        self.memory_spin = self.Spinbox(mem_frame, from_=0, to=100, width=6)
        self.memory_spin.grid(row=1, column=0)
        ttk.Label(mem_frame, text='min length').grid(row=2, column=0, sticky='w')
        self.minlength_spin = self.Spinbox(mem_frame, from_=0, to=1000, width=6)
        self.minlength_spin.grid(row=3, column=0)

        # searchRange box on right (arrange x,y,z horizontally)
        search_frame = ttk.Frame(params_frame)
        search_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)
        ttk.Label(search_frame, text='searchRange').pack()
        search_inner = ttk.Frame(search_frame)
        search_inner.pack()
        ttk.Label(search_inner, text='x_range(m)').grid(row=0, column=0, padx=4)
        ttk.Label(search_inner, text='y_range(m)').grid(row=0, column=1, padx=4)
        ttk.Label(search_inner, text='z_range(m)').grid(row=0, column=2, padx=4)
        self.x_range_entry = ttk.Entry(search_inner, width=8)
        self.x_range_entry.insert(0, '2E-6')
        self.x_range_entry.grid(row=1, column=0, padx=4)
        self.y_range_entry = ttk.Entry(search_inner, width=8)
        self.y_range_entry.insert(0, '2E-6')
        self.y_range_entry.grid(row=1, column=1, padx=4)
        self.z_range_entry = ttk.Entry(search_inner, width=8)
        self.z_range_entry.insert(0, '2E-6')
        self.z_range_entry.grid(row=1, column=2, padx=4)

        # Trajectory controls (placed below link parameters)
        traj_frame = ttk.Frame(parent)
        traj_frame.pack(fill=tk.X, **pad)
        ttk.Label(traj_frame, text='trajectory').pack(anchor='w')
        # Trajectory slider: 0 = all, 1..N map to sorted particle list indices
        self.trajectory_scale = tk.Scale(traj_frame, from_=0, to=0, orient='horizontal', command=self._on_trajectory_scale)
        self.trajectory_scale.set(0)
        self.trajectory_scale.pack(fill=tk.X, padx=6)
        # show current trajectory index
        ttk.Label(traj_frame, textvariable=self.trajectory_index_var).pack(anchor='w', padx=6, pady=(2,0))
        ttk.Label(traj_frame, text='Total trajectories').pack(anchor='w')
        self.total_traj_entry = ttk.Entry(traj_frame, width=8)
        self.total_traj_entry.insert(0, '0')
        # align the total counter under the slider (left-aligned like the label)
        self.total_traj_entry.pack(anchor='w', padx=6)
        # Particle status indicator (shows selected particle ID and point count)
        ttk.Label(traj_frame, textvariable=self.particle_status_var, anchor='w').pack(pady=(6,0))

        # Play / Cancel and history/wait (inline)
        play_frame = ttk.Frame(parent)
        play_frame.pack(fill=tk.X, **pad)
        self.play_btn = self.Button(play_frame, text='Play', bootstyle='success', command=self._play)
        self.play_btn.pack(side=tk.LEFT, padx=6)
        self.cancel_btn = self.Button(play_frame, text='Cancel', bootstyle='danger', command=self._cancel)
        self.cancel_btn.pack(side=tk.LEFT, padx=6)

        ttk.Label(play_frame, text='history length').pack(side=tk.LEFT, padx=(12,2))
        self.history_spin = self.Spinbox(play_frame, from_=0, to=1000, width=6)
        self.history_spin.pack(side=tk.LEFT)
        ttk.Label(play_frame, text='wait time (ms)').pack(side=tk.LEFT, padx=(12,2))
        self.step_spin = self.Spinbox(play_frame, from_=1, to=60000, width=6)
        self.step_spin.pack(side=tk.LEFT)

        # Bottom Save / Exit / Help
        bottom_frame = ttk.Frame(parent)
        # avoid vertical expanding at the bottom so the left panel doesn't reserve
        # a large empty area; keep controls compact
        bottom_frame.pack(fill=tk.X, expand=False, **pad)
        self.save_btn = self.Button(bottom_frame, text='SAVE TRAJECTORIES', bootstyle='info', command=self._save)
        self.save_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6, pady=12)
        self.help_btn = self.Button(bottom_frame, text='HELP', bootstyle='primary', command=self._show_help)
        self.help_btn.pack(side=tk.LEFT, fill=tk.X, padx=6, pady=12)
        self.exit_btn = self.Button(bottom_frame, text='EXIT', bootstyle='danger', command=self._exit)
        self.exit_btn.pack(side=tk.RIGHT, fill=tk.X, padx=6, pady=12)
        # status label: place under the Save/Exit buttons so it spans the left panel width
        try:
            self.status_lbl = ttk.Label(parent, textvariable=self.status_var, anchor='w')
            self.status_lbl.pack(fill=tk.X, padx=6, pady=(6,2))
        except Exception:
            pass

    def _build_plot_area(self, parent):
        # Create a matplotlib 3D axes and place it in the right frame
        fig = Figure(figsize=(6,6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X position (\u03BCm)')
        ax.set_ylabel('Y position (\u03BCm)')
        ax.set_zlabel('Z position (\u03BCm)')

        # Make grid like in screenshot
        ax.grid(True)
        ax.set_box_aspect((1,1,1))
        # Ensure tick formatting is clear when numbers are very small.
        # We display data in µm (labels already show µm). Use a FuncFormatter so labels
        # show scientific notation inline (no separate multiplier text like "1e-5").
        def sci_fmt(val, pos):
            # show 3 significant digits in scientific notation
            return f"{val:.3e}"
        sci = FuncFormatter(sci_fmt)
        ax.xaxis.set_major_formatter(sci)
        ax.yaxis.set_major_formatter(sci)
        ax.zaxis.set_major_formatter(sci)

        # Initial empty plot limits similar to screenshot
        ax.set_xlim(0,200)
        ax.set_ylim(0,200)
        ax.set_zlim(0,120)

        self.fig = fig
        self.ax = ax

        # Add a small figure-level text to make the display units explicit
        # This makes it obvious that plotted coordinates are shown in µm
        self.fig.text(0.78, 0.96, 'Displayed in \u03BCm', fontsize=9, ha='right')

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        toolbar.pack(side=tk.RIGHT, anchor='se')

        # keep references
        self.canvas = canvas

    # --- Callbacks (stubs) ---
    def _browse_feature_file(self):
        path = filedialog.askopenfilename(title='Select localisation file', filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
        if path:
            try:
                self.feature_path_text.delete(0, tk.END)
                self.feature_path_text.insert(0, path)
            except Exception:
                pass
            self.status_var.set(f'Feature file: {os.path.basename(path)}')

    def _open_localisation_file(self):
        # Read file path from the PATH tab text widget (no file dialog)
        path = self.feature_path_text.get().strip()
        if not path:
            messagebox.showwarning('No path', 'Please specify a localisation file path in the "feature localisation file path" field.')
            return

        # Validate file exists
        if not os.path.isfile(path):
            messagebox.showerror('File not found', f'File not found: {path}')
            return

        # Update displayed path (normalize) and disable controls while loading
        try:
            try:
                self.feature_path_text.delete(0, tk.END)
                self.feature_path_text.insert(0, path)
            except Exception:
                pass
        except Exception:
            pass

        self.open_loc_btn.state(['disabled'])
        self.link_btn.state(['disabled'])
        self.status_var.set(f'Loading localisation file: {os.path.basename(path)}')

        # Submit background load
        future = self.executor.submit(self._load_localisation_file, path)
        # attach callback to be processed on main thread
        future.add_done_callback(lambda f: self._cb_queue.put(('load_done', f)))
        self._futures.append(future)

    def _load_localisation_file(self, path):
        """Background worker: delegate CSV loading to processor module."""
    
        df = processor.load_localisation_csv(path)
        return df

    def _link_action(self):
        # Start linking: ensure data loaded
        if self.loaded_df is None:
            messagebox.showwarning('No data', 'Load a localisation file first')
            return

        if not TRACKPY_AVAILABLE:
            messagebox.showerror('Missing dependency', 'trackpy not available. Install with `pip install trackpy`')
            return
        # Read parameters: build per-axis search_range tuple from meter inputs (keep in meters)
        xr = float(self.x_range_entry.get())
        yr = float(self.y_range_entry.get())
        zr = float(self.z_range_entry.get())

        # search_range stays in meters (processor.link_df expects meters)
        search_range = (xr, yr, zr)

        try:
            memory = int(self.memory_spin.get())
        except Exception:
            memory = 3
        try:
            minlength = int(self.minlength_spin.get())
        except Exception:
            minlength = 0

        # Disable LINK while running
        self.link_btn.state(['disabled'])
        self.status_var.set('Linking trajectories...')

        # Prepare dataframe for trackpy: uses columns frame, x, y (units: meters)
        df = self.loaded_df.copy()
        # trackpy expects frame numbers starting at 0 or 1, that's fine
        # Submit background task
        future = self.executor.submit(self._do_link, df, search_range, memory, minlength)
        future.add_done_callback(lambda f: self._cb_queue.put(('link_done', f)))
        self._futures.append(future)

    def _play(self):
        """Animate windows of trajectories.

        Behavior requested:
        - let H = history length (from UI)
        - On play, show trajectories in windows:
            1..H, 1..H+1, 2..H+1, 2..H+2, 3..H+2, ...
          i.e. successive windows of length H sliding forward by one until the end.
        - wait time between frames is taken from 'wait time (ms)'.
        - allow cancellation via _cancel (which stops the after-loop).
        """
        # cancel any existing animation
        try:
            if hasattr(self, '_play_after_id') and self._play_after_id is not None:
                try:
                    self.master.after_cancel(self._play_after_id)
                except Exception:
                    pass
                self._play_after_id = None
        except Exception:
            pass

        # get parameters
        try:
            history = int(self.history_spin.get())
        except Exception:
            history = 1
        try:
            wait_ms = int(self.step_spin.get())
        except Exception:
            wait_ms = 200

        # Prepare trajectory df (must contain frame info)
        if self.trajectories_df is None or 'particle' not in self.trajectories_df.columns or 'frame' not in self.trajectories_df.columns:
            self.status_var.set('No trajectories to play')
            return

        df = self.trajectories_df.copy()

        # Determine frame range available
        min_frame = int(df['frame'].min())
        max_frame = int(df['frame'].max())
        if history <= 0:
            history = 1

        # Start frames from min_frame up to max_frame - history + 1
        max_start = max_frame - history + 1
        if max_start < min_frame:
            max_start = min_frame

        self.status_var.set('Playing')

        # internal state for animation: current start frame
        self._play_state = {'start_frame': min_frame, 'history': history, 'min_frame': min_frame, 'max_frame': max_frame, 'max_start': max_start}

        def _step():
            st = self._play_state['start_frame']
            H = self._play_state['history']
            end_frame = st + H - 1
            if end_frame > self._play_state['max_frame']:
                end_frame = self._play_state['max_frame']

            # select points whose frame is within [st, end_frame]
            sel = df[(df['frame'] >= st) & (df['frame'] <= end_frame)]

            # plot all trajectories but only the points in the window
            self.ax.clear()
            self.ax.set_xlabel('X position (\u03BCm)')
            self.ax.set_ylabel('Y position (\u03BCm)')
            self.ax.set_zlabel('Z position (\u03BCm)')

            for pid, group in sel.groupby('particle'):
                gx = group['x'] * 1e6
                gy = group['y'] * 1e6
                gz = group['z'] * 1e6 if 'z' in group else 0
                # use stable color per particle if available
                try:
                    col = self._traj_color_map.get(int(pid)) if hasattr(self, '_traj_color_map') else None
                except Exception:
                    col = None
                try:
                    self.ax.plot(gx, gy, gz, linewidth=1, color=col)
                except Exception:
                    try:
                        self.ax.plot(gx, gy, [0]*len(gx), linewidth=1, color=col)
                    except Exception:
                        pass

            # keep global limits if available
            if hasattr(self, '_traj_xlim') and self._traj_xlim is not None:
                self.ax.set_xlim(*self._traj_xlim)
                self.ax.set_ylim(*self._traj_ylim)
                self.ax.set_zlim(*self._traj_zlim)
            else:
                try:
                    self.ax.autoscale()
                except Exception:
                    pass

            # update status and redraw
            try:
                self.status_var.set(f'Frames {st}..{end_frame}')
            except Exception:
                pass
            try:
                self.canvas.draw()
            except Exception:
                pass

            # advance to next start frame
            if st < self._play_state['max_start']:
                self._play_state['start_frame'] = st + 1
                try:
                    self._play_after_id = self.master.after(wait_ms, _step)
                except Exception:
                    self._play_after_id = None
            else:
                self.status_var.set('Play finished')
                self._play_after_id = None

        # start the loop
        try:
            self._play_after_id = self.master.after(wait_ms, _step)
        except Exception:
            self._play_after_id = None

    def _cancel(self):
        # Cancel any running play animation
        try:
            if hasattr(self, '_play_after_id') and self._play_after_id is not None:
                try:
                    self.master.after_cancel(self._play_after_id)
                except Exception:
                    pass
                self._play_after_id = None
        except Exception:
            pass
        self.status_var.set('Play cancelled')

    def _save(self):
        """Save trajectories to a new CSV file.

        Behavior:
        - Read the original localisation CSV from the path in the feature path entry.
        - Create an output file named TRAJECTORIES_YYYYMMDD_HHMMSS.csv next to the original.
        - The output file is the original CSV but with the column 'OBJECT NUMBER' replaced
          by a column named 'TRAJECTORY NUMBER' containing the particle id assigned by trackpy
          (rows without an assigned trajectory will receive -1).
        """
        if self.trajectories_df is None:
            messagebox.showwarning('No trajectories', 'No trajectories available to save. Run LINK first.')
            return

        # get original path
        path = self.feature_path_text.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror('File not found', 'Original localisation file not found. Cannot save trajectories.')
            return

        try:
            orig = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror('Read error', f'Could not read original file: {e}')
            return

        # Build a robust mapping from original row index -> particle id
        try:
            traj = self.trajectories_df
            if 'particle' not in traj.columns:
                messagebox.showerror('Save error', 'Trajectories do not contain a particle column')
                return

            out = orig.copy()
            nrows = len(out)
            # initialize with -1 (no trajectory)
            traj_nums = [-1] * nrows

            # iterate through trajectories and map each row index to its particle id
            for idx, pid in zip(traj.index, traj['particle']):
                try:
                    pos = int(idx)
                except Exception:
                    # if index cannot be interpreted as integer, skip
                    continue
                if 0 <= pos < nrows:
                    try:
                        traj_nums[pos] = int(pid)
                    except Exception:
                        traj_nums[pos] = -1

            out['TRAJECTORY NUMBER'] = traj_nums

            # Count how many were assigned; if too few, attempt a fallback match by (frame,x,y,z)
            assigned = sum(1 for v in traj_nums if v != -1)
            if assigned < max(10, int(0.5 * len(traj))):
                # Attempt robust matching by key (frame, x, y, z) using rounding to avoid float noise
                def pick_column(df, candidates):
                    for c in candidates:
                        if c in df.columns:
                            return c
                    return None

                frame_col = pick_column(orig, ['HOLOGRAM NUMBER', 'frame', 'Frame', 'HOLOGRAM_NUMBER'])
                x_col = pick_column(orig, ['X POSITION (m)', 'x', 'X_POSITION_(m)', 'X'])
                y_col = pick_column(orig, ['Y POSITION (m)', 'y', 'Y_POSITION_(m)', 'Y'])
                z_col = pick_column(orig, ['Z POSITION (m)', 'z', 'Z_POSITION_(m)', 'Z'])

                if frame_col and x_col and y_col:
                    # build mapping key -> list of positions
                    orig_keys = {}
                    for pos, row in orig.iterrows():
                        try:
                            f = int(row[frame_col])
                        except Exception:
                            continue
                        try:
                            xval = float(row[x_col])
                        except Exception:
                            xval = 0.0
                        try:
                            yval = float(row[y_col])
                        except Exception:
                            yval = 0.0
                        try:
                            zval = float(row[z_col]) if z_col and z_col in row else 0.0
                        except Exception:
                            zval = 0.0
                        key = (f, round(xval, 9), round(yval, 9), round(zval, 9))
                        orig_keys.setdefault(key, []).append(pos)

                    # assign from trajectories by matching keys
                    traj_nums2 = [-1] * nrows
                    used = 0
                    for _, r in traj.iterrows():
                        try:
                            f = int(r.get('frame', r.get('Frame', None)))
                        except Exception:
                            continue
                        xval = float(r.get('x', 0.0)) if not pd.isna(r.get('x', 0.0)) else 0.0
                        yval = float(r.get('y', 0.0)) if not pd.isna(r.get('y', 0.0)) else 0.0
                        zval = float(r.get('z', 0.0)) if not pd.isna(r.get('z', 0.0)) else 0.0
                        key = (f, round(xval, 9), round(yval, 9), round(zval, 9))
                        lst = orig_keys.get(key)
                        if lst and len(lst) > 0:
                            pos = lst.pop(0)
                            try:
                                traj_nums2[pos] = int(r['particle'])
                                used += 1
                            except Exception:
                                traj_nums2[pos] = -1

                    # prefer fallback mapping if it assigned more
                    if used > assigned:
                        out['TRAJECTORY NUMBER'] = traj_nums2
                        assigned = used

            # Remove original OBJECT NUMBER if present
            if 'OBJECT NUMBER' in out.columns:
                out.drop(columns=['OBJECT NUMBER'], inplace=True)

            # Build output filename
            dt = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_path = os.path.join(os.path.dirname(path), f'TRAJECTORIES_{dt}.csv')
            out.to_csv(out_path, index=False)
            messagebox.showinfo('Saved', f'Trajectories saved to: {out_path}')
            self.status_var.set(f'Saved trajectories to {os.path.basename(out_path)}')
        except Exception as e:
            messagebox.showerror('Save error', f'Error saving trajectories: {e}')

    def _show_help(self):
        """Display comprehensive help dialog with step-by-step procedure for trajectory linking"""
        help_window = tk.Toplevel(self.master)
        help_window.title("HoloTracker Link - User Guide")
        help_window.geometry("900x700")
        help_window.transient(self.master)
        help_window.grab_set()
        
        # Center the window
        help_window.update_idletasks()
        x = (help_window.winfo_screenwidth() // 2) - (900 // 2)
        y = (help_window.winfo_screenheight() // 2) - (700 // 2)
        help_window.geometry(f"900x700+{x}+{y}")
        
        # Create scrollable text widget
        text_frame = ttk.Frame(help_window)
        text_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        text_widget = tk.Text(text_frame, wrap="word", font=("Arial", 11), padx=10, pady=10)
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Help content
        help_text = """HoloTracker Link - Particle Trajectory Linking User Guide

OVERVIEW:
HoloTracker Link is designed to link particle detections from HoloTracker Locate across multiple hologram frames into continuous trajectories. It uses the powerful Trackpy library to perform particle tracking and provides interactive 3D visualization of trajectories.

PREREQUISITE:
Before using HoloTracker Link, you must first process your hologram sequence with HoloTracker Locate to generate a CSV file containing particle localizations with columns:
- HOLOGRAM NUMBER (frame number)
- OBJECT NUMBER (particle ID per frame)
- X POSITION (m)
- Y POSITION (m)
- Z POSITION (m)
- NUMBER OF VOXEL (particle size)

STEP-BY-STEP PROCEDURE:

1. SELECT LOCALISATION FILE:
   • In the PATH tab, click "Browse" to select the CSV file generated by HoloTracker Locate
   • The file should contain particle positions from all processed holograms
   • Verify the file path is displayed correctly in the text field
   • Return to the ACTIONS tab to proceed

2. OPEN LOCALISATION FILE:
   • Click "Open localisation File" button (large green button at top)
   • The system will load and validate the CSV file
   • Status bar will show the number of loaded particles
   • Counters will update:
     - Total holograms: Number of frames in your sequence
     - Total trajectories: Will show 0 until linking is performed
   • A 3D scatter plot of all particle positions will appear on the right
   • Feature hologram slider allows browsing individual frames before linking

3. CONFIGURE LINK PARAMETERS:
   These parameters control how particles are linked across frames:
   
   SEARCH RANGE (most critical parameters):
   • x_range (m): Maximum distance a particle can move in X direction between frames
     - Example: 2E-6 means particles can move up to 2 micrometers in X
   • y_range (m): Maximum distance a particle can move in Y direction between frames
     - Example: 2E-6 means particles can move up to 2 micrometers in Y
   • z_range (m): Maximum distance a particle can move in Z direction between frames
     - Example: 2E-6 means particles can move up to 2 micrometers in Z
   • Set these based on your particle velocity and frame rate
   • Too small: trajectories will be fragmented (broken into multiple short tracks)
   • Too large: different particles may be incorrectly linked together
   • Formula: search_range ≈ max_velocity × time_between_frames × safety_factor
     (safety_factor typically 1.5 to 2.0)
   
   MEMORY:
   • Number of frames a particle can be "missing" and still be linked to its trajectory
   • Useful when particles temporarily disappear (out of focus, occlusion)
   • Example: memory=2 allows a particle to vanish for 2 frames and still be tracked
   • memory=0: No gap tolerance (stricter tracking)
   • memory=5-10: Tolerates temporary disappearances (more forgiving)
   
   MIN LENGTH:
   • Minimum number of frames a trajectory must span to be kept
   • Filters out spurious detections and very short trajectories
   • Example: min_length=5 keeps only trajectories appearing in 5+ frames
   • Higher values = cleaner data but may discard real short-lived particles
   • Lower values = more trajectories but potentially more noise

4. PERFORM TRAJECTORY LINKING:
   • Click the "LINK" button to start trajectory linking
   • Trackpy will process all particles using your parameters
   • Progress appears in status bar
   • When complete:
     - Total trajectories counter updates with number of linked trajectories
     - 3D plot updates to show color-coded trajectories
     - Trajectory slider becomes active
     - Each trajectory is assigned a unique particle ID

5. VISUALIZE AND EXPLORE TRAJECTORIES:
   
   TRAJECTORY SLIDER:
   • Position 0: Display ALL trajectories simultaneously
   • Position 1 to N: Display individual trajectory (sorted by particle ID)
   • Use this to inspect specific particle paths
   • Particle status shows: "Particle ID: X (Y points)" where Y is trajectory length
   
   FEATURE HOLOGRAM SLIDER:
   • Browse through individual hologram frames
   • Shows all particles detected in that specific frame
   • Useful to verify detection quality before linking
   
   3D VISUALIZATION:
   • Rotate: Click and drag with left mouse button
   • Zoom: Scroll wheel or right-click drag
   • Pan: Middle-click drag or Shift + left-click drag
   • Navigation toolbar (bottom-right) provides additional controls:
     - Home: Reset view to initial state
     - Pan/Zoom: Toggle pan/zoom mode
     - Save: Export current view as image
   • Color coding:
     - Before linking: Single color for all particles
     - After linking: Each trajectory has unique color
     - Colormap wraps for datasets with many trajectories

6. ANIMATION AND PLAYBACK:
   
   PLAY BUTTON:
   • Animates trajectory evolution over time
   • Shows how particles move frame-by-frame
   • Displays trajectory history (trails behind particles)
   
   HISTORY LENGTH:
   • Number of previous positions to display during animation
   • history=0: Show only current frame positions (no trails)
   • history=10: Show last 10 positions for each particle (short trails)
   • history=100: Show last 100 positions (long trails revealing full paths)
   • Adjust based on desired visualization style
   
   WAIT TIME (ms):
   • Delay between animation frames in milliseconds
   • Lower values (50-100 ms): Fast animation
   • Higher values (500-1000 ms): Slow, detailed animation
   • Adjust based on trajectory complexity and viewing preference
   
   CANCEL BUTTON:
   • Stops ongoing animation
   • Returns to static display mode

7. SAVE TRAJECTORIES:
   • Click "SAVE TRAJECTORIES" button when satisfied with linking results
   • Output CSV file is created with timestamp: TRAJECTORIES_YYYYMMDD_HHMMSS.csv
   • Saved in same directory as input localisation file
   • Output file contains all columns from input PLUS:
     - TRAJECTORY NUMBER: Unique ID for each linked trajectory
     - Particles not assigned to trajectories have TRAJECTORY NUMBER = -1
   • Status bar confirms save location
   • Use this file for further analysis in external software (Excel, Python, MATLAB, etc.)

PARAMETER OPTIMIZATION WORKFLOW:

1. Start with conservative search ranges (slightly larger than expected particle movement)
2. Use memory=0 and min_length=2 initially to see all possible trajectories
3. Click LINK and examine results:
   - Too few trajectories? Increase search range or reduce min_length
   - Too many short fragments? Increase memory or search range
   - Incorrect links between different particles? Reduce search range
4. Iterate by adjusting parameters and re-linking until results look correct
5. Once satisfied, increase min_length to filter out noise
6. Use trajectory slider to inspect individual particle paths
7. Save final trajectories

UNDERSTANDING THE DISPLAY:

COORDINATE SYSTEM:
• All positions displayed in micrometers (µm) for readability
• X, Y: Transverse positions (perpendicular to beam)
• Z: Axial position (along beam propagation direction)
• Origin (0, 0, 0) corresponds to top-left corner of first reconstruction plane

STATUS MESSAGES:
• "Loaded X particles from Y holograms": File successfully opened
• "Linking trajectories...": Trackpy is processing (may take time for large datasets)
• "Found X trajectories": Linking complete
• "Saved trajectories to...": Export successful

TROUBLESHOOTING:

PROBLEM: No trajectories found after linking
SOLUTION: 
- Increase search range (particles moving too fast)
- Reduce min_length (requirement too strict)
- Check that input file has correct format and multiple frames

PROBLEM: All particles linked into one giant trajectory
SOLUTION:
- Reduce search range (too permissive, linking unrelated particles)
- Check particle density (too many particles in small volume)
- Verify data quality from HoloTracker Locate

PROBLEM: Trajectories are fragmented (same particle has multiple IDs)
SOLUTION:
- Increase search range
- Increase memory parameter (allows gaps)
- Reduce min_length to see all fragments first

PROBLEM: Linking is very slow
SOLUTION:
- Large datasets take time (normal for 1000+ particles per frame)
- Consider filtering input data to reduce particle count
- Process smaller time windows separately

PROBLEM: 3D plot is cluttered
SOLUTION:
- Use trajectory slider to view one trajectory at a time
- Adjust plot view angle using mouse rotation
- Increase min_length to show only longer trajectories
- Use animation with limited history length

TECHNICAL NOTES:

• Trackpy uses a nearest-neighbor approach with predictive algorithms
• Search range defines a search ellipsoid around each particle's last known position
• Memory parameter uses a cost-minimization algorithm for gap closing
• Trajectory IDs are assigned sequentially based on first appearance frame
• Color assignment uses a cyclic colormap (repeats after ~20 colors)
• Very large datasets (>100,000 points) may require significant RAM

TYPICAL PARAMETER RANGES:

Low-speed particles (< 10 µm/s):
- search_range: 1E-6 to 5E-6 m
- memory: 2-5
- min_length: 5-10

High-speed particles (> 50 µm/s):
- search_range: 5E-6 to 20E-6 m  
- memory: 0-2
- min_length: 3-5

Dense particle fields:
- Use smaller search_range to avoid incorrect links
- Use lower memory to prevent bridging
- Use higher min_length to filter noise

Sparse particle fields:
- Can use larger search_range safely
- Higher memory helps with intermittent detection
- Lower min_length keeps more trajectories

For technical support or questions, contact: simon.becker@univ-lorraine.fr or simbecker@gmail.com

AUTHOR:
Simon Becker (Université de Lorraine)

CITATION:
If HoloTracker Link has been useful for your research, please cite it in your publications along with Trackpy:
- Trackpy: doi.org/10.5281/zenodo.4682814
- HoloTracker Link: Developed by Simon Becker

This software builds upon the excellent Trackpy library developed by the soft-matter community."""
        
        text_widget.insert("1.0", help_text)
        text_widget.config(state="disabled")  # Make read-only
        
        # Close button
        close_frame = ttk.Frame(help_window)
        close_frame.pack(pady=10)
        ttk.Button(close_frame, text="Close", command=help_window.destroy).pack()
        
        # Focus on the window
        help_window.focus_set()

    def _exit(self):
        # shutdown executor
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass
        # Save params before quitting
        try:
            self._save_params()
        except Exception:
            pass
        self.master.quit()

    def _load_params(self):
        """Load persistent parameters from last_params_link.json if present."""
        if not os.path.isfile(self._params_path):
            return
        with open(self._params_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)

        # Apply params if present
        try:
            fp = data.get('feature_path')
            if fp:
                try:
                    self.feature_path_text.delete(0, tk.END)
                    self.feature_path_text.insert(0, fp)
                except Exception:
                    pass

            mem = data.get('memory')
            if mem is not None:
                try:
                    self.memory_spin.delete(0, tk.END)
                    self.memory_spin.insert(0, str(mem))
                except Exception:
                    pass

            minlength = data.get('minlength')
            if minlength is not None:
                try:
                    self.minlength_spin.delete(0, tk.END)
                    self.minlength_spin.insert(0, str(minlength))
                except Exception:
                    pass

            for axis in ('x_range', 'y_range', 'z_range'):
                v = data.get(axis)
                if v is not None:
                    try:
                        getattr(self, f'{axis}_entry').delete(0, tk.END)
                        getattr(self, f'{axis}_entry').insert(0, str(v))
                    except Exception:
                        pass

            history = data.get('history')
            if history is not None:
                try:
                    self.history_spin.delete(0, tk.END)
                    self.history_spin.insert(0, str(history))
                except Exception:
                    pass

            step = data.get('step')
            if step is not None:
                try:
                    self.step_spin.delete(0, tk.END)
                    self.step_spin.insert(0, str(step))
                except Exception:
                    pass
        except Exception:
            # don't fail loading on any per-field error
            pass

    def _save_params(self):
        """Save selected parameters to last_params_link.json"""
        data = {}
        try:
            data['feature_path'] = self.feature_path_text.get().strip()
        except Exception:
            data['feature_path'] = ''

        try:
            data['memory'] = int(self.memory_spin.get())
        except Exception:
            data['memory'] = None

        try:
            data['minlength'] = int(self.minlength_spin.get())
        except Exception:
            data['minlength'] = None

        try:
            data['x_range'] = self.x_range_entry.get()
        except Exception:
            data['x_range'] = None
        try:
            data['y_range'] = self.y_range_entry.get()
        except Exception:
            data['y_range'] = None
        try:
            data['z_range'] = self.z_range_entry.get()
        except Exception:
            data['z_range'] = None

        try:
            data['history'] = int(self.history_spin.get())
        except Exception:
            data['history'] = None
        try:
            data['step'] = int(self.step_spin.get())
        except Exception:
            data['step'] = None

        try:
            with open(self._params_path, 'w', encoding='utf-8') as fh:
                json.dump(data, fh, indent=2)
        except Exception:
            # ignore save errors
            pass

    # --- Background task handlers ---
    def _process_cb_queue(self):
        try:
            while True:
                item = self._cb_queue.get_nowait()
                if item[0] == 'load_done':
                    future = item[1]
                    try:
                        df = future.result()
                        self._on_file_loaded(df)
                    except Exception as e:
                        messagebox.showerror('Load error', f'Error loading file: {e}')
                        self.status_var.set('Load failed')
                    finally:
                        # Re-enable open button
                        try:
                            self.open_loc_btn.state(['!disabled'])
                        except Exception:
                            pass
                elif item[0] == 'link_done':
                    future = item[1]
                    try:
                        traj_df = future.result()
                        self._on_link_complete(traj_df)
                    except Exception as e:
                        messagebox.showerror('Link error', f'Error linking trajectories: {e}')
                        self.status_var.set('Link failed')
                        try:
                            self.link_btn.state(['!disabled'])
                        except Exception:
                            pass
                self._cb_queue.task_done()
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self._process_cb_queue)

    def _on_file_loaded(self, df):
        self.loaded_df = df
        # update counters
        holo_ids = sorted(df['frame'].unique())
        total_holo = len(holo_ids)
        self.total_holo_entry.delete(0, tk.END)
        self.total_holo_entry.insert(0, str(total_holo))
        # set spinbox range to available hologram indices (min->max)
        if total_holo > 0:
            min_frame = int(holo_ids[0])
            max_frame = int(holo_ids[-1])
            try:
                self.feature_holo_scale.configure(from_=min_frame, to=max_frame)
            except Exception:
                pass
            # set to first available
            try:
                self.feature_holo_scale.set(min_frame)
                # update displayed value
                try:
                    self.feature_index_var.set(str(min_frame))
                except Exception:
                    pass
            except Exception:
                pass
            self._plot_hologram(min_frame)
        self.status_var.set('Load complete')
        # enable LINK
        try:
            self.link_btn.state(['!disabled'])
        except Exception:
            pass

    def _on_hologram_spin(self):
        # kept for compatibility with older bindings
        try:
            val = int(self.feature_holo_scale.get())
            self._plot_hologram(val)
        except Exception:
            pass

    def _on_hologram_scale(self, value):
        try:
            val = int(float(value))
            # update displayed numeric indicator
            try:
                self.feature_index_var.set(str(val))
            except Exception:
                pass
            self._plot_hologram(val)
        except Exception:
            pass

    def _on_trajectory_spin(self):
        """Callback when trajectory spinbox changes: show only selected trajectory if linking was performed."""
        try:
            if self.trajectories_df is None:
                return
            val = int(self.trajectory_scale.get())
            # slider value 0 means show all
            if val == 0:
                self._plot_all_trajectories()
            else:
                # map slider index to particle id
                if hasattr(self, '_particle_id_list') and 1 <= val <= len(self._particle_id_list):
                    pid = self._particle_id_list[val-1]
                    self._plot_trajectory(pid)
                else:
                    # fallback: try treat val as pid
                    self._plot_trajectory(val)
        except Exception:
            pass

    def _on_trajectory_scale(self, value):
        try:
            v = int(float(value))
            # update displayed numeric indicator
            try:
                self.trajectory_index_var.set(str(v))
            except Exception:
                pass
            if v == 0:
                self._plot_all_trajectories()
            else:
                if hasattr(self, '_particle_id_list') and 1 <= v <= len(self._particle_id_list):
                    pid = self._particle_id_list[v-1]
                    self._plot_trajectory(pid)
                else:
                    self._plot_trajectory(v)
        except Exception:
            pass

    def _plot_all_trajectories(self):
        """Plot all trajectories using stored global limits if available."""
        if self.trajectories_df is None:
            return
        traj_df = self.trajectories_df
        try:
            self.ax.clear()
            self.ax.set_xlabel('X position (\u03BCm)')
            self.ax.set_ylabel('Y position (\u03BCm)')
            self.ax.set_zlabel('Z position (\u03BCm)')
            for pid, group in traj_df.groupby('particle'):
                gx = group['x'] * 1e6
                gy = group['y'] * 1e6
                gz = group.get('z', 0) * 1e6 if 'z' in group else 0
                try:
                    col = self._traj_color_map.get(int(pid)) if hasattr(self, '_traj_color_map') else None
                except Exception:
                    col = None
                # thinner lines for better visibility
                self.ax.plot(gx, gy, gz, linewidth=0.8, color=col)

            # compute and store global limits
            all_x = (traj_df['x'] * 1e6)
            all_y = (traj_df['y'] * 1e6)
            all_z = (traj_df['z'] * 1e6) if 'z' in traj_df else (traj_df['x']*0)
            xmin, xmax = float(all_x.min()), float(all_x.max())
            ymin, ymax = float(all_y.min()), float(all_y.max())
            zmin, zmax = float(all_z.min()), float(all_z.max())

            def _expand(a, b):
                if a == b:
                    delta = abs(a) * 0.05 if a != 0 else 1.0
                    return a - delta, b + delta
                rng = b - a
                pad = rng * 0.05
                return a - pad, b + pad

            xmin, xmax = _expand(xmin, xmax)
            ymin, ymax = _expand(ymin, ymax)
            zmin, zmax = _expand(zmin, zmax)

            self._traj_xlim = (xmin, xmax)
            self._traj_ylim = (ymin, ymax)
            self._traj_zlim = (zmin, zmax)

            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            self.ax.set_zlim(zmin, zmax)

            self.canvas.draw()
        except Exception:
            pass

    def _plot_trajectory(self, particle_id):
        """Plot only the trajectory with given particle id (particle ids from trackpy)."""
        if self.trajectories_df is None:
            return
        df = self.trajectories_df
        if 'particle' not in df.columns:
            return
        sel = df[df['particle'] == int(particle_id)]
        n = len(sel)
        try:
            self.status_var.set(f'Trajectory {particle_id}: {n} points')
        except Exception:
            pass

        # Clear axes and labels
        self.ax.clear()
        self.ax.set_xlabel('X position (\u03BCm)')
        self.ax.set_ylabel('Y position (\u03BCm)')
        self.ax.set_zlabel('Z position (\u03BCm)')

        if n > 0:
            # convert meters -> µm only for display
            xs = (sel['x'].dropna() * 1e6)
            ys = (sel['y'].dropna() * 1e6)
            zs = (sel['z'].dropna() * 1e6) if 'z' in sel else pd.Series([0]*len(xs))

            # If global trajectory limits were computed when plotting all trajectories,
            # reuse those so individual-trajectory views keep the same scale (no zooming).
            if hasattr(self, '_traj_xlim') and self._traj_xlim is not None:
                xmin, xmax = self._traj_xlim
                ymin, ymax = self._traj_ylim
                zmin, zmax = self._traj_zlim
            else:
                xmin, xmax = float(xs.min()), float(xs.max())
                ymin, ymax = float(ys.min()), float(ys.max())
                zmin, zmax = float(zs.min()), float(zs.max())

                def _expand(a, b):
                    if a == b:
                        delta = abs(a) * 0.05 if a != 0 else 1.0
                        return a - delta, b + delta
                    rng = b - a
                    pad = rng * 0.05
                    return a - pad, b + pad

                xmin, xmax = _expand(xmin, xmax)
                ymin, ymax = _expand(ymin, ymax)
                zmin, zmax = _expand(zmin, zmax)

            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            self.ax.set_zlim(zmin, zmax)

            try:
                col = self._traj_color_map.get(int(particle_id)) if hasattr(self, '_traj_color_map') else None
            except Exception:
                col = None

            try:
                # plot as line with small markers using stable color
                self.ax.plot(xs, ys, zs, linewidth=0.8, marker='o', markersize=3, color=col)
            except Exception:
                try:
                    self.ax.plot(xs, ys, zs if len(zs) == len(xs) else [0] * len(xs), linewidth=0.8, color=col)
                except Exception:
                    pass
        else:
            # no points: set a default view
            self.ax.set_xlim(0, 200)
            self.ax.set_ylim(0, 200)
            self.ax.set_zlim(0, 120)

        try:
            self.canvas.draw()
        except Exception:
            pass

    def _plot_hologram(self, frame_idx):
        if self.loaded_df is None:
            return
        df = self.loaded_df
        sel = df[df['frame'] == int(frame_idx)]
        # Count selected objects and report
        n = len(sel)
        try:
            self.status_var.set(f'Frame {frame_idx}: {n} objects')
        except Exception:
            pass
        try:
            print(f'[CodeLink] Plotting frame {frame_idx}: {n} objects')
        except Exception:
            pass

        # Clear axes and labels
        self.ax.clear()
        self.ax.set_xlabel('X position (\u03BCm)')
        self.ax.set_ylabel('Y position (\u03BCm)')
        self.ax.set_zlabel('Z position (\u03BCm)')

        # If we have points, autoscale axes to their extents with padding
        if n > 0:
            # convert meters -> µm only for display
            xs = (sel['x'].dropna() * 1e6)
            ys = (sel['y'].dropna() * 1e6)
            zs = (sel['z'].dropna() * 1e6)
            if len(xs) == 0:
                xs = pd.Series([0.0])
            if len(ys) == 0:
                ys = pd.Series([0.0])
            if len(zs) == 0:
                zs = pd.Series([0.0])

            xmin, xmax = float(xs.min()), float(xs.max())
            ymin, ymax = float(ys.min()), float(ys.max())
            zmin, zmax = float(zs.min()), float(zs.max())

            # handle zero-range by expanding a small fraction
            def _expand(a, b):
                if a == b:
                    delta = abs(a) * 0.05 if a != 0 else 1.0
                    return a - delta, b + delta
                rng = b - a
                pad = rng * 0.05
                return a - pad, b + pad

            xmin, xmax = _expand(xmin, xmax)
            ymin, ymax = _expand(ymin, ymax)
            zmin, zmax = _expand(zmin, zmax)

            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            self.ax.set_zlim(zmin, zmax)

            # scatter points
            try:
                self.ax.scatter(xs, ys, zs, c='r', s=10)
            except Exception:
                # fallback: try plotting without z
                try:
                    self.ax.scatter(xs, ys, [0]*len(xs), c='r', s=10)
                except Exception:
                    pass
        else:
            # no points: set a default view
            self.ax.set_xlim(0, 200)
            self.ax.set_ylim(0, 200)
            self.ax.set_zlim(0, 120)

        # redraw
        try:
            self.canvas.draw()
        except Exception:
            pass

    def _do_link(self, df, search_range_m, memory, minlength=0):
        # Delegate linking to processor.link_df which links the entire dataframe
        # NOTE: search_range_m is a tuple in meters (x_range, y_range, z_range)
        try:
            from code_link import processor
        except Exception:
            # fallback to local trackpy if processor import fails
            processor = None

        if processor is not None:
            # processor.link_df expects df with frame,x,y,(z) in meters
            return processor.link_df(df, search_range_m, memory, minlength=minlength)

        # Fallback: call trackpy directly
        df_tp = df[['frame', 'x', 'y', 'z']].copy()
        df_tp.rename(columns={'frame': 'frame', 'x': 'x', 'y': 'y'}, inplace=True)
        # tp.link_df expects the search radius in the same units as the x,y columns (meters here)
        trajectories = tp.link_df(df_tp, search_range_m, memory=memory)
        return trajectories

    def _on_link_complete(self, traj_df):
        self.trajectories_df = traj_df
        n_traj = traj_df['particle'].nunique() if 'particle' in traj_df.columns else 0
        self.total_traj_entry.delete(0, tk.END)
        self.total_traj_entry.insert(0, str(n_traj))
        self.status_var.set(f'Link complete - {n_traj} trajectories')
        # Re-enable LINK
        self.link_btn.state(['!disabled'])

        # Configure trajectory scale range and mapping: 0 => all, 1..N map to sorted particle ids
        if 'particle' in traj_df.columns and n_traj > 0:
            pids = sorted(traj_df['particle'].unique())
            self._particle_id_list = [int(x) for x in pids]
            self.trajectory_scale.configure(from_=0, to=len(self._particle_id_list))
            self.trajectory_scale.set(0)
            self.trajectory_index_var.set('0')
            self._traj_color_map = {}
            for i, pid in enumerate(self._particle_id_list):
                col = mcolors.to_hex(cm.tab20(i % 20))
                self._traj_color_map[pid] = col
            self._plot_all_trajectories()

        # Optionally plot trajectories (first few)
        # plot trajectories as lines for a subset
        self.ax.clear()
        self.ax.set_xlabel('X position (\u03BCm)')
        self.ax.set_ylabel('Y position (\u03BCm)')
        self.ax.set_zlabel('Z position (\u03BCm)')
        for pid, group in traj_df.groupby('particle'):
            # plot all trajectories but keep small marker size
            # convert meters -> µm for display
            gx = group['x'] * 1e6
            gy = group['y'] * 1e6
            gz = group.get('z', 0) * 1e6 if 'z' in group else 0
            try:
                col = self._traj_color_map.get(int(pid)) if hasattr(self, '_traj_color_map') else None
            except Exception:
                col = None
            self.ax.plot(gx, gy, gz, linewidth=0.8, color=col)
        self.canvas.draw()



if __name__ == '__main__':
    root = tk.Tk()
    app = CodeLinkGUI(root)
    root.mainloop()
