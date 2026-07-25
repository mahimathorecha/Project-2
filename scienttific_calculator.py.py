"""
Advanced Scientific Calculator  —  Tkinter
Recreation of the provided dark-theme UI mockup.

Run:  python scientific_calculator.py
Requires: Python 3.8+ (tkinter ships with standard CPython).

Notes
-----
* The CALC keypad is fully wired: numbers, operators, trig (DEG/RAD aware),
  powers, roots, logs, constants, factorial, ABS, etc.  Press '=' to evaluate.
* MATRIX / STAT / GRAPH panels + the CONVERT/SOLVE tabs are laid out to match
  the mockup. STAT computes live from its data row and GRAPH draws a real sine
  curve; the other side-panel items are stubs you can extend.
* Tkinter has no native rounded corners / gradients, so those are approximated
  with flat fills and border frames. Everything else is faithful to the mockup.
"""

import math
import re
import tkinter as tk
from tkinter import font as tkfont


# --------------------------------------------------------------------------- #
#  Palette (sampled from the mockup)
# --------------------------------------------------------------------------- #
C = {
    "bg":            "#0a0e17",   # app background (near black)
    "panel":         "#0d1220",   # inset panels
    "btn":           "#1b2130",   # standard key
    "btn_hi":        "#252c3d",   # standard key hover
    "btn_edge":      "#2a3346",
    "num":           "#20283a",   # number keys (slightly lighter)
    "num_hi":        "#2b3550",
    "display_bg":    "#e9eef1",   # light LCD
    "display_ink":   "#0c1118",
    "green":         "#22c55e",   # DEG / constants header
    "blue":          "#2563eb",   # CALC active / '=' / graph
    "blue_hi":       "#3b82f6",
    "gold":          "#9a7b17",   # SHIFT
    "gold_hi":       "#b8952a",
    "gold_txt":      "#f3c04a",
    "purple":        "#6d28d9",   # ALPHA / matrix
    "purple_hi":     "#7c3aed",
    "red":           "#7f1d2d",   # AC / DEL
    "red_hi":        "#9a2540",
    "amber":         "#d9a441",   # stat panel / shift-labels
    "sub":           "#8b97a8",   # muted text
    "sub2":          "#5c6b80",
    "hist_blue":     "#4f9bff",
    "white":         "#e7edf5",
}

FONT      = "DejaVu Sans"          # broad unicode coverage on Linux
FONT_MONO = "DejaVu Sans Mono"


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #
def bordered(parent, color, bg, pad=1):
    """A frame with a 1px colored 'border' (outer colored frame + inner fill)."""
    outer = tk.Frame(parent, bg=color)
    inner = tk.Frame(outer, bg=bg)
    inner.pack(fill="both", expand=True, padx=pad, pady=pad)
    return outer, inner


# --------------------------------------------------------------------------- #
#  Matrix math (pure Python, no numpy dependency)
# --------------------------------------------------------------------------- #
def _shape(M):
    return (len(M), len(M[0]) if M else 0)


def _square(M):
    if _shape(M)[0] != _shape(M)[1]:
        raise ValueError("Requires a square matrix")


def m_identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def m_add(A, B):
    if _shape(A) != _shape(B):
        raise ValueError("A and B must be the same size")
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def m_sub(A, B):
    if _shape(A) != _shape(B):
        raise ValueError("A and B must be the same size")
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def m_mul(A, B):
    if len(A[0]) != len(B):
        raise ValueError("A cols must equal B rows")
    return [[sum(A[i][k] * B[k][j] for k in range(len(B)))
             for j in range(len(B[0]))] for i in range(len(A))]


def m_scalar(A, k):
    return [[k * x for x in row] for row in A]


def m_transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def m_det(M):
    _square(M)
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    total = 0.0
    for c in range(n):
        minor = [row[:c] + row[c + 1:] for row in M[1:]]
        total += ((-1) ** c) * M[0][c] * m_det(minor)
    return total


def m_cofactor(M):
    n = len(M)
    if n == 1:
        return [[1.0]]
    return [[((-1) ** (i + j)) *
             m_det([row[:j] + row[j + 1:] for k, row in enumerate(M) if k != i])
             for j in range(n)] for i in range(n)]


def m_adjoint(M):
    _square(M)
    return m_transpose(m_cofactor(M))


def m_inverse(M):
    _square(M)
    d = m_det(M)
    if abs(d) < 1e-12:
        raise ValueError("Not Invertible")
    adj = m_adjoint(M)
    n = len(M)
    return [[adj[i][j] / d for j in range(n)] for i in range(n)]


def m_trace(M):
    _square(M)
    return sum(M[i][i] for i in range(len(M)))


def m_power(M, n):
    _square(M)
    if n < 0:
        M, n = m_inverse(M), -n
    R = m_identity(len(M))
    for _ in range(n):
        R = m_mul(R, M)
    return R


def m_rref(M):
    A = [[float(x) for x in row] for row in M]
    rows, cols, lead = len(A), len(A[0]), 0
    for r in range(rows):
        if lead >= cols:
            break
        i = r
        while abs(A[i][lead]) < 1e-12:
            i += 1
            if i == rows:
                i, lead = r, lead + 1
                if lead == cols:
                    return A
        A[i], A[r] = A[r], A[i]
        lv = A[r][lead]
        A[r] = [x / lv for x in A[r]]
        for i in range(rows):
            if i != r:
                f = A[i][lead]
                A[i] = [a - f * b for a, b in zip(A[i], A[r])]
        lead += 1
    return A


def m_rank(M):
    R = m_rref(M)
    return sum(1 for row in R if any(abs(x) > 1e-9 for x in row))


def nfmt(v):
    """Format a number for matrix cells: ints plain, floats trimmed."""
    try:
        v = round(float(v) + 0.0, 6)
    except (TypeError, ValueError):
        return str(v)
    if v == int(v):
        return str(int(v))
    return f"{v:g}"


# --------------------------------------------------------------------------- #
#  Hoverable flat "button" built from a Frame (lets us stack a shift-label)
# --------------------------------------------------------------------------- #
class Key(tk.Frame):
    def __init__(self, parent, main, command=None, *, shift=None,
                 bg=C["btn"], hi=C["btn_hi"], fg=C["white"],
                 shift_fg=C["amber"], font_size=13, bold=False, height=42):
        super().__init__(parent, bg=bg, height=height, cursor="hand2",
                         highlightthickness=1, highlightbackground=C["btn_edge"])
        self.pack_propagate(False)
        self._bg, self._hi = bg, hi
        self.command = command
        self.shift_lbl = None

        if shift:
            self.shift_lbl = tk.Label(self, text=shift, bg=bg, fg=shift_fg,
                                      font=(FONT, 8))
            self.shift_lbl.pack(anchor="ne", padx=4, pady=(1, 0))
            pady_main = (0, 4)
        else:
            pady_main = 0

        weight = "bold" if bold else "normal"
        self.lbl = tk.Label(self, text=main, bg=bg, fg=fg,
                            font=(FONT, font_size, weight))
        self.lbl.pack(expand=True, pady=pady_main)

        for w in (self, self.lbl):
            w.bind("<Button-1>", self._click)
            w.bind("<Enter>", self._enter)
            w.bind("<Leave>", self._leave)
        for child in self.winfo_children():
            child.bind("<Button-1>", self._click)
            child.bind("<Enter>", self._enter)
            child.bind("<Leave>", self._leave)

    def _paint(self, color):
        self.config(bg=color)
        for c in self.winfo_children():
            c.config(bg=color)

    def _enter(self, _): self._paint(self._hi)
    def _leave(self, _): self._paint(self._bg)

    def set_base(self, bg, hi):
        """Change the resting/hover colours (used for toggle state)."""
        self._bg, self._hi = bg, hi
        self._paint(bg)

    def _click(self, _):
        if self.command:
            self.command()


# --------------------------------------------------------------------------- #
#  Main application
# --------------------------------------------------------------------------- #
class SciCalc(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Advanced Scientific Calculator")
        self.configure(bg=C["bg"])
        self._pick_fonts()      # choose best available medium-weight family
        # Full layout is ~1440px tall; open at a screen-friendly height and let
        # the user scroll down to the Matrix / Statistics / Graph panels.
        self.geometry("880x1000")
        self.minsize(760, 560)

        self.expr = ""          # raw display expression (pretty symbols)
        self.result = "0"
        self.angle = "DEG"      # DEG / RAD
        self.history = []       # list of (expr, value) tuples
        self.just_evaluated = False
        self.shift_on = False
        self.alpha_on = False
        self.memory = 0.0

        # ---- scrollable shell so nothing gets cut off on shorter screens ----
        shell = tk.Frame(self, bg=C["bg"])
        shell.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(shell, bg=C["bg"], highlightthickness=0)
        vsb = tk.Scrollbar(shell, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        root = tk.Frame(self._canvas, bg=C["bg"])
        self._root_id = self._canvas.create_window((0, 0), window=root,
                                                   anchor="nw")
        # keep inner frame width matched to the canvas, and scrollregion fresh
        self._canvas.bind(
            "<Configure>",
            lambda e: self._canvas.itemconfigure(self._root_id, width=e.width))
        root.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._bind_mousewheel()

        # inner padding lives on a child so it doesn't fight the scroll width
        root = tk.Frame(root, bg=C["bg"])
        root.pack(fill="both", expand=True, padx=9, pady=8)

        self._build_titlebar(root)
        self._build_display(root)
        self._build_modebar(root)

        # ---------------- swappable body: one frame per mode ----------------
        self.body = tk.Frame(root, bg=C["bg"])
        self.body.pack(fill="both", expand=True)

        # CALC view (keypad + sidebar + bottom panels)
        self.calc_view = tk.Frame(self.body, bg=C["bg"])
        mid = tk.Frame(self.calc_view, bg=C["bg"])
        mid.pack(fill="both", expand=True, pady=(6, 0))
        self._build_keypad(mid)
        self._build_sidebar(mid)
        self._build_bottom_panels(self.calc_view)

        # MATRIX view (built lazily on first switch)
        self.matrix_view = None
        self.stat_view = None
        self.graph_view = None
        self.convert_view = None
        self.solve_view = None
        self._mode_views = {"CALC": self.calc_view}

        self._set_mode("CALC")
        self._refresh_display()

    def _pick_fonts(self):
        """Pick the nicest available UI + mono family (Tk has no 'medium'
        weight, so a well-hinted family at normal weight gives the medium look
        seen in the reference)."""
        global FONT, FONT_MONO
        fams = set(tkfont.families(self))
        for f in ("Segoe UI", "Roboto", "Helvetica Neue", "Inter",
                  "Arial", "DejaVu Sans"):
            if f in fams:
                FONT = f
                break
        for f in ("Cascadia Code", "Consolas", "SF Mono", "Menlo",
                  "DejaVu Sans Mono", "Courier New"):
            if f in fams:
                FONT_MONO = f
                break

    def _bind_mousewheel(self):
        def _on_wheel(event):
            if event.num == 4 or event.delta > 0:          # scroll up
                self._canvas.yview_scroll(-3, "units")
            elif event.num == 5 or event.delta < 0:        # scroll down
                self._canvas.yview_scroll(3, "units")
        # Windows / macOS
        self.bind_all("<MouseWheel>", _on_wheel)
        # Linux (X11)
        self.bind_all("<Button-4>", _on_wheel)
        self.bind_all("<Button-5>", _on_wheel)

    # ----------------------------------------------------------------- title
    def _build_titlebar(self, parent):
        bar = tk.Frame(parent, bg=C["bg"])
        bar.pack(fill="x")
        tk.Label(bar, text="ADVANCED SCIENTIFIC CALCULATOR", bg=C["bg"],
                 fg=C["white"], font=(FONT, 13, "bold")).pack(side="left")

        right = tk.Frame(bar, bg=C["bg"])
        right.pack(side="right")
        for icon in ("\u2600", "\U0001F509"):     # sun, speaker
            tk.Label(right, text=icon, bg=C["bg"], fg=C["sub"],
                     font=(FONT, 12)).pack(side="left", padx=6)
        self.deg_lbl = tk.Label(right, text=f"{self.angle} \u25be", bg=C["bg"],
                                fg=C["green"], font=(FONT, 11, "bold"),
                                cursor="hand2")
        self.deg_lbl.pack(side="left", padx=6)
        self.deg_lbl.bind("<Button-1>", lambda e: self._toggle_angle())
        tk.Label(right, text="\u2630", bg=C["bg"], fg=C["sub"],
                 font=(FONT, 14)).pack(side="left", padx=6)

    # --------------------------------------------------------------- display
    def _build_display(self, parent):
        outer, inner = bordered(parent, "#20304a", C["display_bg"], pad=2)
        self.display_outer = outer
        outer.pack(fill="x", pady=(8, 0))
        inner.configure(height=200)
        inner.pack_propagate(False)

        left = tk.Frame(inner, bg=C["display_bg"])
        left.pack(side="left", fill="both", expand=True, padx=14, pady=10)

        topline = tk.Frame(left, bg=C["display_bg"])
        topline.pack(anchor="w", fill="x")
        tk.Label(topline, text="DEG", bg=C["display_bg"], fg=C["green"],
                 font=(FONT, 10, "bold")).pack(side="left")
        self.deg_display = topline.winfo_children()[-1]
        self.mem_indicator = tk.Label(topline, text="", bg=C["display_bg"],
                                      fg=C["blue"], font=(FONT, 10, "bold"))
        self.mem_indicator.pack(side="left", padx=10)

        self.expr_lbl = tk.Label(left, text="", bg=C["display_bg"],
                                 fg=C["display_ink"], font=(FONT, 24),
                                 anchor="w", justify="left", wraplength=440)
        self.expr_lbl.pack(anchor="w", pady=(6, 0), fill="x")

        self.res_lbl = tk.Label(left, text="0", bg=C["display_bg"],
                                fg=C["display_ink"], font=(FONT, 40, "bold"),
                                anchor="e")
        self.res_lbl.pack(side="bottom", anchor="e", fill="x")

        # history column
        hist = tk.Frame(inner, bg=C["display_bg"], width=230)
        hist.pack(side="right", fill="y", padx=(0, 14), pady=10)
        hist.pack_propagate(False)
        tk.Label(hist, text="HISTORY", bg=C["display_bg"], fg=C["sub2"],
                 font=(FONT, 10, "bold")).pack(anchor="w")
        self.hist_box = tk.Frame(hist, bg=C["display_bg"])
        self.hist_box.pack(fill="both", expand=True, pady=(6, 0))
        self.sum_lbl = tk.Label(hist, text="", bg=C["display_bg"],
                                fg=C["hist_blue"], font=(FONT, 13),
                                anchor="w")
        self.sum_lbl.pack(side="bottom", anchor="w")
        self._hist_entries = []          # [(pretty_term, value_str), ...]

    def _render_history(self):
        for w in self.hist_box.winfo_children():
            w.destroy()
        for e, v in self._hist_entries[-5:]:
            row = tk.Frame(self.hist_box, bg=C["display_bg"])
            row.pack(fill="x", pady=2)
            tk.Label(row, text=e, bg=C["display_bg"], fg=C["display_ink"],
                     font=(FONT, 12), anchor="w").pack(side="left")
            tk.Label(row, text=f"=  {v}", bg=C["display_bg"], fg=C["display_ink"],
                     font=(FONT, 12), anchor="e").pack(side="right")

    # --------------------------------------------------------------- modebar
    def _build_modebar(self, parent):
        bar = tk.Frame(parent, bg=C["bg"])
        bar.pack(fill="x", pady=(8, 0))
        tk.Label(bar, text="MODE", bg=C["bg"], fg=C["sub"],
                 font=(FONT, 11, "bold")).pack(side="left", padx=(3, 12))
        self.mode = "CALC"
        self._mode_btns = {}
        for name in ("CALC", "MATRIX", "STAT", "GRAPH", "CONVERT", "SOLVE"):
            active = name == self.mode
            b = tk.Label(bar, text=name, bg=C["blue"] if active else C["btn"],
                         fg="white" if active else C["sub"],
                         font=(FONT, 11, "bold"), padx=16, pady=7,
                         cursor="hand2")
            b.pack(side="left", padx=4)
            b.bind("<Button-1>", lambda e, n=name: self._set_mode(n))
            self._mode_btns[name] = b

    def _set_mode(self, name):
        self.mode = name
        for n, b in self._mode_btns.items():
            on = n == name
            b.config(bg=C["blue"] if on else C["btn"],
                     fg="white" if on else C["sub"])

        # build views lazily
        if name == "MATRIX" and self.matrix_view is None:
            self.matrix_view = tk.Frame(self.body, bg=C["bg"])
            self._build_matrix_view(self.matrix_view)
            self._mode_views["MATRIX"] = self.matrix_view
        if name == "STAT" and getattr(self, "stat_view", None) is None:
            self.stat_view = tk.Frame(self.body, bg=C["bg"])
            self._build_stat_view(self.stat_view)
            self._mode_views["STAT"] = self.stat_view
        if name == "GRAPH" and getattr(self, "graph_view", None) is None:
            self.graph_view = tk.Frame(self.body, bg=C["bg"])
            self._build_graph_view(self.graph_view)
            self._mode_views["GRAPH"] = self.graph_view
        if name == "CONVERT" and getattr(self, "convert_view", None) is None:
            self.convert_view = tk.Frame(self.body, bg=C["bg"])
            self._build_convert_view(self.convert_view)
            self._mode_views["CONVERT"] = self.convert_view
        if name == "SOLVE" and getattr(self, "solve_view", None) is None:
            self.solve_view = tk.Frame(self.body, bg=C["bg"])
            self._build_solve_view(self.solve_view)
            self._mode_views["SOLVE"] = self.solve_view

        # hide every view, then show the selected one (fall back to CALC)
        for v in self._mode_views.values():
            v.pack_forget()
        view = self._mode_views.get(name)
        if view is None:                          # not-yet-implemented modes
            view = self._placeholder(name)
            self._mode_views[name] = view
        view.pack(fill="both", expand=True)

        # these modes carry their own result display, so hide the CALC LCD
        if name in ("MATRIX", "STAT", "GRAPH", "CONVERT", "SOLVE"):
            self.display_outer.pack_forget()
        else:
            self.display_outer.pack(fill="x", pady=(8, 0), before=self.body)

    def _placeholder(self, name):
        f = tk.Frame(self.body, bg=C["bg"], height=200)
        tk.Label(f, text=f"{name} mode \u2014 coming soon", bg=C["bg"],
                 fg=C["sub"], font=(FONT, 14)).pack(pady=40)
        return f

    # ===================================================================== #
    #  MATRIX MODE
    # ===================================================================== #
    def _build_matrix_view(self, root):
        self.mx_size = {"A": 3, "B": 3}
        self.mx_cells = {"A": [], "B": []}
        self.mx_last = None            # last result (matrix or scalar)
        self.mx_history = []           # [(label, result), ...]
        defaults = {"A": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    "B": [[2, 0, 1], [1, 2, 0], [3, 4, 1]]}

        # ---------- matrix result display (light LCD) ----------
        o, disp = bordered(root, "#20304a", C["display_bg"], pad=2)
        o.pack(fill="x", pady=(8, 0))
        disp.configure(height=190)
        disp.pack_propagate(False)

        dl = tk.Frame(disp, bg=C["display_bg"])
        dl.pack(side="left", fill="both", expand=True, padx=14, pady=10)
        tk.Label(dl, text="MATRIX", bg=C["display_bg"], fg=C["blue"],
                 font=(FONT, 11, "bold")).pack(anchor="w")
        tk.Label(dl, text="Operation", bg=C["display_bg"], fg=C["sub2"],
                 font=(FONT, 9)).pack(anchor="w", pady=(6, 0))
        self.mx_op_lbl = tk.Label(dl, text="\u2014", bg=C["display_bg"],
                                  fg=C["display_ink"], font=(FONT, 22, "bold"))
        self.mx_op_lbl.pack(anchor="w")
        tk.Label(dl, text="Result", bg=C["display_bg"], fg=C["sub2"],
                 font=(FONT, 9)).pack(anchor="w", pady=(6, 0))
        self.mx_result_frame = tk.Frame(dl, bg=C["display_bg"])
        self.mx_result_frame.pack(anchor="w", pady=(2, 0))

        dr = tk.Frame(disp, bg=C["display_bg"], width=250)
        dr.pack(side="right", fill="y", padx=(0, 14), pady=10)
        dr.pack_propagate(False)
        tk.Label(dr, text="HISTORY", bg=C["display_bg"], fg=C["sub2"],
                 font=(FONT, 10, "bold")).pack(anchor="w")
        self.mx_hist_frame = tk.Frame(dr, bg=C["display_bg"])
        self.mx_hist_frame.pack(fill="both", expand=True, pady=(6, 0))

        # ---------- editors + operations ----------
        row1 = tk.Frame(root, bg=C["bg"]); row1.pack(fill="x", pady=(10, 0))
        for i in range(3):
            row1.grid_columnconfigure(i, weight=1, uniform="mx")
        self._mx_editor(row1, "A", 0, C["blue_hi"], defaults["A"])
        self._mx_editor(row1, "B", 1, C["green"], defaults["B"])
        self._mx_ops_panel(row1, 2)

        # ---------- info + result history ----------
        row2 = tk.Frame(root, bg=C["bg"]); row2.pack(fill="x", pady=(10, 0))
        row2.grid_columnconfigure(0, weight=3, uniform="mx2")
        row2.grid_columnconfigure(1, weight=2, uniform="mx2")
        self._mx_info_panel(row2, 0)
        self._mx_reshist_panel(row2, 1)

        self._mx_refresh_info()

    def _mx_editor(self, parent, which, col, accent, default):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 6, 0))
        head = tk.Frame(p, bg=C["panel"]); head.pack(fill="x", padx=10, pady=(9, 4))
        tk.Label(head, text=f"MATRIX {which}", bg=C["panel"], fg=accent,
                 font=(FONT, 11, "bold")).pack(side="left")
        var = tk.StringVar(value=f"{self.mx_size[which]}\u00d7{self.mx_size[which]}")
        om = tk.OptionMenu(head, var, "2\u00d72", "3\u00d73", "4\u00d74",
                           command=lambda v, w=which: self._mx_resize(w, int(v[0])))
        om.config(bg=C["btn"], fg=C["white"], font=(FONT, 9), bd=0,
                  highlightthickness=0, activebackground=C["btn_hi"],
                  activeforeground=C["white"], width=4)
        om["menu"].config(bg=C["panel"], fg=C["white"])
        om.pack(side="right")
        self.mx_cells[which + "_var"] = var

        grid_holder = tk.Frame(p, bg=C["panel"]); grid_holder.pack(padx=10, pady=6)
        self.mx_cells[which + "_holder"] = grid_holder
        self._mx_fill_grid(which, default)

        btns = tk.Frame(p, bg=C["panel"]); btns.pack(fill="x", padx=8, pady=(4, 10))
        for label, cmd, col_ in (
                ("Identity", lambda w=which: self._mx_preset(w, "I"), C["btn"]),
                ("Zero",     lambda w=which: self._mx_preset(w, "0"), C["btn"]),
                ("Clear",    lambda w=which: self._mx_preset(w, "c"), C["red"])):
            Key(btns, label, command=cmd, bg=col_,
                hi=C["red_hi"] if col_ == C["red"] else C["btn_hi"],
                font_size=9, height=26).pack(side="left", expand=True,
                                             fill="x", padx=2)

    def _mx_fill_grid(self, which, values):
        holder = self.mx_cells[which + "_holder"]
        for w in holder.winfo_children():
            w.destroy()
        n = self.mx_size[which]
        cells = []
        for r in range(n):
            rowc = []
            for c in range(n):
                e = tk.Entry(holder, width=4, justify="center", bg=C["btn"],
                             fg=C["white"], font=(FONT_MONO, 12), relief="flat",
                             insertbackground=C["white"], highlightthickness=1,
                             highlightbackground=C["btn_edge"],
                             highlightcolor=C["blue_hi"])
                val = values[r][c] if r < len(values) and c < len(values[0]) else 0
                e.insert(0, nfmt(val))
                e.grid(row=r, column=c, padx=2, pady=2, ipady=3)
                rowc.append(e)
            cells.append(rowc)
        self.mx_cells[which] = cells

    def _mx_resize(self, which, n):
        old = self._mx_read(which, silent=True) or []
        self.mx_size[which] = n
        self._mx_fill_grid(which, old)

    def _mx_preset(self, which, kind):
        n = self.mx_size[which]
        if kind == "I":
            vals = m_identity(n)
        elif kind == "0":
            vals = [[0] * n for _ in range(n)]
        else:                       # clear -> blanks
            holder = self.mx_cells[which + "_holder"]
            for w in holder.winfo_children():
                w.destroy()
            self._mx_fill_grid(which, [[0] * n for _ in range(n)])
            for rowc in self.mx_cells[which]:
                for e in rowc:
                    e.delete(0, "end")
            return
        self._mx_fill_grid(which, vals)

    def _mx_read(self, which, silent=False):
        cells = self.mx_cells[which]
        try:
            return [[float(e.get() or 0) for e in row] for row in cells]
        except ValueError:
            if silent:
                return None
            raise ValueError(f"Matrix {which} has a non-numeric entry")

    OPS = [("+",  "Add (A + B)",          "Add"),
           ("\u2212", "Subtract (A \u2212 B)", "Subtract"),
           ("x\u02e3", "Multiply (A \u00d7 B)", "Multiply"),
           ("T\u1d40", "Transpose (A\u1d40)",   "Transpose"),
           ("A\u207b\u00b9", "Inverse (A\u207b\u00b9)", "Inverse"),
           ("|A|", "Determinant (|A|)",   "Determinant"),
           ("adj", "Adjoint (adj A)",     "Adjoint"),
           ("rk",  "Rank (rank A)",       "Rank"),
           ("tr",  "Trace (tr A)",        "Trace"),
           ("kA",  "Scalar Multiply (kA)", "Scalar"),
           ("A\u207f", "Power (A\u207f)",  "Power"),
           ("rref", "RREF (Row Echelon)", "RREF")]

    def _mx_ops_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(6, 0))
        tk.Label(p, text="MATRIX OPERATIONS", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=10, pady=(9, 6))
        grid = tk.Frame(p, bg=C["panel"]); grid.pack(fill="both", expand=True,
                                                     padx=8, pady=(0, 10))
        for i in range(2):
            grid.grid_columnconfigure(i, weight=1, uniform="ops")
        for idx, (badge, label, op) in enumerate(self.OPS):
            r, c = divmod(idx, 2)
            b = tk.Frame(grid, bg=C["btn"], cursor="hand2", highlightthickness=1,
                         highlightbackground=C["btn_edge"])
            b.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
            tk.Label(b, text=badge, bg=C["btn"], fg=C["blue_hi"],
                     font=(FONT, 9, "bold"), width=4).pack(side="left", padx=(6, 2), pady=6)
            tk.Label(b, text=label, bg=C["btn"], fg=C["white"],
                     font=(FONT, 9), anchor="w").pack(side="left")
            for w in (b, *b.winfo_children()):
                w.bind("<Button-1>", lambda e, o_=op: self._mx_operate(o_))
                w.bind("<Enter>", lambda e, bb=b: [x.config(bg=C["btn_hi"]) for x in (bb, *bb.winfo_children())])
                w.bind("<Leave>", lambda e, bb=b: [x.config(bg=C["btn"]) for x in (bb, *bb.winfo_children())])

    def _mx_info_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(0, 6))
        tk.Label(p, text="MATRIX INFORMATION", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        body = tk.Frame(p, bg=C["panel"]); body.pack(fill="x", padx=12, pady=(0, 12))
        self.mx_info = {}
        fields = ["Size", "Determinant", "Rank", "Trace", "Last Operation", "Status"]
        for i, f in enumerate(fields):
            tk.Label(body, text=f + " :", bg=C["panel"], fg=C["sub"],
                     font=(FONT, 10)).grid(row=i, column=0, sticky="w", pady=2)
            v = tk.Label(body, text="\u2014", bg=C["panel"], fg=C["white"],
                         font=(FONT, 10))
            v.grid(row=i, column=1, sticky="w", padx=14, pady=2)
            self.mx_info[f] = v

    def _mx_reshist_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(6, 0))
        head = tk.Frame(p, bg=C["panel"]); head.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(head, text="RESULT HISTORY", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(side="left")
        tk.Label(head, text="Clear", bg=C["panel"], fg=C["sub"],
                 font=(FONT, 9), cursor="hand2").pack(side="right")
        self.mx_reshist = tk.Frame(p, bg=C["panel"])
        self.mx_reshist.pack(fill="both", expand=True, padx=8, pady=(0, 10))

    # ---------- operation dispatch ----------
    def _mx_operate(self, op):
        try:
            A = self._mx_read("A")
            need_b = op in ("Add", "Subtract", "Multiply")
            B = self._mx_read("B") if need_b else None
            label, result = op, None

            if op == "Add":            label, result = "A + B", m_add(A, B)
            elif op == "Subtract":     label, result = "A \u2212 B", m_sub(A, B)
            elif op == "Multiply":     label, result = "A \u00d7 B", m_mul(A, B)
            elif op == "Transpose":    label, result = "A\u1d40", m_transpose(A)
            elif op == "Inverse":      label, result = "A\u207b\u00b9", m_inverse(A)
            elif op == "Adjoint":      label, result = "adj(A)", m_adjoint(A)
            elif op == "RREF":         label, result = "rref(A)", m_rref(A)
            elif op == "Determinant":  label, result = "det(A)", m_det(A)
            elif op == "Rank":         label, result = "rank(A)", m_rank(A)
            elif op == "Trace":        label, result = "tr(A)", m_trace(A)
            elif op == "Scalar":
                from tkinter import simpledialog
                k = simpledialog.askfloat("Scalar Multiply", "k =", parent=self)
                if k is None:
                    return
                label, result = f"{nfmt(k)}\u00b7A", m_scalar(A, k)
            elif op == "Power":
                from tkinter import simpledialog
                n = simpledialog.askinteger("Matrix Power", "n =", parent=self)
                if n is None:
                    return
                label, result = f"A^{n}", m_power(A, n)
            else:
                raise ValueError(f"{op}: not available")

            self.mx_last = result
            self.mx_op_lbl.config(text=label)
            self._mx_render_result(self.mx_result_frame, result, C["display_ink"],
                                   C["display_bg"], big=True)
            self.mx_history.insert(0, (label, result))
            self._mx_render_history()
            self._mx_render_reshist()
            self._mx_refresh_info(last=label, status="Success",
                                  status_col=C["green"])
        except Exception as exc:
            self.mx_op_lbl.config(text=op)
            for w in self.mx_result_frame.winfo_children():
                w.destroy()
            tk.Label(self.mx_result_frame, text=str(exc), bg=C["display_bg"],
                     fg="#c0392b", font=(FONT, 12)).pack(anchor="w")
            self._mx_refresh_info(last=op, status=str(exc), status_col="#e07a7a")

    # ---------- rendering helpers ----------
    def _mx_render_result(self, frame, result, ink, bg, big=False):
        for w in frame.winfo_children():
            w.destroy()
        if not isinstance(result, list):                 # scalar
            tk.Label(frame, text=nfmt(result), bg=bg, fg=ink,
                     font=(FONT, 30 if big else 12, "bold")).pack(anchor="w")
            return
        self._draw_bracket_matrix(frame, result, ink, bg,
                                  fsize=18 if big else 10)

    def _draw_bracket_matrix(self, frame, M, ink, bg, fsize=12):
        grid = tk.Frame(frame, bg=bg); grid.pack(anchor="w")
        tk.Label(grid, text="[", bg=bg, fg=ink,
                 font=(FONT, fsize + len(M) * 6)).grid(row=0, column=0, rowspan=len(M))
        for r, row in enumerate(M):
            for c, val in enumerate(row):
                tk.Label(grid, text=nfmt(val), bg=bg, fg=ink,
                         font=(FONT_MONO, fsize), width=max(3, len(nfmt(val)) + 1),
                         anchor="e").grid(row=r, column=c + 1, padx=3, pady=1)
        tk.Label(grid, text="]", bg=bg, fg=ink,
                 font=(FONT, fsize + len(M) * 6)).grid(row=0, column=len(M[0]) + 1,
                                                       rowspan=len(M))

    def _mx_render_history(self):
        for w in self.mx_hist_frame.winfo_children():
            w.destroy()
        # show A, B, then the two most recent results
        rows = [("A", self._mx_read("A", silent=True)),
                ("B", self._mx_read("B", silent=True))]
        for lab, res in self.mx_history[:2]:
            rows.append((lab, res))
        for lab, res in rows[:4]:
            if res is None:
                continue
            r = tk.Frame(self.mx_hist_frame, bg=C["display_bg"])
            r.pack(fill="x", anchor="w", pady=1)
            tk.Label(r, text=f"{lab} =", bg=C["display_bg"], fg=C["display_ink"],
                     font=(FONT, 10)).pack(side="left", padx=(0, 4))
            self._mx_render_result(r, res, C["display_ink"], C["display_bg"])

    def _mx_render_reshist(self):
        for w in self.mx_reshist.winfo_children():
            w.destroy()
        for lab, res in self.mx_history[:6]:
            r = tk.Frame(self.mx_reshist, bg=C["panel"])
            r.pack(fill="x", pady=3)
            tk.Label(r, text="\u25cf", bg=C["panel"], fg=C["green"],
                     font=(FONT, 8)).pack(side="left", padx=(4, 8))
            tk.Label(r, text=lab, bg=C["panel"], fg=C["white"], font=(FONT, 10),
                     width=10, anchor="w").pack(side="left")
            box = tk.Frame(r, bg=C["panel"]); box.pack(side="left")
            self._mx_render_result(box, res, C["white"], C["panel"])

    def _mx_refresh_info(self, last="\u2014", status="Ready", status_col=None):
        A = self._mx_read("A", silent=True)
        size = f"{self.mx_size['A']} \u00d7 {self.mx_size['A']}"
        det = rank = trace = "\u2014"
        if A:
            try:
                det = nfmt(m_det(A)); trace = nfmt(m_trace(A))
            except Exception:
                det = trace = "n/a"
            try:
                rank = str(m_rank(A))
            except Exception:
                rank = "n/a"
        info = getattr(self, "mx_info", None)
        if not info:
            return
        info["Size"].config(text=size)
        info["Determinant"].config(text=det)
        info["Rank"].config(text=rank)
        info["Trace"].config(text=trace)
        info["Last Operation"].config(text=last)
        info["Status"].config(text=status, fg=status_col or C["white"])

    # ===================================================================== #
    #  STAT MODE
    # ===================================================================== #
    def _build_stat_view(self, root):
        self.stat_data = [[10, 1], [12, 2], [14, 3], [16, 2],
                          [18, 1], [20, 1], [22, 1], [24, 1]]
        self.stat_entry_rows = []
        self.stat_selected = "Mean"
        self.stat_chart_kind = "Bar"
        self.stat_cards = {}
        self.stat_summary = {}

        # ---------- display (light LCD) ----------
        o, disp = bordered(root, "#20304a", C["display_bg"], pad=2)
        o.pack(fill="x", pady=(8, 0))
        disp.configure(height=190)
        disp.pack_propagate(False)

        dl = tk.Frame(disp, bg=C["display_bg"])
        dl.pack(side="left", fill="both", expand=True, padx=14, pady=10)
        tk.Label(dl, text="STATISTICS", bg=C["display_bg"], fg=C["blue"],
                 font=(FONT, 11, "bold")).pack(anchor="w")
        drow = tk.Frame(dl, bg=C["display_bg"]); drow.pack(anchor="w", pady=(4, 0))
        tk.Label(drow, text="Data Set :", bg=C["display_bg"], fg=C["sub2"],
                 font=(FONT, 9)).pack(side="left")
        tk.Label(drow, text="Data 1 \u25be", bg=C["display_bg"], fg=C["display_ink"],
                 font=(FONT, 9)).pack(side="left", padx=6)
        self.stat_sel_lbl = tk.Label(dl, text="Mean", bg=C["display_bg"],
                                     fg=C["display_ink"], font=(FONT, 22))
        self.stat_sel_lbl.pack(anchor="w", pady=(4, 0))
        self.stat_val_lbl = tk.Label(dl, text="0", bg=C["display_bg"],
                                     fg=C["display_ink"], font=(FONT, 40, "bold"),
                                     anchor="e")
        self.stat_val_lbl.pack(side="bottom", anchor="e", fill="x")
        flags = tk.Frame(dl, bg=C["display_bg"]); flags.pack(side="bottom", anchor="w")
        for t, col in (("NORM", C["sub2"]), ("MATH", C["sub2"]), ("STAT", C["blue"])):
            tk.Label(flags, text=t, bg=C["display_bg"], fg=col,
                     font=(FONT, 9, "bold")).pack(side="left", padx=(0, 14))

        dr = tk.Frame(disp, bg=C["display_bg"], width=260)
        dr.pack(side="right", fill="y", padx=(0, 14), pady=10)
        dr.pack_propagate(False)
        tk.Label(dr, text="SUMMARY", bg=C["display_bg"], fg=C["sub2"],
                 font=(FONT, 10, "bold")).pack(anchor="w")
        self.stat_summary_frame = tk.Frame(dr, bg=C["display_bg"])
        self.stat_summary_frame.pack(fill="both", expand=True, pady=(6, 0))

        # ---------- data set | results | tools ----------
        row1 = tk.Frame(root, bg=C["bg"]); row1.pack(fill="x", pady=(10, 0))
        row1.grid_columnconfigure(0, weight=2, uniform="s1")
        row1.grid_columnconfigure(1, weight=3, uniform="s1")
        row1.grid_columnconfigure(2, weight=2, uniform="s1")
        self._stat_dataset_panel(row1, 0)
        self._stat_results_panel(row1, 1)
        self._stat_tools_panel(row1, 2)

        # ---------- visualization ----------
        row2 = tk.Frame(root, bg=C["bg"]); row2.pack(fill="x", pady=(10, 0))
        self._stat_viz_panel(row2)

        # ---------- summary | frequency dist | preview ----------
        row3 = tk.Frame(root, bg=C["bg"]); row3.pack(fill="x", pady=(10, 0))
        for i in range(3):
            row3.grid_columnconfigure(i, weight=1, uniform="s3")
        self._stat_summary_panel(row3, 0)
        self._stat_freq_panel(row3, 1)
        self._stat_preview_panel(row3, 2)

        self._stat_build_table()
        self._stat_recompute()

    # ---- data set editable table ----
    def _stat_dataset_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(0, 6))
        head = tk.Frame(p, bg=C["panel"]); head.pack(fill="x", padx=10, pady=(9, 4))
        tk.Label(head, text="DATA SET", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(side="left")
        tk.Label(head, text="Data 1 \u25be", bg=C["panel"], fg=C["sub"],
                 font=(FONT, 9)).pack(side="right")

        hdr = tk.Frame(p, bg=C["panel"]); hdr.pack(fill="x", padx=10)
        for t, wdt in (("No.", 4), ("x", 8), ("f", 6)):
            tk.Label(hdr, text=t, bg=C["panel"], fg=C["sub"], font=(FONT, 9, "bold"),
                     width=wdt, anchor="w").pack(side="left", padx=2)
        self.stat_rows_frame = tk.Frame(p, bg=C["panel"])
        self.stat_rows_frame.pack(fill="x", padx=10, pady=(2, 4))

        btns = tk.Frame(p, bg=C["panel"]); btns.pack(fill="x", padx=8, pady=(4, 10))
        Key(btns, "+ Add Row", command=self._stat_add_row, bg=C["btn"],
            hi=C["btn_hi"], font_size=9, height=26).pack(side="left", expand=True,
                                                         fill="x", padx=2)
        Key(btns, "Delete", command=self._stat_del_row, bg=C["red"],
            hi=C["red_hi"], font_size=9, height=26).pack(side="left", expand=True,
                                                         fill="x", padx=2)

    def _stat_build_table(self):
        for w in self.stat_rows_frame.winfo_children():
            w.destroy()
        self.stat_entry_rows = []
        for i, (x, f) in enumerate(self.stat_data, 1):
            r = tk.Frame(self.stat_rows_frame, bg=C["panel"])
            r.pack(fill="x", pady=1)
            tk.Label(r, text=str(i), bg=C["panel"], fg=C["sub"], font=(FONT, 9),
                     width=4, anchor="w").pack(side="left", padx=2)
            ex = self._stat_cell(r, x)
            ef = self._stat_cell(r, f)
            self.stat_entry_rows.append((ex, ef))

    def _stat_cell(self, parent, value):
        e = tk.Entry(parent, width=7, justify="center", bg=C["btn"], fg=C["white"],
                     font=(FONT_MONO, 10), relief="flat", insertbackground=C["white"],
                     highlightthickness=1, highlightbackground=C["btn_edge"],
                     highlightcolor=C["blue_hi"])
        e.insert(0, nfmt(value))
        e.pack(side="left", padx=2, ipady=2)
        e.bind("<KeyRelease>", lambda ev: self._stat_recompute())
        return e

    def _stat_read(self):
        data = []
        for ex, ef in self.stat_entry_rows:
            try:
                x = float(ex.get()); f = float(ef.get() or 0)
            except ValueError:
                continue
            if f > 0:
                data.append((x, f))
        return data

    def _stat_add_row(self):
        self.stat_data = self._stat_read() + [[0, 1]]
        self._stat_build_table(); self._stat_recompute()

    def _stat_del_row(self):
        d = self._stat_read()
        if d:
            self.stat_data = d[:-1]
            self._stat_build_table(); self._stat_recompute()

    # ---- results cards ----
    _RESULT_CARDS = [("Mean", "x\u0304"), ("Median", "M"), ("Mode", "Mo"),
                     ("Std Dev", "\u03c3"), ("Variance", "\u03c3\u00b2"),
                     ("Min", "Min"), ("Max", "Max"), ("Range", "R"),
                     ("\u03a3x", "\u03a3x"), ("\u03a3x\u00b2", "\u03a3x\u00b2"),
                     ("n (Count)", "n")]

    def _stat_results_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=6)
        tk.Label(p, text="STATISTICS RESULTS", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=10, pady=(9, 6))
        grid = tk.Frame(p, bg=C["panel"]); grid.pack(fill="both", expand=True,
                                                     padx=8, pady=(0, 8))
        for i in range(2):
            grid.grid_columnconfigure(i, weight=1, uniform="cards")
        for idx, (name, badge) in enumerate(self._RESULT_CARDS):
            r, c = divmod(idx, 2)
            card = tk.Frame(grid, bg=C["btn"], cursor="hand2", highlightthickness=1,
                            highlightbackground=C["btn_edge"])
            card.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
            top = tk.Frame(card, bg=C["btn"]); top.pack(fill="x", padx=8, pady=(5, 0))
            tk.Label(top, text=name, bg=C["btn"], fg=C["sub"],
                     font=(FONT, 9)).pack(side="left")
            tk.Label(top, text=badge, bg=C["btn"], fg=C["blue_hi"],
                     font=(FONT, 9, "bold")).pack(side="right")
            val = tk.Label(card, text="0", bg=C["btn"], fg=C["white"],
                           font=(FONT, 15, "bold"), anchor="w")
            val.pack(anchor="w", padx=8, pady=(0, 6))
            self.stat_cards[name] = val
            for w in (card, top, *top.winfo_children(), val):
                w.bind("<Button-1>", lambda e, n=name: self._stat_select(n))
        Key(p, "More Statistics \u25be", command=lambda: None, bg=C["btn"],
            hi=C["btn_hi"], font_size=9, height=26).pack(fill="x", padx=10, pady=(0, 10))

    def _stat_select(self, name):
        self.stat_selected = name
        self._stat_recompute()

    # ---- stat tools ----
    _STAT_TOOLS = ["1-Variable Statistics", "2-Variable Statistics",
                   "Regression (Linear)", "Regression (Polynomial)",
                   "Correlation", "Moving Average", "Frequency Distribution",
                   "Z-Score", "Probability", "Hypothesis Testing"]

    def _stat_tools_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(6, 0))
        tk.Label(p, text="STAT TOOLS", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=10, pady=(9, 6))
        for i, name in enumerate(self._STAT_TOOLS):
            active = i == 0
            row = tk.Frame(p, bg=C["btn"] if active else C["panel"], cursor="hand2",
                           highlightthickness=1,
                           highlightbackground=C["blue_hi"] if active else C["btn_edge"])
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=name, bg=row["bg"],
                     fg=C["white"] if active else C["sub"],
                     font=(FONT, 10)).pack(side="left", padx=8, pady=6)
            tk.Label(row, text="\u203a", bg=row["bg"], fg=C["sub2"],
                     font=(FONT, 12)).pack(side="right", padx=8)

    # ---- visualization ----
    def _stat_viz_panel(self, parent):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.pack(fill="x")
        tk.Label(p, text="DATA VISUALIZATION", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        self.stat_canvas = tk.Canvas(p, bg=C["panel"], height=220,
                                     highlightthickness=0)
        self.stat_canvas.pack(fill="x", padx=12, pady=4)
        self.stat_canvas.bind("<Configure>", lambda e: self._stat_draw_chart())
        btns = tk.Frame(p, bg=C["panel"]); btns.pack(fill="x", padx=12, pady=(2, 12))
        self.stat_chart_btns = {}
        for kind in ("Bar Chart", "Pie Chart", "Histogram", "Ogive"):
            k = kind.split()[0]
            b = Key(btns, kind, command=lambda kk=k: self._stat_set_chart(kk),
                    bg=C["btn"], hi=C["btn_hi"], font_size=10, height=30)
            b.pack(side="left", expand=True, fill="x", padx=3)
            self.stat_chart_btns[k] = b

    def _stat_set_chart(self, kind):
        self.stat_chart_kind = kind
        self._stat_draw_chart()

    # ---- bottom summary/freq/preview ----
    def _stat_summary_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(0, 6))
        tk.Label(p, text="DATA SUMMARY", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        body = tk.Frame(p, bg=C["panel"]); body.pack(fill="x", padx=12, pady=(0, 12))
        self.stat_dsum = {}
        for i, f in enumerate(["Data Set Name", "Variable", "Type",
                               "Count (n)", "Missing Values"]):
            tk.Label(body, text=f + " :", bg=C["panel"], fg=C["sub"],
                     font=(FONT, 10)).grid(row=i, column=0, sticky="w", pady=2)
            v = tk.Label(body, text="\u2014", bg=C["panel"], fg=C["white"],
                         font=(FONT, 10))
            v.grid(row=i, column=1, sticky="w", padx=12, pady=2)
            self.stat_dsum[f] = v

    def _stat_freq_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=6)
        tk.Label(p, text="FREQUENCY DISTRIBUTION", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        self.stat_freq_frame = tk.Frame(p, bg=C["panel"])
        self.stat_freq_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def _stat_preview_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(6, 0))
        tk.Label(p, text="DATA PREVIEW", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        self.stat_preview_lbl = tk.Label(p, text="", bg=C["panel"], fg=C["white"],
                                         font=(FONT_MONO, 10), wraplength=240,
                                         justify="left", anchor="w")
        self.stat_preview_lbl.pack(anchor="w", padx=12)
        self.stat_preview_meta = tk.Label(p, text="", bg=C["panel"], fg=C["sub"],
                                          font=(FONT, 10))
        self.stat_preview_meta.pack(anchor="w", padx=12, pady=(8, 12))

    # ---- the statistics engine ----
    def _stat_stats(self, data):
        expanded = []
        for x, f in data:
            expanded.extend([x] * int(round(f)))
        expanded.sort()
        n = len(expanded)
        if n == 0:
            return None
        sx = sum(expanded)
        sx2 = sum(v * v for v in expanded)
        mean = sx / n
        if n % 2:
            median = expanded[n // 2]
        else:
            median = (expanded[n // 2 - 1] + expanded[n // 2]) / 2
        # mode = x with highest frequency
        fmax = max(f for _, f in data)
        modes = [x for x, f in data if f == fmax]
        mode = modes[0]
        var = sx2 / n - mean * mean
        var = max(var, 0.0)
        std = math.sqrt(var)
        return {"n (Count)": n, "\u03a3x": sx, "\u03a3x\u00b2": sx2,
                "Mean": mean, "Median": median, "Mode": mode,
                "Std Dev": std, "Variance": var,
                "Min": min(expanded), "Max": max(expanded),
                "Range": max(expanded) - min(expanded)}

    def _stat_recompute(self):
        data = self._stat_read()
        self.stat_data = [list(d) for d in data]
        s = self._stat_stats(data)
        if s is None:
            return

        # cards
        for name, w in self.stat_cards.items():
            w.config(text=nfmt(s[name]))
        # big selected value
        self.stat_sel_lbl.config(text=self.stat_selected)
        self.stat_val_lbl.config(text=nfmt(s.get(self.stat_selected, 0)))

        # summary (right of display)
        for w in self.stat_summary_frame.winfo_children():
            w.destroy()
        for name, key in (("n", "n (Count)"), ("\u03a3x", "\u03a3x"),
                          ("Mean", "Mean"), ("Median", "Median"), ("Mode", "Mode"),
                          ("Std Dev", "Std Dev"), ("Variance", "Variance")):
            r = tk.Frame(self.stat_summary_frame, bg=C["display_bg"])
            r.pack(fill="x")
            tk.Label(r, text=name, bg=C["display_bg"], fg=C["display_ink"],
                     font=(FONT, 10), width=9, anchor="w").pack(side="left")
            tk.Label(r, text="= " + nfmt(s[key]), bg=C["display_bg"],
                     fg=C["display_ink"], font=(FONT, 10)).pack(side="left")

        # data summary panel
        self.stat_dsum["Data Set Name"].config(text="Data 1")
        self.stat_dsum["Variable"].config(text="x")
        self.stat_dsum["Type"].config(text="Frequency Table")
        self.stat_dsum["Count (n)"].config(text=str(s["n (Count)"]))
        self.stat_dsum["Missing Values"].config(text="0")

        # frequency distribution table (x, f, cf, rf%)
        for w in self.stat_freq_frame.winfo_children():
            w.destroy()
        total = s["n (Count)"]
        header = tk.Frame(self.stat_freq_frame, bg=C["panel"]); header.pack(fill="x")
        for t in ("x", "f", "cf", "rf (%)"):
            tk.Label(header, text=t, bg=C["panel"], fg=C["sub"], font=(FONT, 9, "bold"),
                     width=7, anchor="w").pack(side="left")
        cf = 0
        for x, f in sorted(data):
            cf += f
            row = tk.Frame(self.stat_freq_frame, bg=C["panel"]); row.pack(fill="x")
            for val in (nfmt(x), nfmt(f), nfmt(cf), f"{f/total*100:.2f}"):
                tk.Label(row, text=val, bg=C["panel"], fg=C["white"], font=(FONT, 9),
                         width=7, anchor="w").pack(side="left")
        trow = tk.Frame(self.stat_freq_frame, bg=C["panel"]); trow.pack(fill="x", pady=(2, 0))
        for val in ("Total", nfmt(total), "-", "100.00"):
            tk.Label(trow, text=val, bg=C["panel"], fg=C["green"], font=(FONT, 9, "bold"),
                     width=7, anchor="w").pack(side="left")

        # preview
        expanded = []
        for x, f in sorted(data):
            expanded.extend([x] * int(round(f)))
        shown = ", ".join(nfmt(v) for v in expanded[:10])
        if len(expanded) > 10:
            shown += ", ..."
        self.stat_preview_lbl.config(text=shown)
        self.stat_preview_meta.config(
            text=f"Count : {s['n (Count)']}     Min : {nfmt(s['Min'])}"
                 f"     Max : {nfmt(s['Max'])}")

        self._stat_draw_chart()

    # ---- chart rendering ----
    def _stat_draw_chart(self):
        cv = getattr(self, "stat_canvas", None)
        if cv is None:
            return
        cv.delete("all")
        data = sorted(self._stat_read())
        if not data:
            return
        w = cv.winfo_width() or 600
        h = int(cv["height"])
        pl, pr, pt, pb = 34, 16, 16, 26
        pw, ph = w - pl - pr, h - pt - pb
        xs = [d[0] for d in data]
        fs = [d[1] for d in data]
        fmax = max(fs) or 1
        kind = self.stat_chart_kind

        def X(i, n):
            return pl + pw * (i + 0.5) / n
        def Y(v):
            return pt + ph * (1 - v / fmax)

        # axes
        cv.create_line(pl, pt, pl, pt + ph, fill=C["sub2"])
        cv.create_line(pl, pt + ph, pl + pw, pt + ph, fill=C["sub2"])
        for gy in range(int(fmax) + 1):
            y = Y(gy)
            cv.create_text(pl - 8, y, text=str(gy), fill=C["sub"], font=(FONT, 8))
            cv.create_line(pl, y, pl + pw, y, fill="#1a2233")

        n = len(data)
        if kind == "Pie":
            self._stat_pie(cv, data, w, h)
            return
        if kind == "Ogive":
            cf = 0; pts = []
            total = sum(fs)
            for i, (x, f) in enumerate(data):
                cf += f
                px = X(i, n); py = pt + ph * (1 - cf / total)
                pts += [px, py]
                cv.create_oval(px - 3, py - 3, px + 3, py + 3, fill=C["blue_hi"],
                               outline="")
                cv.create_text(px, pt + ph + 12, text=nfmt(x), fill=C["sub"],
                               font=(FONT, 8))
            if len(pts) >= 4:
                cv.create_line(*pts, fill="#e08a3c", width=2)
            return

        # Bar / Histogram
        bw = pw / n * (0.98 if kind == "Histogram" else 0.55)
        for i, (x, f) in enumerate(data):
            cx = X(i, n)
            cv.create_rectangle(cx - bw / 2, Y(f), cx + bw / 2, pt + ph,
                                fill=C["blue"], outline=C["blue_hi"])
            cv.create_text(cx, pt + ph + 12, text=nfmt(x), fill=C["sub"],
                           font=(FONT, 8))
        if kind == "Bar":       # frequency polygon overlay
            pts = []
            for i, (x, f) in enumerate(data):
                pts += [X(i, n), Y(f)]
            if len(pts) >= 4:
                cv.create_line(*pts, fill="#e08a3c", width=2)
                for i in range(0, len(pts), 2):
                    cv.create_oval(pts[i] - 3, pts[i + 1] - 3, pts[i] + 3,
                                   pts[i + 1] + 3, fill="#e08a3c", outline="")

    def _stat_pie(self, cv, data, w, h):
        total = sum(f for _, f in data)
        cx, cy, rad = w / 2, h / 2, min(w, h) / 2 - 24
        start = 0.0
        palette = ["#2563eb", "#22c55e", "#e08a3c", "#a855f7", "#ef4444",
                   "#14b8a6", "#eab308", "#ec4899"]
        for i, (x, f) in enumerate(data):
            extent = 360 * f / total
            cv.create_arc(cx - rad, cy - rad, cx + rad, cy + rad, start=start,
                          extent=extent, fill=palette[i % len(palette)],
                          outline=C["panel"])
            start += extent
    # ===================================================================== #
    #  GRAPH MODE
    # ===================================================================== #
    GRAPH_COLORS = ["#3b82f6", "#22c55e", "#ef4444", "#e08a3c", "#a855f7"]

    def _to_eval(self, s):
        """Like _to_python but also inserts implicit multiplication so
        expressions like '2x^3', '4x', '2sin(x)' evaluate correctly."""
        expr = self._to_python(str(s))
        expr = re.sub(r"(\d)([A-Za-z(])", r"\1*\2", expr)   # 2x -> 2*x, 2( -> 2*(
        expr = re.sub(r"(\))([A-Za-z(0-9])", r"\1*\2", expr)  # )( -> )*(, )x -> )*x
        return expr

    def _plot_ns(self):
        """Radian namespace for plotting f(x)."""
        return {
            "__builtins__": {}, "pi": math.pi, "e": math.e, "tau": math.tau,
            "sqrt": math.sqrt, "cbrt": lambda v: math.copysign(abs(v) ** (1 / 3), v),
            "abs": abs, "exp": math.exp,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
            "log": math.log10, "ln": math.log,
            "csc": lambda v: 1 / math.sin(v), "sec": lambda v: 1 / math.cos(v),
            "cot": lambda v: 1 / math.tan(v),
        }

    def _eval_num(self, s, default=0.0):
        """Evaluate a window-setting expression like '-2π' or 'π/2'."""
        try:
            expr = self._to_python(str(s))
            # implicit multiplication: 2pi -> 2*pi, 2( -> 2*(, )( -> )*(
            expr = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", expr)
            expr = re.sub(r"(\))(\()", r"\1*\2", expr)
            return float(eval(expr, self._plot_ns(), {}))
        except Exception:
            return default

    def _build_graph_view(self, root):
        self.graph_eqs = [
            {"expr": "sin(x)", "on": True},
            {"expr": "cos(x)", "on": True},
            {"expr": "x^2-4",  "on": True},
            {"expr": "e^x",    "on": True},
            {"expr": "",       "on": False},
        ]
        self.graph_win = {"Xmin": "-2\u03c0", "Xmax": "2\u03c0", "Xscl": "\u03c0/2",
                          "Ymin": "-5", "Ymax": "5", "Yscl": "1"}
        self.graph_trace_x = math.pi / 3          # 1.0472
        self.graph_trace_on = True

        # ---------- display (light LCD): equations | plot | value ----------
        o, disp = bordered(root, "#20304a", C["display_bg"], pad=2)
        o.pack(fill="x", pady=(8, 0))
        disp.configure(height=250)
        disp.pack_propagate(False)

        dleft = tk.Frame(disp, bg=C["display_bg"], width=220)
        dleft.pack(side="left", fill="y", padx=12, pady=10)
        dleft.pack_propagate(False)
        tk.Label(dleft, text="GRAPH", bg=C["display_bg"], fg=C["blue"],
                 font=(FONT, 11, "bold")).pack(anchor="w")
        self.graph_disp_eqs = tk.Frame(dleft, bg=C["display_bg"])
        self.graph_disp_eqs.pack(anchor="w", fill="x", pady=(6, 0))
        flags = tk.Frame(dleft, bg=C["display_bg"]); flags.pack(side="bottom", anchor="w")
        for t, col in (("NORM", C["sub2"]), ("MATH", C["sub2"]),
                       ("FRAC", C["sub2"]), ("GRAPH", C["blue"])):
            tk.Label(flags, text=t, bg=C["display_bg"], fg=col,
                     font=(FONT, 8, "bold")).pack(side="left", padx=(0, 8))

        self.graph_disp_canvas = tk.Canvas(disp, bg="#f4f7f9", highlightthickness=0)
        self.graph_disp_canvas.pack(side="left", fill="both", expand=True,
                                    padx=6, pady=10)
        self.graph_disp_canvas.bind(
            "<Configure>", lambda e: self._graph_draw(self.graph_disp_canvas, light=True))

        dval = tk.Frame(disp, bg=C["display_bg"], width=170)
        dval.pack(side="right", fill="y", padx=12, pady=10)
        dval.pack_propagate(False)
        tk.Label(dval, text="VALUE", bg=C["display_bg"], fg=C["sub2"],
                 font=(FONT, 10, "bold")).pack(anchor="w")
        self.graph_value_frame = tk.Frame(dval, bg=C["display_bg"])
        self.graph_value_frame.pack(anchor="w", fill="x", pady=(6, 0))

        # ---------- equations | window | tools ----------
        row1 = tk.Frame(root, bg=C["bg"]); row1.pack(fill="x", pady=(10, 0))
        row1.grid_columnconfigure(0, weight=2, uniform="g1")
        row1.grid_columnconfigure(1, weight=3, uniform="g1")
        row1.grid_columnconfigure(2, weight=2, uniform="g1")
        self._graph_eq_panel(row1, 0)
        self._graph_window_panel(row1, 1)
        self._graph_tools_panel(row1, 2)

        # ---------- window settings ----------
        row2 = tk.Frame(root, bg=C["bg"]); row2.pack(fill="x", pady=(10, 0))
        self._graph_settings_panel(row2)

        # ---------- table | analysis | info ----------
        row3 = tk.Frame(root, bg=C["bg"]); row3.pack(fill="x", pady=(10, 0))
        for i in range(3):
            row3.grid_columnconfigure(i, weight=1, uniform="g3")
        self._graph_table_panel(row3, 0)
        self._graph_analysis_panel(row3, 1)
        self._graph_info_panel(row3, 2)

        self._graph_render_eq_rows()
        self._graph_refresh()

    # ---- equations editor ----
    def _graph_eq_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(0, 6))
        tk.Label(p, text="EQUATIONS", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=10, pady=(9, 6))
        self.graph_eq_rows = tk.Frame(p, bg=C["panel"])
        self.graph_eq_rows.pack(fill="x", padx=8)
        btns = tk.Frame(p, bg=C["panel"]); btns.pack(fill="x", padx=8, pady=(8, 10))
        Key(btns, "+ Add Equation", command=self._graph_add_eq, bg=C["btn"],
            hi=C["btn_hi"], font_size=9, height=26).pack(side="left", expand=True,
                                                         fill="x", padx=2)
        Key(btns, "Clear All", command=self._graph_clear, bg=C["red"],
            hi=C["red_hi"], font_size=9, height=26).pack(side="left", expand=True,
                                                         fill="x", padx=2)

    def _graph_render_eq_rows(self):
        for w in self.graph_eq_rows.winfo_children():
            w.destroy()
        self.graph_eq_entries = []
        for i, eq in enumerate(self.graph_eqs):
            color = self.GRAPH_COLORS[i % len(self.GRAPH_COLORS)]
            r = tk.Frame(self.graph_eq_rows, bg=C["panel"]); r.pack(fill="x", pady=3)
            var = tk.BooleanVar(value=eq["on"])
            cb = tk.Checkbutton(r, variable=var, bg=C["panel"], activebackground=C["panel"],
                                selectcolor=color, bd=0, highlightthickness=0,
                                command=lambda idx=i, v=var: self._graph_toggle(idx, v))
            cb.pack(side="left")
            tk.Label(r, text=f"y{chr(0x2080 + i + 1)} =", bg=C["panel"], fg=color,
                     font=(FONT, 11, "bold")).pack(side="left", padx=(2, 4))
            e = tk.Entry(r, bg=C["btn"], fg=C["white"], font=(FONT_MONO, 11),
                         relief="flat", insertbackground=C["white"],
                         highlightthickness=1, highlightbackground=C["btn_edge"],
                         highlightcolor=color)
            e.insert(0, eq["expr"])
            e.pack(side="left", fill="x", expand=True, ipady=2)
            e.bind("<KeyRelease>", lambda ev, idx=i: self._graph_edit(idx))
            e.bind("<Return>", lambda ev: self._graph_refresh())
            tk.Frame(r, bg=color, width=18, height=3).pack(side="left", padx=6)
            self.graph_eq_entries.append(e)

    def _graph_toggle(self, idx, var):
        self.graph_eqs[idx]["on"] = bool(var.get())
        self._graph_refresh()

    def _graph_edit(self, idx):
        self.graph_eqs[idx]["expr"] = self.graph_eq_entries[idx].get()
        self._graph_refresh()

    def _graph_add_eq(self):
        for i, e in enumerate(self.graph_eq_entries):
            self.graph_eqs[i]["expr"] = e.get()
        if len(self.graph_eqs) < 8:
            self.graph_eqs.append({"expr": "", "on": True})
            self._graph_render_eq_rows(); self._graph_refresh()

    def _graph_clear(self):
        for eq in self.graph_eqs:
            eq["expr"], eq["on"] = "", False
        self._graph_render_eq_rows(); self._graph_refresh()

    # ---- graph window (big plot) ----
    def _graph_window_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=6)
        tk.Label(p, text="GRAPH WINDOW", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=10, pady=(9, 4))
        self.graph_canvas = tk.Canvas(p, bg=C["panel"], height=300,
                                      highlightthickness=0)
        self.graph_canvas.pack(fill="both", expand=True, padx=10, pady=4)
        self.graph_canvas.bind("<Configure>", lambda e: self._graph_draw(self.graph_canvas))
        self.graph_canvas.bind("<Button-1>", self._graph_click_trace)
        btns = tk.Frame(p, bg=C["panel"]); btns.pack(fill="x", padx=8, pady=(2, 10))
        for label, cmd in (("Auto", self._graph_autoscale),
                           ("Default", self._graph_default),
                           ("Zoom In", lambda: self._graph_zoom(0.7)),
                           ("Zoom Out", lambda: self._graph_zoom(1.4))):
            Key(btns, label, command=cmd, bg=C["btn"], hi=C["btn_hi"],
                font_size=9, height=28).pack(side="left", expand=True, fill="x", padx=2)

    _GRAPH_TOOLS = ["Trace", "Zoom", "Intercept", "Minimum", "Maximum", "Root",
                    "Y-Intercept", "Derivative", "Integral", "Tangent", "Normal"]

    def _graph_tools_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(6, 0))
        tk.Label(p, text="GRAPH TOOLS", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=10, pady=(9, 6))
        for i, name in enumerate(self._GRAPH_TOOLS):
            active = name == "Trace"
            row = tk.Frame(p, bg=C["btn"] if active else C["panel"], cursor="hand2",
                           highlightthickness=1,
                           highlightbackground=C["blue_hi"] if active else C["btn_edge"])
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=name, bg=row["bg"],
                     fg=C["white"] if active else C["sub"],
                     font=(FONT, 10)).pack(side="left", padx=8, pady=5)
            for w in (row, *row.winfo_children()):
                w.bind("<Button-1>", lambda e, n=name: self._graph_tool(n))

    def _graph_tool(self, name):
        if name == "Zoom":
            self._graph_zoom(0.7)
        elif name == "Trace":
            self.graph_trace_on = not self.graph_trace_on
            self._graph_refresh()
        else:                      # analysis tools -> run scan for first eq
            self._graph_analyze(force=name)

    # ---- window settings ----
    def _graph_settings_panel(self, parent):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.pack(fill="x")
        tk.Label(p, text="WINDOW SETTINGS", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        grid = tk.Frame(p, bg=C["panel"]); grid.pack(fill="x", padx=12)
        self.graph_win_entries = {}
        for i, key in enumerate(("Xmin", "Xmax", "Xscl", "Ymin", "Ymax", "Yscl")):
            r, c = divmod(i, 3)
            cell = tk.Frame(grid, bg=C["panel"]); cell.grid(row=r, column=c,
                                                            sticky="w", padx=8, pady=4)
            tk.Label(cell, text=key, bg=C["panel"], fg=C["sub"], font=(FONT, 10),
                     width=5, anchor="w").pack(side="left")
            e = tk.Entry(cell, width=8, bg=C["btn"], fg=C["white"], font=(FONT_MONO, 10),
                         relief="flat", insertbackground=C["white"], justify="center",
                         highlightthickness=1, highlightbackground=C["btn_edge"],
                         highlightcolor=C["blue_hi"])
            e.insert(0, self.graph_win[key])
            e.pack(side="left", ipady=2)
            e.bind("<KeyRelease>", lambda ev: self._graph_refresh())
            self.graph_win_entries[key] = e
        btns = tk.Frame(p, bg=C["panel"]); btns.pack(fill="x", padx=12, pady=(6, 12))
        Key(btns, "Auto Scale", command=self._graph_autoscale, bg=C["btn"],
            hi=C["btn_hi"], font_size=10, height=28).pack(side="left", expand=True,
                                                          fill="x", padx=3)
        Key(btns, "Default", command=self._graph_default, bg=C["btn"],
            hi=C["btn_hi"], font_size=10, height=28).pack(side="left", expand=True,
                                                          fill="x", padx=3)

    def _win(self, key):
        e = self.graph_win_entries.get(key)
        return self._eval_num(e.get() if e else self.graph_win[key])

    def _set_win(self, **kw):
        for k, v in kw.items():
            self.graph_win_entries[k].delete(0, "end")
            self.graph_win_entries[k].insert(0, v)
        self._graph_refresh()

    def _graph_default(self):
        self._set_win(Xmin="-2\u03c0", Xmax="2\u03c0", Xscl="\u03c0/2",
                      Ymin="-5", Ymax="5", Yscl="1")

    def _graph_zoom(self, factor):
        xmin, xmax = self._win("Xmin"), self._win("Xmax")
        ymin, ymax = self._win("Ymin"), self._win("Ymax")
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        hx, hy = (xmax - xmin) / 2 * factor, (ymax - ymin) / 2 * factor
        self._set_win(Xmin=nfmt(cx - hx), Xmax=nfmt(cx + hx),
                      Ymin=nfmt(cy - hy), Ymax=nfmt(cy + hy))

    def _graph_autoscale(self):
        xmin, xmax = self._win("Xmin"), self._win("Xmax")
        ns = self._plot_ns()
        lo, hi = float("inf"), float("-inf")
        for eq in self.graph_eqs:
            if not (eq["on"] and eq["expr"].strip()):
                continue
            py = self._to_eval(eq["expr"])
            for k in range(121):
                xv = xmin + (xmax - xmin) * k / 120
                try:
                    yv = float(eval(py, ns, {"x": xv}))
                    if math.isfinite(yv):
                        lo, hi = min(lo, yv), max(hi, yv)
                except Exception:
                    pass
        if lo < hi:
            pad = (hi - lo) * 0.1 or 1
            self._set_win(Ymin=nfmt(lo - pad), Ymax=nfmt(hi + pad))

    # ---- table / analysis / info ----
    def _graph_table_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(0, 6))
        tk.Label(p, text="TABLE VIEW", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        self.graph_table_title = tk.Label(p, text="y\u2081 = sin(x)", bg=C["panel"],
                                          fg=C["blue_hi"], font=(FONT, 9))
        self.graph_table_title.pack(anchor="w", padx=12)
        self.graph_table_frame = tk.Frame(p, bg=C["panel"])
        self.graph_table_frame.pack(fill="both", expand=True, padx=12, pady=(4, 12))

    def _graph_analysis_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=6)
        tk.Label(p, text="ANALYSIS", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        self.graph_analysis_frame = tk.Frame(p, bg=C["panel"])
        self.graph_analysis_frame.pack(fill="both", expand=True, padx=12, pady=(2, 12))

    def _graph_info_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(6, 0))
        tk.Label(p, text="GRAPH INFO", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        body = tk.Frame(p, bg=C["panel"]); body.pack(fill="x", padx=12, pady=(0, 12))
        self.graph_info = {}
        for i, f in enumerate(["Equations Plotted", "View Mode", "Angle Unit",
                               "Grid Style", "Trace Point", "Memory Used"]):
            tk.Label(body, text=f + " :", bg=C["panel"], fg=C["sub"],
                     font=(FONT, 10)).grid(row=i, column=0, sticky="w", pady=2)
            v = tk.Label(body, text="\u2014", bg=C["panel"], fg=C["white"],
                         font=(FONT, 10))
            v.grid(row=i, column=1, sticky="w", padx=10, pady=2)
            self.graph_info[f] = v

    # ---- the plotter ----
    def _graph_draw(self, cv, light=False):
        cv.delete("all")
        w = cv.winfo_width() or 400
        h = int(cv["height"]) or 300
        xmin, xmax = self._win("Xmin"), self._win("Xmax")
        ymin, ymax = self._win("Ymin"), self._win("Ymax")
        if xmax <= xmin or ymax <= ymin:
            return
        axis = "#222" if light else C["sub"]
        gridc = "#dbe2e8" if light else "#1a2233"
        txt = "#333" if light else C["sub"]

        def PX(x): return (x - xmin) / (xmax - xmin) * w
        def PY(y): return h - (y - ymin) / (ymax - ymin) * h

        xscl = self._win("Xscl") or (xmax - xmin) / 8
        yscl = self._win("Yscl") or (ymax - ymin) / 8
        # grid + tick labels
        k = math.ceil(xmin / xscl)
        while k * xscl <= xmax + 1e-9:
            gx = PX(k * xscl)
            cv.create_line(gx, 0, gx, h, fill=gridc)
            if abs(k) > 0:
                cv.create_text(gx, PY(0) + 10, text=self._pi_lbl(k * xscl),
                               fill=txt, font=(FONT, 8))
            k += 1
        k = math.ceil(ymin / yscl)
        while k * yscl <= ymax + 1e-9:
            gy = PY(k * yscl)
            cv.create_line(0, gy, w, gy, fill=gridc)
            if abs(k) > 0:
                cv.create_text(PX(0) - 10, gy, text=nfmt(k * yscl),
                               fill=txt, font=(FONT, 8))
            k += 1
        # axes
        cv.create_line(0, PY(0), w, PY(0), fill=axis, width=1)
        cv.create_line(PX(0), 0, PX(0), h, fill=axis, width=1)
        cv.create_text(w - 6, PY(0) - 8, text="x", fill=txt, font=(FONT, 9))
        cv.create_text(PX(0) + 8, 8, text="y", fill=txt, font=(FONT, 9))

        # curves
        ns = self._plot_ns()
        for i, eq in enumerate(self.graph_eqs):
            if not (eq["on"] and eq["expr"].strip()):
                continue
            py = self._to_eval(eq["expr"])
            color = self.GRAPH_COLORS[i % len(self.GRAPH_COLORS)]
            seg = []
            for px in range(w + 1):
                xv = xmin + (xmax - xmin) * px / w
                try:
                    yv = float(eval(py, ns, {"x": xv}))
                except Exception:
                    yv = None
                if yv is None or not math.isfinite(yv) or abs(yv) > 1e6:
                    if len(seg) >= 4:
                        cv.create_line(*seg, fill=color, width=2)
                    seg = []
                    continue
                pyv = PY(yv)
                # break on huge vertical jumps (asymptotes)
                if seg and abs(pyv - seg[-1]) > h:
                    if len(seg) >= 4:
                        cv.create_line(*seg, fill=color, width=2)
                    seg = []
                seg += [px, pyv]
            if len(seg) >= 4:
                cv.create_line(*seg, fill=color, width=2)

        # trace point on the big canvas
        if self.graph_trace_on and not light:
            tx = self.graph_trace_x
            if xmin <= tx <= xmax:
                cv.create_line(PX(tx), 0, PX(tx), h, fill=C["sub2"], dash=(3, 3))
                for i, eq in enumerate(self.graph_eqs):
                    if not (eq["on"] and eq["expr"].strip()):
                        continue
                    try:
                        yv = float(eval(self._to_eval(eq["expr"]), ns, {"x": tx}))
                        if ymin <= yv <= ymax:
                            cv.create_oval(PX(tx) - 4, PY(yv) - 4, PX(tx) + 4,
                                           PY(yv) + 4, fill="white",
                                           outline=self.GRAPH_COLORS[i])
                    except Exception:
                        pass

    @staticmethod
    def _pi_lbl(v):
        """Label a tick as a multiple of π when close to one."""
        r = v / math.pi
        rr = round(r * 2) / 2          # nearest half
        if abs(r - rr) < 1e-6 and rr != 0:
            if rr == 1: return "\u03c0"
            if rr == -1: return "-\u03c0"
            if rr == int(rr): return f"{int(rr)}\u03c0"
            return f"{nfmt(rr)}\u03c0"
        return nfmt(v)

    def _graph_click_trace(self, event):
        w = self.graph_canvas.winfo_width() or 400
        xmin, xmax = self._win("Xmin"), self._win("Xmax")
        self.graph_trace_x = xmin + (xmax - xmin) * event.x / w
        self.graph_trace_on = True
        self._graph_refresh()

    def _graph_refresh(self):
        # sync entry expressions into model
        if getattr(self, "graph_eq_entries", None):
            for i, e in enumerate(self.graph_eq_entries):
                self.graph_eqs[i]["expr"] = e.get()
        self._graph_draw(self.graph_canvas)
        if getattr(self, "graph_disp_canvas", None):
            self._graph_draw(self.graph_disp_canvas, light=True)
        self._graph_value_panel()
        self._graph_disp_eq_list()
        self._graph_table()
        self._graph_analyze()
        self._graph_update_info()

    def _graph_value_panel(self):
        for w in self.graph_value_frame.winfo_children():
            w.destroy()
        tx = self.graph_trace_x
        ns = self._plot_ns()
        r = tk.Frame(self.graph_value_frame, bg=C["display_bg"]); r.pack(anchor="w")
        tk.Label(r, text="x  =", bg=C["display_bg"], fg=C["display_ink"],
                 font=(FONT_MONO, 11)).pack(side="left")
        tk.Label(r, text=f"{tx:.4f}", bg=C["display_bg"], fg=C["display_ink"],
                 font=(FONT_MONO, 11)).pack(side="left", padx=4)
        for i, eq in enumerate(self.graph_eqs):
            if not (eq["on"] and eq["expr"].strip()):
                continue
            try:
                yv = float(eval(self._to_eval(eq["expr"]), ns, {"x": tx}))
                vs = f"{yv:.4f}"
            except Exception:
                vs = "\u2014"
            color = self.GRAPH_COLORS[i]
            rr = tk.Frame(self.graph_value_frame, bg=C["display_bg"]); rr.pack(anchor="w")
            tk.Label(rr, text=f"y{chr(0x2080 + i + 1)} =", bg=C["display_bg"],
                     fg=color, font=(FONT_MONO, 11, "bold")).pack(side="left")
            tk.Label(rr, text=vs, bg=C["display_bg"], fg=color,
                     font=(FONT_MONO, 11)).pack(side="left", padx=4)

    def _graph_disp_eq_list(self):
        for w in self.graph_disp_eqs.winfo_children():
            w.destroy()
        for i, eq in enumerate(self.graph_eqs):
            if not eq["expr"].strip():
                continue
            color = self.GRAPH_COLORS[i]
            r = tk.Frame(self.graph_disp_eqs, bg=C["display_bg"]); r.pack(anchor="w", pady=1)
            tk.Label(r, text="\u25a0" if eq["on"] else "\u25a1", bg=C["display_bg"],
                     fg=color, font=(FONT, 10)).pack(side="left")
            tk.Label(r, text=f"y{chr(0x2080 + i + 1)} = {eq['expr']}",
                     bg=C["display_bg"], fg=C["display_ink"],
                     font=(FONT_MONO, 10)).pack(side="left", padx=4)

    def _graph_first_on(self):
        for i, eq in enumerate(self.graph_eqs):
            if eq["on"] and eq["expr"].strip():
                return i, eq
        return None, None

    def _graph_table(self):
        for w in self.graph_table_frame.winfo_children():
            w.destroy()
        i, eq = self._graph_first_on()
        if eq is None:
            return
        self.graph_table_title.config(text=f"y{chr(0x2080 + i + 1)} = {eq['expr']}")
        ns = self._plot_ns(); py = self._to_eval(eq["expr"])
        xmin = self._win("Xmin"); xscl = self._win("Xscl") or math.pi / 2
        hdr = tk.Frame(self.graph_table_frame, bg=C["panel"]); hdr.pack(fill="x")
        tk.Label(hdr, text="x", bg=C["panel"], fg=C["sub"], font=(FONT, 9, "bold"),
                 width=10, anchor="w").pack(side="left")
        tk.Label(hdr, text=f"y{chr(0x2080 + i + 1)}", bg=C["panel"], fg=C["sub"],
                 font=(FONT, 9, "bold"), width=8, anchor="w").pack(side="left")
        for kk in range(9):
            xv = xmin + kk * xscl
            try:
                yv = nfmt(float(eval(py, ns, {"x": xv})))
            except Exception:
                yv = "\u2014"
            row = tk.Frame(self.graph_table_frame, bg=C["panel"]); row.pack(fill="x")
            tk.Label(row, text=self._pi_lbl(xv), bg=C["panel"], fg=C["white"],
                     font=(FONT_MONO, 9), width=10, anchor="w").pack(side="left")
            tk.Label(row, text=yv, bg=C["panel"], fg=C["white"],
                     font=(FONT_MONO, 9), width=8, anchor="w").pack(side="left")

    def _graph_analyze(self, force=None):
        for w in self.graph_analysis_frame.winfo_children():
            w.destroy()
        i, eq = self._graph_first_on()
        if eq is None:
            return
        ns = self._plot_ns(); py = self._to_eval(eq["expr"])
        xmin, xmax = self._win("Xmin"), self._win("Xmax")
        N = 400
        xs, ys = [], []
        for k in range(N + 1):
            xv = xmin + (xmax - xmin) * k / N
            try:
                yv = float(eval(py, ns, {"x": xv}))
            except Exception:
                yv = None
            xs.append(xv); ys.append(yv if (yv is not None and math.isfinite(yv)) else None)
        roots, maxima, minima = [], [], []
        for k in range(1, N + 1):
            a, b = ys[k - 1], ys[k]
            if a is not None and b is not None and a == 0:
                roots.append(xs[k - 1])
            elif a is not None and b is not None and a * b < 0:
                roots.append((xs[k - 1] + xs[k]) / 2)
        for k in range(1, N):
            a, b, c = ys[k - 1], ys[k], ys[k + 1]
            if None in (a, b, c):
                continue
            if b > a and b > c:
                maxima.append((xs[k], b))
            elif b < a and b < c:
                minima.append((xs[k], b))
        label = f"y{chr(0x2080 + i + 1)} = {eq['expr']}"

        def line(txt, col=C["white"]):
            tk.Label(self.graph_analysis_frame, text=txt, bg=C["panel"], fg=col,
                     font=(FONT, 9), anchor="w", justify="left",
                     wraplength=260).pack(anchor="w")

        line(f"Function: {label}", C["blue_hi"])
        if roots:
            line("Roots (x): " + ", ".join(nfmt(round(r, 3)) for r in roots[:6]))
        else:
            line("Roots: none in view")
        if maxima:
            mx = max(maxima, key=lambda t: t[1])
            line(f"Maximum: x={nfmt(round(mx[0],3))}, y={nfmt(round(mx[1],3))}")
        if minima:
            mn = min(minima, key=lambda t: t[1])
            line(f"Minimum: x={nfmt(round(mn[0],3))}, y={nfmt(round(mn[1],3))}")

    def _graph_update_info(self):
        plotted = sum(1 for e in self.graph_eqs if e["on"] and e["expr"].strip())
        self.graph_info["Equations Plotted"].config(text=str(plotted))
        self.graph_info["View Mode"].config(text="Rectangular")
        self.graph_info["Angle Unit"].config(text="RAD")
        self.graph_info["Grid Style"].config(text="Dotted")
        tx = self.graph_trace_x
        i, eq = self._graph_first_on()
        ty = "\u2014"
        if eq is not None:
            try:
                ty = f"{float(eval(self._to_eval(eq['expr']), self._plot_ns(), {'x': tx})):.4f}"
            except Exception:
                pass
        self.graph_info["Trace Point"].config(text=f"({tx:.4f}, {ty})")
        self.graph_info["Memory Used"].config(
            text=f"{len([e for e in self.graph_eqs if e['expr'].strip()])} / 10")

    # ===================================================================== #
    #  CONVERT MODE
    # ===================================================================== #
    # category -> list of (name, symbol, factor-to-base).  Temperature is
    # handled specially (affine), everything else is linear.
    CV_UNITS = {
        "Length": [("Millimeter", "mm", 1e-3), ("Centimeter", "cm", 1e-2),
                   ("Meter", "m", 1.0), ("Kilometer", "km", 1e3),
                   ("Inch", "in", 0.0254), ("Foot", "ft", 0.3048),
                   ("Yard", "yd", 0.9144), ("Mile", "mi", 1609.344),
                   ("Nautical Mile", "nmi", 1852.0), ("Angstrom", "\u00c5", 1e-10),
                   ("Furlong", "fur", 201.168), ("Light Year", "ly", 9.4607e15)],
        "Area": [("Square Meter", "m\u00b2", 1.0), ("Square Kilometer", "km\u00b2", 1e6),
                 ("Square Centimeter", "cm\u00b2", 1e-4), ("Hectare", "ha", 1e4),
                 ("Acre", "ac", 4046.8564224), ("Square Foot", "ft\u00b2", 0.09290304),
                 ("Square Inch", "in\u00b2", 6.4516e-4), ("Square Mile", "mi\u00b2", 2589988.11)],
        "Volume": [("Milliliter", "mL", 1e-3), ("Liter", "L", 1.0),
                   ("Cubic Meter", "m\u00b3", 1e3), ("Gallon (US)", "gal", 3.785411784),
                   ("Quart", "qt", 0.946352946), ("Pint", "pt", 0.473176473),
                   ("Cup", "cup", 0.2365882365), ("Fluid Ounce", "fl oz", 0.0295735296),
                   ("Cubic Inch", "in\u00b3", 0.016387064)],
        "Mass": [("Milligram", "mg", 1e-6), ("Gram", "g", 1e-3),
                 ("Kilogram", "kg", 1.0), ("Tonne", "t", 1e3),
                 ("Pound", "lb", 0.45359237), ("Ounce", "oz", 0.028349523125),
                 ("Stone", "st", 6.35029318)],
        "Time": [("Millisecond", "ms", 1e-3), ("Second", "s", 1.0),
                 ("Minute", "min", 60.0), ("Hour", "h", 3600.0),
                 ("Day", "d", 86400.0), ("Week", "wk", 604800.0),
                 ("Year", "yr", 31557600.0)],
        "Speed": [("Meter/second", "m/s", 1.0), ("Kilometer/hour", "km/h", 0.2777778),
                  ("Mile/hour", "mph", 0.44704), ("Foot/second", "ft/s", 0.3048),
                  ("Knot", "kn", 0.5144444)],
        "Temperature": [("Celsius", "\u00b0C", 1.0), ("Fahrenheit", "\u00b0F", 1.0),
                        ("Kelvin", "K", 1.0)],
        "Energy": [("Joule", "J", 1.0), ("Kilojoule", "kJ", 1e3),
                   ("Calorie", "cal", 4.184), ("Kilocalorie", "kcal", 4184.0),
                   ("Watt-hour", "Wh", 3600.0), ("Kilowatt-hour", "kWh", 3.6e6),
                   ("Electronvolt", "eV", 1.602176634e-19), ("BTU", "BTU", 1055.06)],
        "Pressure": [("Pascal", "Pa", 1.0), ("Kilopascal", "kPa", 1e3),
                     ("Bar", "bar", 1e5), ("Atmosphere", "atm", 101325.0),
                     ("PSI", "psi", 6894.757), ("Torr", "Torr", 133.322)],
        "Angle": [("Radian", "rad", 1.0), ("Degree", "\u00b0", 0.01745329252),
                  ("Gradian", "grad", 0.015707963), ("Arcminute", "'", 2.908882e-4),
                  ("Arcsecond", "''", 4.848137e-6), ("Revolution", "rev", 6.283185307)],
        "Data Storage": [("Bit", "b", 0.125), ("Byte", "B", 1.0),
                         ("Kilobyte", "KB", 1024.0), ("Megabyte", "MB", 1048576.0),
                         ("Gigabyte", "GB", 1073741824.0), ("Terabyte", "TB", 1.099511628e12)],
        "Force": [("Newton", "N", 1.0), ("Kilonewton", "kN", 1e3),
                  ("Dyne", "dyn", 1e-5), ("Pound-force", "lbf", 4.4482216),
                  ("Kilogram-force", "kgf", 9.80665)],
        "Power": [("Watt", "W", 1.0), ("Kilowatt", "kW", 1e3),
                  ("Megawatt", "MW", 1e6), ("Horsepower", "hp", 745.7),
                  ("BTU/hour", "BTU/h", 0.293071)],
    }

    def _build_convert_view(self, root):
        self.cv_cat = "Length"
        self.cv_recent = []
        self.cv_favorites = [("cm", "in"), ("kg", "lb"), ("\u00b0C", "\u00b0F"),
                             ("km", "mi"), ("kWh", "J")]

        # ---------- display (light LCD) ----------
        o, disp = bordered(root, "#20304a", C["display_bg"], pad=2)
        o.pack(fill="x", pady=(8, 0))
        disp.configure(height=210)
        disp.pack_propagate(False)

        dl = tk.Frame(disp, bg=C["display_bg"])
        dl.pack(side="left", fill="both", expand=True, padx=14, pady=10)
        tk.Label(dl, text="CONVERT", bg=C["display_bg"], fg=C["blue"],
                 font=(FONT, 11, "bold")).pack(anchor="w")
        fr = tk.Frame(dl, bg=C["display_bg"]); fr.pack(anchor="w", pady=(4, 0))
        tk.Label(fr, text="From", bg=C["display_bg"], fg=C["sub2"],
                 font=(FONT, 9)).pack(side="left")
        self.cv_cat_var = tk.StringVar(value=self.cv_cat)
        cm = tk.OptionMenu(fr, self.cv_cat_var, *self.CV_UNITS.keys(),
                           command=lambda v: self._cv_set_cat(v))
        cm.config(bg=C["display_bg"], fg=C["display_ink"], font=(FONT, 9), bd=0,
                  highlightthickness=0, activebackground="#dfe6ec")
        cm.pack(side="left", padx=6)
        self.cv_disp_from = tk.Label(dl, text="100", bg=C["display_bg"],
                                     fg=C["display_ink"], font=(FONT, 26, "bold"))
        self.cv_disp_from.pack(anchor="w")
        self.cv_disp_result = tk.Label(dl, text="0", bg=C["display_bg"], fg=C["blue"],
                                       font=(FONT, 34, "bold"))
        self.cv_disp_result.pack(anchor="w")
        self.cv_disp_rel = tk.Label(dl, text="", bg=C["display_bg"], fg=C["sub2"],
                                    font=(FONT, 9))
        self.cv_disp_rel.pack(anchor="w")
        flags = tk.Frame(dl, bg=C["display_bg"]); flags.pack(side="bottom", anchor="w")
        for t, col in (("NORM", C["sub2"]), ("MATH", C["sub2"]),
                       ("FRAC", C["sub2"]), ("CONVERT", C["blue"])):
            tk.Label(flags, text=t, bg=C["display_bg"], fg=col,
                     font=(FONT, 8, "bold")).pack(side="left", padx=(0, 8))

        dr = tk.Frame(disp, bg=C["display_bg"], width=250)
        dr.pack(side="right", fill="y", padx=(0, 14), pady=10)
        dr.pack_propagate(False)
        tk.Label(dr, text="RECENT CONVERSIONS", bg=C["display_bg"], fg=C["sub2"],
                 font=(FONT, 9, "bold")).pack(anchor="w")
        self.cv_recent_frame = tk.Frame(dr, bg=C["display_bg"])
        self.cv_recent_frame.pack(fill="both", expand=True, pady=(6, 0))

        # ---------- categories | converter | unit list ----------
        row1 = tk.Frame(root, bg=C["bg"]); row1.pack(fill="x", pady=(10, 0))
        row1.grid_columnconfigure(0, weight=2, uniform="c1")
        row1.grid_columnconfigure(1, weight=3, uniform="c1")
        row1.grid_columnconfigure(2, weight=2, uniform="c1")
        self._cv_categories_panel(row1, 0)
        self._cv_converter_panel(row1, 1)
        self._cv_unitlist_panel(row1, 2)

        # ---------- formula | favorites | tips ----------
        row2 = tk.Frame(root, bg=C["bg"]); row2.pack(fill="x", pady=(10, 0))
        for i in range(3):
            row2.grid_columnconfigure(i, weight=1, uniform="c2")
        self._cv_formula_panel(row2, 0)
        self._cv_favorites_panel(row2, 1)
        self._cv_tips_panel(row2, 2)

        self._cv_load_category()

    def _cv_categories_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(0, 6))
        tk.Label(p, text="CONVERSION CATEGORIES", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=10, pady=(9, 6))
        self.cv_cat_rows = {}
        for cat in self.CV_UNITS:
            active = cat == self.cv_cat
            row = tk.Frame(p, bg=C["btn"] if active else C["panel"], cursor="hand2",
                           highlightthickness=1,
                           highlightbackground=C["blue_hi"] if active else C["btn_edge"])
            row.pack(fill="x", padx=8, pady=2)
            lbl = tk.Label(row, text=cat, bg=row["bg"],
                           fg=C["white"] if active else C["sub"], font=(FONT, 10))
            lbl.pack(side="left", padx=8, pady=5)
            arr = tk.Label(row, text="\u203a", bg=row["bg"], fg=C["sub2"], font=(FONT, 12))
            arr.pack(side="right", padx=8)
            for w in (row, lbl, arr):
                w.bind("<Button-1>", lambda e, c=cat: self._cv_set_cat(c))
            self.cv_cat_rows[cat] = (row, lbl, arr)

    def _cv_converter_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=6)
        self.cv_title = tk.Label(p, text="LENGTH", bg=C["panel"], fg=C["blue_hi"],
                                 font=(FONT, 11, "bold"))
        self.cv_title.pack(anchor="w", padx=12, pady=(10, 6))

        tk.Label(p, text="From Unit", bg=C["panel"], fg=C["sub"],
                 font=(FONT, 9)).pack(anchor="w", padx=12)
        self.cv_from_var = tk.StringVar()
        self.cv_from_menu = tk.OptionMenu(p, self.cv_from_var, "")
        self.cv_from_menu.config(bg=C["btn"], fg=C["white"], font=(FONT, 10), bd=0,
                                 highlightthickness=1, highlightbackground=C["btn_edge"],
                                 activebackground=C["btn_hi"], anchor="w")
        self.cv_from_menu.pack(fill="x", padx=12, pady=(2, 6))

        self.cv_input = tk.Entry(p, bg=C["btn"], fg=C["white"], font=(FONT, 18),
                                 relief="flat", insertbackground=C["white"],
                                 highlightthickness=1, highlightbackground=C["btn_edge"],
                                 highlightcolor=C["blue_hi"])
        self.cv_input.insert(0, "100")
        self.cv_input.pack(fill="x", padx=12, ipady=6)
        self.cv_input.bind("<KeyRelease>", lambda e: self._cv_convert_now())

        Key(p, "\u21c5  Swap", command=self._cv_swap, bg=C["btn"], hi=C["btn_hi"],
            font_size=11, height=30).pack(pady=6)

        tk.Label(p, text="To Unit", bg=C["panel"], fg=C["sub"],
                 font=(FONT, 9)).pack(anchor="w", padx=12)
        self.cv_to_var = tk.StringVar()
        self.cv_to_menu = tk.OptionMenu(p, self.cv_to_var, "")
        self.cv_to_menu.config(bg=C["btn"], fg=C["white"], font=(FONT, 10), bd=0,
                               highlightthickness=1, highlightbackground=C["btn_edge"],
                               activebackground=C["btn_hi"], anchor="w")
        self.cv_to_menu.pack(fill="x", padx=12, pady=(2, 6))

        res = tk.Frame(p, bg=C["btn"], highlightthickness=1,
                       highlightbackground=C["btn_edge"])
        res.pack(fill="x", padx=12, pady=(0, 4))
        self.cv_result_lbl = tk.Label(res, text="0", bg=C["btn"], fg=C["blue_hi"],
                                      font=(FONT, 22, "bold"))
        self.cv_result_lbl.pack(side="left", padx=10, pady=8)
        self.cv_result_sym = tk.Label(res, text="", bg=C["btn"], fg=C["sub"],
                                      font=(FONT, 12))
        self.cv_result_sym.pack(side="right", padx=10)
        self.cv_rel_lbl = tk.Label(p, text="", bg=C["panel"], fg=C["sub"],
                                   font=(FONT, 9))
        self.cv_rel_lbl.pack(anchor="w", padx=12, pady=(0, 8))

        tk.Label(p, text="COMMON UNITS", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 10, "bold")).pack(anchor="w", padx=12, pady=(4, 4))
        self.cv_common_frame = tk.Frame(p, bg=C["panel"])
        self.cv_common_frame.pack(fill="x", padx=10, pady=(0, 10))

    def _cv_unitlist_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(6, 0))
        tk.Label(p, text="UNIT LIST", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=10, pady=(9, 6))
        self.cv_search = tk.Entry(p, bg=C["btn"], fg=C["white"], font=(FONT, 10),
                                  relief="flat", insertbackground=C["white"],
                                  highlightthickness=1, highlightbackground=C["btn_edge"],
                                  highlightcolor=C["blue_hi"])
        self.cv_search.pack(fill="x", padx=10, ipady=4)
        self.cv_search.bind("<KeyRelease>", lambda e: self._cv_fill_unitlist())
        self.cv_unitlist_frame = tk.Frame(p, bg=C["panel"])
        self.cv_unitlist_frame.pack(fill="both", expand=True, padx=10, pady=8)

    def _cv_formula_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(0, 6))
        tk.Label(p, text="CONVERSION FORMULA", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        self.cv_formula1 = tk.Label(p, text="", bg=C["panel"], fg=C["green"],
                                    font=(FONT, 10), anchor="w", justify="left",
                                    wraplength=260)
        self.cv_formula1.pack(anchor="w", padx=12)
        self.cv_formula2 = tk.Label(p, text="", bg=C["panel"], fg=C["white"],
                                    font=(FONT, 10), anchor="w", justify="left",
                                    wraplength=260)
        self.cv_formula2.pack(anchor="w", padx=12, pady=(2, 12))

    def _cv_favorites_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=6)
        tk.Label(p, text="FAVORITES", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        for a, b in self.cv_favorites:
            r = tk.Frame(p, bg=C["panel"]); r.pack(fill="x", padx=12, pady=2)
            tk.Label(r, text="\u2605", bg=C["panel"], fg="#eab308",
                     font=(FONT, 10)).pack(side="left")
            tk.Label(r, text=f"{a}  \u2194  {b}", bg=C["panel"], fg=C["white"],
                     font=(FONT, 10)).pack(side="left", padx=8)
        tk.Frame(p, bg=C["panel"], height=6).pack()

    def _cv_tips_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(6, 0))
        tk.Label(p, text="\U0001f4a1 TIPS", bg=C["panel"], fg="#eab308",
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        tk.Label(p, text="Use the Swap button to quickly exchange the From and "
                         "To units. Type in the input box for a live result, and "
                         "click any unit in the list to set the target.",
                 bg=C["panel"], fg=C["sub"], font=(FONT, 10), justify="left",
                 wraplength=260, anchor="w").pack(anchor="w", padx=12, pady=(0, 12))

    # ---- category / unit handling ----
    def _cv_label(self, u):
        return f"{u[0]} ({u[1]})"

    def _cv_set_cat(self, cat):
        self.cv_cat = cat
        self.cv_cat_var.set(cat)
        for c, (row, lbl, arr) in self.cv_cat_rows.items():
            on = c == cat
            bg = C["btn"] if on else C["panel"]
            row.config(bg=bg, highlightbackground=C["blue_hi"] if on else C["btn_edge"])
            lbl.config(bg=bg, fg=C["white"] if on else C["sub"])
            arr.config(bg=bg)
        self._cv_load_category()

    def _cv_load_category(self):
        units = self.CV_UNITS[self.cv_cat]
        labels = [self._cv_label(u) for u in units]
        self.cv_title.config(text=self.cv_cat.upper())

        # rebuild From / To option menus
        for menu, var, default_i in ((self.cv_from_menu, self.cv_from_var, 0),
                                     (self.cv_to_menu, self.cv_to_var,
                                      1 if len(units) > 1 else 0)):
            m = menu["menu"]; m.delete(0, "end")
            for lab in labels:
                m.add_command(label=lab,
                              command=lambda l=lab, v=var: (v.set(l), self._cv_convert_now()))
            var.set(labels[default_i])

        # common unit quick buttons
        for w in self.cv_common_frame.winfo_children():
            w.destroy()
        for idx, u in enumerate(units[:8]):
            r, c = divmod(idx, 4)
            Key(self.cv_common_frame, u[1],
                command=lambda lab=self._cv_label(u): (self.cv_to_var.set(lab),
                                                       self._cv_convert_now()),
                bg=C["btn"], hi=C["btn_hi"], font_size=10, height=26
                ).grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
        for c in range(4):
            self.cv_common_frame.grid_columnconfigure(c, weight=1, uniform="cu")

        self._cv_fill_unitlist()
        self._cv_convert_now()

    def _cv_fill_unitlist(self):
        for w in self.cv_unitlist_frame.winfo_children():
            w.destroy()
        q = (self.cv_search.get() if hasattr(self, "cv_search") else "").lower()
        to_lab = self.cv_to_var.get()
        for u in self.CV_UNITS[self.cv_cat]:
            lab = self._cv_label(u)
            if q and q not in u[0].lower() and q not in u[1].lower():
                continue
            sel = lab == to_lab
            r = tk.Frame(self.cv_unitlist_frame, bg=C["btn"] if sel else C["panel"],
                         cursor="hand2")
            r.pack(fill="x", pady=1)
            n = tk.Label(r, text=u[0], bg=r["bg"],
                         fg=C["white"] if sel else C["sub"], font=(FONT, 10))
            n.pack(side="left", padx=6, pady=3)
            s = tk.Label(r, text=u[1], bg=r["bg"],
                         fg=C["blue_hi"] if sel else C["sub2"], font=(FONT, 10))
            s.pack(side="right", padx=6)
            for w in (r, n, s):
                w.bind("<Button-1>", lambda e, l=lab: (self.cv_to_var.set(l),
                                                       self._cv_convert_now()))

    def _cv_lookup(self, label):
        for u in self.CV_UNITS[self.cv_cat]:
            if self._cv_label(u) == label:
                return u
        return None

    def _cv_convert_num(self, cat, fu, tu, val):
        if cat == "Temperature":
            c = {"Celsius": val, "Fahrenheit": (val - 32) * 5 / 9,
                 "Kelvin": val - 273.15}[fu[0]]
            return {"Celsius": c, "Fahrenheit": c * 9 / 5 + 32,
                    "Kelvin": c + 273.15}[tu[0]]
        return val * fu[2] / tu[2]

    def _cv_convert_now(self):
        fu = self._cv_lookup(self.cv_from_var.get())
        tu = self._cv_lookup(self.cv_to_var.get())
        if not fu or not tu:
            return
        try:
            val = float(self.cv_input.get())
        except ValueError:
            self.cv_result_lbl.config(text="\u2014")
            return
        res = self._cv_convert_num(self.cv_cat, fu, tu, val)
        rs = f"{res:.6g}"
        self.cv_result_lbl.config(text=rs)
        self.cv_result_sym.config(text=tu[1])
        # one-unit relation
        one = self._cv_convert_num(self.cv_cat, fu, tu, 1.0)
        rel = f"1 {fu[1]} = {one:.6g} {tu[1]}"
        self.cv_rel_lbl.config(text=rel)

        # mirror to the top display
        self.cv_disp_from.config(text=f"{val:g} {fu[1]}")
        self.cv_disp_result.config(text=rs + "  " + tu[1])
        self.cv_disp_rel.config(text=rel)

        # formula
        if self.cv_cat == "Temperature":
            self.cv_formula1.config(text=self._cv_temp_formula(fu, tu))
            self.cv_formula2.config(text=self._cv_temp_formula(tu, fu))
        else:
            mult = fu[2] / tu[2]
            self.cv_formula1.config(
                text=f"{tu[0]} ({tu[1]}) = {fu[0]} ({fu[1]}) \u00d7 {mult:.6g}")
            self.cv_formula2.config(
                text=f"{fu[0]} ({fu[1]}) = {tu[0]} ({tu[1]}) \u00f7 {mult:.6g}")

        # recent conversions
        entry = f"{val:g} {fu[1]}  \u2192  {rs} {tu[1]}"
        if not self.cv_recent or self.cv_recent[0] != entry:
            self.cv_recent.insert(0, entry)
            self.cv_recent = self.cv_recent[:6]
        self._cv_render_recent()
        self._cv_fill_unitlist()

    @staticmethod
    def _cv_temp_formula(a, b):
        pair = (a[0], b[0])
        table = {
            ("Celsius", "Fahrenheit"): "\u00b0F = \u00b0C \u00d7 9/5 + 32",
            ("Fahrenheit", "Celsius"): "\u00b0C = (\u00b0F \u2212 32) \u00d7 5/9",
            ("Celsius", "Kelvin"): "K = \u00b0C + 273.15",
            ("Kelvin", "Celsius"): "\u00b0C = K \u2212 273.15",
            ("Fahrenheit", "Kelvin"): "K = (\u00b0F \u2212 32) \u00d7 5/9 + 273.15",
            ("Kelvin", "Fahrenheit"): "\u00b0F = (K \u2212 273.15) \u00d7 9/5 + 32",
        }
        return table.get(pair, f"{b[1]} = {a[1]}")

    def _cv_swap(self):
        f, t = self.cv_from_var.get(), self.cv_to_var.get()
        self.cv_from_var.set(t); self.cv_to_var.set(f)
        # set the input to the current result so the value stays meaningful
        try:
            cur = float(self.cv_result_lbl.cget("text"))
            self.cv_input.delete(0, "end"); self.cv_input.insert(0, f"{cur:g}")
        except ValueError:
            pass
        self._cv_convert_now()

    def _cv_render_recent(self):
        for w in self.cv_recent_frame.winfo_children():
            w.destroy()
        for e in self.cv_recent:
            r = tk.Frame(self.cv_recent_frame, bg=C["display_bg"]); r.pack(fill="x", pady=2)
            tk.Label(r, text=e, bg=C["display_bg"], fg=C["display_ink"],
                     font=(FONT, 10), anchor="w").pack(side="left")

    # ===================================================================== #
    #  SOLVE MODE
    # ===================================================================== #
    def _build_solve_view(self, root):
        self.solve_history = []

        # ---------- display (light LCD) ----------
        o, disp = bordered(root, "#20304a", C["display_bg"], pad=2)
        o.pack(fill="x", pady=(8, 0))
        disp.configure(height=210)
        disp.pack_propagate(False)

        dl = tk.Frame(disp, bg=C["display_bg"])
        dl.pack(side="left", fill="both", expand=True, padx=14, pady=10)
        tk.Label(dl, text="SOLVE", bg=C["display_bg"], fg=C["blue"],
                 font=(FONT, 11, "bold")).pack(anchor="w")
        tk.Label(dl, text="Equation", bg=C["display_bg"], fg=C["sub2"],
                 font=(FONT, 9)).pack(anchor="w", pady=(6, 0))
        self.solve_eq_disp = tk.Label(dl, text="\u2014", bg=C["display_bg"],
                                      fg=C["display_ink"], font=(FONT, 22, "bold"),
                                      justify="left", anchor="w", wraplength=420)
        self.solve_eq_disp.pack(anchor="w")
        flags = tk.Frame(dl, bg=C["display_bg"]); flags.pack(side="bottom", anchor="w")
        for t, col in (("NORM", C["sub2"]), ("MATH", C["sub2"]), ("SOLVE", C["blue"])):
            tk.Label(flags, text=t, bg=C["display_bg"], fg=col,
                     font=(FONT, 9, "bold")).pack(side="left", padx=(0, 14))

        dm = tk.Frame(disp, bg=C["display_bg"], width=180)
        dm.pack(side="left", fill="y", padx=10, pady=10)
        dm.pack_propagate(False)
        tk.Label(dm, text="SOLUTION", bg=C["display_bg"], fg=C["sub2"],
                 font=(FONT, 10, "bold")).pack(anchor="w")
        self.solve_sol_frame = tk.Frame(dm, bg=C["display_bg"])
        self.solve_sol_frame.pack(anchor="w", fill="both", expand=True, pady=(6, 0))

        dr = tk.Frame(disp, bg=C["display_bg"], width=250)
        dr.pack(side="right", fill="y", padx=(0, 14), pady=10)
        dr.pack_propagate(False)
        tk.Label(dr, text="HISTORY", bg=C["display_bg"], fg=C["sub2"],
                 font=(FONT, 10, "bold")).pack(anchor="w")
        self.solve_hist_frame = tk.Frame(dr, bg=C["display_bg"])
        self.solve_hist_frame.pack(fill="both", expand=True, pady=(6, 0))

        # ---------- input | details | tools ----------
        row1 = tk.Frame(root, bg=C["bg"]); row1.pack(fill="x", pady=(10, 0))
        row1.grid_columnconfigure(0, weight=2, uniform="v1")
        row1.grid_columnconfigure(1, weight=3, uniform="v1")
        row1.grid_columnconfigure(2, weight=2, uniform="v1")
        self._solve_input_panel(row1, 0)
        self._solve_details_panel(row1, 1)
        self._solve_tools_panel(row1, 2)

        # ---------- common equations | formula ref | methods ----------
        row2 = tk.Frame(root, bg=C["bg"]); row2.pack(fill="x", pady=(10, 0))
        for i in range(3):
            row2.grid_columnconfigure(i, weight=1, uniform="v2")
        self._solve_common_panel(row2, 0)
        self._solve_formula_panel(row2, 1)
        self._solve_methods_panel(row2, 2)

        self.solve_input.insert(0, "2x^3-4x^2-22x+24=0")
        self._do_solve()

    def _solve_input_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(0, 6))
        head = tk.Frame(p, bg=C["panel"]); head.pack(fill="x", padx=10, pady=(9, 6))
        tk.Label(head, text="EQUATION INPUT", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(side="left")
        tk.Label(head, text="Clear", bg=C["panel"], fg="#e07a7a", font=(FONT, 9),
                 cursor="hand2").pack(side="right")

        self.solve_input = tk.Entry(p, bg=C["btn"], fg=C["white"], font=(FONT_MONO, 13),
                                    relief="flat", insertbackground=C["white"],
                                    highlightthickness=1, highlightbackground=C["btn_edge"],
                                    highlightcolor=C["blue_hi"])
        self.solve_input.pack(fill="x", padx=10, ipady=6)
        self.solve_input.bind("<Return>", lambda e: self._do_solve())

        syms = tk.Frame(p, bg=C["panel"]); syms.pack(fill="x", padx=8, pady=6)
        for i, s in enumerate(("x", "(", ")", "^", "\u221a", "\u03c0", "e")):
            Key(syms, s, command=lambda t=s: self._solve_insert(t), bg=C["btn"],
                hi=C["btn_hi"], font_size=11, height=28
                ).grid(row=0, column=i, sticky="nsew", padx=2)
        for i in range(7):
            syms.grid_columnconfigure(i, weight=1, uniform="sy")

        info = tk.Frame(p, bg=C["panel"]); info.pack(fill="x", padx=12, pady=(4, 0))
        for i, (lbl, key) in enumerate((("VARIABLE", "var"), ("DEGREE", "deg"),
                                        ("SOLVE FOR", "for"))):
            tk.Label(info, text=lbl, bg=C["panel"], fg=C["blue_hi"],
                     font=(FONT, 9, "bold")).grid(row=i, column=0, sticky="w", pady=3)
        self.solve_var_lbl = tk.Label(info, text="x", bg=C["btn"], fg=C["white"],
                                      font=(FONT, 10), width=8)
        self.solve_var_lbl.grid(row=0, column=1, sticky="e", padx=8, pady=3)
        self.solve_deg_lbl = tk.Label(info, text="\u2014", bg=C["btn"], fg=C["white"],
                                      font=(FONT, 10), width=8)
        self.solve_deg_lbl.grid(row=1, column=1, sticky="e", padx=8, pady=3)
        self.solve_for_lbl = tk.Label(info, text="x", bg=C["btn"], fg=C["white"],
                                      font=(FONT, 10), width=8)
        self.solve_for_lbl.grid(row=2, column=1, sticky="e", padx=8, pady=3)
        info.grid_columnconfigure(0, weight=1)

        Key(p, "Solve  \u203a", command=self._do_solve, bg=C["blue"], hi=C["blue_hi"],
            font_size=13, bold=True, height=40).pack(fill="x", padx=10, pady=12)

    def _solve_details_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=6)
        tk.Label(p, text="SOLUTION DETAILS", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))

        tk.Label(p, text="Exact Roots", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 9, "bold")).pack(anchor="w", padx=12)
        self.solve_exact = tk.Frame(p, bg=C["btn"], highlightthickness=1,
                                    highlightbackground="#2e7d46")
        self.solve_exact.pack(fill="x", padx=12, pady=(2, 6))

        tk.Label(p, text="Factorization", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 9, "bold")).pack(anchor="w", padx=12)
        self.solve_factor = tk.Label(p, text="\u2014", bg=C["btn"], fg=C["white"],
                                     font=(FONT_MONO, 11), anchor="w", padx=10, pady=8)
        self.solve_factor.pack(fill="x", padx=12, pady=(2, 6))

        tk.Label(p, text="Numeric Roots", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 9, "bold")).pack(anchor="w", padx=12)
        self.solve_numeric = tk.Frame(p, bg=C["btn"], highlightthickness=1,
                                      highlightbackground=C["btn_edge"])
        self.solve_numeric.pack(fill="x", padx=12, pady=(2, 6))

        tk.Label(p, text="Graph", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 9, "bold")).pack(anchor="w", padx=12)
        self.solve_canvas = tk.Canvas(p, bg=C["panel"], height=150,
                                      highlightthickness=0)
        self.solve_canvas.pack(fill="x", padx=12, pady=(2, 10))

    def _solve_tools_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(6, 0))
        tk.Label(p, text="SOLVE TOOLS", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=10, pady=(9, 6))
        tools = ["Solve Equation", "Solve System", "Factor Polynomial",
                 "Simplify Expression", "Differentiate", "Integrate", "Limits",
                 "Solve Inequality", "Solve Trigonometry", "Matrix Solve",
                 "Numerical Solve", "Roots & Zeros", "Optimization (Max/Min)"]
        for i, name in enumerate(tools):
            active = i == 0
            row = tk.Frame(p, bg=C["btn"] if active else C["panel"], cursor="hand2",
                           highlightthickness=1,
                           highlightbackground=C["blue_hi"] if active else C["btn_edge"])
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=name, bg=row["bg"],
                     fg=C["white"] if active else C["sub"],
                     font=(FONT, 10)).pack(side="left", padx=8, pady=4)
            tk.Label(row, text="\u203a", bg=row["bg"], fg=C["sub2"],
                     font=(FONT, 11)).pack(side="right", padx=8)

    _COMMON_EQS = ["ax^2 + bx + c = 0", "ax^3 + bx^2 + cx + d = 0",
                   "x^2 - 5x + 6 = 0", "x^3 - 6x^2 + 11x - 6 = 0",
                   "sin(x) = 0", "cos(x) = 0", "e^x = 5", "ln(x) = 1"]

    def _solve_common_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(0, 6))
        tk.Label(p, text="COMMON EQUATIONS", bg=C["panel"], fg="#eab308",
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        for eq in self._COMMON_EQS:
            r = tk.Frame(p, bg=C["panel"], cursor="hand2"); r.pack(fill="x", padx=12, pady=1)
            lab = tk.Label(r, text=eq, bg=C["panel"], fg=C["white"],
                           font=(FONT_MONO, 10), anchor="w")
            lab.pack(side="left")
            for w in (r, lab):
                w.bind("<Button-1>", lambda e, q=eq: self._solve_load(q))
        tk.Frame(p, bg=C["panel"], height=6).pack()

    def _solve_formula_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=6)
        tk.Label(p, text="FORMULA REFERENCE", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        rows = [("Quadratic Formula", "x = (-b \u00b1 \u221a(b\u00b2 - 4ac)) / 2a"),
                ("Cubic (Cardano's Method)", "depressed t\u00b3 + pt + q = 0"),
                ("General Polynomial (Degree n)", "numeric root finding")]
        for title, body in rows:
            tk.Label(p, text=title, bg=C["panel"], fg=C["white"],
                     font=(FONT, 10, "bold")).pack(anchor="w", padx=12, pady=(4, 0))
            tk.Label(p, text=body, bg=C["panel"], fg=C["sub"], font=(FONT_MONO, 9),
                     anchor="w", justify="left", wraplength=250).pack(anchor="w", padx=12)
        tk.Frame(p, bg=C["panel"], height=8).pack()

    def _solve_methods_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(6, 0))
        tk.Label(p, text="SOLUTION METHODS", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        for m in ("Factoring", "Rational Root Theorem", "Synthetic Division",
                  "Numerical (Durand-Kerner)", "Graphical Method"):
            tk.Label(p, text="\u2022  " + m, bg=C["panel"], fg=C["sub"],
                     font=(FONT, 10), anchor="w").pack(anchor="w", padx=12, pady=1)
        tk.Label(p, text="\U0001f4a1 Tip: use lower-degree forms for exact roots.",
                 bg=C["panel"], fg="#eab308", font=(FONT, 9), anchor="w",
                 justify="left", wraplength=250).pack(anchor="w", padx=12, pady=(8, 12))

    # ---- input helpers ----
    def _solve_insert(self, t):
        self.solve_input.insert("insert", t)

    def _solve_load(self, eq):
        self.solve_input.delete(0, "end")
        self.solve_input.insert(0, eq)
        self._do_solve()

    # ---- the solving engine ----
    def _solve_prep(self, s):
        s = s.split("=")
        lhs = s[0]
        rhs = s[1] if len(s) > 1 else "0"
        expr = f"({lhs})-({rhs})"
        expr = self._to_python(expr)
        expr = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", expr)   # 2x -> 2*x, 2( -> 2*(
        expr = re.sub(r"(\))([a-zA-Z0-9(])", r"\1*\2", expr)
        return expr

    def _is_polynomial(self, expr):
        return not any(fn in expr for fn in
                       ("sin", "cos", "tan", "log", "ln", "exp", "sqrt", "csc",
                        "sec", "cot", "sinh", "cosh", "tanh"))

    def _poly_degree(self, expr):
        powers = [int(m) for m in re.findall(r"x\s*\*\*\s*(\d+)", expr)]
        deg = max(powers) if powers else (1 if "x" in expr else 0)
        return min(deg, 8)

    def _poly_coeffs(self, expr, deg):
        ns = self._plot_ns()
        xs = list(range(deg + 1))
        A = [[float(x) ** j for j in range(deg + 1)] for x in xs]
        b = [float(eval(expr, ns, {"x": float(x)})) for x in xs]
        Ainv = m_inverse(A)
        c = [sum(Ainv[i][k] * b[k] for k in range(deg + 1)) for i in range(deg + 1)]
        return [round(v, 9) for v in c]     # c[0] + c[1]x + ... c[deg]x^deg

    @staticmethod
    def _durand_kerner(coeffs):
        import cmath
        c = coeffs[:]
        while len(c) > 1 and abs(c[-1]) < 1e-12:
            c.pop()
        n = len(c) - 1
        if n <= 0:
            return []
        monic = [ci / c[-1] for ci in c]        # a0..1

        def p(z):
            r = 0
            for ci in reversed(monic):
                r = r * z + ci
            return r

        roots = [(0.4 + 0.9j) ** k for k in range(n)]
        for _ in range(200):
            maxd = 0
            for i in range(n):
                num = p(roots[i])
                den = 1
                for j in range(n):
                    if j != i:
                        den *= (roots[i] - roots[j])
                if abs(den) < 1e-15:
                    den = 1e-15
                delta = num / den
                roots[i] -= delta
                maxd = max(maxd, abs(delta))
            if maxd < 1e-13:
                break
        return roots

    def _numeric_roots(self, expr, lo=-30, hi=30, steps=2000):
        ns = self._plot_ns()
        roots, prev_x, prev_y = [], None, None
        for k in range(steps + 1):
            x = lo + (hi - lo) * k / steps
            try:
                y = float(eval(expr, ns, {"x": x}))
            except Exception:
                prev_x, prev_y = None, None
                continue
            if prev_y is not None and math.isfinite(y) and math.isfinite(prev_y):
                if prev_y == 0:
                    roots.append(prev_x)
                elif prev_y * y < 0:                 # bisection in [prev_x, x]
                    a, b = prev_x, x
                    for _ in range(60):
                        m = (a + b) / 2
                        fm = float(eval(expr, ns, {"x": m}))
                        if fm == 0:
                            break
                        if float(eval(expr, ns, {"x": a})) * fm < 0:
                            b = m
                        else:
                            a = m
                    roots.append((a + b) / 2)
            prev_x, prev_y = x, y
        # dedupe
        uniq = []
        for r in roots:
            if not any(abs(r - u) < 1e-6 for u in uniq):
                uniq.append(r)
        return uniq

    @staticmethod
    def _clean(v):
        r = round(v, 6)
        if abs(r - round(r)) < 1e-6:
            return int(round(r))
        return r

    def _do_solve(self):
        raw = self.solve_input.get().strip()
        if not raw:
            return
        self.solve_eq_disp.config(text=self._pretty(raw).replace("=", " = "))
        try:
            expr = self._solve_prep(raw)
            poly = self._is_polynomial(expr)
            deg = self._poly_degree(expr) if poly else 0
            self.solve_deg_lbl.config(text=str(deg) if poly else "\u2014")

            realroots, complexroots, coeffs = [], [], None
            if poly and deg >= 1:
                coeffs = self._poly_coeffs(expr, deg)
                for z in self._durand_kerner(coeffs):
                    if abs(z.imag) < 1e-6:
                        realroots.append(z.real)
                    else:
                        complexroots.append(z)
            else:
                realroots = self._numeric_roots(expr)

            realroots = sorted(self._clean(r) for r in realroots)
            # dedupe cleaned
            dd = []
            for r in realroots:
                if not dd or abs(r - dd[-1]) > 1e-9:
                    dd.append(r)
            realroots = dd

            self._solve_show(raw, expr, realroots, complexroots, coeffs, poly)
        except Exception as exc:
            for f in (self.solve_sol_frame, self.solve_exact, self.solve_numeric):
                for w in f.winfo_children():
                    w.destroy()
            tk.Label(self.solve_exact, text=f"Could not solve: {exc}", bg=C["btn"],
                     fg="#e07a7a", font=(FONT, 10), wraplength=280).pack(padx=8, pady=8)

    def _solve_show(self, raw, expr, roots, croots, coeffs, poly):
        ns = self._plot_ns()

        # --- top display SOLUTION ---
        for w in self.solve_sol_frame.winfo_children():
            w.destroy()
        tk.Label(self.solve_sol_frame, text="Roots", bg=C["display_bg"], fg=C["sub2"],
                 font=(FONT, 9)).pack(anchor="w")
        for i, r in enumerate(roots, 1):
            rr = tk.Frame(self.solve_sol_frame, bg=C["display_bg"]); rr.pack(anchor="w")
            tk.Label(rr, text=f"x{chr(0x2080+i)} =", bg=C["display_bg"], fg=C["blue"],
                     font=(FONT, 11, "bold")).pack(side="left")
            tk.Label(rr, text=f" {nfmt(r)}", bg=C["display_bg"], fg=C["display_ink"],
                     font=(FONT, 11)).pack(side="left")
        if not roots:
            tk.Label(self.solve_sol_frame, text="no real roots", bg=C["display_bg"],
                     fg=C["sub"], font=(FONT, 10)).pack(anchor="w")
        # verification
        if roots:
            lhs = float(eval(expr, ns, {"x": float(roots[0])}))
            ok = abs(lhs) < 1e-6
            vf = tk.Frame(self.solve_sol_frame, bg=C["display_bg"]); vf.pack(anchor="w", pady=(6, 0))
            tk.Label(vf, text="LHS =", bg=C["display_bg"], fg=C["sub2"],
                     font=(FONT, 9)).pack(side="left")
            tk.Label(vf, text=f" {lhs:.2g} {'\u2713' if ok else ''}", bg=C["display_bg"],
                     fg=C["green"] if ok else "#c0392b", font=(FONT, 9)).pack(side="left")

        # --- exact roots ---
        for w in self.solve_exact.winfo_children():
            w.destroy()
        top = tk.Frame(self.solve_exact, bg=C["btn"]); top.pack(fill="x")
        badge = "Exact" if all(isinstance(r, int) for r in roots) and roots else "Approx"
        tk.Label(top, text=badge, bg=C["btn"], fg=C["green"],
                 font=(FONT, 9, "bold")).pack(side="right", padx=8, pady=4)
        for i, r in enumerate(roots, 1):
            tk.Label(self.solve_exact, text=f"x{chr(0x2080+i)} = {nfmt(r)}",
                     bg=C["btn"], fg=C["white"], font=(FONT_MONO, 11),
                     anchor="w").pack(anchor="w", padx=10)
        if croots:
            for i, z in enumerate(croots, len(roots) + 1):
                tk.Label(self.solve_exact,
                         text=f"x{chr(0x2080+i)} = {self._clean(z.real)}{'+' if z.imag>=0 else '-'}{abs(self._clean(z.imag))}i",
                         bg=C["btn"], fg=C["sub"], font=(FONT_MONO, 10),
                         anchor="w").pack(anchor="w", padx=10)
        tk.Frame(self.solve_exact, bg=C["btn"], height=4).pack()

        # --- factorization ---
        if poly and coeffs and roots:
            lead = coeffs[-1]
            parts = []
            for r in roots:
                if r == 0:
                    parts.append("x")
                elif r > 0:
                    parts.append(f"(x - {nfmt(r)})")
                else:
                    parts.append(f"(x + {nfmt(-r)})")
            pref = "" if abs(lead - 1) < 1e-9 else f"{nfmt(self._clean(lead))}"
            self.solve_factor.config(text=(pref + "".join(parts) + " = 0")
                                     if len(parts) == len([r for r in roots]) else "\u2014")
        else:
            self.solve_factor.config(text="\u2014 (numeric)")

        # --- numeric roots ---
        for w in self.solve_numeric.winfo_children():
            w.destroy()
        tk.Label(self.solve_numeric, text="Decimal", bg=C["btn"], fg=C["blue_hi"],
                 font=(FONT, 9, "bold")).pack(anchor="e", padx=8, pady=(4, 0))
        for i, r in enumerate(roots, 1):
            tk.Label(self.solve_numeric, text=f"x{chr(0x2080+i)} = {float(r):.6f}",
                     bg=C["btn"], fg=C["white"], font=(FONT_MONO, 10),
                     anchor="w").pack(anchor="w", padx=10)
        tk.Frame(self.solve_numeric, bg=C["btn"], height=4).pack()

        # --- graph ---
        self._solve_draw(expr, roots)

        # --- history ---
        rootstr = ", ".join(nfmt(r) for r in roots) if roots else "no real roots"
        self.solve_history.insert(0, (self._pretty(raw).replace("=", " = "),
                                      "Roots: " + rootstr))
        self.solve_history = self.solve_history[:5]
        for w in self.solve_hist_frame.winfo_children():
            w.destroy()
        for eq, rs in self.solve_history:
            hr = tk.Frame(self.solve_hist_frame, bg=C["display_bg"]); hr.pack(fill="x", pady=2)
            tk.Label(hr, text=eq, bg=C["display_bg"], fg=C["blue"], font=(FONT, 9),
                     anchor="w").pack(anchor="w")
            tk.Label(hr, text=rs, bg=C["display_bg"], fg=C["display_ink"],
                     font=(FONT, 9), anchor="w").pack(anchor="w")

    def _solve_draw(self, expr, roots):
        cv = self.solve_canvas
        cv.delete("all")
        w = cv.winfo_width() or 300
        h = int(cv["height"])
        ns = self._plot_ns()
        # window around the roots
        if roots:
            lo, hi = min(roots) - 2, max(roots) + 2
        else:
            lo, hi = -10, 10
        if hi - lo < 2:
            lo, hi = lo - 2, hi + 2
        ys = []
        for k in range(w + 1):
            x = lo + (hi - lo) * k / w
            try:
                ys.append(float(eval(expr, ns, {"x": x})))
            except Exception:
                ys.append(None)
        fin = [y for y in ys if y is not None and math.isfinite(y)]
        if not fin:
            return
        ymin, ymax = min(fin), max(fin)
        if ymax - ymin < 1e-9:
            ymax, ymin = ymin + 1, ymin - 1
        pad = (ymax - ymin) * 0.1
        ymin -= pad; ymax += pad

        def PX(x): return (x - lo) / (hi - lo) * w
        def PY(y): return h - (y - ymin) / (ymax - ymin) * h
        cv.create_line(0, PY(0), w, PY(0), fill=C["sub2"])
        if lo <= 0 <= hi:
            cv.create_line(PX(0), 0, PX(0), h, fill=C["sub2"])
        seg = []
        for k in range(w + 1):
            y = ys[k]
            if y is None or not math.isfinite(y):
                if len(seg) >= 4:
                    cv.create_line(*seg, fill=C["blue_hi"], width=2)
                seg = []
                continue
            seg += [k, PY(y)]
        if len(seg) >= 4:
            cv.create_line(*seg, fill=C["blue_hi"], width=2)
        for r in roots:
            if lo <= r <= hi:
                cv.create_oval(PX(r) - 4, PY(0) - 4, PX(r) + 4, PY(0) + 4,
                               fill="white", outline=C["blue_hi"])
                cv.create_text(PX(r), PY(0) + 12, text=nfmt(r), fill=C["sub"],
                               font=(FONT, 8))

    # ===================================================================== #
    #  SOLVE MODE
    # ===================================================================== #
    def _build_solve_view(self, root):
        self.solve_history = []

        # ---------- display (light LCD) ----------
        o, disp = bordered(root, "#20304a", C["display_bg"], pad=2)
        o.pack(fill="x", pady=(8, 0))
        disp.configure(height=250)
        disp.pack_propagate(False)

        dl = tk.Frame(disp, bg=C["display_bg"])
        dl.pack(side="left", fill="both", expand=True, padx=14, pady=10)
        tk.Label(dl, text="SOLVE", bg=C["display_bg"], fg=C["blue"],
                 font=(FONT, 11, "bold")).pack(anchor="w")
        tk.Label(dl, text="Equation", bg=C["display_bg"], fg=C["sub2"],
                 font=(FONT, 9)).pack(anchor="w", pady=(6, 0))
        self.solve_eq_lbl = tk.Label(dl, text="", bg=C["display_bg"],
                                     fg=C["display_ink"], font=(FONT, 22, "bold"),
                                     wraplength=340, justify="left", anchor="w")
        self.solve_eq_lbl.pack(anchor="w")
        flags = tk.Frame(dl, bg=C["display_bg"]); flags.pack(side="bottom", anchor="w")
        for t, col in (("NORM", C["sub2"]), ("MATH", C["sub2"]), ("SOLVE", C["blue"])):
            tk.Label(flags, text=t, bg=C["display_bg"], fg=col,
                     font=(FONT, 9, "bold")).pack(side="left", padx=(0, 14))

        dm = tk.Frame(disp, bg=C["display_bg"], width=180)
        dm.pack(side="left", fill="y", padx=8, pady=10)
        dm.pack_propagate(False)
        tk.Label(dm, text="SOLUTION", bg=C["display_bg"], fg=C["sub2"],
                 font=(FONT, 10, "bold")).pack(anchor="w")
        self.solve_sol_frame = tk.Frame(dm, bg=C["display_bg"])
        self.solve_sol_frame.pack(anchor="w", fill="both", expand=True, pady=(6, 0))

        dr = tk.Frame(disp, bg=C["display_bg"], width=230)
        dr.pack(side="right", fill="y", padx=(0, 14), pady=10)
        dr.pack_propagate(False)
        tk.Label(dr, text="HISTORY", bg=C["display_bg"], fg=C["sub2"],
                 font=(FONT, 10, "bold")).pack(anchor="w")
        self.solve_hist_frame = tk.Frame(dr, bg=C["display_bg"])
        self.solve_hist_frame.pack(fill="both", expand=True, pady=(6, 0))

        # ---------- input | details | tools ----------
        row1 = tk.Frame(root, bg=C["bg"]); row1.pack(fill="x", pady=(10, 0))
        row1.grid_columnconfigure(0, weight=2, uniform="sv1")
        row1.grid_columnconfigure(1, weight=3, uniform="sv1")
        row1.grid_columnconfigure(2, weight=2, uniform="sv1")
        self._solve_input_panel(row1, 0)
        self._solve_details_panel(row1, 1)
        self._solve_tools_panel(row1, 2)

        # ---------- common equations | methods | tips ----------
        row2 = tk.Frame(root, bg=C["bg"]); row2.pack(fill="x", pady=(10, 0))
        for i in range(3):
            row2.grid_columnconfigure(i, weight=1, uniform="sv2")
        self._solve_common_panel(row2, 0)
        self._solve_methods_panel(row2, 1)
        self._solve_tips_panel(row2, 2)

        self.solve_input.insert(0, "2x^3 - 4x^2 - 22x + 24 = 0")
        self._solve_run()

    def _solve_input_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(0, 6))
        head = tk.Frame(p, bg=C["panel"]); head.pack(fill="x", padx=10, pady=(9, 6))
        tk.Label(head, text="EQUATION INPUT", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(side="left")
        tk.Label(head, text="Clear", bg=C["panel"], fg="#e07a7a", font=(FONT, 9),
                 cursor="hand2").pack(side="right")

        self.solve_input = tk.Entry(p, bg=C["btn"], fg=C["white"], font=(FONT_MONO, 13),
                                    relief="flat", insertbackground=C["white"],
                                    highlightthickness=1, highlightbackground=C["btn_edge"],
                                    highlightcolor=C["blue_hi"])
        self.solve_input.pack(fill="x", padx=12, ipady=6)
        self.solve_input.bind("<Return>", lambda e: self._solve_run())

        syms = tk.Frame(p, bg=C["panel"]); syms.pack(fill="x", padx=10, pady=8)
        for i, s in enumerate(("x", "y", "z", "(", ")", "^", "\u221a", "\u221b", "\u03c0", "e")):
            r, c = divmod(i, 5)
            Key(syms, s, command=lambda ss=s: self._solve_insert(ss), bg=C["btn"],
                hi=C["btn_hi"], font_size=11, height=28).grid(
                row=r, column=c, sticky="nsew", padx=2, pady=2)
        for c in range(5):
            syms.grid_columnconfigure(c, weight=1, uniform="sy")

        opts = tk.Frame(p, bg=C["panel"]); opts.pack(fill="x", padx=12, pady=4)
        self.solve_var = tk.StringVar(value="x")
        self.solve_find = tk.StringVar(value="Roots")
        for label, var, vals in (("VARIABLE", self.solve_var, ["x", "y", "z"]),
                                 ("FIND", self.solve_find,
                                  ["Roots", "Factor", "Derivative"])):
            r = tk.Frame(opts, bg=C["panel"]); r.pack(fill="x", pady=3)
            tk.Label(r, text=label, bg=C["panel"], fg=C["blue_hi"],
                     font=(FONT, 9, "bold"), width=10, anchor="w").pack(side="left")
            om = tk.OptionMenu(r, var, *vals, command=lambda v: self._solve_run())
            om.config(bg=C["btn"], fg=C["white"], font=(FONT, 9), bd=0,
                      highlightthickness=0, activebackground=C["btn_hi"], width=10)
            om["menu"].config(bg=C["panel"], fg=C["white"])
            om.pack(side="right")

        Key(p, "Solve  \u203a", command=self._solve_run, bg=C["blue"], hi=C["blue_hi"],
            bold=True, font_size=13, height=38).pack(fill="x", padx=12, pady=(8, 12))

    def _solve_insert(self, s):
        self.solve_input.insert("insert", s)

    def _solve_details_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=6)
        tk.Label(p, text="SOLUTION DETAILS", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))

        tk.Label(p, text="Exact Roots", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 9)).pack(anchor="w", padx=12)
        self.solve_exact = tk.Frame(p, bg=C["btn"], highlightthickness=1,
                                    highlightbackground=C["green"])
        self.solve_exact.pack(fill="x", padx=12, pady=(2, 8))

        tk.Label(p, text="Factorization", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 9)).pack(anchor="w", padx=12)
        self.solve_factor = tk.Label(p, text="", bg=C["btn"], fg=C["white"],
                                     font=(FONT_MONO, 11), anchor="w", justify="left",
                                     wraplength=300)
        self.solve_factor.pack(fill="x", padx=12, ipady=6, pady=(2, 8))

        tk.Label(p, text="Numeric Roots", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 9)).pack(anchor="w", padx=12)
        self.solve_numeric = tk.Frame(p, bg=C["btn"], highlightthickness=1,
                                      highlightbackground=C["btn_edge"])
        self.solve_numeric.pack(fill="x", padx=12, pady=(2, 8))

        tk.Label(p, text="Graph", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 9)).pack(anchor="w", padx=12)
        self.solve_canvas = tk.Canvas(p, bg=C["panel"], height=150,
                                      highlightthickness=1, highlightbackground=C["btn_edge"])
        self.solve_canvas.pack(fill="x", padx=12, pady=(2, 12))
        self.solve_canvas.bind("<Configure>", lambda e: self._solve_draw())

    _SOLVE_TOOLS = ["Solve Equation", "Solve System", "Factor Polynomial",
                    "Simplify Expression", "Differentiate", "Integrate", "Limits",
                    "Solve Inequality", "Solve Trigonometry", "Matrix Solve",
                    "Numerical Solve", "Roots & Zeros", "Optimization (Max/Min)"]

    def _solve_tools_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(6, 0))
        tk.Label(p, text="SOLVE TOOLS", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=10, pady=(9, 6))
        for i, name in enumerate(self._SOLVE_TOOLS):
            active = i == 0
            row = tk.Frame(p, bg=C["btn"] if active else C["panel"], cursor="hand2",
                           highlightthickness=1,
                           highlightbackground=C["blue_hi"] if active else C["btn_edge"])
            row.pack(fill="x", padx=8, pady=1)
            tk.Label(row, text=name, bg=row["bg"],
                     fg=C["white"] if active else C["sub"],
                     font=(FONT, 10)).pack(side="left", padx=8, pady=5)
            tk.Label(row, text="\u203a", bg=row["bg"], fg=C["sub2"],
                     font=(FONT, 11)).pack(side="right", padx=8)

    _COMMON_EQS = ["ax^2 + bx + c = 0", "ax^3 + bx^2 + cx + d = 0",
                   "x^2 - 5x + 6 = 0", "x^3 - 6x^2 + 11x - 6 = 0",
                   "sin(x) = 0", "cos(x) = 0", "e^x = 5", "ln(x) = 1"]

    def _solve_common_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(0, 6))
        tk.Label(p, text="COMMON EQUATIONS", bg=C["panel"], fg="#eab308",
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        for eq in self._COMMON_EQS:
            r = tk.Frame(p, bg=C["panel"], cursor="hand2")
            r.pack(fill="x", padx=12, pady=1)
            lbl = tk.Label(r, text=eq, bg=C["panel"], fg=C["white"],
                           font=(FONT_MONO, 10), anchor="w")
            lbl.pack(side="left")
            for w in (r, lbl):
                w.bind("<Button-1>", lambda e, q=eq: self._solve_load(q))
        tk.Frame(p, bg=C["panel"], height=6).pack()

    def _solve_load(self, eq):
        self.solve_input.delete(0, "end")
        self.solve_input.insert(0, eq)
        self._solve_run()

    def _solve_methods_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=6)
        tk.Label(p, text="SOLUTION METHODS", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        for m in ("Factoring", "Rational Root Theorem", "Synthetic Division",
                  "Numerical (Durand-Kerner)", "Bisection", "Graphical Method"):
            tk.Label(p, text="\u2022  " + m, bg=C["panel"], fg=C["sub"],
                     font=(FONT, 10), anchor="w").pack(anchor="w", padx=14, pady=1)
        tk.Frame(p, bg=C["panel"], height=8).pack()

    def _solve_tips_panel(self, parent, col):
        o, p = bordered(parent, C["btn_edge"], C["panel"])
        o.grid(row=0, column=col, sticky="nsew", padx=(6, 0))
        tk.Label(p, text="\U0001f4a1 TIPS", bg=C["panel"], fg="#eab308",
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        tk.Label(p, text="Enter a polynomial like  2x^3 - 4x^2 - 22x + 24 = 0  and "
                         "press Solve. Real roots are shown exactly when they're "
                         "integers, otherwise numerically. Non-polynomial equations "
                         "(sin(x)=0, e^x=5) are solved numerically over the view.",
                 bg=C["panel"], fg=C["sub"], font=(FONT, 10), justify="left",
                 wraplength=260, anchor="w").pack(anchor="w", padx=12, pady=(0, 12))

    # ---- solver engine ----
    def _parse_poly(self, s, var):
        s = str(s).replace(" ", "").replace("\u2212", "-").replace("\u00d7", "*")
        s = s.replace("^", "**")
        if "=" in s:
            lhs, rhs = s.split("=", 1)
        else:
            lhs, rhs = s, "0"

        def side(expr):
            if not expr:
                return {}
            expr = expr.replace("-", "+-").replace("++", "+")
            if expr.startswith("+"):
                expr = expr[1:]
            coeffs = {}
            for t in expr.split("+"):
                if not t:
                    continue
                if var in t:
                    left, _, right = t.partition(var)
                    left = left.rstrip("*")
                    if left == "":
                        c = 1.0
                    elif left == "-":
                        c = -1.0
                    else:
                        c = float(left)
                    if right.startswith("**"):
                        p = int(right[2:])
                    elif right == "":
                        p = 1
                    else:
                        raise ValueError("not a polynomial")
                    coeffs[p] = coeffs.get(p, 0.0) + c
                else:
                    coeffs[0] = coeffs.get(0, 0.0) + float(t)
            return coeffs

        lc = side(lhs)
        for pw, c in side(rhs).items():
            lc[pw] = lc.get(pw, 0.0) - c
        if not lc:
            return None
        deg = max(lc)
        return [lc.get(pw, 0.0) for pw in range(deg, -1, -1)]

    @staticmethod
    def _poly_roots(coeffs):
        import cmath
        c = list(coeffs)
        while len(c) > 1 and abs(c[0]) < 1e-14:
            c = c[1:]
        n = len(c) - 1
        if n <= 0:
            return []
        if n == 1:
            return [-c[1] / c[0]]
        a = [x / c[0] for x in c]                 # monic, highest-first

        def ev(z):
            r = 0
            for coef in a:
                r = r * z + coef
            return r

        roots = [(0.4 + 0.9j) ** k for k in range(n)]
        for _ in range(500):
            maxd = 0.0
            for i in range(n):
                den = 1 + 0j
                for j in range(n):
                    if j != i:
                        den *= (roots[i] - roots[j])
                if den == 0:
                    continue
                delta = ev(roots[i]) / den
                roots[i] -= delta
                maxd = max(maxd, abs(delta))
            if maxd < 1e-13:
                break
        return roots

    def _numeric_roots(self, s, var):
        if "=" in s:
            lhs, rhs = s.split("=", 1)
        else:
            lhs, rhs = s, "0"
        ns = self._plot_ns()
        lpy, rpy = self._to_eval(lhs), self._to_eval(rhs or "0")

        def f(x):
            return float(eval(lpy, ns, {var: x})) - float(eval(rpy, ns, {var: x}))

        roots, prev, xprev = [], None, None
        for k in range(4001):
            x = -20 + 40 * k / 4000
            try:
                y = f(x)
            except Exception:
                prev = None
                continue
            if prev is not None and prev * y < 0:
                a, b = xprev, x
                for _ in range(60):
                    m = (a + b) / 2
                    try:
                        fm = f(m)
                    except Exception:
                        break
                    if f(a) * fm <= 0:
                        b = m
                    else:
                        a = m
                r = (a + b) / 2
                if all(abs(r - q) > 1e-4 for q in roots):
                    roots.append(r)
            prev, xprev = y, x
        return roots

    def _solve_run(self):
        s = self.solve_input.get().strip()
        var = self.solve_var.get()
        self.solve_eq_lbl.config(text=self._pretty(s) if s else "")
        if not s:
            return
        real_roots, complex_roots, coeffs = [], [], None
        try:
            coeffs = self._parse_poly(s, var)
        except Exception:
            coeffs = None
        try:
            if coeffs and len(coeffs) >= 2:
                for z in self._poly_roots(coeffs):
                    if abs(z.imag if hasattr(z, "imag") else 0) < 1e-6:
                        real_roots.append(float(getattr(z, "real", z)))
                    else:
                        complex_roots.append(z)
            else:
                real_roots = self._numeric_roots(s, var)
        except Exception:
            real_roots = []
        # round near-integers, sort, dedupe
        cleaned = []
        for r in sorted(real_roots):
            rr = round(r)
            r = rr if abs(r - rr) < 1e-6 else round(r, 6)
            if all(abs(r - q) > 1e-6 for q in cleaned):
                cleaned.append(r)
        real_roots = cleaned

        self._solve_show(s, var, real_roots, complex_roots, coeffs)

    def _solve_show(self, s, var, roots, croots, coeffs):
        # --- top SOLUTION panel ---
        for w in self.solve_sol_frame.winfo_children():
            w.destroy()
        tk.Label(self.solve_sol_frame, text="Roots", bg=C["display_bg"],
                 fg=C["sub2"], font=(FONT, 9)).pack(anchor="w")
        for i, r in enumerate(roots[:6], 1):
            row = tk.Frame(self.solve_sol_frame, bg=C["display_bg"]); row.pack(anchor="w")
            tk.Label(row, text=f"x{chr(0x2080 + i)} =", bg=C["display_bg"], fg=C["blue"],
                     font=(FONT_MONO, 11)).pack(side="left")
            tk.Label(row, text=nfmt(r), bg=C["display_bg"], fg=C["display_ink"],
                     font=(FONT_MONO, 11)).pack(side="left", padx=4)
        if not roots:
            tk.Label(self.solve_sol_frame, text="no real roots", bg=C["display_bg"],
                     fg=C["sub2"], font=(FONT, 10)).pack(anchor="w")
        # verification
        vok = self._solve_verify(s, var, roots[0]) if roots else None
        vf = tk.Frame(self.solve_sol_frame, bg=C["display_bg"]); vf.pack(anchor="w", pady=(6, 0))
        if vok is not None:
            tk.Label(vf, text="LHS = 0" + ("  \u2713" if vok else "  \u2717"),
                     bg=C["display_bg"], fg="#1a7f37" if vok else "#c0392b",
                     font=(FONT, 10)).pack(side="left")

        # --- exact roots box ---
        for w in self.solve_exact.winfo_children():
            w.destroy()
        badge = tk.Label(self.solve_exact, text="Exact" if roots else "\u2014",
                         bg=C["green"] if roots else C["btn"], fg="white",
                         font=(FONT, 8, "bold"))
        badge.pack(anchor="ne", padx=6, pady=4)
        for i, r in enumerate(roots[:6], 1):
            tk.Label(self.solve_exact, text=f"x{chr(0x2080 + i)} = {nfmt(r)}",
                     bg=C["btn"], fg=C["white"], font=(FONT_MONO, 11),
                     anchor="w").pack(anchor="w", padx=8)
        tk.Frame(self.solve_exact, bg=C["btn"], height=4).pack()

        # --- factorization ---
        self.solve_factor.config(text=self._factor_string(coeffs, roots, croots, var))

        # --- numeric roots ---
        for w in self.solve_numeric.winfo_children():
            w.destroy()
        for i, r in enumerate(roots[:6], 1):
            tk.Label(self.solve_numeric, text=f"x{chr(0x2080 + i)} = {r:.6f}",
                     bg=C["btn"], fg=C["white"], font=(FONT_MONO, 11),
                     anchor="w").pack(anchor="w", padx=8)
        for z in croots[:4]:
            tk.Label(self.solve_numeric, text=f"    {z.real:.4f} + {z.imag:.4f}i",
                     bg=C["btn"], fg=C["sub"], font=(FONT_MONO, 10),
                     anchor="w").pack(anchor="w", padx=8)
        tk.Frame(self.solve_numeric, bg=C["btn"], height=4).pack()

        self._solve_roots = roots
        self._solve_draw()

        # --- history ---
        rtext = ", ".join(nfmt(r) for r in roots) if roots else "no real roots"
        self.solve_history.insert(0, (s, "Roots: " + rtext))
        self.solve_history = self.solve_history[:5]
        for w in self.solve_hist_frame.winfo_children():
            w.destroy()
        for eq, res in self.solve_history:
            r = tk.Frame(self.solve_hist_frame, bg=C["display_bg"]); r.pack(anchor="w", fill="x", pady=2)
            tk.Label(r, text=self._pretty(eq), bg=C["display_bg"], fg=C["blue"],
                     font=(FONT, 9), anchor="w", wraplength=210, justify="left").pack(anchor="w")
            tk.Label(r, text=res, bg=C["display_bg"], fg=C["display_ink"],
                     font=(FONT, 9), anchor="w", wraplength=210, justify="left").pack(anchor="w")

    def _solve_verify(self, s, var, r):
        if "=" in s:
            lhs, rhs = s.split("=", 1)
        else:
            lhs, rhs = s, "0"
        try:
            ns = self._plot_ns()
            v = eval(self._to_eval(lhs), ns, {var: r}) - \
                eval(self._to_eval(rhs or "0"), ns, {var: r})
            return abs(v) < 1e-4
        except Exception:
            return None

    def _factor_string(self, coeffs, roots, croots, var):
        if not coeffs or croots or len(roots) != (len(coeffs) - 1):
            return "\u2014 (not fully factorable over reals)"
        lead = coeffs[0]
        parts = []
        for r in roots:
            if r == 0:
                parts.append(f"{var}")
            elif r > 0:
                parts.append(f"({var} - {nfmt(r)})")
            else:
                parts.append(f"({var} + {nfmt(-r)})")
        pre = "" if abs(lead - 1) < 1e-9 else nfmt(lead)
        return pre + "".join(parts) + " = 0"

    def _solve_draw(self):
        cv = self.solve_canvas
        cv.delete("all")
        s = self.solve_input.get()
        var = self.solve_var.get()
        if "=" in s:
            lhs, rhs = s.split("=", 1)
        else:
            lhs, rhs = s, "0"
        ns = self._plot_ns()
        lpy, rpy = self._to_eval(lhs), self._to_eval(rhs or "0")

        def f(x):
            return float(eval(lpy, ns, {var: x})) - float(eval(rpy, ns, {var: x}))

        roots = getattr(self, "_solve_roots", [])
        if roots:
            span = max(abs(min(roots)), abs(max(roots))) + 2
        else:
            span = 8
        xmin, xmax = -span, span
        w = cv.winfo_width() or 300
        h = int(cv["height"])
        ys = []
        for k in range(w + 1):
            x = xmin + (xmax - xmin) * k / w
            try:
                ys.append(f(x))
            except Exception:
                ys.append(None)
        fin = [y for y in ys if y is not None and math.isfinite(y)]
        if not fin:
            return
        ymin, ymax = min(fin), max(fin)
        if ymax - ymin < 1e-9:
            ymin, ymax = ymin - 1, ymax + 1
        pad = (ymax - ymin) * 0.1
        ymin, ymax = ymin - pad, ymax + pad

        def PX(x): return (x - xmin) / (xmax - xmin) * w
        def PY(y): return h - (y - ymin) / (ymax - ymin) * h

        cv.create_line(0, PY(0), w, PY(0), fill=C["sub2"])
        cv.create_line(PX(0), 0, PX(0), h, fill=C["sub2"])
        seg = []
        for k in range(w + 1):
            y = ys[k]
            if y is None or not math.isfinite(y):
                if len(seg) >= 4:
                    cv.create_line(*seg, fill=C["blue_hi"], width=2)
                seg = []
                continue
            seg += [k, PY(y)]
        if len(seg) >= 4:
            cv.create_line(*seg, fill=C["blue_hi"], width=2)
        for r in roots:
            if xmin <= r <= xmax:
                cv.create_oval(PX(r) - 4, PY(0) - 4, PX(r) + 4, PY(0) + 4,
                               fill="white", outline=C["blue_hi"])

    # ---------------------------------------------------------------- keypad
    FS_FUNC = 13        # function / trig keys
    FS_NUM  = 17        # digits
    FS_OP   = 18        # + - x div
    FS_SIDE = 11        # CONST / CONV / UNIT
    KEY_H   = 44

    def _build_keypad(self, parent):
        pad = tk.Frame(parent, bg=C["bg"])
        pad.pack(side="left", fill="both", expand=True)
        ins = self._insert          # shorthand

        # ================= function grid : 8 equal columns =================
        fg = tk.Frame(pad, bg=C["bg"]); fg.pack(fill="x")
        for c in range(8):
            fg.grid_columnconfigure(c, weight=1, uniform="fk")

        def F(r, c, main, cmd=None, **kw):
            kw.setdefault("font_size", self.FS_FUNC)
            kw.setdefault("height", self.KEY_H)
            k = Key(fg, main, command=cmd, **kw)
            k.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
            return k

        # memory / modifier row
        self.shift_key = F(0, 0, "SHIFT", self._toggle_shift, bg=C["gold"],
                           hi=C["gold_hi"], bold=True)
        self.alpha_key = F(0, 1, "ALPHA", self._toggle_alpha, bg=C["purple"],
                           hi=C["purple_hi"], bold=True)
        mem = [("MC", self._mem_clear), ("MR", self._mem_recall),
               ("M+", self._mem_add), ("M-", self._mem_sub),
               ("MS", self._mem_store), ("M\u25be", self._mem_show)]
        for i, (t, cmd) in enumerate(mem):
            F(0, 2 + i, t, cmd)

        F(1, 0, "x!", self._key_fact, shift="nPr")
        F(1, 1, "nCr", lambda: ins("C"))
        F(1, 2, "d/dx", self._key_deriv, shift="\u222bdx")
        F(1, 3, "\u222bdx", lambda: ins("\u222b("))
        F(1, 4, "x\u207b\u00b9", lambda: ins("^(-1)"))
        F(1, 5, "x\u02b8", lambda: ins("^"))
        F(1, 6, "10\u02e3", lambda: ins("10^"))
        F(1, 7, "e\u02e3", lambda: ins("e^"))

        F(2, 0, "\u221ax", lambda: ins("\u221a("))
        F(2, 1, "\u221b\u0078", lambda: ins("cbrt("))
        F(2, 2, "x\u00b2", lambda: ins("^2"))
        F(2, 3, "x\u00b3", lambda: ins("^3"))
        F(2, 4, "x\u02b8", lambda: ins("^"))
        F(2, 5, "log", lambda: ins("log("))
        F(2, 6, "ln", lambda: ins("ln("))
        F(2, 7, "log\u2090x", lambda: ins("logb("))

        trig = [("sin", "sin\u207b\u00b9"), ("cos", "cos\u207b\u00b9"),
                ("tan", "tan\u207b\u00b9"), ("csc", "csc\u207b\u00b9"),
                ("sec", "sec\u207b\u00b9"), ("cot", "cot\u207b\u00b9")]
        for i, (fn, sh) in enumerate(trig):
            F(3, i, fn, lambda f=fn: self._key_trig(f), shift=sh)
        F(3, 6, "%", lambda: ins("%"))
        F(3, 7, "Mod", lambda: ins(" mod "))

        F(4, 0, "\u03c0", lambda: ins("\u03c0"))
        F(4, 1, "e", lambda: ins("e"))
        F(4, 2, "(", lambda: ins("("))
        F(4, 3, ")", lambda: ins(")"))
        F(4, 4, ",", lambda: ins(","))
        F(4, 5, "ABS", lambda: ins("abs("))
        F(4, 6, "ENG", self._eng)
        F(4, 7, "RCL", self._rcl)

        # ============= number grid : 7 equal (wider) columns =============
        # Separate grid so digits sit in their own wider columns, exactly
        # like the reference — they do NOT line up under the function keys.
        ng = tk.Frame(pad, bg=C["bg"]); ng.pack(fill="x", pady=(2, 0))
        for c in range(7):
            ng.grid_columnconfigure(c, weight=1, uniform="nk")

        def N(r, c, main, cmd=None, **kw):
            kw.setdefault("height", self.KEY_H)
            k = Key(ng, main, command=cmd, **kw)
            k.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
            return k

        num = dict(bg=C["num"], hi=C["num_hi"], font_size=self.FS_NUM)
        op  = dict(font_size=self.FS_OP)

        # row 0
        N(0, 0, "AC", self.clear_all, bg=C["red"], hi=C["red_hi"],
          shift="ON", shift_fg="#f0a6b4", bold=True)
        N(0, 1, "DEL", self.backspace, bg=C["red"], hi=C["red_hi"], bold=True)
        N(0, 2, "7", lambda: ins("7"), **num)
        N(0, 3, "8", lambda: ins("8"), **num)
        N(0, 4, "9", lambda: ins("9"), **num)
        N(0, 5, "\u00f7", lambda: ins("\u00f7"), **op)
        N(0, 6, "CONST \u25be", self._const_menu, font_size=self.FS_SIDE)

        # row 1  (col 1 left empty -> the gap under DEL, as in the reference)
        N(1, 0, "+/\u2212", self.negate)
        N(1, 2, "4", lambda: ins("4"), **num)
        N(1, 3, "5", lambda: ins("5"), **num)
        N(1, 4, "6", lambda: ins("6"), **num)
        N(1, 5, "\u00d7", lambda: ins("\u00d7"), **op)
        N(1, 6, "CONV \u25be", lambda: self._set_mode("CONVERT"),
          font_size=self.FS_SIDE)

        # row 2
        N(2, 0, "\u207f/\u2094", lambda: ins("/"))
        N(2, 2, "1", lambda: ins("1"), **num)
        N(2, 3, "2", lambda: ins("2"), **num)
        N(2, 4, "3", lambda: ins("3"), **num)
        N(2, 5, "\u2212", lambda: ins("\u2212"), **op)
        N(2, 6, "UNIT \u25be", lambda: self._set_mode("CONVERT"),
          font_size=self.FS_SIDE)

        # row 3
        self.shift_ind = N(3, 0, "SHIFT", self._toggle_shift, shift="OFF",
                           shift_fg=C["sub2"])
        N(3, 2, "0", lambda: ins("0"), **num)
        N(3, 3, ".", lambda: ins("."), **num)
        N(3, 4, "Ans", self.insert_ans, bg=C["num"], hi=C["num_hi"])
        N(3, 5, "+", lambda: ins("+"), **op)
        N(3, 6, "=", self.evaluate, bg=C["blue"], hi=C["blue_hi"],
          bold=True, font_size=self.FS_OP)

    # ---- modifier keys (SHIFT / ALPHA) ----
    def _toggle_shift(self):
        self.shift_on = not self.shift_on
        self.alpha_on = False
        self._update_modifiers()

    def _toggle_alpha(self):
        self.alpha_on = not self.alpha_on
        self.shift_on = False
        self._update_modifiers()

    def _reset_shift(self):
        if self.shift_on or self.alpha_on:
            self.shift_on = self.alpha_on = False
            self._update_modifiers()

    def _update_modifiers(self):
        if getattr(self, "shift_key", None):
            self.shift_key.set_base(C["gold_hi"] if self.shift_on else C["gold"],
                                    C["gold_hi"])
        if getattr(self, "alpha_key", None):
            self.alpha_key.set_base(C["purple_hi"] if self.alpha_on else C["purple"],
                                    C["purple_hi"])
        ind = getattr(self, "shift_ind", None)
        if ind and ind.shift_lbl:
            ind.shift_lbl.config(text="ON" if self.shift_on else "OFF",
                                 fg=C["gold_txt"] if self.shift_on else C["sub2"])

    # ---- shifted function dispatchers ----
    def _key_trig(self, fn):
        inv = {"sin": "asin", "cos": "acos", "tan": "atan",
               "csc": "acsc", "sec": "asec", "cot": "acot"}
        self._insert((inv[fn] if self.shift_on else fn) + "(")

    def _key_fact(self):
        self._insert("P" if self.shift_on else "!")

    def _key_deriv(self):
        self._insert("\u222b(" if self.shift_on else "d/dx(")

    # ---- memory register ----
    def _current_value(self):
        try:
            if self.just_evaluated or not self.expr:
                return float(self.result.replace("\u2212", "-"))
            expr = self.expr
            miss = expr.count("(") - expr.count(")")
            if miss > 0:
                expr += ")" * miss
            return float(eval(self._to_python(self._preprocess_calculus(expr)),
                              self._namespace(), {}))
        except Exception:
            return 0.0

    def _update_mem_indicator(self):
        ind = getattr(self, "mem_indicator", None)
        if ind:
            ind.config(text=(f"M = {nfmt(self.memory)}" if self.memory != 0 else ""))

    def _mem_store(self):
        self.memory = self._current_value()
        self._update_mem_indicator()

    def _mem_clear(self):
        self.memory = 0.0
        self._update_mem_indicator()

    def _mem_add(self):
        self.memory += self._current_value()
        self._update_mem_indicator()

    def _mem_sub(self):
        self.memory -= self._current_value()
        self._update_mem_indicator()

    def _mem_recall(self):
        self._insert(nfmt(self.memory))

    def _mem_show(self):
        """M▾ opens a small memory menu."""
        m = tk.Menu(self, tearoff=0, bg=C["panel"], fg=C["white"],
                    activebackground=C["blue"], activeforeground="white", bd=0)
        m.add_command(label=f"Memory  =  {nfmt(self.memory)}", state="disabled")
        m.add_separator()
        m.add_command(label="Recall (MR)", command=self._mem_recall)
        m.add_command(label="Store (MS)", command=self._mem_store)
        m.add_command(label="Add (M+)", command=self._mem_add)
        m.add_command(label="Subtract (M-)", command=self._mem_sub)
        m.add_command(label="Clear (MC)", command=self._mem_clear)
        try:
            m.tk_popup(self.winfo_pointerx(), self.winfo_pointery())
        finally:
            m.grab_release()

    def _rcl(self):
        self._insert(nfmt(self.memory))

    # ---- engineering notation of the current value ----
    def _eng(self):
        try:
            v = self._current_value()
            if v == 0:
                s = "0"
            else:
                exp = int(math.floor(math.log10(abs(v)) / 3) * 3)
                s = f"{v / (10 ** exp):g}\u00d710^{exp}"
            self.result = s
            self.res_lbl.config(text=s)
            self.just_evaluated = True
        except Exception:
            pass

    # ---- CONST popup menu ----
    def _const_menu(self):
        m = tk.Menu(self, tearoff=0, bg=C["panel"], fg=C["white"],
                    activebackground=C["blue"], activeforeground="white",
                    bd=0)
        consts = [("\u03c0  (pi)", math.pi), ("e  (Euler)", math.e),
                  ("\u03c6  (golden)", 1.618033988749895),
                  ("\u221a2", math.sqrt(2)),
                  ("c  speed of light", 299792458.0),
                  ("g  gravity", 9.80665),
                  ("h  Planck", 6.62607015e-34),
                  ("N\u2090  Avogadro", 6.02214076e23),
                  ("k  Boltzmann", 1.380649e-23)]
        for name, val in consts:
            m.add_command(label=f"{name}  =  {val:g}",
                          command=lambda v=val: self._insert(repr(v)))
        try:
            m.tk_popup(self.winfo_pointerx(), self.winfo_pointery())
        finally:
            m.grab_release()

    # ---- numerical calculus (d/dx( , ∫( ) used from the keypad ----
    def _preprocess_calculus(self, s):
        for token, handler in (("d/dx(", self._num_deriv),
                               ("\u222b(", self._num_integ)):
            while token in s:
                i = s.index(token)
                j = i + len(token)
                depth = 1
                while j < len(s) and depth:
                    depth += 1 if s[j] == "(" else -1 if s[j] == ")" else 0
                    j += 1
                inner = s[i + len(token): j - 1]
                val = handler(self._split_top_commas(inner))
                s = s[:i] + ("%.12g" % val) + s[j:]
        return s

    @staticmethod
    def _split_top_commas(s):
        parts, depth, cur = [], 0, ""
        for ch in s:
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur); cur = ""
            else:
                cur += ch
        parts.append(cur)
        return parts

    def _num_deriv(self, args):
        expr = args[0]
        a = float(eval(self._to_eval(args[1]), self._plot_ns(), {})) if len(args) > 1 else 0.0
        f = lambda x: float(eval(self._to_eval(expr), self._plot_ns(), {"x": x}))
        h = 1e-6
        return (f(a + h) - f(a - h)) / (2 * h)

    def _num_integ(self, args):
        expr = args[0]
        a = float(eval(self._to_eval(args[1]), self._plot_ns(), {}))
        b = float(eval(self._to_eval(args[2]), self._plot_ns(), {}))
        f = lambda x: float(eval(self._to_eval(expr), self._plot_ns(), {"x": x}))
        n = 1000
        hstep = (b - a) / n
        total = f(a) + f(b)
        for k in range(1, n):
            total += (4 if k % 2 else 2) * f(a + k * hstep)
        return total * hstep / 3

    # --------------------------------------------------------------- sidebar
    def _build_sidebar(self, parent):
        # Size to content (top-aligned). Fixing the height here clipped the
        # CONSTANTS panel once the keypad got shorter, so let it grow instead;
        # the outer scroll canvas handles any overflow.
        side = tk.Frame(parent, bg=C["bg"], width=210)
        side.pack(side="right", fill="y", padx=(8, 0), anchor="n")
        side.pack_propagate(True)

        # --- CALCULATIONS ---
        o1, calc = bordered(side, C["btn_edge"], C["panel"])
        o1.pack(fill="x")
        tk.Label(calc, text="CALCULATIONS", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 10, "bold")).pack(anchor="w", padx=10, pady=(9, 4))
        items = [("\U0001F5A9", "Equation Solver"), ("\u25b3", "Quadratic Solver"),
                 ("\u2699", "System Solver"), ("\u25a6", "Matrix Calculator"),
                 ("\U0001F4CA", "Statistics"), ("\u25a4", "Unit Converter"),
                 ("\u25a7", "Base Converter"), ("\u2645", "Graph Plotter")]
        for icon, name in items:
            row = tk.Frame(calc, bg=C["panel"], cursor="hand2")
            row.pack(fill="x", padx=8, pady=3)
            ic = tk.Label(row, text=icon, bg=C["panel"], fg=C["sub"], font=(FONT, 10))
            ic.pack(side="left", padx=(3, 7))
            nm = tk.Label(row, text=name, bg=C["panel"], fg=C["white"], font=(FONT, 10))
            nm.pack(side="left")
            ar = tk.Label(row, text="\u203a", bg=C["panel"], fg=C["sub2"], font=(FONT, 12))
            ar.pack(side="right", padx=5)
            for w in (row, ic, nm, ar):
                w.bind("<Button-1>", lambda e, n=name: self._calc_tool(n))
                w.bind("<Enter>", lambda e, ws=(row, ic, nm, ar):
                       [x.config(bg=C["btn"]) for x in ws])
                w.bind("<Leave>", lambda e, ws=(row, ic, nm, ar):
                       [x.config(bg=C["panel"]) for x in ws])
        tk.Frame(calc, bg=C["panel"], height=5).pack()

        # --- CONSTANTS ---
        o2, cons = bordered(side, C["btn_edge"], C["panel"])
        o2.pack(fill="x", pady=(8, 0))
        tk.Label(cons, text="CONSTANTS", bg=C["panel"], fg=C["green"],
                 font=(FONT, 10, "bold")).pack(anchor="w", padx=10, pady=(9, 4))
        consts = [("\u03c0", "3.141592653589793", "\u03c0"),
                  ("e", "2.718281828459045", "e"),
                  ("\u03c6", "1.618033988749895", "1.618033988749895"),
                  ("\u03b3", "0.577215664901533", "0.577215664901533")]
        for sym, val, token in consts:
            row = tk.Frame(cons, bg=C["panel"], cursor="hand2")
            row.pack(fill="x", padx=10, pady=3)
            lbls = [row]
            lbls.append(tk.Label(row, text=sym, bg=C["panel"], fg=C["green"],
                                 font=(FONT, 11, "bold"), width=2))
            lbls[-1].pack(side="left")
            lbls.append(tk.Label(row, text=val, bg=C["panel"], fg=C["white"],
                                 font=(FONT_MONO, 8)))
            lbls[-1].pack(side="left", padx=6)
            for w in lbls:
                w.bind("<Button-1>", lambda e, t=token: self._insert(t))
        mc = tk.Label(cons, text="More Constants \u25be", bg=C["btn"],
                      fg=C["sub"], font=(FONT, 9), pady=6, cursor="hand2")
        mc.pack(fill="x", padx=10, pady=9)
        mc.bind("<Button-1>", lambda e: self._open_more_constants())

    # ---- CALCULATIONS sidebar routing ----
    def _calc_tool(self, name):
        routes = {"Equation Solver": "SOLVE", "System Solver": "SOLVE",
                  "Matrix Calculator": "MATRIX", "Statistics": "STAT",
                  "Unit Converter": "CONVERT", "Graph Plotter": "GRAPH"}
        if name == "Base Converter":
            self._base_converter_popup()
            return
        if name == "Quadratic Solver":
            self._set_mode("SOLVE")
            if getattr(self, "solve_input", None):
                self._solve_load("x^2 - 5x + 6 = 0")
            return
        self._set_mode(routes.get(name, "CALC"))

    def _base_converter_popup(self):
        if getattr(self, "_bc_win", None) and self._bc_win.winfo_exists():
            self._bc_win.lift(); return
        win = tk.Toplevel(self); self._bc_win = win
        win.title("Base Converter"); win.configure(bg=C["panel"])
        win.geometry("380x300"); win.transient(self)

        tk.Label(win, text="BASE CONVERTER", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 12, "bold")).pack(anchor="w", padx=16, pady=(14, 6))
        top = tk.Frame(win, bg=C["panel"]); top.pack(fill="x", padx=16)
        tk.Label(top, text="Value", bg=C["panel"], fg=C["sub"],
                 font=(FONT, 10)).pack(side="left")
        base_var = tk.StringVar(value="Decimal")
        om = tk.OptionMenu(top, base_var, "Decimal", "Binary", "Octal",
                           "Hexadecimal")
        om.config(bg=C["btn"], fg=C["white"], font=(FONT, 9), bd=0,
                  highlightthickness=0, activebackground=C["btn_hi"], width=12)
        om["menu"].config(bg=C["panel"], fg=C["white"])
        om.pack(side="right")

        entry = tk.Entry(win, bg=C["btn"], fg=C["white"], font=(FONT_MONO, 14),
                         relief="flat", insertbackground=C["white"],
                         highlightthickness=1, highlightbackground=C["btn_edge"],
                         highlightcolor=C["blue_hi"])
        entry.pack(fill="x", padx=16, pady=8, ipady=6)
        entry.insert(0, "255")

        out = tk.Frame(win, bg=C["panel"]); out.pack(fill="x", padx=16, pady=6)
        rows = {}
        for label in ("Decimal", "Binary", "Octal", "Hexadecimal"):
            r = tk.Frame(out, bg=C["panel"]); r.pack(fill="x", pady=3)
            tk.Label(r, text=label, bg=C["panel"], fg=C["sub"], font=(FONT, 10),
                     width=12, anchor="w").pack(side="left")
            v = tk.Label(r, text="", bg=C["panel"], fg=C["white"], font=(FONT_MONO, 12))
            v.pack(side="left")
            rows[label] = v

        def convert(*_):
            base = {"Decimal": 10, "Binary": 2, "Octal": 8, "Hexadecimal": 16}[base_var.get()]
            try:
                n = int(entry.get().strip(), base)
                rows["Decimal"].config(text=str(n), fg=C["white"])
                rows["Binary"].config(text=bin(n)[2:], fg=C["white"])
                rows["Octal"].config(text=oct(n)[2:], fg=C["white"])
                rows["Hexadecimal"].config(text=hex(n)[2:].upper(), fg=C["white"])
            except ValueError:
                for k in rows:
                    rows[k].config(text="invalid", fg="#c0392b")

        entry.bind("<KeyRelease>", convert)
        base_var.trace_add("write", convert)
        convert()


    EXTRA_CONSTANTS = [
        ("\u221a2",            "Square Root of 2",   "1.4142135623730951"),
        ("\u221a3",            "Square Root of 3",   "1.7320508075688772"),
        ("\u221b2",            "Cube Root of 2",     "1.2599210498948732"),
        ("ln(2)",              "Natural Log of 2",   "0.6931471805599453"),
        ("ln(10)",             "Natural Log of 10",  "2.302585092994046"),
        ("log\u2081\u2080(e)", "Base-10 Log of e",   "0.4342944819032518"),
        ("log\u2082(e)",       "Base-2 Log of e",    "1.4426950408889634"),
        ("G",                  "Catalan Constant",   "0.915965594177219"),
        ("\u03b6(3)",          "Apery's Constant",   "1.202056903159594"),
        ("K",                  "Khinchin's Constant","2.685452001065306"),
        ("C\u2082",            "Twin Prime Constant","0.660161815846869"),
        ("\u03b4",             "Feigenbaum Delta",   "4.66920160910299"),
        ("\u03b1",             "Feigenbaum Alpha",   "2.5029078750958928"),
    ]

    def _open_more_constants(self):
        if getattr(self, "_mc_win", None) and self._mc_win.winfo_exists():
            self._mc_win.lift()
            return
        win = tk.Toplevel(self)
        self._mc_win = win
        win.title("More Constants")
        win.configure(bg=C["panel"])
        win.geometry("560x460")
        win.transient(self)

        tk.Label(win, text="MORE CONSTANTS", bg=C["panel"], fg=C["green"],
                 font=(FONT, 12, "bold")).pack(anchor="w", padx=16, pady=(14, 2))
        tk.Label(win, text="Tap a constant to insert its value.", bg=C["panel"],
                 fg=C["sub"], font=(FONT, 9)).pack(anchor="w", padx=16, pady=(0, 8))

        # scrollable list ---------------------------------------------------
        wrap = tk.Frame(win, bg=C["panel"])
        wrap.pack(fill="both", expand=True, padx=8, pady=(0, 12))
        cv = tk.Canvas(wrap, bg=C["panel"], highlightthickness=0)
        sb = tk.Scrollbar(wrap, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(cv, bg=C["panel"])
        iid = cv.create_window((0, 0), window=inner, anchor="nw")
        cv.bind("<Configure>", lambda e: cv.itemconfigure(iid, width=e.width))
        inner.bind("<Configure>",
                   lambda e: cv.configure(scrollregion=cv.bbox("all")))

        def _wheel(e):
            cv.yview_scroll(-1 if (e.num == 4 or e.delta > 0) else 1, "units")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            cv.bind_all(seq, _wheel)
        win.bind("<Destroy>", lambda e: [cv.unbind_all(s) for s in
                 ("<MouseWheel>", "<Button-4>", "<Button-5>")])

        def insert_and_close(value):
            self._insert(value)
            win.destroy()

        for i, (sym, name, val) in enumerate(self.EXTRA_CONSTANTS):
            row = tk.Frame(inner, bg=C["panel"], cursor="hand2")
            row.pack(fill="x")
            cells = [
                tk.Label(row, text=sym, bg=C["panel"], fg=C["green"],
                         font=(FONT, 12, "bold"), width=7, anchor="w"),
                tk.Label(row, text=name, bg=C["panel"], fg=C["white"],
                         font=(FONT, 11), width=22, anchor="w"),
                tk.Label(row, text=val, bg=C["panel"], fg=C["sub"],
                         font=(FONT_MONO, 10), anchor="w"),
            ]
            cells[0].pack(side="left", padx=(12, 6), pady=9)
            cells[1].pack(side="left", padx=6)
            cells[2].pack(side="left", padx=6)
            # hover highlight + click to insert
            def on_enter(e, r=row, cs=cells):
                r.config(bg=C["btn"]);  [c.config(bg=C["btn"]) for c in cs]
            def on_leave(e, r=row, cs=cells):
                r.config(bg=C["panel"]); [c.config(bg=C["panel"]) for c in cs]
            for w in (row, *cells):
                w.bind("<Enter>", on_enter)
                w.bind("<Leave>", on_leave)
                w.bind("<Button-1>", lambda e, v=val: insert_and_close(v))
            tk.Frame(inner, bg=C["btn_edge"], height=1).pack(fill="x")

    # --------------------------------------------------------- bottom panels
    def _build_bottom_panels(self, parent):
        row = tk.Frame(parent, bg=C["bg"])
        row.pack(fill="x", pady=(8, 0))
        for i in range(3):
            row.grid_columnconfigure(i, weight=1, uniform="p")
        self._build_matrix_panel(row)
        self._build_stats_panel(row)
        self._build_graph_panel(row)

    # ============================ MATRIX ============================
    def _build_matrix_panel(self, row):
        o, m = bordered(row, C["purple_hi"], C["panel"])
        o.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        head = tk.Frame(m, bg=C["panel"]); head.pack(fill="x", padx=8, pady=(7, 3))
        tk.Label(head, text="MATRIX", bg=C["panel"], fg=C["purple_hi"],
                 font=(FONT, 10, "bold")).pack(side="left")
        tk.Label(head, text="2 \u00d7 2 \u25be", bg=C["panel"], fg=C["sub"],
                 font=(FONT, 9)).pack(side="right")

        body = tk.Frame(m, bg=C["panel"]); body.pack(padx=8, pady=4)
        self.matA = self._matrix_editor(body, "A", [[1, 2], [3, 4]])
        self.matB = self._matrix_editor(body, "B", [[5, 6], [7, 8]])

        ops = tk.Frame(m, bg=C["panel"]); ops.pack(fill="x", padx=8, pady=(4, 3))
        actions = [("A + B", lambda: self._mat_binop("+")),
                   ("A \u2212 B", lambda: self._mat_binop("-")),
                   ("A \u00d7 B", lambda: self._mat_binop("*")),
                   ("A\u207b\u00b9", self._mat_inverse),
                   ("|A|", self._mat_det)]
        for label, cmd in actions:
            self._panel_btn(ops, label, cmd, C["purple"], C["purple_hi"])

        self.mat_result = tk.Label(m, text="Result appears here", bg=C["panel"],
                                   fg=C["sub"], font=(FONT_MONO, 9),
                                   justify="left", anchor="w")
        self.mat_result.pack(fill="x", padx=8, pady=(2, 8))

    def _matrix_editor(self, parent, name, mat):
        """Return a 2x2 list of Entry widgets pre-filled with `mat`."""
        f = tk.Frame(parent, bg=C["panel"]); f.pack(side="left", padx=5)
        tk.Label(f, text=f"{name} =", bg=C["panel"], fg=C["white"],
                 font=(FONT, 10)).pack(side="left")
        grid = tk.Frame(f, bg=C["panel"]); grid.pack(side="left", padx=4)
        cells = []
        for r in range(2):
            rowcells = []
            for c in range(2):
                e = tk.Entry(grid, width=3, justify="center",
                             bg=C["btn"], fg=C["white"], font=(FONT_MONO, 10),
                             relief="flat", insertbackground=C["white"],
                             highlightthickness=1, highlightbackground=C["btn_edge"],
                             highlightcolor=C["purple_hi"])
                e.insert(0, str(mat[r][c]))
                e.grid(row=r, column=c, padx=2, pady=2)
                rowcells.append(e)
            cells.append(rowcells)
        return cells

    def _read_matrix(self, cells):
        return [[float(cells[r][c].get()) for c in range(2)] for r in range(2)]

    @staticmethod
    def _fmt_matrix(mat):
        def cell(v):
            v = round(v, 6)
            return f"{int(v) if v == int(v) else v:g}"
        return "\n".join("  ".join(cell(v).rjust(7) for v in r) for r in mat)

    def _mat_binop(self, op):
        try:
            A, B = self._read_matrix(self.matA), self._read_matrix(self.matB)
            if op == "+":
                R = [[A[r][c] + B[r][c] for c in range(2)] for r in range(2)]
            elif op == "-":
                R = [[A[r][c] - B[r][c] for c in range(2)] for r in range(2)]
            else:  # matrix multiply
                R = [[sum(A[r][k] * B[k][c] for k in range(2))
                      for c in range(2)] for r in range(2)]
            self.mat_result.config(text=self._fmt_matrix(R), fg=C["white"])
        except ValueError:
            self.mat_result.config(text="Enter valid numbers", fg=C["amber"])

    def _mat_inverse(self):
        try:
            A = self._read_matrix(self.matA)
            det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
            if abs(det) < 1e-12:
                self.mat_result.config(text="A is singular (det = 0)", fg=C["amber"])
                return
            inv = [[A[1][1] / det, -A[0][1] / det],
                   [-A[1][0] / det, A[0][0] / det]]
            self.mat_result.config(text=self._fmt_matrix(inv), fg=C["white"])
        except ValueError:
            self.mat_result.config(text="Enter valid numbers", fg=C["amber"])

    def _mat_det(self):
        try:
            A = self._read_matrix(self.matA)
            det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
            det = round(det, 6)
            self.mat_result.config(
                text=f"|A| = {int(det) if det == int(det) else det:g}",
                fg=C["white"])
        except ValueError:
            self.mat_result.config(text="Enter valid numbers", fg=C["amber"])

    # ========================== STATISTICS ==========================
    def _build_stats_panel(self, row):
        o, s = bordered(row, C["amber"], C["panel"])
        o.grid(row=0, column=1, sticky="nsew", padx=4)
        tk.Label(s, text="STATISTICS", bg=C["panel"], fg=C["amber"],
                 font=(FONT, 10, "bold")).pack(anchor="w", padx=8, pady=(7, 3))

        drow = tk.Frame(s, bg=C["panel"]); drow.pack(fill="x", padx=8)
        tk.Label(drow, text="Data:", bg=C["panel"], fg=C["white"],
                 font=(FONT, 9)).pack(side="left")
        self.stat_entry = tk.Entry(drow, bg=C["btn"], fg=C["white"],
                                   font=(FONT, 9), relief="flat",
                                   insertbackground=C["white"],
                                   highlightthickness=1,
                                   highlightbackground=C["btn_edge"],
                                   highlightcolor=C["amber"])
        self.stat_entry.insert(0, "2, 4, 6, 8, 10")
        self.stat_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.stat_entry.bind("<KeyRelease>", lambda e: self._recompute_stats())

        self.stat_rows = tk.Frame(s, bg=C["panel"]); self.stat_rows.pack(fill="x")
        self.stat_more = False
        self.more_btn = self._panel_btn(
            s, "More Stats \u25be", self._toggle_more_stats, C["btn"], C["btn_hi"])
        self.more_btn.pack_configure(fill="x", padx=8, pady=7)
        self._recompute_stats()

    def _toggle_more_stats(self):
        self.stat_more = not self.stat_more
        arrow = "\u25b4" if self.stat_more else "\u25be"
        self.more_btn.lbl.config(text=f"{'Less' if self.stat_more else 'More'} Stats {arrow}")
        self._recompute_stats()

    def _recompute_stats(self):
        raw = self.stat_entry.get().replace(";", ",")
        try:
            data = [float(x) for x in raw.split(",") if x.strip() != ""]
            if not data:
                raise ValueError
        except ValueError:
            self._render_stat_rows([("Data", "invalid \u2014 use commas")])
            return
        self._render_stat_rows(self._stats(data, self.stat_more))

    def _render_stat_rows(self, pairs):
        for w in self.stat_rows.winfo_children():
            w.destroy()
        for name, val in pairs:
            r = tk.Frame(self.stat_rows, bg=C["panel"])
            r.pack(fill="x", padx=8, pady=0)
            tk.Label(r, text=name, bg=C["panel"], fg=C["sub"],
                     font=(FONT, 9)).pack(side="left")
            tk.Label(r, text=val, bg=C["panel"], fg=C["white"],
                     font=(FONT, 9)).pack(side="right")

    # ============================= GRAPH ============================
    def _build_graph_panel(self, row):
        self.graph_scale = 1.0          # 1.0 => x spans -2pi..2pi
        self.graph_expr = "sin(x)"
        o, g = bordered(row, C["blue"], C["panel"])
        o.grid(row=0, column=2, sticky="nsew", padx=(4, 0))
        head = tk.Frame(g, bg=C["panel"]); head.pack(fill="x", padx=8, pady=(7, 3))
        tk.Label(head, text="GRAPH", bg=C["panel"], fg=C["blue_hi"],
                 font=(FONT, 10, "bold")).pack(side="left")
        self.graph_title = tk.Label(head, text="y = sin(x)", bg=C["panel"],
                                    fg=C["sub"], font=(FONT, 9))
        self.graph_title.pack(side="right")

        frow = tk.Frame(g, bg=C["panel"]); frow.pack(fill="x", padx=8, pady=(0, 2))
        tk.Label(frow, text="y =", bg=C["panel"], fg=C["white"],
                 font=(FONT, 9)).pack(side="left")
        self.graph_entry = tk.Entry(frow, bg=C["btn"], fg=C["white"],
                                    font=(FONT_MONO, 9), relief="flat",
                                    insertbackground=C["white"],
                                    highlightthickness=1,
                                    highlightbackground=C["btn_edge"],
                                    highlightcolor=C["blue_hi"])
        self.graph_entry.insert(0, "sin(x)")
        self.graph_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.graph_entry.bind("<Return>", lambda e: self._plot_expr())

        self.graph_cv = tk.Canvas(g, bg=C["panel"], height=88,
                                  highlightthickness=0)
        self.graph_cv.pack(fill="x", padx=8, pady=3)
        self.graph_cv.bind("<Configure>", lambda e: self._draw_graph())

        btns = tk.Frame(g, bg=C["panel"]); btns.pack(fill="x", padx=8, pady=(3, 8))
        for label, cmd in (("Zoom In", lambda: self._graph_zoom(0.7)),
                           ("Zoom Out", lambda: self._graph_zoom(1.4)),
                           ("Reset", self._graph_reset)):
            self._panel_btn(btns, label, cmd, C["btn"], C["btn_hi"]
                            ).pack_configure(expand=True, fill="x")

    def _plot_expr(self):
        self.graph_expr = self.graph_entry.get().strip() or "sin(x)"
        self.graph_title.config(text=f"y = {self.graph_expr}")
        self._draw_graph()

    def _graph_zoom(self, factor):
        self.graph_scale = max(0.05, min(20, self.graph_scale * factor))
        self._draw_graph()

    def _graph_reset(self):
        self.graph_scale = 1.0
        self.graph_entry.delete(0, "end"); self.graph_entry.insert(0, "sin(x)")
        self.graph_expr = "sin(x)"
        self.graph_title.config(text="y = sin(x)")
        self._draw_graph()

    def _graph_func(self):
        """Compile current expr into a f(x); trig here is always radians."""
        py = self._to_python(self.graph_expr)
        ns = self._namespace()
        ns["sin"] = math.sin; ns["cos"] = math.cos; ns["tan"] = math.tan
        def f(x):
            local = {"x": x}
            return eval(py, ns, local)      # noqa: S307 (sandboxed ns)
        return f

    def _draw_graph(self):
        cv = self.graph_cv
        cv.delete("all")
        w = cv.winfo_width() or 240
        h = int(cv["height"]); midy = h / 2
        span = 2 * math.pi * self.graph_scale          # half-width in x
        yspan = span                                   # keep aspect squarish
        cv.create_line(0, midy, w, midy, fill=C["sub2"])
        cv.create_line(w / 2, 0, w / 2, h, fill=C["sub2"])
        try:
            f = self._graph_func()
        except Exception:
            cv.create_text(w / 2, midy, text="invalid expression",
                           fill=C["amber"], font=(FONT, 11)); return
        pts, seg = [], []
        for px in range(w):
            x = (px - w / 2) / (w / 2) * span
            try:
                y = f(x)
                py = midy - (y / yspan) * (h / 2 - 8)
                if -h < py < 2 * h:
                    seg.extend((px, py)); continue
            except Exception:
                pass
            if len(seg) >= 4:
                pts.append(seg)
            seg = []
        if len(seg) >= 4:
            pts.append(seg)
        for s in pts:
            cv.create_line(*s, fill=C["blue_hi"], width=2, smooth=True)

    # ---------------------------------------------------------------- stats
    @staticmethod
    def _stats(data, more=False):
        n = len(data)
        mean = sum(data) / n
        srt = sorted(data)
        median = (srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2)
        from collections import Counter
        mode = Counter(data).most_common(1)[0][0]
        var = sum((x - mean) ** 2 for x in data) / n
        std = math.sqrt(var)
        g = lambda v: f"{round(v, 6):g}"
        base = [("Mean", g(mean)), ("Median", g(median)), ("Mode", g(mode)),
                ("Std Dev", g(std)), ("Variance", g(var))]
        if not more:
            return base

        def quantile(p):
            if n == 1:
                return srt[0]
            pos = p * (n - 1)
            lo = int(math.floor(pos)); frac = pos - lo
            hi = min(lo + 1, n - 1)
            return srt[lo] + (srt[hi] - srt[lo]) * frac
        q1, q3 = quantile(0.25), quantile(0.75)
        sstd = math.sqrt(sum((x - mean) ** 2 for x in data) / (n - 1)) if n > 1 else 0
        base += [("Count", g(n)), ("Sum", g(sum(data))),
                 ("Min", g(min(data))), ("Max", g(max(data))),
                 ("Range", g(max(data) - min(data))),
                 ("Q1", g(q1)), ("Q3", g(q3)), ("IQR", g(q3 - q1)),
                 ("Sample SD", g(sstd))]
        return base

    # --------------------------------------------------- panel button helper
    def _panel_btn(self, parent, label, cmd, bg, hi):
        b = Key(parent, label, command=cmd, bg=bg, hi=hi, fg=C["white"],
                font_size=9, height=28)
        b.pack(side="left", expand=True, fill="x", padx=2)
        return b

    # ------------------------------------------------------ input handlers
    def _insert(self, token):
        if self.just_evaluated and token not in "+\u2212\u00d7\u00f7^":
            self.expr = ""          # start fresh after '='
        self.just_evaluated = False
        self.expr += token
        self._refresh_display()
        self._reset_shift()         # SHIFT / ALPHA are one-shot modifiers

    def insert_ans(self):
        self._insert(self.result)

    def negate(self):
        MINUS = "\u2212"
        # after '=', negate the result and start a fresh expression
        if self.just_evaluated:
            r = self.result.lstrip("-" + MINUS)
            self.expr = r if self.result[:1] in ("-", MINUS) else MINUS + r
            self.just_evaluated = False
            self._refresh_display()
            return

        e = self.expr
        # find the start of the trailing number (digits / decimal point)
        i = len(e)
        while i > 0 and (e[i - 1].isdigit() or e[i - 1] == "."):
            i -= 1
        # a MINUS right before the number is unary if it sits at the start or
        # follows an operator / open-paren / comma -> toggle it off; else add it
        if i > 0 and e[i - 1] == MINUS and (i - 1 == 0 or e[i - 2] in "+" + MINUS + "\u00d7\u00f7^(,"):
            self.expr = e[:i - 1] + e[i:]
        else:
            self.expr = e[:i] + MINUS + e[i:]
        self._refresh_display()

    def backspace(self):
        self.expr = self.expr[:-1]
        self.just_evaluated = False
        self._refresh_display()

    def clear_all(self):
        self.expr = ""
        self.result = "0"
        self.just_evaluated = False
        self._refresh_display()

    def _toggle_angle(self):
        self.angle = "RAD" if self.angle == "DEG" else "DEG"
        self.deg_lbl.config(text=f"{self.angle} \u25be")
        self.deg_display.config(text=self.angle)

    # ----------------------------------------------------------- evaluation
    def _to_python(self, s):
        s = s.replace("\u00d7", "*").replace("\u00f7", "/")
        s = s.replace("\u2212", "-").replace("^", "**")
        s = s.replace("\u03c0", "pi").replace("\u221a", "sqrt")
        # infix combinatorics:  5C2 -> nCr(5,2) ,  5P2 -> nPr(5,2)
        s = re.sub(r"(\d+\.?\d*)C(\d+\.?\d*)", r"nCr(\1,\2)", s)
        s = re.sub(r"(\d+\.?\d*)P(\d+\.?\d*)", r"nPr(\1,\2)", s)
        # Mod = modulo (Python %) ; a bare % = percent (÷100)
        s = s.replace(" mod ", "\x00")     # park modulo
        s = s.replace("%", "/100")         # percent
        s = s.replace("\x00", "%")         # restore modulo
        s = self._apply_factorial(s)       # handles  5!  and  (2+3)!
        return s

    @staticmethod
    def _apply_factorial(s):
        """Turn a trailing '!' into factorial(...), wrapping either a run of
        digits (5!) or a balanced parenthesised group ((2+3)!)."""
        while "!" in s:
            i = s.index("!")
            j = i - 1
            if j < 0:                     # leading '!', nothing to wrap
                s = s[:i] + s[i + 1:]
                continue
            if s[j] == ")":               # walk back to the matching '('
                depth, k = 0, j
                while k >= 0:
                    depth += 1 if s[k] == ")" else -1 if s[k] == "(" else 0
                    if depth == 0:
                        break
                    k -= 1
                start = k
            else:                         # grab the trailing number
                k = j
                while k >= 0 and (s[k].isdigit() or s[k] == "."):
                    k -= 1
                start = k + 1
            operand = s[start:i]
            if operand == "":             # can't resolve -> drop the '!'
                s = s[:i] + s[i + 1:]
                continue
            s = s[:start] + "factorial(" + operand + ")" + s[i + 1:]
        return s

    def _namespace(self):
        deg = self.angle == "DEG"
        conv = math.radians if deg else (lambda x: x)
        aconv = math.degrees if deg else (lambda x: x)
        return {
            "__builtins__": {},
            "pi": math.pi, "e": math.e, "tau": math.tau,
            "sqrt": math.sqrt, "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
            "abs": abs, "factorial": lambda n: math.factorial(int(n)),
            "nCr": lambda n, r: math.comb(int(n), int(r)),
            "nPr": lambda n, r: math.perm(int(n), int(r)),
            "log": math.log10, "ln": math.log, "exp": math.exp,
            "logb": lambda x, b: math.log(x, b),
            "sin": lambda x: math.sin(conv(x)), "cos": lambda x: math.cos(conv(x)),
            "tan": lambda x: math.tan(conv(x)),
            "csc": lambda x: 1 / math.sin(conv(x)),
            "sec": lambda x: 1 / math.cos(conv(x)),
            "cot": lambda x: 1 / math.tan(conv(x)),
            "asin": lambda x: aconv(math.asin(x)),
            "acos": lambda x: aconv(math.acos(x)),
            "atan": lambda x: aconv(math.atan(x)),
            "acsc": lambda x: aconv(math.asin(1 / x)),
            "asec": lambda x: aconv(math.acos(1 / x)),
            "acot": lambda x: aconv(math.atan(1 / x)),
        }

    # ---- pretty math typesetting for the display (mirrors the mockup) ----
    _SUP = {"0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3",
            "4": "\u2074", "5": "\u2075", "6": "\u2076", "7": "\u2077",
            "8": "\u2078", "9": "\u2079", "-": "\u207b", "+": "\u207a",
            "(": "\u207d", ")": "\u207e", "n": "\u207f", "x": "\u02e3",
            "y": "\u02b8"}

    def _pretty(self, s):
        """Turn the raw expression into math notation for the LCD:
        log(->log10(, sqrt without parens, ^n as superscripts, spaced ops."""
        if not s:
            return " "
        out = s
        out = out.replace("log(", "log\u2081\u2080(")     # log -> log₁₀
        out = out.replace("cbrt(", "\u221b(")             # cbrt -> ∛
        out = out.replace("asin(", "sin\u207b\u00b9(").replace("acos(", "cos\u207b\u00b9(")
        out = out.replace("atan(", "tan\u207b\u00b9(").replace("acsc(", "csc\u207b\u00b9(")
        out = out.replace("asec(", "sec\u207b\u00b9(").replace("acot(", "cot\u207b\u00b9(")
        out = out.replace(" mod ", " mod ")
        out = re.sub(r"\u221a\(([^()]*)\)", lambda m: "\u221a" + m.group(1), out)   # √(16) -> √16
        # ^(...) group  and  ^<signed digits>  -> superscript
        out = re.sub(r"\^\(([^()]*)\)",
                     lambda m: "".join(self._SUP.get(c, c) for c in "(" + m.group(1) + ")"),
                     out)
        out = re.sub(r"\^(-?[0-9a-z]+)",
                     lambda m: "".join(self._SUP.get(c, c) for c in m.group(1)), out)
        # spaces around the binary operators (× ÷ +)
        out = re.sub(r"\s*([\u00d7\u00f7+])\s*", r" \1 ", out)
        # spaced minus, but not a leading unary one
        out = re.sub(r"(?<=[\w\)\u03c0])\s*\u2212\s*", " \u2212 ", out)
        return out.strip()

    @staticmethod
    def _fmt_num(val):
        if isinstance(val, float):
            val = round(val, 12)
            if val == int(val):
                val = int(val)
        return f"{val:g}" if isinstance(val, float) else str(val)

    def _split_terms(self, s):
        """Split on top-level + / − , returning [(sign, body), ...]."""
        terms, depth, cur, sign = [], 0, "", "+"
        for ch in s:
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            if depth == 0 and ch in "+\u2212" and cur.strip():
                terms.append((sign, cur))
                sign, cur = ("-" if ch == "\u2212" else "+"), ""
            else:
                cur += ch
        if cur.strip():
            terms.append((sign, cur))
        return terms

    def evaluate(self):
        if not self.expr:
            return
        expr = self.expr
        missing = expr.count("(") - expr.count(")")
        if missing > 0:                                   # auto-close brackets
            expr = expr + ")" * missing
        try:
            ns = self._namespace()
            eval_expr = self._preprocess_calculus(expr)   # d/dx( , ∫( -> numbers
            val = eval(self._to_python(eval_expr), ns, {})     # noqa: S307
            self.result = self._fmt_num(val)
            self.expr = expr

            # ---- HISTORY: break the expression into its additive terms ----
            entries, pieces = [], []
            for sign, body in self._split_terms(eval_expr):
                b = body.lstrip("+\u2212")
                tv = eval(self._to_python(b), ns, {})
                tvs = self._fmt_num(tv)
                entries.append((self._pretty(b), tvs))
                pieces.append(("\u2212 " if sign == "-" else "" if not pieces else "+ ") + tvs)
            self._hist_entries = entries
            self._render_history()
            self.sum_lbl.config(text=" ".join(pieces)[:40] if len(entries) > 1 else "")
            self.just_evaluated = True
        except ZeroDivisionError:
            self.result = "\u00f7 by zero"
        except (SyntaxError, TypeError):
            self.result = "Syntax error"
        except Exception:
            self.result = "Error"
        self._refresh_display()

    def _refresh_display(self):
        self.expr_lbl.config(text=self._pretty(self.expr))
        self.res_lbl.config(text=self.result)


if __name__ == "__main__":
    SciCalc().mainloop()