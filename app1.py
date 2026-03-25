import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.feature import blob_log
from skimage.filters import gaussian
import io
import base64

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HRTEM · Bravais Lattice Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:        #030810;
    --surface:   #080f1e;
    --surface2:  #0d1a2e;
    --border:    #112240;
    --accent:    #00d4ff;
    --accent2:   #7c3aed;
    --accent3:   #10b981;
    --warn:      #f59e0b;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --glow:      0 0 40px rgba(0,212,255,.15);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,212,255,.07) 0%, transparent 60%),
        radial-gradient(ellipse 40% 30% at 90% 80%, rgba(124,58,237,.06) 0%, transparent 50%),
        var(--bg) !important;
}

/* hide default streamlit chrome */
[data-testid="stHeader"],
[data-testid="stToolbar"],
footer { display: none !important; }

/* main content padding */
[data-testid="stMain"] > div:first-child { padding-top: 0 !important; }
.block-container { padding: 0 2rem 2rem !important; max-width: 1400px !important; }

/* ── hero bar ── */
.hero {
    padding: 2.8rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
    position: relative;
}
.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: .72rem;
    letter-spacing: .22em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: .5rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.05;
    background: linear-gradient(135deg, #e2e8f0 0%, #00d4ff 60%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 .6rem;
}
.hero-sub {
    color: var(--muted);
    font-size: .95rem;
    font-weight: 400;
    letter-spacing: .01em;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,212,255,.08);
    border: 1px solid rgba(0,212,255,.25);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: .68rem;
    letter-spacing: .12em;
    padding: .2rem .65rem;
    border-radius: 2px;
    text-transform: uppercase;
    margin-top: .8rem;
}

/* ── panels ── */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    opacity: .6;
}
.panel-title {
    font-family: 'Space Mono', monospace;
    font-size: .7rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.panel-title .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    display: inline-block;
    box-shadow: 0 0 8px var(--accent);
}

/* ── result card ── */
.result-card {
    background: linear-gradient(135deg, rgba(0,212,255,.05), rgba(124,58,237,.05));
    border: 1px solid rgba(0,212,255,.3);
    border-radius: 10px;
    padding: 1.8rem;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: var(--glow);
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: .7rem;
    letter-spacing: .2em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: .4rem;
}
.result-name {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
    text-shadow: 0 0 30px rgba(0,212,255,.4);
    letter-spacing: -.02em;
}
.result-full {
    font-size: 1rem;
    color: var(--text);
    margin-top: .35rem;
    font-weight: 600;
}
.confidence-bar-wrap {
    margin-top: 1.2rem;
}
.confidence-label {
    font-family: 'Space Mono', monospace;
    font-size: .68rem;
    color: var(--muted);
    letter-spacing: .1em;
    display: flex;
    justify-content: space-between;
    margin-bottom: .4rem;
}
.confidence-track {
    background: rgba(255,255,255,.05);
    border-radius: 2px;
    height: 6px;
    overflow: hidden;
}
.confidence-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    transition: width .6s ease;
    box-shadow: 0 0 10px rgba(0,212,255,.5);
}

/* ── meta chips ── */
.meta-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: .8rem;
    margin-top: .5rem;
}
.meta-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: .7rem 1rem;
}
.meta-chip .label {
    font-family: 'Space Mono', monospace;
    font-size: .6rem;
    color: var(--muted);
    letter-spacing: .12em;
    text-transform: uppercase;
}
.meta-chip .value {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    margin-top: .15rem;
}

/* ── upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 8px !important;
    transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] label { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { color: var(--muted) !important; }

/* ── top-k table ── */
.topk-row {
    display: flex;
    align-items: center;
    gap: .8rem;
    padding: .55rem 0;
    border-bottom: 1px solid var(--border);
}
.topk-row:last-child { border-bottom: none; }
.topk-rank {
    font-family: 'Space Mono', monospace;
    font-size: .65rem;
    color: var(--muted);
    width: 1.5rem;
    flex-shrink: 0;
}
.topk-name {
    font-family: 'Space Mono', monospace;
    font-size: .82rem;
    color: var(--text);
    width: 2.8rem;
    flex-shrink: 0;
}
.topk-bar-wrap { flex: 1; background: rgba(255,255,255,.04); border-radius: 2px; height: 5px; }
.topk-bar { height: 100%; border-radius: 2px; }
.topk-pct {
    font-family: 'Space Mono', monospace;
    font-size: .72rem;
    color: var(--muted);
    width: 3.5rem;
    text-align: right;
    flex-shrink: 0;
}

/* ── info panel ── */
.lattice-info-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: .4rem;
}
.lattice-desc {
    font-size: .88rem;
    color: var(--muted);
    line-height: 1.6;
}
.prop-row {
    display: flex;
    justify-content: space-between;
    padding: .5rem 0;
    border-bottom: 1px solid var(--border);
    font-size: .85rem;
}
.prop-row .k { color: var(--muted); font-family: 'Space Mono', monospace; font-size: .75rem; }
.prop-row .v { color: var(--text); font-weight: 600; }

/* ── image renders ── */
.img-label {
    font-family: 'Space Mono', monospace;
    font-size: .65rem;
    letter-spacing: .14em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: .5rem;
}
img { border-radius: 4px; }

/* ── hint box ── */
.hint-box {
    background: rgba(245,158,11,.04);
    border: 1px solid rgba(245,158,11,.2);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-top: 1.2rem;
}
.hint-box p { font-size: .83rem; color: var(--muted); margin: 0; line-height: 1.6; }
.hint-box strong { color: var(--warn); font-family: 'Space Mono', monospace; font-size: .75rem; letter-spacing: .1em; }

/* ── scanline texture overlay ── */
.scanlines {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,.03) 2px, rgba(0,0,0,.03) 4px);
    pointer-events: none;
    z-index: 9999;
}

/* ── streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,212,255,.1), rgba(124,58,237,.1)) !important;
    border: 1px solid rgba(0,212,255,.35) !important;
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: .72rem !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    padding: .5rem 1.2rem !important;
    border-radius: 4px !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    background: rgba(0,212,255,.15) !important;
    box-shadow: 0 0 20px rgba(0,212,255,.2) !important;
}
.stSpinner > div { color: var(--accent) !important; }

/* ── divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

</style>
<div class="scanlines"></div>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────────
BRAVAIS_LABELS = {
    'cP': 0, 'cI': 1, 'cF': 2,
    'tP': 3, 'tI': 4,
    'oP': 5, 'oI': 6, 'oF': 7, 'oC': 8,
    'hP': 9, 'hR': 10,
    'mP': 11, 'mC': 12,
    'aP': 13,
}
LABEL_TO_BRAVAIS = {v: k for k, v in BRAVAIS_LABELS.items()}

BRAVAIS_META = {
    'cP': ('Cubic Primitive',       'Simple cubic. One lattice point per unit cell. Least efficient packing.',       'Cubic',       'Primitive',    'a = b = c, α=β=γ=90°'),
    'cI': ('Cubic Body-Centered',   'BCC. Two lattice points per cell. Common in metals like Fe, W, Cr.',            'Cubic',       'Body-Centered','a = b = c, α=β=γ=90°'),
    'cF': ('Cubic Face-Centered',   'FCC. Four lattice points per cell. Close-packed; Cu, Al, Au, Ni.',              'Cubic',       'Face-Centered','a = b = c, α=β=γ=90°'),
    'tP': ('Tetragonal Primitive',  'Square base with different height. One lattice point per cell.',                'Tetragonal',  'Primitive',    'a = b ≠ c, α=β=γ=90°'),
    'tI': ('Tetragonal Body-Centered','BCC tetragonal. Two points per cell. Found in In, Sn (white).',              'Tetragonal',  'Body-Centered','a = b ≠ c, α=β=γ=90°'),
    'oP': ('Orthorhombic Primitive', 'Three unequal axes, all right angles. One point per cell.',                   'Orthorhombic','Primitive',    'a ≠ b ≠ c, α=β=γ=90°'),
    'oI': ('Orthorhombic BCC',      'Body-centered orthorhombic. Two lattice points per cell.',                     'Orthorhombic','Body-Centered','a ≠ b ≠ c, α=β=γ=90°'),
    'oF': ('Orthorhombic FCC',      'All-face-centered orthorhombic. Four points per cell.',                        'Orthorhombic','Face-Centered','a ≠ b ≠ c, α=β=γ=90°'),
    'oC': ('Orthorhombic Base-Centered','C-face-centered orthorhombic. Two points per cell.',                       'Orthorhombic','Base-Centered','a ≠ b ≠ c, α=β=γ=90°'),
    'hP': ('Hexagonal Primitive',   'Hexagonal symmetry. Graphite, Mg, Ti, Zn crystallise here.',                  'Hexagonal',   'Primitive',    'a = b ≠ c, α=β=90°, γ=120°'),
    'hR': ('Rhombohedral',          'Three equal axes, equal non-right angles. Calcite, Bi, Sb.',                   'Trigonal',    'Rhombohedral', 'a = b = c, α=β=γ ≠ 90°'),
    'mP': ('Monoclinic Primitive',  'One unique angle ≠ 90°. Very common in minerals and organics.',                'Monoclinic',  'Primitive',    'a ≠ b ≠ c, α=γ=90° ≠ β'),
    'mC': ('Monoclinic Base-Centered','C-face-centered monoclinic. Two lattice points per cell.',                   'Monoclinic',  'Base-Centered','a ≠ b ≠ c, α=γ=90° ≠ β'),
    'aP': ('Triclinic Primitive',   'Lowest symmetry. All axes and angles differ. Feldspar, kaolinite.',            'Triclinic',   'Primitive',    'a ≠ b ≠ c, α ≠ β ≠ γ ≠ 90°'),
}

CRYSTAL_SYSTEMS = {
    'Cubic': '#00d4ff', 'Tetragonal': '#7c3aed', 'Orthorhombic': '#10b981',
    'Hexagonal': '#f59e0b', 'Trigonal': '#ef4444', 'Monoclinic': '#8b5cf6', 'Triclinic': '#ec4899',
}

PHYS_DIM   = 128
TOP_N      = 8
N_CLASSES  = 14
DEVICE     = "cpu"


# ─── Model definition ────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class Encoder(nn.Module):
    def __init__(self, in_channels=2, feat_dim=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
        )
        self.layer1 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.down1  = nn.Sequential(nn.Conv2d(64,  128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.GELU())
        self.layer2 = nn.Sequential(ResBlock(128), ResBlock(128))
        self.down2  = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.GELU())
        self.layer3 = nn.Sequential(ResBlock(256), ResBlock(256))
        self.down3  = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(512), nn.GELU())
        self.pool   = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.down1(x)
        x = self.layer2(x); x = self.down2(x)
        x = self.layer3(x); x = self.down3(x)
        return self.pool(x).view(x.size(0), -1)


class Classifier(nn.Module):
    def __init__(self, encoder, phys_dim=PHYS_DIM, n_classes=N_CLASSES, feat_dim=512):
        super().__init__()
        self.encoder  = encoder
        self.phys_net = nn.Sequential(
            nn.Linear(phys_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim + 128, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, 256),            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x, phys):
        return self.head(torch.cat([self.encoder(x), self.phys_net(phys)], dim=1))


@st.cache_resource
def load_model():
    encoder = Encoder(in_channels=2, feat_dim=512).to(DEVICE)
    model   = Classifier(encoder).to(DEVICE)
    state   = torch.load("best_clf.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# ─── Signal processing ───────────────────────────────────────────────────────────
def preprocess(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = gaussian(img, sigma=1.0, preserve_range=True)
    img = np.log(img + 1e-6)
    return (img - img.min()) / (img.max() - img.min() + 1e-6)


def detect_spots(img_pp: np.ndarray) -> np.ndarray:
    blobs = blob_log(img_pp, min_sigma=1.5, max_sigma=5.0, num_sigma=8, threshold=0.08)
    return blobs.astype(np.float32) if len(blobs) > 0 else np.zeros((0, 3), dtype=np.float32)


def extract_physics_features(image: np.ndarray, image_size: int = 128) -> np.ndarray:
    img_pp = preprocess(image)
    spots  = detect_spots(img_pp)[:TOP_N]
    R = (np.linalg.norm(spots[:, :2] - image_size / 2, axis=1) / (image_size / 2 + 1e-6)
         if len(spots) > 0 else np.array([]))
    feat = np.zeros(PHYS_DIM, dtype=np.float32)
    feat[:len(R)] = R[:PHYS_DIM]
    return feat


def add_fft(x: torch.Tensor) -> torch.Tensor:
    fft_mag = torch.log(torch.abs(torch.fft.fft2(x)) + 1e-6)
    fft_mag = (fft_mag - fft_mag.min()) / (fft_mag.max() - fft_mag.min() + 1e-6)
    return torch.cat([x, fft_mag], dim=0)


# ─── Render helpers ──────────────────────────────────────────────────────────────
def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150,
                facecolor="none", transparent=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def render_image_panel(arr: np.ndarray, cmap="inferno", title="") -> str:
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.imshow(arr, cmap=cmap, aspect="equal", interpolation="nearest")
    ax.axis("off")
    fig.patch.set_alpha(0)
    plt.tight_layout(pad=0)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return f'<div class="img-label">{title}</div><img src="data:image/png;base64,{b64}" style="width:100%; border-radius:6px;">'


def render_spots_panel(img: np.ndarray, title="Blob detection") -> str:
    img_pp = preprocess(img)
    blobs  = detect_spots(img_pp)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.imshow(img_pp, cmap="gray", aspect="equal", interpolation="nearest")
    for blob in blobs[:TOP_N]:
        y, x, r = blob
        circle = plt.Circle((x, y), r * 1.5, color="#00d4ff", linewidth=1.2,
                             fill=False, alpha=0.9)
        ax.add_patch(circle)
    ax.axis("off")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    fig.patch.set_alpha(0)
    plt.tight_layout(pad=0)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    n_spots = min(len(blobs), TOP_N)
    return (f'<div class="img-label">{title} &nbsp;<span style="color:#00d4ff;font-family:\'Space Mono\',monospace;font-size:.65rem">'
            f'{n_spots} spots</span></div>'
            f'<img src="data:image/png;base64,{b64}" style="width:100%; border-radius:6px;">')


def render_fft_panel(img_tensor: torch.Tensor, title="FFT magnitude") -> str:
    fft_raw = torch.fft.fft2(img_tensor)
    fft_shifted = torch.fft.fftshift(fft_raw)
    fft_mag = torch.log(torch.abs(fft_shifted) + 1e-6).numpy()
    fft_mag = (fft_mag - fft_mag.min()) / (fft_mag.max() - fft_mag.min() + 1e-6)
    return render_image_panel(fft_mag, cmap="magma", title=title)


def render_topk(probs: torch.Tensor, top_k: int = 7) -> str:
    vals, idxs = torch.topk(probs, top_k)
    system_colors = {k: CRYSTAL_SYSTEMS[v[2]] for k, v in BRAVAIS_META.items()}
    rows = ""
    for rank, (idx, val) in enumerate(zip(idxs, vals)):
        name  = LABEL_TO_BRAVAIS[idx.item()]
        pct   = val.item() * 100
        color = system_colors.get(name, "#64748b")
        width = f"{pct:.1f}%"
        rows += f"""
        <div class="topk-row">
            <span class="topk-rank">#{rank+1}</span>
            <span class="topk-name" style="color:{color}">{name}</span>
            <div class="topk-bar-wrap">
                <div class="topk-bar" style="width:{width}; background:{color}; opacity:.75;"></div>
            </div>
            <span class="topk-pct">{pct:.2f}%</span>
        </div>"""
    return rows


def render_polar_chart(probs: torch.Tensor) -> str:
    labels = list(BRAVAIS_LABELS.keys())
    values = probs.numpy()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values_plot = np.concatenate([values, [values[0]]])
    angles_plot = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(3.8, 3.8))
    ax  = fig.add_subplot(111, projection="polar")
    ax.set_facecolor("#080f1e")
    fig.patch.set_facecolor("#080f1e")

    # fill
    ax.fill(angles_plot, values_plot, alpha=0.2, color="#00d4ff")
    ax.plot(angles_plot, values_plot, color="#00d4ff", linewidth=1.5, alpha=0.9)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=7, color="#94a3b8",
                       fontfamily="monospace")
    ax.set_yticklabels([])
    ax.grid(color="#112240", linewidth=0.6)
    ax.spines["polar"].set_color("#112240")
    ax.tick_params(pad=4)

    plt.tight_layout(pad=.5)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return f'<div class="img-label">Probability radar</div><img src="data:image/png;base64,{b64}" style="width:100%;">'


def render_phys_bar(phys: np.ndarray) -> str:
    active = phys[phys > 0]
    if len(active) == 0:
        return '<div class="img-label">No spots detected</div>'

    fig, ax = plt.subplots(figsize=(4, 1.8))
    ax.set_facecolor("#080f1e")
    fig.patch.set_facecolor("#080f1e")
    x = np.arange(len(active))
    colors = plt.cm.cool(np.linspace(0.2, 0.9, len(active)))
    ax.bar(x, active, color=colors, width=0.7, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{i+1}" for i in x], fontsize=6.5, color="#64748b",
                       fontfamily="monospace")
    ax.yaxis.set_tick_params(labelcolor="#64748b", labelsize=6.5)
    ax.grid(axis="y", color="#112240", linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#112240")
    ax.set_ylabel("norm. radius", color="#64748b", fontsize=6.5)
    plt.tight_layout(pad=0.4)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return (f'<div class="img-label">Spot radii &nbsp;<span style="color:#00d4ff;'
            f'font-family:\'Space Mono\',monospace;font-size:.65rem">{len(active)} detected</span></div>'
            f'<img src="data:image/png;base64,{b64}" style="width:100%; border-radius:6px;">')


# ─── Hero ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Electron Microscopy · AI Analysis</div>
    <div class="hero-title">HRTEM Lattice Classifier</div>
    <div class="hero-sub">Upload a .pt sample file to identify the Bravais lattice type using a ResNet encoder + physics feature fusion model.</div>
    <span class="hero-badge">14 lattice classes &nbsp;·&nbsp; CNN + FFT + blob detection</span>
</div>
""", unsafe_allow_html=True)

# ─── Layout ──────────────────────────────────────────────────────────────────────
col_upload, col_result, col_info = st.columns([1.1, 1.4, 1.5])

with col_upload:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><span class="dot"></span>Input</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a .pt file here",
        type=["pt"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div class="hint-box">
        <p><strong>FORMAT</strong><br>
        PyTorch dict with keys:<br>
        <code style="color:#f59e0b;font-size:.75rem">image</code> · tensor (128×128)<br>
        <code style="color:#f59e0b;font-size:.75rem">phys</code> · tensor (128,) optional</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Crystal system legend
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><span class="dot" style="background:#7c3aed;box-shadow:0 0 8px #7c3aed"></span>Crystal systems</div>', unsafe_allow_html=True)
    legend_html = ""
    for sys_name, color in CRYSTAL_SYSTEMS.items():
        members = [k for k, v in BRAVAIS_META.items() if v[2] == sys_name]
        chips   = " ".join(f'<span style="background:rgba(255,255,255,.05);border:1px solid {color}33;color:{color};font-family:\'Space Mono\',monospace;font-size:.6rem;padding:.1rem .35rem;border-radius:2px">{m}</span>' for m in members)
        legend_html += f'<div style="margin-bottom:.6rem"><span style="color:{color};font-size:.75rem;font-weight:700">{sys_name}</span><br><div style="margin-top:.25rem;display:flex;flex-wrap:wrap;gap:.25rem">{chips}</div></div>'
    st.markdown(legend_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ─── Run inference when file uploaded ────────────────────────────────────────────
if uploaded is not None:
    with st.spinner("Running model..."):
        try:
            model = load_model()
            data  = torch.load(uploaded, map_location=DEVICE)
            img   = data["image"].float().numpy()

            img_tensor  = add_fft(torch.from_numpy(img).unsqueeze(0).float()).unsqueeze(0).to(DEVICE)
            phys_np     = data["phys"].float().numpy() if "phys" in data else extract_physics_features(img)
            phys_tensor = torch.from_numpy(phys_np).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(img_tensor, phys_tensor)
                probs  = F.softmax(logits, dim=1).squeeze()

            pred_idx    = probs.argmax().item()
            pred_name   = LABEL_TO_BRAVAIS[pred_idx]
            confidence  = probs[pred_idx].item()
            true_label  = LABEL_TO_BRAVAIS.get(data.get("label", -1), "—")
            fold        = data.get("fold", "—")
            meta        = BRAVAIS_META[pred_name]
            sys_color   = CRYSTAL_SYSTEMS[meta[2]]

            # ── Result column ──────────────────────────────────────────────────
            with col_result:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<div class="panel-title"><span class="dot" style="background:#10b981;box-shadow:0 0 8px #10b981"></span>Prediction</div>', unsafe_allow_html=True)

                correct_mark = ""
                if true_label != "—":
                    correct_mark = (" ✓" if true_label == pred_name else " ✗")

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Bravais lattice type</div>
                    <div class="result-name" style="color:{sys_color};text-shadow:0 0 30px {sys_color}66">{pred_name}</div>
                    <div class="result-full">{meta[0]}{correct_mark}</div>
                    <div class="confidence-bar-wrap">
                        <div class="confidence-label">
                            <span>Confidence</span>
                            <span style="color:{sys_color}">{confidence:.2%}</span>
                        </div>
                        <div class="confidence-track">
                            <div class="confidence-fill" style="width:{confidence*100:.1f}%;background:linear-gradient(90deg,{sys_color},{sys_color}99)"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # meta chips
                st.markdown(f"""
                <div class="meta-grid">
                    <div class="meta-chip">
                        <div class="label">Crystal system</div>
                        <div class="value" style="color:{sys_color}">{meta[2]}</div>
                    </div>
                    <div class="meta-chip">
                        <div class="label">Centering</div>
                        <div class="value">{meta[3]}</div>
                    </div>
                    <div class="meta-chip">
                        <div class="label">Ground truth</div>
                        <div class="value" style="color:{'#10b981' if true_label==pred_name else '#ef4444'}">{true_label}</div>
                    </div>
                    <div class="meta-chip">
                        <div class="label">Fold</div>
                        <div class="value">{fold}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # top-k ranking
                st.markdown('<div class="panel-title" style="margin-top:.5rem"><span class="dot" style="background:#f59e0b;box-shadow:0 0 8px #f59e0b"></span>Top-7 probabilities</div>', unsafe_allow_html=True)
                st.markdown(render_topk(probs, 7), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(render_polar_chart(probs), unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            # ── Info column ────────────────────────────────────────────────────
            with col_info:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<div class="panel-title"><span class="dot" style="background:#7c3aed;box-shadow:0 0 8px #7c3aed"></span>Visualizations</div>', unsafe_allow_html=True)

                img_tensor_single = torch.from_numpy(img).float()

                v1, v2 = st.columns(2)
                with v1:
                    st.markdown(render_image_panel(img, cmap="inferno", title="Raw image"), unsafe_allow_html=True)
                with v2:
                    st.markdown(render_fft_panel(img_tensor_single, title="FFT magnitude"), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                v3, v4 = st.columns(2)
                with v3:
                    st.markdown(render_spots_panel(img, title="Blob detection"), unsafe_allow_html=True)
                with v4:
                    st.markdown(render_image_panel(preprocess(img), cmap="viridis", title="Preprocessed"), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(render_phys_bar(phys_np), unsafe_allow_html=True)

                st.markdown("<hr>", unsafe_allow_html=True)

                # lattice info
                st.markdown(f"""
                <div class="lattice-info-header" style="color:{sys_color}">{meta[0]}</div>
                <div class="lattice-desc">{meta[1]}</div>
                <div style="margin-top:1rem">
                    <div class="prop-row">
                        <span class="k">Parameters</span>
                        <span class="v">{meta[4]}</span>
                    </div>
                    <div class="prop-row">
                        <span class="k">Crystal system</span>
                        <span class="v" style="color:{sys_color}">{meta[2]}</span>
                    </div>
                    <div class="prop-row">
                        <span class="k">Centering type</span>
                        <span class="v">{meta[3]}</span>
                    </div>
                    <div class="prop-row">
                        <span class="k">Symbol</span>
                        <span class="v" style="font-family:'Space Mono',monospace">{pred_name}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

else:
    # ── Empty states ──────────────────────────────────────────────────────────
    with col_result:
        st.markdown("""
        <div class="panel" style="min-height:320px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;gap:1rem">
            <div style="font-size:3rem;opacity:.15">⬡</div>
            <div style="font-family:'Space Mono',monospace;font-size:.7rem;letter-spacing:.15em;color:var(--muted);text-transform:uppercase">Awaiting sample</div>
            <div style="font-size:.82rem;color:var(--muted);max-width:220px;line-height:1.6">Upload a .pt file to run inference and see the classification result.</div>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown("""
        <div class="panel" style="min-height:320px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;gap:1rem">
            <div style="font-size:3rem;opacity:.15">◈</div>
            <div style="font-family:'Space Mono',monospace;font-size:.7rem;letter-spacing:.15em;color:var(--muted);text-transform:uppercase">No data yet</div>
            <div style="font-size:.82rem;color:var(--muted);max-width:260px;line-height:1.6">Visualizations (raw image, FFT, blob detection, spot radii) will appear here after classification.</div>
        </div>
        """, unsafe_allow_html=True)
