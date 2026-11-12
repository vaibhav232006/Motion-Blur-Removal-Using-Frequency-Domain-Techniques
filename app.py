import base64
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="OPTIX // Frequency-Domain Deconvolution Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. PRO DARK-MODE WORKSTATION CSS (Hides default Streamlit bloat)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* Global Dark Theme Overrides */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background-color: #090d16;
    color: #e2e8f0;
}

/* Hide default Streamlit top bar, hamburger menu, and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: rgba(9, 13, 22, 0.8);
    backdrop-filter: blur(10px);
}

/* Tighten main container padding */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1450px;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #0d1322 !important;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
}

/* Top Studio Navbar */
.studio-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(135deg, #0f172a 0%, #131c31 100%);
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.1rem 1.6rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
}
.studio-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}
.brand-badge {
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 6px 10px;
    border-radius: 8px;
    letter-spacing: 0.08em;
}
.brand-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0;
    letter-spacing: -0.02em;
}
.brand-sub {
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 2px 0 0 0;
}
.status-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.25);
    padding: 6px 12px;
    border-radius: 999px;
}

/* Custom KPI Metric Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 1.25rem;
}
.kpi-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    transition: border-color 0.2s ease;
}
.kpi-card:hover {
    border-color: #3b82f6;
}
.kpi-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 6px;
    font-weight: 600;
}
.kpi-val-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    color: #f8fafc;
}
.kpi-delta-pos {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    color: #10b981;
    background: rgba(16, 185, 129, 0.12);
    padding: 2px 8px;
    border-radius: 6px;
}
.kpi-delta-neg {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    color: #f43f5e;
    background: rgba(244, 63, 94, 0.12);
    padding: 2px 8px;
    border-radius: 6px;
}

/* Image Frame Cards */
.img-panel {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 12px;
}
.img-panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e293b;
}
.img-panel-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #e2e8f0;
}
.img-panel-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #94a3b8;
    background: #1e293b;
    padding: 2px 8px;
    border-radius: 5px;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0f172a;
    padding: 6px;
    border-radius: 10px;
    border: 1px solid #1e293b;
    gap: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    color: #94a3b8;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background-color: #1e293b !important;
    color: #38bdf8 !important;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background-color: #0f172a;
    border: 1px dashed #334155;
    border-radius: 12px;
    padding: 10px;
}

/* Buttons */
.stDownloadButton > button, .stButton > button {
    background: linear-gradient(135deg, #2563eb, #0284c7);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.55rem 1.2rem;
    transition: all 0.2s ease;
}
.stDownloadButton > button:hover, .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. IMAGE PROCESSING & SIGNAL ALGORITHMS
# -----------------------------------------------------------------------------
def generate_calibration_chart(width: int = 720, height: int = 480) -> np.ndarray:
    """Creates a high-contrast synthetic test target with sharp text, grids, and geometry."""
    img = np.full((height, width, 3), 18, dtype=np.uint8)
    # Subtle grid
    for x in range(0, width, 40):
        cv2.line(img, (x, 0), (x, height), (35, 45, 65), 1)
    for y in range(0, height, 40):
        cv2.line(img, (0, y), (width, y), (35, 45, 65), 1)

    # Geometric shapes
    cv2.circle(img, (170, 220), 95, (56, 189, 248), 4)
    cv2.circle(img, (170, 220), 55, (244, 63, 94), -1)
    cv2.rectangle(img, (320, 125), (640, 315), (16, 185, 129), 3)

    # Resolution bars
    for i in range(10):
        x_pos = 345 + i * 26
        cv2.rectangle(img, (x_pos, 150), (x_pos + 11, 230), (248, 250, 252), -1)

    # Sharp typography
    cv2.putText(img, "OPTIX // PSF TEST TARGET", (40, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (248, 250, 252), 2, cv2.LINE_AA)
    cv2.putText(img, "FFT WIENER DECONVOLUTION 2026", (342, 285),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (56, 189, 248), 1, cv2.LINE_AA)
    cv2.putText(img, "EDGE SHARPNESS & RINGING VERIFICATION", (40, 425),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (148, 163, 184), 2, cv2.LINE_AA)
    return img

def create_motion_psf(size: int, angle: float) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    center = (size / 2 - 0.5, size / 2 - 0.5)
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (size, size))
    s = np.sum(kernel)
    return kernel / s if s != 0 else kernel

def deconvolve_channel(channel: np.ndarray, psf: np.ndarray, K: float) -> np.ndarray:
    h, w = channel.shape
    kh, kw = psf.shape
    H = np.zeros((h, w), dtype=np.float32)
    H[:kh, :kw] = psf
    H = np.roll(H, -(kh // 2), axis=0)
    H = np.roll(H, -(kw // 2), axis=1)

    G = np.fft.fft2(channel.astype(np.float32))
    H_fft = np.fft.fft2(H)
    W = np.conj(H_fft) / (np.abs(H_fft) ** 2 + K)
    restored = np.abs(np.fft.ifft2(W * G))
    return np.clip(restored, 0, 255).astype(np.uint8)

def compute_metrics(ref: np.ndarray, test: np.ndarray):
    mse = float(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2))
    psnr = 99.99 if mse == 0 else float(10 * np.log10((255.0 ** 2) / mse))
    return mse, psnr

def img_to_base64(img_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) if img_rgb.ndim == 3 else img_rgb
    _, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf).decode("utf-8")

def render_split_slider(before_rgb: np.ndarray, after_rgb: np.ndarray, height_px: int = 440):
    """Renders an interactive Before/After draggable comparison slider."""
    b64_before = img_to_base64(before_rgb)
    b64_after = img_to_base64(after_rgb)
    html_code = f"""
    <div style="position:relative; width:100%; height:{height_px}px; overflow:hidden; border-radius:12px; border:1px solid #1e293b; background:#05080f; user-select:none;">
        <img src="data:image/png;base64,{b64_after}" style="position:absolute; top:0; left:0; width:100%; height:100%; object-fit:contain;" />
        <div id="clipBox" style="position:absolute; top:0; left:0; width:50%; height:100%; overflow:hidden; border-right:2px solid #38bdf8;">
            <img src="data:image/png;base64,{b64_before}" style="position:absolute; top:0; left:0; width:100vw; max-width:none; height:100%; object-fit:contain;" id="beforeImg" />
        </div>
        <div style="position:absolute; top:14px; left:14px; background:rgba(15,23,42,0.85); color:#f43f5e; font-family:monospace; font-size:12px; padding:4px 10px; border-radius:6px; border:1px solid rgba(244,63,94,0.4);">
            MOTION BLURRED
        </div>
        <div style="position:absolute; top:14px; right:14px; background:rgba(15,23,42,0.85); color:#10b981; font-family:monospace; font-size:12px; padding:4px 10px; border-radius:6px; border:1px solid rgba(16,185,129,0.4);">
            WIENER RESTORED
        </div>
        <input type="range" min="0" max="100" value="50" id="sliderRange"
               style="position:absolute; top:0; left:0; width:100%; height:100%; opacity:0; cursor:ew-resize; z-index:10; margin:0;" />
        <div id="handleDot" style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:36px; height:36px; border-radius:50%; background:#38bdf8; color:#090d16; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:14px; box-shadow:0 0 15px rgba(56,189,248,0.8); pointer-events:none;">
            ⇄
        </div>
    </div>
    <script>
        const slider = document.getElementById('sliderRange');
        const clipBox = document.getElementById('clipBox');
        const handle = document.getElementById('handleDot');
        const beforeImg = document.getElementById('beforeImg');
        function syncWidth() {{
            beforeImg.style.width = slider.offsetWidth + 'px';
        }}
        window.addEventListener('resize', syncWidth);
        syncWidth();
        slider.addEventListener('input', (e) => {{
            const val = e.target.value + '%';
            clipBox.style.width = val;
            handle.style.left = val;
        }});
    </script>
    """
    components.html(html_code, height=height_px + 8)

# -----------------------------------------------------------------------------
# 4. TOP STUDIO HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="studio-topbar">
    <div class="studio-brand">
        <div class="brand-badge">OPTIX // FFT</div>
        <div>
            <div class="brand-title">Frequency-Domain Motion Deconvolution Studio</div>
            <div class="brand-sub">2D Fast Fourier Transform • Point Spread Function Modeling • Parametric Wiener Filtering</div>
        </div>
    </div>
    <div class="status-pill">● ENGINE READY // REAL-TIME DSP</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. SIDEBAR WORKSTATION CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ DSP Control Console")
    
    source_choice = st.radio(
        "Input Source",
        ["Upload Custom Image", "Built-in Calibration Target"],
        index=0
    )

    color_mode = st.selectbox(
        "Color Space Pipeline",
        ["RGB Full-Spectrum (3-Channel)", "Luminance Grayscale (1-Channel)"]
    )

    st.markdown("---")
    st.markdown("#### 1️⃣ Motion Degradation (PSF)")
    kernel_size = st.slider("Blur Trajectory Length (px)", 5, 49, 17, step=2)
    blur_angle = st.slider("Motion Vector Angle (°)", 0, 180, 15, step=5)

    st.markdown("---")
    st.markdown("#### 2️⃣ Restoration Filter")
    filter_type = st.radio(
        "Deconvolution Algorithm",
        ["Wiener Filter (Regularized)", "Naive Inverse Filter (Unregularized)"],
        help="Compare regularized Wiener filtering against unregularized inverse filtering (K ≈ 0)."
    )

    if filter_type == "Wiener Filter (Regularized)":
        k_log = st.slider("Regularization Constant K (10ˣ)", -4.5, -0.5, -2.1, step=0.1)
        K = 10 ** k_log
        st.caption(f"Noise-to-Signal Ratio: `K = {K:.6f}`")
    else:
        K = 1e-7
        st.warning("⚠️ Unregularized mode (`K = 1e-7`) demonstrates severe noise amplification at PSF zero-crossings.")

# -----------------------------------------------------------------------------
# 6. LOAD IMAGE (UPLOADED OR CALIBRATION TARGET)
# -----------------------------------------------------------------------------
original_rgb = None

if source_choice == "Upload Custom Image":
    uploaded = st.file_uploader(
        "Drop a JPG or PNG image here to begin frequency-domain restoration",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded is not None:
        arr = np.frombuffer(uploaded.read(), np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
else:
    original_rgb = generate_calibration_chart()

if original_rgb is None:
    st.info("👈 Upload an image above or switch **Input Source** in the sidebar to **Built-in Calibration Target** for an instant demo.")
    st.stop()

# Resize very large images slightly for instantaneous real-time slider responsiveness
max_dim = 900
h0, w0 = original_rgb.shape[:2]
if max(h0, w0) > max_dim:
    scale = max_dim / max(h0, w0)
    original_rgb = cv2.resize(original_rgb, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

is_gray = color_mode.startswith("Luminance")
proc_orig = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY) if is_gray else original_rgb

# Apply Motion Blur
psf = create_motion_psf(kernel_size, blur_angle)
proc_blur = cv2.filter2D(proc_orig, -1, psf)

# Apply Deconvolution
if is_gray:
    proc_rest = deconvolve_channel(proc_blur, psf, K)
    gray_blur = proc_blur
    gray_rest = proc_rest
else:
    proc_rest = cv2.merge([deconvolve_channel(c, psf, K) for c in cv2.split(proc_blur)])
    gray_blur = cv2.cvtColor(proc_blur, cv2.COLOR_RGB2GRAY)
    gray_rest = cv2.cvtColor(proc_rest, cv2.COLOR_RGB2GRAY)

# Frequency Spectra
def get_spectrum(gray_img: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(gray_img.astype(np.float32)))
    mag = 20 * np.log(np.abs(f) + 1)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

spec_blur = get_spectrum(gray_blur)
spec_rest = get_spectrum(gray_rest)

# Metrics
mse_b, psnr_b = compute_metrics(proc_orig, proc_blur)
mse_r, psnr_r = compute_metrics(proc_orig, proc_rest)
psnr_gain = psnr_r - psnr_b
mse_drop = mse_r - mse_b

# -----------------------------------------------------------------------------
# 7. KPI TELEMETRY CARDS
# -----------------------------------------------------------------------------
psnr_badge = f'<span class="kpi-delta-pos">{psnr_gain:+.2f} dB</span>' if psnr_gain >= 0 else f'<span class="kpi-delta-neg">{psnr_gain:+.2f} dB</span>'
mse_badge = f'<span class="kpi-delta-pos">{mse_drop:+.1f}</span>' if mse_drop <= 0 else f'<span class="kpi-delta-neg">{mse_drop:+.1f}</span>'

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Degraded PSNR</div>
        <div class="kpi-val-row">
            <div class="kpi-value">{psnr_b:.2f} dB</div>
            <span class="img-panel-tag">INPUT</span>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Restored PSNR</div>
        <div class="kpi-val-row">
            <div class="kpi-value">{psnr_r:.2f} dB</div>
            {psnr_badge}
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Degraded MSE</div>
        <div class="kpi-val-row">
            <div class="kpi-value">{mse_b:.1f}</div>
            <span class="img-panel-tag">ERROR</span>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Restored MSE</div>
        <div class="kpi-val-row">
            <div class="kpi-value">{mse_r:.1f}</div>
            {mse_badge}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 8. MULTI-TAB STUDIO WORKSPACE
# -----------------------------------------------------------------------------
tab_split, tab_tri, tab_fft, tab_math = st.tabs([
    "⚡ Interactive Split Inspector",
    "🖼️ 3-Stage Pipeline Grid",
    "🌊 Fourier Spectrum & PSF",
    "📐 Mathematical Theory"
])

with tab_split:
    st.caption("Drag the blue slider handle left and right to inspect edge recovery between the blurred input and restored output.")
    disp_blur = cv2.cvtColor(proc_blur, cv2.COLOR_GRAY2RGB) if is_gray else proc_blur
    disp_rest = cv2.cvtColor(proc_rest, cv2.COLOR_GRAY2RGB) if is_gray else proc_rest
    render_split_slider(disp_blur, disp_rest, height_px=460)

with tab_tri:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="img-panel"><div class="img-panel-header"><span class="img-panel-title">01. Ground Truth</span><span class="img-panel-tag">ORIGINAL</span></div></div>', unsafe_allow_html=True)
        st.image(proc_orig, use_container_width=True)
    with c2:
        st.markdown(f'<div class="img-panel"><div class="img-panel-header"><span class="img-panel-title">02. Motion Blurred</span><span class="img-panel-tag">{kernel_size}px @ {blur_angle}°</span></div></div>', unsafe_allow_html=True)
        st.image(proc_blur, use_container_width=True)
    with c3:
        st.markdown(f'<div class="img-panel"><div class="img-panel-header"><span class="img-panel-title">03. Restored Output</span><span class="img-panel-tag">K={K:.1e}</span></div></div>', unsafe_allow_html=True)
        st.image(proc_rest, use_container_width=True)

with tab_fft:
    f1, f2, f3 = st.columns([1, 1.4, 1.4])
    with f1:
        st.markdown('<div class="img-panel"><div class="img-panel-header"><span class="img-panel-title">Point Spread Function</span><span class="img-panel-tag">SPATIAL PSF</span></div></div>', unsafe_allow_html=True)
        psf_vis = cv2.normalize(psf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        psf_vis = cv2.resize(psf_vis, (280, 280), interpolation=cv2.INTER_NEAREST)
        st.image(psf_vis, use_container_width=True)
    with f2:
        st.markdown('<div class="img-panel"><div class="img-panel-header"><span class="img-panel-title">Blurred Log-Spectrum</span><span class="img-panel-tag">|G(u,v)|</span></div></div>', unsafe_allow_html=True)
        st.image(spec_blur, use_container_width=True)
    with f3:
        st.markdown('<div class="img-panel"><div class="img-panel-header"><span class="img-panel-title">Restored Log-Spectrum</span><span class="img-panel-tag">|F̂(u,v)|</span></div></div>', unsafe_allow_html=True)
        st.image(spec_rest, use_container_width=True)

with tab_math:
    st.markdown("### Frequency-Domain Degradation & Wiener Deconvolution")
    st.latex(r"g(x,y) = (f * h)(x,y) + \eta(x,y) \xrightarrow{\mathcal{F}} G(u,v) = F(u,v)H(u,v) + N(u,v)")
    st.latex(r"\hat{F}(u,v) = \left[\frac{H^*(u,v)}{|H(u,v)|^2 + K}\right] G(u,v), \quad K = \frac{S_{\eta}(u,v)}{S_{f}(u,v)}")
    st.markdown("""
    - **Why Naive Inverse Filtering Fails:** When $K \\to 0$, zeros in $H(u,v)$ cause division by near-zero values, amplifying additive sensor noise $\\eta(x,y)$ across the entire image.
    - **Why Wiener Filtering Succeeds:** Adding $K$ dampens frequencies where $|H(u,v)|^2$ is small while faithfully inverting high-SNR frequency components.
    """)

# -----------------------------------------------------------------------------
# 9. EXPORT BAR
# -----------------------------------------------------------------------------
st.markdown("---")
ex1, ex2, ex3 = st.columns([1, 1, 2])
out_bgr = proc_rest if is_gray else cv2.cvtColor(proc_rest, cv2.COLOR_RGB2BGR)
blur_bgr = proc_blur if is_gray else cv2.cvtColor(proc_blur, cv2.COLOR_RGB2BGR)

_, enc_rest = cv2.imencode(".png", out_bgr)
_, enc_blur = cv2.imencode(".png", blur_bgr)

ex1.download_button(
    "⬇️ Export Restored PNG",
    data=enc_rest.tobytes(),
    file_name="optix_restored.png",
    mime="image/png",
    use_container_width=True
)
ex2.download_button(
    "⬇️ Export Blurred PNG",
    data=enc_blur.tobytes(),
    file_name="optix_blurred.png",
    mime="image/png",
    use_container_width=True
)