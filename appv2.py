"""
Streamlit UI for the CBIR TrashNet project.

Run with:
    streamlit run app.py

Requires:
    - cbir_index_v2.pkl (built by the notebook)
    - The TrashNet dataset images at the paths stored in the pickle
"""
import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import kagglehub

# Download dataset automatically on Streamlit Cloud
@st.cache_resource(show_spinner="Downloading dataset...")
def get_dataset_root():
    path = kagglehub.dataset_download("asdasdasasdas/garbage-classification")
    # Find the actual image root
    for p in Path(path).rglob("cardboard"):
        return p.parent
    return Path(path)

DATASET_ROOT = get_dataset_root()
# ═══════════════════════════════════════════════════════════════════
# CONFIG — must match the notebook exactly
# ═══════════════════════════════════════════════════════════════════
INDEX_FILE = 'cbir_index_v2.pkl'

RESIZE_DIM   = (256, 256)
H_BINS, S_BINS, V_BINS = 16, 16, 8
GRAY_LEVEL   = 32
HOG_ORIENTATIONS    = 9
HOG_PIX_PER_CELL    = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

W_COLOR_HIST = 0.25
W_COLOR_MOM  = 0.10
W_GLCM       = 0.20
W_LBP        = 0.15
W_HOG        = 0.30

ROCCHIO_ALPHA = 1.0
ROCCHIO_BETA  = 0.75
ROCCHIO_GAMMA = 0.15

FEATURE_KEYS = ['color_hist', 'color_mom', 'glcm', 'lbp', 'hog']

# ═══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTORS — copied verbatim from the notebook
# ═══════════════════════════════════════════════════════════════════
def create_foreground_mask(image_bgr):
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError('create_foreground_mask received an empty image')
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    if mask.sum() / 255 < (mask.shape[0] * mask.shape[1] * 0.05):
        mask = np.ones_like(mask) * 255
    return mask


def extract_color_histogram(image_bgr):
    img  = cv2.resize(image_bgr, RESIZE_DIM)
    mask = create_foreground_mask(img)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask,
                        [H_BINS, S_BINS, V_BINS],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def extract_color_moments(image_bgr):
    img  = cv2.resize(image_bgr, RESIZE_DIM)
    mask = create_foreground_mask(img)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
    mask_bool = mask > 0
    if mask_bool.sum() == 0:
        return np.zeros(9, dtype=np.float32)
    moments = []
    for ch in range(3):
        pixels = hsv[:, :, ch][mask_bool]
        mean = pixels.mean()
        std  = pixels.std()
        diff = pixels - mean
        skew_raw = np.mean(diff ** 3)
        skew = np.sign(skew_raw) * (abs(skew_raw) ** (1/3))
        moments.extend([mean, std, skew])
    return np.array(moments, dtype=np.float32)


def _build_glcm_vectorised(gray, dx, dy, levels, mask=None):
    h, w = gray.shape
    if dy >= 0:
        rows_a = slice(0, h - dy);    rows_b = slice(dy, h)
    else:
        rows_a = slice(-dy, h);       rows_b = slice(0, h + dy)
    if dx >= 0:
        cols_a = slice(0, w - dx);    cols_b = slice(dx, w)
    else:
        cols_a = slice(-dx, w);       cols_b = slice(0, w + dx)
    a = gray[rows_a, cols_a].ravel()
    b = gray[rows_b, cols_b].ravel()
    if mask is not None:
        ma = mask[rows_a, cols_a].ravel() > 0
        mb = mask[rows_b, cols_b].ravel() > 0
        valid = ma & mb
        a = a[valid]; b = b[valid]
    if a.size == 0:
        return np.zeros((levels, levels), dtype=np.float64)
    pair_codes = a.astype(np.int64) * levels + b.astype(np.int64)
    counts = np.bincount(pair_codes, minlength=levels * levels)
    glcm = counts.reshape(levels, levels).astype(np.float64)
    total = glcm.sum()
    if total > 0:
        glcm /= total
    return glcm


def _haralick(glcm):
    levels = glcm.shape[0]
    i_idx, j_idx = np.meshgrid(range(levels), range(levels), indexing='ij')
    diff = (i_idx - j_idx).astype(np.float64)
    asm = float(np.sum(glcm ** 2))
    con = float(np.sum(diff ** 2 * glcm))
    nz  = glcm[glcm > 0]
    ent = float(-np.sum(nz * np.log2(nz))) if nz.size > 0 else 0.0
    idm = float(np.sum(glcm / (1.0 + diff ** 2)))
    return [asm, con, ent, idm]


def extract_glcm_features(image_bgr):
    img  = cv2.resize(image_bgr, RESIZE_DIM)
    mask = create_foreground_mask(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = (gray.astype(np.int32) * GRAY_LEVEL // 256).clip(0, GRAY_LEVEL - 1).astype(np.int32)
    offsets = [(1, 0), (1, -1), (0, -1)]
    features = []
    for dx, dy in offsets:
        glcm = _build_glcm_vectorised(gray, dx, dy, GRAY_LEVEL, mask=mask)
        features.extend(_haralick(glcm))
    return np.array(features, dtype=np.float32)


_LBP_LUT = None
def _get_lbp_lut():
    global _LBP_LUT
    if _LBP_LUT is None:
        lut = np.zeros(256, dtype=np.uint8)
        for x in range(256):
            bits = [(x >> i) & 1 for i in range(8)]
            t = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
            lut[x] = bin(x).count('1') if t <= 2 else 9
        _LBP_LUT = lut
    return _LBP_LUT


def extract_lbp_features(image_bgr):
    img  = cv2.resize(image_bgr, RESIZE_DIM)
    mask = create_foreground_mask(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = np.pad(gray, 1, mode='edge').astype(np.int32)
    centre = g[1:-1, 1:-1]
    neighbours = [
        g[0:-2, 0:-2], g[0:-2, 1:-1], g[0:-2, 2:],
        g[1:-1, 2:],
        g[2:,   2:],   g[2:,   1:-1], g[2:,   0:-2],
        g[1:-1, 0:-2],
    ]
    code = np.zeros_like(centre, dtype=np.uint8)
    for bit, nb in enumerate(neighbours):
        code |= ((nb >= centre).astype(np.uint8) << bit)
    uniform_code = _get_lbp_lut()[code]
    fg = mask > 0
    vals = uniform_code[fg]
    if vals.size == 0:
        return np.zeros(10, dtype=np.float32)
    hist = np.bincount(vals, minlength=10).astype(np.float32)
    hist /= hist.sum()
    return hist


def extract_hog_features(image_bgr):
    img  = cv2.resize(image_bgr, RESIZE_DIM)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    win_size = RESIZE_DIM
    block_size = (HOG_CELLS_PER_BLOCK[0] * HOG_PIX_PER_CELL[0],
                  HOG_CELLS_PER_BLOCK[1] * HOG_PIX_PER_CELL[1])
    hog = cv2.HOGDescriptor(win_size, block_size, HOG_PIX_PER_CELL,
                            HOG_PIX_PER_CELL, HOG_ORIENTATIONS)
    feat = hog.compute(gray).flatten()
    n = np.linalg.norm(feat)
    if n > 0:
        feat = feat / n
    return feat.astype(np.float32)


def extract_all_features(image_bgr):
    return {
        'color_hist': extract_color_histogram(image_bgr),
        'color_mom':  extract_color_moments(image_bgr),
        'glcm':       extract_glcm_features(image_bgr),
        'lbp':        extract_lbp_features(image_bgr),
        'hog':        extract_hog_features(image_bgr),
    }


# ═══════════════════════════════════════════════════════════════════
# DISTANCE + RETRIEVAL
# ═══════════════════════════════════════════════════════════════════
def euclidean_distance(v1, v2):
    a, b = np.asarray(v1), np.asarray(v2)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def chi_square_distance(h1, h2):
    a = np.asarray(h1, dtype=np.float64)
    b = np.asarray(h2, dtype=np.float64)
    denom = a + b
    mask  = denom > 0
    return float(np.sum(((a[mask] - b[mask]) ** 2) / denom[mask]))


def compute_combined_distances(q_feat, index):
    """
    Returns (paths, combined_dist_array, per_component_dict).
    Per-component distances are min-max normalised across the candidate set.
    """
    paths = list(index.keys())
    n = len(paths)

    d_color_hist = np.zeros(n, dtype=np.float64)
    d_color_mom  = np.zeros(n, dtype=np.float64)
    d_glcm       = np.zeros(n, dtype=np.float64)
    d_lbp        = np.zeros(n, dtype=np.float64)
    d_hog        = np.zeros(n, dtype=np.float64)

    for i, p in enumerate(paths):
        f = index[p]
        d_color_hist[i] = chi_square_distance(q_feat['color_hist'], f['color_hist'])
        d_color_mom [i] = euclidean_distance (q_feat['color_mom'],  f['color_mom'])
        d_glcm      [i] = euclidean_distance (q_feat['glcm'],       f['glcm'])
        d_lbp       [i] = chi_square_distance(q_feat['lbp'],        f['lbp'])
        d_hog       [i] = euclidean_distance (q_feat['hog'],        f['hog'])

    def _mm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    dn_color_hist = _mm(d_color_hist)
    dn_color_mom  = _mm(d_color_mom)
    dn_glcm       = _mm(d_glcm)
    dn_lbp        = _mm(d_lbp)
    dn_hog        = _mm(d_hog)

    combined = (W_COLOR_HIST * dn_color_hist +
                W_COLOR_MOM  * dn_color_mom  +
                W_GLCM       * dn_glcm       +
                W_LBP        * dn_lbp        +
                W_HOG        * dn_hog)

    per_component = {
        'color_hist': dn_color_hist,
        'color_mom' : dn_color_mom,
        'glcm'      : dn_glcm,
        'lbp'       : dn_lbp,
        'hog'       : dn_hog,
    }
    return paths, combined, per_component


def normalise_query_features(raw_feat, norm_params):
    out = {}
    for key in FEATURE_KEYS:
        vec = np.asarray(raw_feat[key], dtype=np.float32)
        p = norm_params[key]
        out[key] = np.clip((vec - p['min']) / p['range'], 0, 1).astype(np.float32)
    return out


def rocchio_refine(q_feat, relevant_paths, nonrelevant_paths, index):
    new_q = {}
    for key in FEATURE_KEYS:
        q_vec = np.asarray(q_feat[key], dtype=np.float64)
        if relevant_paths:
            rel_mean = np.mean([np.asarray(index[p][key], dtype=np.float64)
                                for p in relevant_paths], axis=0)
        else:
            rel_mean = np.zeros_like(q_vec)
        if nonrelevant_paths:
            nrel_mean = np.mean([np.asarray(index[p][key], dtype=np.float64)
                                 for p in nonrelevant_paths], axis=0)
        else:
            nrel_mean = np.zeros_like(q_vec)
        updated = (ROCCHIO_ALPHA * q_vec +
                   ROCCHIO_BETA  * rel_mean -
                   ROCCHIO_GAMMA * nrel_mean)
        new_q[key] = np.clip(updated, 0, 1).astype(np.float32)
    return new_q


# ═══════════════════════════════════════════════════════════════════
# INDEX LOADING
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner='Loading CBIR index...')
def load_index(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    index = data['index']
    norm_params = data['norm_params']
    categories = {}
    for p in index.keys():
        cat = Path(p).parent.name.lower()
        categories.setdefault(cat, []).append(p)
    return index, norm_params, categories


def get_category(filepath):
    return Path(filepath).parent.name.lower()


def read_rgb(rel_path):
    full = DATASET_ROOT / rel_path
    img = cv2.imread(str(full))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ═══════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title='CBIR — TrashNet',
    page_icon='🔍',
    layout='wide'
)

st.title('🔍 CBIR — Waste Classification Retrieval')
st.caption('Content-Based Image Retrieval using HSV histogram, colour moments, '
           'GLCM, LBP and HOG features. Aligned with SDG 11 & 12.')

# ── Load index ─────────────────────────────────────────
if not os.path.isfile(INDEX_FILE):
    st.error(f'Index file `{INDEX_FILE}` not found in this folder.\n\n'
             f'Run the notebook first to build the index, then place '
             f'this `app.py` next to `{INDEX_FILE}`.')
    st.stop()

index, norm_params, categories = load_index(INDEX_FILE)
total_images = len(index)
all_categories = sorted(categories.keys())

# ── Session state ──────────────────────────────────────
defaults = {
    'query_bgr': None,
    'query_path': None,
    'query_source': None,
    'results': None,
    'per_component': None,
    'feedback': {},
    'rocchio_round': 0,
    'original_q_feat': None,
    'search_time': 0.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tab_search, tab_stats = st.tabs(['Search', 'Dataset & Stats'])

# ───────────────────────────── SEARCH TAB ─────────────────────────────
with tab_search:
    col_left, col_right = st.columns([1, 2], gap='large')

    # ── LEFT: query selection ──────────────────────────
    with col_left:
        st.subheader('1. Choose a query image')

        source = st.radio(
            'Query source',
            ['Upload your own', 'Pick from dataset'],
            horizontal=True,
            key='source_radio'
        )

        new_query_loaded = False

        if source == 'Upload your own':
            uploaded = st.file_uploader(
                'Upload an image',
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
            )
            if uploaded is not None:
                file_bytes = np.frombuffer(uploaded.read(), np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    st.error('Could not decode uploaded image.')
                else:
                    if st.session_state.query_source != f'upload:{uploaded.name}':
                        st.session_state.query_bgr = img_bgr
                        st.session_state.query_path = None
                        st.session_state.query_source = f'upload:{uploaded.name}'
                        new_query_loaded = True

        else:
            pick_cat = st.selectbox('Category', all_categories)
            pool = categories[pick_cat]
            pick_idx = st.selectbox(
                f'Image ({len(pool)} available)',
                range(len(pool)),
                format_func=lambda i: Path(pool[i]).name
            )
            picked_path = pool[pick_idx]
            if st.session_state.query_source != f'dataset:{picked_path}':
                img_bgr = cv2.imread(picked_path)
                if img_bgr is not None:
                    st.session_state.query_bgr = img_bgr
                    st.session_state.query_path = picked_path
                    st.session_state.query_source = f'dataset:{picked_path}'
                    new_query_loaded = True

        if new_query_loaded:
            st.session_state.results = None
            st.session_state.per_component = None
            st.session_state.feedback = {}
            st.session_state.rocchio_round = 0
            st.session_state.original_q_feat = None

        # Query preview
        if st.session_state.query_bgr is not None:
            st.markdown('---')
            preview = cv2.cvtColor(st.session_state.query_bgr, cv2.COLOR_BGR2RGB)
            st.image(preview, caption='Query image', use_container_width=True)

            if st.session_state.query_path:
                st.caption(f'Ground-truth category: **{get_category(st.session_state.query_path)}**')

            with st.expander('Show foreground mask (Otsu)'):
                q_resized = cv2.resize(st.session_state.query_bgr, RESIZE_DIM)
                mask = create_foreground_mask(q_resized)
                masked = cv2.bitwise_and(q_resized, q_resized, mask=mask)
                masked_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
                fg_pct = (mask.sum() / 255) / (mask.shape[0] * mask.shape[1]) * 100
                st.image(masked_rgb,
                         caption=f'Foreground: {fg_pct:.0f}% of image',
                         use_container_width=True)

        st.markdown('---')
        st.subheader('2. Retrieval settings')
        top_k = st.slider('Top-K results', min_value=3, max_value=20, value=10, step=1)

        if st.button('Search', type='primary', use_container_width=True,
                     disabled=st.session_state.query_bgr is None):
            with st.spinner('Extracting features and searching...'):
                t0 = time.time()
                raw = extract_all_features(st.session_state.query_bgr)
                q_feat = normalise_query_features(raw, norm_params)
                st.session_state.original_q_feat = q_feat

                paths, combined, per_comp = compute_combined_distances(q_feat, index)

                order = np.argsort(combined)
                sorted_results = []
                sorted_per_comp = {k: [] for k in FEATURE_KEYS}
                for idx in order:
                    p = paths[idx]
                    if p == st.session_state.query_path:
                        continue
                    sorted_results.append((p, float(combined[idx])))
                    for k in FEATURE_KEYS:
                        sorted_per_comp[k].append(float(per_comp[k][idx]))
                    if len(sorted_results) >= top_k:
                        break

                st.session_state.results = sorted_results
                st.session_state.per_component = sorted_per_comp
                st.session_state.feedback = {i: None for i in range(len(sorted_results))}
                st.session_state.rocchio_round = 1
                st.session_state.search_time = time.time() - t0

    # ── RIGHT: results ─────────────────────────────────
    with col_right:
        if st.session_state.results is None:
            st.info('Pick or upload a query image on the left, then click **Search**.')
        else:
            round_label = (f'Round {st.session_state.rocchio_round}'
                           if st.session_state.rocchio_round > 0 else 'Initial')
            st.subheader(f'Results — {round_label}')
            st.caption(f'Retrieved {len(st.session_state.results)} images in '
                       f'{st.session_state.search_time:.2f}s · '
                       f'Metric: combined (5 features, per-component normalised)')

            if st.session_state.query_path is not None:
                q_cat = get_category(st.session_state.query_path)
                hits = sum(1 for p, _ in st.session_state.results
                           if get_category(p) == q_cat)
                p_at_k = hits / len(st.session_state.results)
                col_a, col_b, col_c = st.columns(3)
                col_a.metric('Query class', q_cat)
                col_b.metric(f'P@{len(st.session_state.results)}', f'{p_at_k:.2f}')
                col_c.metric('Hits', f'{hits} / {len(st.session_state.results)}')

            st.markdown('---')
            st.markdown('**Mark results as 👍 relevant / 👎 not relevant, '
                        'then click _Refine with feedback_ below.**')

            n_cols = 5
            results = st.session_state.results
            rows = (len(results) + n_cols - 1) // n_cols

            for row in range(rows):
                cols = st.columns(n_cols)
                for col_idx in range(n_cols):
                    i = row * n_cols + col_idx
                    if i >= len(results):
                        break
                    path, dist = results[i]
                    cat = get_category(path)
                    img = read_rgb(path)

                    with cols[col_idx]:
                        if img is not None:
                            st.image(img, use_container_width=True)
                        st.caption(f'**#{i+1}** · {cat} · d={dist:.3f}')

                        current = st.session_state.feedback.get(i)
                        bcols = st.columns(3)
                        if bcols[0].button('👍', key=f'up_{i}_{st.session_state.rocchio_round}',
                                           type='primary' if current == 'rel' else 'secondary',
                                           use_container_width=True):
                            st.session_state.feedback[i] = 'rel'
                            st.rerun()
                        if bcols[1].button('👎', key=f'down_{i}_{st.session_state.rocchio_round}',
                                           type='primary' if current == 'nrel' else 'secondary',
                                           use_container_width=True):
                            st.session_state.feedback[i] = 'nrel'
                            st.rerun()
                        if bcols[2].button('✖', key=f'clear_{i}_{st.session_state.rocchio_round}',
                                           use_container_width=True):
                            st.session_state.feedback[i] = None
                            st.rerun()

            st.markdown('---')
            n_rel  = sum(1 for v in st.session_state.feedback.values() if v == 'rel')
            n_nrel = sum(1 for v in st.session_state.feedback.values() if v == 'nrel')
            col_fb1, col_fb2 = st.columns([3, 1])
            with col_fb1:
                st.write(f'**Feedback:** 👍 {n_rel} relevant, 👎 {n_nrel} not relevant')
            with col_fb2:
                refine = st.button('Refine with feedback',
                                   type='primary',
                                   disabled=(n_rel == 0 and n_nrel == 0),
                                   use_container_width=True)

            if refine:
                with st.spinner('Applying Rocchio refinement...'):
                    rel_paths  = [st.session_state.results[i][0]
                                  for i, v in st.session_state.feedback.items() if v == 'rel']
                    nrel_paths = [st.session_state.results[i][0]
                                  for i, v in st.session_state.feedback.items() if v == 'nrel']

                    refined_q = rocchio_refine(
                        st.session_state.original_q_feat,
                        rel_paths, nrel_paths, index
                    )

                    paths, combined, per_comp = compute_combined_distances(refined_q, index)
                    order = np.argsort(combined)
                    sorted_results = []
                    sorted_per_comp = {k: [] for k in FEATURE_KEYS}
                    for idx in order:
                        p = paths[idx]
                        if p == st.session_state.query_path:
                            continue
                        sorted_results.append((p, float(combined[idx])))
                        for k in FEATURE_KEYS:
                            sorted_per_comp[k].append(float(per_comp[k][idx]))
                        if len(sorted_results) >= top_k:
                            break

                    st.session_state.original_q_feat = refined_q
                    st.session_state.results = sorted_results
                    st.session_state.per_component = sorted_per_comp
                    st.session_state.feedback = {i: None for i in range(len(sorted_results))}
                    st.session_state.rocchio_round += 1
                    st.rerun()

            with st.expander('Per-component distance breakdown'):
                st.markdown(
                    'Shows how each of the 5 features scored the top results. '
                    'Values are min-max normalised across the full candidate set '
                    '(lower = more similar).'
                )

                per_comp = st.session_state.per_component
                df_rows = []
                for i, (path, dist) in enumerate(st.session_state.results):
                    row = {
                        '#': i + 1,
                        'category': get_category(path),
                        'combined': dist,
                    }
                    for k in FEATURE_KEYS:
                        row[k] = per_comp[k][i]
                    df_rows.append(row)

                df = pd.DataFrame(df_rows)

                st.markdown('**Weight × distance contribution to combined score (top result)**')
                weights = {
                    'color_hist': W_COLOR_HIST,
                    'color_mom' : W_COLOR_MOM,
                    'glcm'      : W_GLCM,
                    'lbp'       : W_LBP,
                    'hog'       : W_HOG,
                }
                contribs = {k: weights[k] * per_comp[k][0] for k in FEATURE_KEYS}

                fig, ax = plt.subplots(figsize=(8, 2.5))
                keys   = list(contribs.keys())
                values = list(contribs.values())
                bars   = ax.barh(keys, values,
                                 color=['#4ECDC4', '#FF9F43', '#FFD166',
                                        '#A78BFA', '#F87171'])
                for bar, v in zip(bars, values):
                    ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                            f'{v:.3f}', va='center', fontsize=9)
                ax.set_xlabel('Weight × normalised distance')
                ax.set_xlim(0, max(values) * 1.2 if max(values) > 0 else 1)
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.markdown('**Full table**')
                st.dataframe(
                    df.style.format({
                        'combined'  : '{:.3f}',
                        'color_hist': '{:.3f}',
                        'color_mom' : '{:.3f}',
                        'glcm'      : '{:.3f}',
                        'lbp'       : '{:.3f}',
                        'hog'       : '{:.3f}',
                    }),
                    use_container_width=True,
                    hide_index=True,
                )


# ───────────────────────────── STATS TAB ─────────────────────────────
with tab_stats:
    st.subheader('Dataset overview')

    cat_counts = {c: len(categories[c]) for c in all_categories}

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total images', total_images)
    col2.metric('Categories', len(all_categories))
    col3.metric('Largest class',
                f'{max(cat_counts, key=cat_counts.get)} ({max(cat_counts.values())})')
    col4.metric('Smallest class',
                f'{min(cat_counts, key=cat_counts.get)} ({min(cat_counts.values())})')

    fig, ax = plt.subplots(figsize=(10, 4))
    cats  = list(cat_counts.keys())
    counts = list(cat_counts.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(cats)))
    bars = ax.bar(cats, counts, color=colors, edgecolor='black', linewidth=0.6)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, c + 6,
                f'{c}', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(np.mean(counts), color='red', linestyle='--', alpha=0.6,
               label=f'Mean = {np.mean(counts):.0f}')
    ax.set_ylabel('Number of images')
    ax.set_title('Class distribution')
    ax.set_ylim(0, max(counts) * 1.18)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown('---')
    st.subheader('Feature vector breakdown')
    feat_info = pd.DataFrame([
        {'Feature': 'HSV Histogram', 'Type': 'Colour',  'Dimensions': 2048, 'Weight': W_COLOR_HIST},
        {'Feature': 'Colour Moments','Type': 'Colour',  'Dimensions': 9,    'Weight': W_COLOR_MOM},
        {'Feature': 'GLCM Haralick', 'Type': 'Texture', 'Dimensions': 12,   'Weight': W_GLCM},
        {'Feature': 'LBP (uniform)', 'Type': 'Texture', 'Dimensions': 10,   'Weight': W_LBP},
        {'Feature': 'HOG',           'Type': 'Shape',   'Dimensions': 8100, 'Weight': W_HOG},
    ])
    st.dataframe(feat_info, hide_index=True, use_container_width=True)

    st.markdown('---')
    st.subheader('Evaluation results (from notebook)')
    st.caption('Numbers from the full evaluation: 30 queries × 6 categories × 3 metrics. '
               'Re-run the notebook to refresh.')

    eval_df = pd.DataFrame([
        {'Metric': 'Euclidean', 'P@5': 0.4356, 'R@5': 0.0061, 'R@50': 0.0428, 'mAP': 0.4476, 'MRR': 0.6336},
        {'Metric': 'Cosine',    'P@5': 0.4333, 'R@5': 0.0060, 'R@50': 0.0420, 'mAP': 0.4460, 'MRR': 0.6227},
        {'Metric': 'Combined',  'P@5': 0.6989, 'R@5': 0.0098, 'R@50': 0.0634, 'mAP': 0.6496, 'MRR': 0.8620},
    ])
    st.dataframe(
        eval_df.style.format({
            'P@5': '{:.4f}', 'R@5': '{:.4f}', 'R@50': '{:.4f}',
            'mAP': '{:.4f}', 'MRR': '{:.4f}',
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown('---')
    st.subheader('Leave-one-feature-out ablation')
    st.caption('How much does each feature contribute to the combined metric? '
               'Negative Δ = dropping that feature hurt performance.')

    abl_df = pd.DataFrame([
        {'Configuration': 'ALL features (baseline)', 'P@5': 0.6989, 'mAP': 0.6496, 'ΔP@5':  0.0000, 'ΔmAP':  0.0000},
        {'Configuration': 'drop color_hist',         'P@5': 0.6600, 'mAP': 0.6167, 'ΔP@5': -0.0389, 'ΔmAP': -0.0329},
        {'Configuration': 'drop color_mom',          'P@5': 0.6500, 'mAP': 0.6047, 'ΔP@5': -0.0489, 'ΔmAP': -0.0449},
        {'Configuration': 'drop glcm',               'P@5': 0.6711, 'mAP': 0.6415, 'ΔP@5': -0.0278, 'ΔmAP': -0.0081},
        {'Configuration': 'drop lbp',                'P@5': 0.6589, 'mAP': 0.6208, 'ΔP@5': -0.0400, 'ΔmAP': -0.0288},
        {'Configuration': 'drop hog',                'P@5': 0.6767, 'mAP': 0.6398, 'ΔP@5': -0.0222, 'ΔmAP': -0.0098},
    ])
    st.dataframe(
        abl_df.style.format({
            'P@5': '{:.4f}', 'mAP': '{:.4f}', 'ΔP@5': '{:+.4f}', 'ΔmAP': '{:+.4f}'
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.info('**Most important feature: colour_mom** (dropping it costs ΔmAP = −0.0449). '
            'Despite being only 9-dimensional, it contributes more to ranking quality than '
            'the 8100-dim HOG feature — a classic IR finding that simple, well-chosen '
            'statistics often outperform high-dimensional representations.')
