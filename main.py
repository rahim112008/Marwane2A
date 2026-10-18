"""
🐄 Bovine SNP Platform v5.0 (Version Complète & Intégrale)
Pipeline complet de bioinformatique pour puces SNP bovines.
Conforme au MANUSCRIT DE FORMATION + Modules Pro (GBLUP, Post-GWAS, Scan).
"""
import gzip
import io
import json
import os
import warnings
import zipfile
from collections import Counter
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats import t as t_dist
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import MDS as SklearnMDS

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="🐄 Bovine SNP Platform v5.0",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_THRESHOLDS = {
    "geno": 0.05, "mind": 0.05, "maf": 0.05, "hwe": 1e-6, "het_sd": 3.0,
    "king_cutoff": 0.354, "ld_r2": 0.2,
}

HWE_EXACT_MAX_SNP = 20_000

ARS_UCD12_LENGTHS = {
    "1": 158_534_110, "2": 136_231_102, "3": 121_005_158, "4": 120_000_166,
    "5": 120_089_699, "6": 117_806_340, "7": 110_682_743, "8": 113_384_748,
    "9": 105_708_134, "10": 103_308_737, "11": 107_310_763, "12": 91_163_125,
    "13": 84_246_514, "14": 84_648_346, "15": 85_207_080, "16": 81_726_628,
    "17": 75_176_999, "18": 66_059_976, "19": 64_089_169, "20": 72_042_983,
    "21": 71_599_096, "22": 61_416_492, "23": 52_531_573, "24": 62_384_193,
    "25": 42_959_610, "26": 51_680_158, "27": 46_772_073, "28": 46_333_854,
    "29": 51_319_414, "X": 139_009_144, "Y": 50_927_933, "MT": 16_338,
}
BOVINE_AUTOSOMES = [str(i) for i in range(1, 30)]

# ============================================================
# CACHE STREAMLIT
# ============================================================
def _hash_ndarray(x):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return "empty"
    if x.dtype.kind in ("U", "S", "O"):
        preview = "|".join(map(str, x.ravel()[:2000]))
        return f"strarr|{x.shape}|{preview}"
    return (f"{x.shape}|{x.dtype}|{float(np.nansum(x))}|"
            f"{float(np.nansum(np.abs(x)))}|{float(np.nansum(x * x))}")

def _hash_any(x):
    try:
        return _hash_ndarray(np.asarray(x))
    except Exception:
        try:
            return f"{type(x).__name__}|{len(x)}"
        except Exception:
            return type(x).__name__

HASH_FUNCS = {np.ndarray: _hash_ndarray, pd.Series: _hash_any, pd.Index: _hash_any}
for _name in ("ArrowStringArray", "StringArray", "IntegerArray", "FloatingArray",
              "BooleanArray", "NumpyExtensionArray", "PandasArray", "DatetimeArray",
              "TimedeltaArray", "PeriodArray", "IntervalArray", "Categorical"):
    _cls = getattr(pd.arrays, _name, None)
    if _cls is None:
        try:
            _cls = getattr(pd.core.arrays, _name, None)
        except Exception:
            _cls = None
    if _cls is not None:
        HASH_FUNCS[_cls] = _hash_any

def cache_data(func=None, **kw):
    def _decorate(f):
        return st.cache_data(show_spinner=False, hash_funcs=HASH_FUNCS, **kw)(f)
    if func is None:
        return _decorate
    return _decorate(func)

# ============================================================
# UTILITAIRES
# ============================================================
def impute_mean(gt):
    gt2 = gt.astype(np.float32, copy=True)
    col_mean = np.nanmean(gt2, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    nan_mask = np.isnan(gt2)
    if not nan_mask.any():
        return gt2
    gt2[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return gt2

def _autosome_weights():
    lens = np.array([ARS_UCD12_LENGTHS[c] for c in BOVINE_AUTOSOMES], dtype=float)
    return lens / lens.sum()

def _chr_sort_key(chrom):
    s = str(chrom).upper().replace("CHR", "").replace("CHROMOSOME", "")
    if s.isdigit():
        return (0, int(s), "")
    special = {"X": 100, "Y": 101, "MT": 102, "M": 102}
    if s in special:
        return (1, special[s], "")
    return (2, 0, s)

def _detect_encoding(raw):
    if len(raw) >= 2 and raw[:2] == b"\x1f\x8b":
        return "gzip"
    if len(raw) >= 2 and raw[:2] == b"\x6c\x1b":
        return "plink_binary"
    return "text"

def step_header(numero, titre, ce_qu_on_fait, pourquoi, attendu):
    with st.expander(f"📖 **Étape {numero} — {titre}**", expanded=False):
        st.markdown(f"**🎯 Ce qu'on fait :** {ce_qu_on_fait}")
        st.markdown(f"**❓ Pourquoi :** {pourquoi}")
        st.markdown(f"**✅ Ce qu'on attend :** {attendu}")

# ============================================================
# CONVERTISSEUR AXIOM → PED
# ============================================================
def sample_to_fid_iid(sample_name, fid_parts=2):
    name = str(sample_name).strip()
    for ext in (".CEL", ".cel", ".TXT", ".txt", ".gz", ".GZ"):
        if name.endswith(ext):
            name = name[:-len(ext)]
    name = (name.replace("(", "_").replace(")", "")
                .replace("[", "_").replace("]", "")
                .replace(" ", "_").replace("/", "_"))
    while "__" in name:
        name = name.replace("__", "_")
    name = name.strip("_")
    parts = name.split("_")
    fid = "_".join(parts[:fid_parts]) if len(parts) >= fid_parts else parts[0]
    return fid, name

def _clean_allele(a):
    if a is None:
        return "0"
    a = str(a).strip()
    if a in ("", ".", "-", "NA", "nan", "None", "0", "00"):
        return "0"
    return a

def parse_raw_axiom_file(raw_bytes, filename=""):
    if (filename.endswith(".gz") or (len(raw_bytes) >= 2 and raw_bytes[:2] == b"\x1f\x8b")):
        try:
            raw = gzip.decompress(raw_bytes)
        except Exception as e:
            raise ValueError(f"Fichier .gz corrompu : {e}")
    else:
        raw = raw_bytes

    text = raw.decode("utf-8", errors="replace")
    lines = [l.rstrip("\r\n") for l in text.splitlines()]
    lines = [l for l in lines if l.strip()]
    if not lines:
        raise ValueError("Fichier vide.")

    first = lines[0]
    is_header = (first.startswith("#") or first.startswith("Sample") or "Sample Filename" in first[:50])
    start = 1 if is_header else 0

    samples, geno_rows = [], []
    for line in lines[start:]:
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split()
            if len(parts) < 2:
                continue
        sample_name = parts[0].strip()
        cells = [c.strip() for c in parts[1:]]
        alleles = []
        for cell in cells:
            toks = cell.split()
            if len(toks) == 2:
                alleles.append((_clean_allele(toks[0]), _clean_allele(toks[1])))
            elif len(toks) == 1:
                s = toks[0]
                if len(s) == 2 and all(ch in "ACGT0" for ch in s):
                    alleles.append((s[0], s[1]))
                else:
                    alleles.append((_clean_allele(s), _clean_allele(s)))
            else:
                alleles.append(("0", "0"))
        samples.append(sample_name)
        geno_rows.append(alleles)

    if not samples:
        raise ValueError("Aucun échantillon trouvé dans le fichier.")

    counts = [len(r) for r in geno_rows]
    common = Counter(counts).most_common(1)[0][0]
    for i, r in enumerate(geno_rows):
        if len(r) > common:
            geno_rows[i] = r[:common]
        elif len(r) < common:
            geno_rows[i] = r + [("0", "0")] * (common - len(r))

    return samples, geno_rows, common

def build_ped_map_from_raw(samples, geno_rows, fid_parts=2):
    n_ind = len(samples)
    n_snp = len(geno_rows[0]) if n_ind else 0
    map_lines = [f"1\tSNP_{i+1:06d}\t0\t{i+1}" for i in range(n_snp)]
    map_text = "\n".join(map_lines) + "\n"

    ped_lines = []
    for i, sample in enumerate(samples):
        fid, iid = sample_to_fid_iid(sample, fid_parts=fid_parts)
        cols = [fid, iid, "0", "0", "0", "-9"]
        for a1, a2 in geno_rows[i]:
            cols.append(a1)
            cols.append(a2)
        ped_lines.append("\t".join(cols))
    return "\n".join(ped_lines) + "\n", map_text, n_snp

def preview_ped_dataframe(samples, geno_rows, max_rows=10, max_snp=20):
    rows = []
    n_snp = len(geno_rows[0]) if geno_rows else 0
    n_show = min(max_snp, n_snp)
    for sample, alleles in zip(samples[:max_rows], geno_rows[:max_rows]):
        fid, iid = sample_to_fid_iid(sample)
        row = {"FID": fid, "IID": iid, "PID": 0, "MID": 0, "Sex": 0, "Pheno": -9}
        for j in range(n_show):
            a1, a2 = alleles[j]
            row[f"SNP_{j+1:05d}_A1"] = a1
            row[f"SNP_{j+1:05d}_A2"] = a2
        rows.append(row)
    return pd.DataFrame(rows)

def render_axiom_converter_page():
    st.header("🔄 Convertisseur Axiom → PED")
    st.markdown("""
Transforme un fichier brut de génotypage Axiom/Thermo Fisher (cellules `A1 A2`) en fichiers **PED** et **MAP** au format PLINK.
    """)
    st.divider()
    raw_f = st.file_uploader("Fichier brut Axiom (.ped, .txt, .tsv, .csv, .gz)", type=["ped", "txt", "tsv", "csv", "gz"], key="conv_raw_file")
    c1, c2 = st.columns(2)
    fid_parts = c1.number_input("Nombre de parties du nom pour le FID", min_value=1, max_value=5, value=2, step=1, key="conv_fid_parts")
    sep_out = c2.selectbox("Séparateur en sortie", ["Tabulation (PLINK standard)", "Espace"], key="conv_sep")

    if raw_f is not None:
        if st.button("🔄 Convertir", type="primary", use_container_width=True, key="conv_btn_run"):
            try:
                with st.spinner("Parsing du fichier brut..."):
                    samples, geno_rows, n_snp = parse_raw_axiom_file(raw_f.read(), raw_f.name)
                with st.spinner("Génération PED + MAP..."):
                    ped_text, map_text, _ = build_ped_map_from_raw(samples, geno_rows, fid_parts=int(fid_parts))
                if sep_out.startswith("Espace"):
                    ped_text, map_text = ped_text.replace("\t", " "), map_text.replace("\t", " ")
                st.session_state.update({"_conv_samples": samples, "_conv_geno": geno_rows, "_conv_n_snp": n_snp, "_conv_ped": ped_text, "_conv_map": map_text})
                st.success(f"✅ {len(samples)} individus × {n_snp} SNPs convertis.")
            except Exception as e:
                st.error(f"❌ {e}")
                import traceback
                st.code(traceback.format_exc())

    if st.session_state.get("_conv_ped"):
        samples = st.session_state["_conv_samples"]
        geno_rows = st.session_state["_conv_geno"]
        n_snp = st.session_state["_conv_n_snp"]
        ped_text = st.session_state["_conv_ped"]
        map_text = st.session_state["_conv_map"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Individus", len(samples)); c2.metric("SNPs", n_snp); c3.metric("Colonnes PED", 6 + 2 * n_snp)
        st.subheader("👁️ Aperçu (10 lignes × 20 SNPs)")
        st.dataframe(preview_ped_dataframe(samples, geno_rows), use_container_width=True, key="conv_df_preview")

        with st.expander("📄 Aperçu brut PED (5 lignes)"): st.code("\n".join(ped_text.splitlines()[:5]), language="text")
        with st.expander("📄 Aperçu MAP (10 premières lignes)"): st.code("\n".join(map_text.splitlines()[:10]), language="text")

        st.subheader("💾 Téléchargement")
        c1, c2, c3 = st.columns(3)
        with c1: st.download_button("⬇ Fichier .ped", data=ped_text.encode("utf-8"), file_name="converted.ped", mime="text/plain", use_container_width=True, key="conv_dl_ped")
        with c2: st.download_button("⬇ Fichier .map", data=map_text.encode("utf-8"), file_name="converted.map", mime="text/plain", use_container_width=True, key="conv_dl_map")
        with c3:
            bio = BytesIO()
            with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("converted.ped", ped_text)
                zf.writestr("converted.map", map_text)
            st.download_button("⬇ .zip (PED + MAP)", data=bio.getvalue(), file_name="converted_ped_map.zip", mime="application/zip", use_container_width=True, key="conv_dl_zip")

# ============================================================
# PARSING MAP & PED
# ============================================================
def parse_map(map_bytes):
    enc = _detect_encoding(map_bytes)
    if enc == "gzip":
        try: text = gzip.decompress(map_bytes).decode("utf-8", errors="replace")
        except Exception as e: raise ValueError(f"Fichier .map.gz corrompu : {e}")
    elif enc == "plink_binary": raise ValueError("❌ Le fichier .map est en binaire PLINK.")
    else: text = map_bytes.decode("utf-8", errors="replace")

    rows, rejected = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        parts = line.split()
        if len(parts) < 4:
            rejected += 1
            continue
        try: cm = float(parts[2]) if parts[2] not in ("0", ".", "NA") else 0.0
        except ValueError: cm = 0.0
        try: bp = int(float(parts[3]))
        except (ValueError, TypeError):
            rejected += 1
            continue
        rows.append({"CHR": str(parts[0]), "SNP": parts[1], "CM": cm, "BP": bp})
    if not rows: raise ValueError("Fichier .map vide ou invalide.")
    return pd.DataFrame(rows), rejected

def parse_ped(ped_bytes, n_snp_map):
    enc = _detect_encoding(ped_bytes)
    if enc == "gzip":
        try: text = gzip.decompress(ped_bytes).decode("utf-8", errors="replace")
        except Exception as e: raise ValueError(f"Fichier .ped.gz corrompu : {e}")
    elif enc == "plink_binary": raise ValueError("❌ Ce fichier est un PLINK binaire (.bed) renommé.")
    else: text = ped_bytes.decode("utf-8", errors="replace")

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines: raise ValueError("Fichier .ped vide.")
    n_cols_first = len(lines[0].split())
    n_snp_ped = max(0, (n_cols_first - 6) // 2)
    if n_cols_first < 7: raise ValueError(f"❌ Format PED invalide : {n_cols_first} colonnes.")
    if 0 < n_snp_ped < 10 and len(lines) > 100: raise ValueError("⚠️ Fichier probablement transposé.")

    n_snp_eff = min(n_snp_map, n_snp_ped) if n_snp_ped != n_snp_map and n_snp_ped > 0 else n_snp_map
    expected_cols_eff = 6 + 2 * n_snp_eff
    fids, iids, geno_rows = [], [], []

    for line in lines:
        parts = line.split()
        if len(parts) < 7: continue
        if len(parts) < expected_cols_eff: continue
        fids.append(parts[0]); iids.append(parts[1]); geno_rows.append(parts[6:6 + 2 * n_snp_eff])

    n_ind = len(iids)
    if n_ind == 0: raise ValueError(f"❌ Aucun individu chargé. Colonnes attendues : {expected_cols_eff}")

    allele_counts = [Counter() for _ in range(n_snp_eff)]
    for row in geno_rows:
        for j in range(n_snp_eff):
            a1, a2 = row[2 * j], row[2 * j + 1]
            if a1 == "0" or a2 == "0": continue
            allele_counts[j][a1] += 1; allele_counts[j][a2] += 1

    minor = [min(c, key=c.get) if len(c) >= 2 else None for c in allele_counts]
    gt = np.full((n_ind, n_snp_eff), np.nan, dtype=np.float32)
    for i, row in enumerate(geno_rows):
        for j in range(n_snp_eff):
            m = minor[j]
            if m is None: continue
            a1, a2 = row[2 * j], row[2 * j + 1]
            if a1 == "0" or a2 == "0": continue
            gt[i, j] = (1 if a1 == m else 0) + (1 if a2 == m else 0)

    ind_df = pd.DataFrame({"FID": fids, "IID": iids})
    return gt, ind_df, 0

# ============================================================
# PARSING VCF & DEMO
# ============================================================
def parse_vcf(vcf_bytes):
    is_gz = len(vcf_bytes) >= 2 and vcf_bytes[:2] == b"\x1f\x8b"
    if is_gz:
        try: text = gzip.decompress(vcf_bytes).decode("utf-8", errors="replace")
        except Exception as e: raise ValueError(f"Fichier .vcf.gz corrompu : {e}")
    else: text = vcf_bytes.decode("utf-8", errors="replace")

    samples, rows = [], []
    for line in text.splitlines():
        line = line.rstrip("\n\r")
        if not line or line.startswith("##"): continue
        if line.startswith("#CHROM"):
            samples = line.split("\t")[9:]
            continue
        if line.startswith("#"): continue
        parts = line.split("\t")
        if len(parts) < 10: continue
        chrom, pos_s, vid, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
        if "," in alt: continue
        try: pos = int(pos_s)
        except ValueError: continue
        fmt = parts[8].split(":")
        if "GT" not in fmt: continue
        gt_idx = fmt.index("GT")

        gts = np.full(len(samples), np.nan, dtype=np.float32)
        for k, s in enumerate(parts[9:]):
            fields = s.split(":")
            if gt_idx >= len(fields): continue
            alleles = fields[gt_idx].replace("|", "/").split("/")
            if any(a in (".", "") for a in alleles): continue
            try: gts[k] = sum(int(a) for a in alleles)
            except ValueError: continue

        rows.append({"CHR": str(chrom), "SNP": vid or f"{chrom}:{pos}", "CM": 0.0, "BP": pos, "A1": ref, "A2": alt, "GT": gts})

    if not rows: raise ValueError("Aucun variant biallélique trouvé dans le VCF.")
    n_ind, n_snp = len(samples), len(rows)
    gt = np.zeros((n_ind, n_snp), dtype=np.float32)
    for j, r in enumerate(rows):
        gt[:, j] = r["GT"]
        valid = gt[:, j][~np.isnan(gt[:, j])]
        if len(valid) > 0 and valid.mean() / 2.0 > 0.5:
            gt[:, j] = np.where(np.isnan(gt[:, j]), np.nan, 2.0 - gt[:, j])

    snp_df = pd.DataFrame({"CHR": [r["CHR"] for r in rows], "SNP": [r["SNP"] for r in rows], "CM": 0.0, "BP": [r["BP"] for r in rows], "A1": [r["A1"] for r in rows], "A2": [r["A2"] for r in rows]})
    ind_df = pd.DataFrame({"FID": samples, "IID": samples})
    return gt, ind_df, snp_df, 0, 0

def generate_demo_data(n_ind=150, n_snp=800, n_pop=4, seed=42):
    rng = np.random.default_rng(seed)
    pool = ["AND", "EBG", "ELN", "EZP", "FGN", "LJR", "NAR", "NBD", "NKA", "PRS", "PSH", "PTP", "SHO", "YBS"]
    n_pop = max(1, min(n_pop, len(pool)))
    pop_names = pool[:n_pop]
    counts = [n_ind // n_pop + (1 if i < n_ind % n_pop else 0) for i in range(n_pop)]
    n_ind_eff = sum(counts)

    p_anc = rng.beta(0.6, 0.6, n_snp)
    drift = 0.15
    gt = np.zeros((n_ind_eff, n_snp), dtype=np.float32)
    ind_rows = []
    k = 0
    for pop, n_i in zip(pop_names, counts):
        p_k = np.clip(p_anc + rng.normal(0, drift, n_snp), 0.02, 0.98)
        for i in range(n_i):
            gt[k] = (rng.random(n_snp) < p_k).astype(np.float32) + (rng.random(n_snp) < p_k).astype(np.float32)
            ind_rows.append({"FID": pop, "IID": f"{pop}_{i + 1:03d}"})
            k += 1

    gt[rng.random(gt.shape) < 0.02] = np.nan
    ind_df = pd.DataFrame(ind_rows)
    chr_names = rng.choice(BOVINE_AUTOSOMES, n_snp, p=_autosome_weights())
    bps = np.array([rng.integers(1, ARS_UCD12_LENGTHS[c]) for c in chr_names], dtype=np.int64)
    snp_df = pd.DataFrame({"CHR": chr_names, "SNP": [f"rs{i:07d}" for i in range(n_snp)], "CM": bps / 1_000_000, "BP": bps})
    snp_df["_k"] = snp_df["CHR"].map(_chr_sort_key)
    snp_df = (snp_df.sort_values(["_k", "BP"]).drop(columns="_k").reset_index(drop=True))
    return gt, ind_df, snp_df

# ============================================================
# MÉTRIQUES QC
# ============================================================
def missingness_per_ind(gt): return np.isnan(gt).mean(axis=1)
def missingness_per_snp(gt): return np.isnan(gt).mean(axis=0)
def allele_freq(gt): return np.nanmean(gt, axis=0) / 2.0
def maf(gt): 
    p = allele_freq(gt)
    return np.minimum(p, 1.0 - p)
def heterozygosity(gt): 
    with np.errstate(invalid="ignore"):
        return np.nanmean(gt == 1, axis=1)

def hwe_exact_p(n_het, n_hom1, n_hom2):
    n = n_het + n_hom1 + n_hom2
    if n == 0: return np.nan
    if n_het == 0 and (n_hom1 == 0 or n_hom2 == 0): return 1.0
    rare = 2 * min(n_hom1, n_hom2) + n_het
    mid = (rare * (2 * n - rare)) // (2 * n)
    if mid % 2 != rare % 2: mid += 1
    probs = np.zeros(rare + 1)
    probs[mid] = 1.0
    mysum = 1.0
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch <= rare - 2:
        probs[ch + 2] = probs[ch] * 4 * chr_ * chc / ((ch + 2) * (ch + 1))
        mysum += probs[ch + 2]
        ch += 2; chr_ -= 1; chc -= 1
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch >= 2:
        probs[ch - 2] = probs[ch] * ch * (ch - 1) / (4 * (chr_ + 1) * (chc + 1))
        mysum += probs[ch - 2]
        ch -= 2; chr_ += 1; chc += 1
    p_obs = probs[n_het] if n_het < len(probs) else 0.0
    return float(min(probs[probs <= p_obs + 1e-7].sum() / mysum, 1.0))

@cache_data
def hwe_pvalues(gt):
    _, n_snp = gt.shape
    pvals = np.full(n_snp, np.nan, dtype=np.float64)
    nan_mask = np.isnan(gt)
    n0 = ((gt == 0) & ~nan_mask).sum(axis=0); n1 = ((gt == 1) & ~nan_mask).sum(axis=0); n2 = ((gt == 2) & ~nan_mask).sum(axis=0)
    n_valid = n0 + n1 + n2
    if n_snp <= HWE_EXACT_MAX_SNP:
        for j in range(n_snp):
            if n_valid[j] < 5: continue
            pvals[j] = hwe_exact_p(int(n1[j]), int(n0[j]), int(n2[j]))
    else:
        p = (n1 + 2 * n2) / (2.0 * np.where(n_valid > 0, n_valid, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            exp_het = 2.0 * p * (1.0 - p) * n_valid
            valid = (exp_het >= 5) & np.isfinite(exp_het)
            chi2 = (np.abs(n1 - exp_het) - 0.5) ** 2 / exp_het
            pvals[valid] = 1.0 - stats.chi2.cdf(chi2[valid], df=1)
    return pvals

def reconstruct_fid_from_iid(ind_df):
    df = ind_df.copy()
    unique_fids = df["FID"].nunique() if "FID" in df.columns else 0
    if unique_fids <= 1:
        df["FID"] = df["IID"].astype(str).apply(lambda x: next((x.split(sep)[0] for sep in ("_", "-", ".") if sep in x), x))
        st.info(f"🔧 FID reconstruit depuis IID → **{df['FID'].nunique()}** races.")
    return df

def filter_autosomes(gt, snp_df, keep_auto_only=True):
    chr_str = snp_df["CHR"].astype(str).str.upper().str.replace("CHR", "")
    keep = chr_str.isin([str(i) for i in range(1, 30)] + ([] if keep_auto_only else ["X", "Y", "MT"]))
    n_before = len(snp_df)
    gt2, snp2 = gt[:, keep.values], snp_df[keep].reset_index(drop=True)
    st.info(f"🧬 Autosomes 1–29 : {n_before - len(snp2)} SNPs exclus → **{len(snp2)}** conservés.")
    return gt2, snp2

# ============================================================
# LD PRUNING & QC FILTRAGE
# ============================================================
@cache_data
def ld_pruning(gt, window=50, step=5, r2_thr=0.2):
    n_snp = gt.shape[1]
    if n_snp < 3: return np.ones(n_snp, dtype=bool)
    keep = np.ones(n_snp, dtype=bool)
    X = impute_mean(gt)
    X = X - X.mean(axis=0)
    i = 0
    while i < n_snp:
        win_end = min(i + step, n_snp)
        while win_end < n_snp and (win_end - i) < window:
            j = win_end
            if keep[j] and keep[i]:
                a, b = X[:, i], X[:, j]
                if a.std() > 1e-8 and b.std() > 1e-8:
                    if np.corrcoef(a, b)[0, 1]**2 > r2_thr: keep[j] = False
            win_end += 1
        i += step
    return keep

def _align_shapes(gt, ind_df, snp_df):
    n_gt, m_gt = gt.shape
    n_ind, n_snp = min(n_gt, len(ind_df)), min(m_gt, len(snp_df))
    if (n_gt, m_gt) != (len(ind_df), len(snp_df)):
        st.warning(f"⚠️ Alignement : gt={gt.shape} → ({n_ind},{n_snp})")
    return gt[:n_ind, :n_snp], ind_df.iloc[:n_ind].reset_index(drop=True), snp_df.iloc[:n_snp].reset_index(drop=True)

def apply_qc_filters(gt, ind_df, snp_df, params, remove_outliers=True):
    gt, ind_df, snp_df = _align_shapes(gt, ind_df, snp_df)
    n0, m0 = gt.shape
    trace = {}

    keep = missingness_per_snp(gt) <= params["geno"]
    gt = gt[:, keep]; snp_df = snp_df[keep].reset_index(drop=True)
    trace["geno_exclus"] = int(m0 - gt.shape[1])

    keep = missingness_per_ind(gt) <= params["mind"]
    gt = gt[keep]; ind_df = ind_df[keep].reset_index(drop=True)
    trace["mind_exclus"] = int(n0 - gt.shape[0])

    if gt.shape[1] == 0 or gt.shape[0] == 0: raise ValueError("Tous les SNPs ou individus exclus (missingness).")

    m_before = gt.shape[1]; m = maf(gt)
    keep = np.isfinite(m) & (m >= params["maf"])
    gt = gt[:, keep]; snp_df = snp_df[keep].reset_index(drop=True)
    trace["maf_exclus"] = int(m_before - gt.shape[1])
    if gt.shape[1] == 0: raise ValueError("Tous les SNPs exclus par MAF.")

    m_before = gt.shape[1]; pv = hwe_pvalues(gt)
    keep = np.isnan(pv) | (pv >= 1e-6)
    gt = gt[:, keep]; snp_df = snp_df[keep].reset_index(drop=True)
    if gt.shape[1] > 0:
        pv2 = hwe_pvalues(gt); keep = np.isnan(pv2) | (pv2 >= params["hwe"])
        gt = gt[:, keep]; snp_df = snp_df[keep].reset_index(drop=True)
    trace["hwe_exclus"] = int(m_before - gt.shape[1])
    if gt.shape[1] == 0: raise ValueError("Tous les SNPs exclus par HWE.")

    # SAUVEGARDE PRE-OUTLIERS
    gt_pre, ind_pre, snp_pre = gt.copy(), ind_df.copy(), snp_df.copy()

    if remove_outliers:
        het = heterozygosity(gt); keep = np.ones(len(het), dtype=bool)
        for fid in ind_df["FID"].unique():
            mask = (ind_df["FID"] == fid).values; sub = het[mask]
            if len(sub) < 5 or np.nanstd(sub) < 1e-9: continue
            z = (sub - np.nanmean(sub)) / np.nanstd(sub)
            idx = np.where(mask)[0]
            keep[idx[~np.abs(z) <= params["het_sd"]]] = False
        n_before = gt.shape[0]
        gt = gt[keep]; ind_df = ind_df[keep].reset_index(drop=True)
        trace["het_exclus"] = int(n_before - gt.shape[0])
    else:
        trace["het_exclus"] = 0

    if gt.shape[0] == 0: raise ValueError("Tous les individus exclus (hétérozygotie).")

    return gt, ind_df, snp_df, gt_pre, ind_pre, snp_pre, {
        "n_ind_init": int(n0), "n_snp_init": int(m0), "n_ind_final": int(gt.shape[0]),
        "n_snp_final": int(gt.shape[1]), "excluded_ind": int(n0 - gt.shape[0]),
        "excluded_snp": int(m0 - gt.shape[1]), "trace": trace,
    }
  # ============================================================
# POPULATION GENETICS & ANALYSES
# ============================================================
@cache_data
def fst_per_snp(gt, pop_labels):
    pops = np.unique(np.asarray(pop_labels))
    if len(pops) < 2: return np.full(gt.shape[1], np.nan)
    fst = np.full(gt.shape[1], np.nan, dtype=np.float64)
    masks = {p: (np.asarray(pop_labels) == p) for p in pops}
    for j in range(gt.shape[1]):
        pl, nl = [], []
        for p in pops:
            vals = gt[masks[p], j]; vals = vals[~np.isnan(vals)]
            if len(vals) < 3: continue
            pl.append(vals.mean() / 2.0); nl.append(len(vals))
        if len(pl) < 2: continue
        p_arr, n_arr = np.asarray(pl), np.asarray(nl)
        p_bar = np.average(p_arr, weights=n_arr)
        h_s = np.average(2.0 * p_arr * (1.0 - p_arr), weights=n_arr)
        h_t = 2.0 * p_bar * (1.0 - p_bar)
        if h_t > 1e-9: fst[j] = (h_t - h_s) / h_t
    return fst

@cache_data
def fst_pairwise(gt, pop_labels):
    labels = np.asarray(pop_labels)
    pops = sorted(np.unique(labels)); K = len(pops)
    matrix = np.full((K, K), np.nan); np.fill_diagonal(matrix, 0.0)
    masks = {p: (labels == p) for p in pops}
    for i in range(K):
        for j in range(i + 1, K):
            g1 = gt[masks[pops[i]]]; g2 = gt[masks[pops[j]]]
            p1 = np.nanmean(g1, 0) / 2.0; p2 = np.nanmean(g2, 0) / 2.0
            n1 = (~np.isnan(g1)).sum(0); n2 = (~np.isnan(g2)).sum(0)
            denom = np.where(n1 + n2 > 0, n1 + n2, np.nan)
            p_bar = (p1 * n1 + p2 * n2) / denom
            hs = (2 * p1 * (1 - p1) * n1 + 2 * p2 * (1 - p2) * n2) / denom
            ht = 2 * p_bar * (1 - p_bar)
            with np.errstate(divide="ignore", invalid="ignore"):
                fst_j = (ht - hs) / ht
            matrix[i, j] = matrix[j, i] = float(np.nanmean(fst_j))
    return matrix, pops

@cache_data
def pca_analysis(gt, n_components=10):
    X = impute_mean(gt); X = X - X.mean(axis=0)
    pca = PCA(n_components=min(n_components, X.shape[0] - 1, X.shape[1]))
    return pca.fit_transform(X), pca.explained_variance_ratio_ * 100.0

@cache_data
def mds_analysis(gt, n_components=5):
    X = impute_mean(gt); n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n): D[i] = np.abs(X - X[i]).sum(1) / X.shape[1]
    return SklearnMDS(n_components=n_components, dissimilarity="precomputed", random_state=42, n_init=1, max_iter=300, normalized_stress=False).fit_transform(D)

@cache_data
def ld_decay(gt, snp_bp, max_kb=1000, max_snp=1500, seed=42):
    n_snp = gt.shape[1]
    if n_snp < 2: return pd.DataFrame(columns=["dist_kb", "r2"])
    idx = np.sort(np.random.default_rng(seed).choice(n_snp, max_snp, replace=False)) if n_snp > max_snp else np.arange(n_snp)
    gt_sub = impute_mean(gt[:, idx]); bp_sub = np.asarray(snp_bp)[idx].astype(np.float64)
    X = gt_sub - gt_sub.mean(0); std = gt_sub.std(0); std[std < 1e-8] = np.nan; X = X / std
    C = (X.T @ X) / X.shape[0]; R2 = C ** 2
    iu, ju = np.triu_indices(len(idx), k=1)
    dist_kb = (bp_sub[ju] - bp_sub[iu]) / 1000.0
    mask = (dist_kb > 0) & (dist_kb <= max_kb)
    return pd.DataFrame({"dist_kb": dist_kb[mask], "r2": R2[iu[mask], ju[mask]]})

@cache_data
def kinship_matrix(gt):
    X = impute_mean(gt); p = np.clip(X.mean(0) / 2.0, 1e-3, 1 - 1e-3)
    Z = X - 2 * p; Zn = Z / np.sqrt(2 * p * (1 - p))
    return (Zn @ Zn.T) / X.shape[1]

@cache_data
def king_kinship(gt):
    n_ind = gt.shape[0]; het = np.nansum(gt == 1, axis=1).astype(np.float64)
    K = np.zeros((n_ind, n_ind))
    for i in range(n_ind):
        for j in range(i + 1, n_ind):
            both_het = np.nansum((gt[i] == 1) & (gt[j] == 1))
            denom = het[i] + het[j]
            K[i, j] = K[j, i] = (denom - 2 * both_het) / denom if denom > 0 else 0.0
    np.fill_diagonal(K, 0.5)
    return K

def apply_king_cutoff(K, ind_df, cutoff=0.354):
    n = K.shape[0]; to_remove = np.zeros(n, dtype=bool)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if K[i, j] > cutoff and not to_remove[i] and not to_remove[j]]
    for i, j in pairs: to_remove[j] = True
    return ~to_remove, int(to_remove.sum())

@cache_data
def admixture_nmf(gt, K=3, seed=42, max_iter=500):
    X = np.clip(impute_mean(gt), 0.0, 2.0)
    model = NMF(n_components=K, init="nndsvda", random_state=seed, max_iter=max_iter)
    W = model.fit_transform(X); s = W.sum(1, keepdims=True); s[s == 0] = 1.0
    return W / s, model.components_

@cache_data
def admix_cv_error(gt, K_range=(2, 6), n_reps=3, holdout=0.05, seed=42):
    rng = np.random.default_rng(seed); X = np.clip(impute_mean(gt), 0.0, 2.0)
    n, m = X.shape; mask = rng.random((n, m)) < holdout
    X_train = X.copy(); X_train[mask] = 0.0
    results = {}
    for K in range(K_range[0], K_range[1]):
        errs = []
        for rep in range(n_reps):
            model = NMF(n_components=K, init="nndsvda", random_state=seed + rep, max_iter=300)
            W = model.fit_transform(X_train); H = model.components_
            errs.append(np.mean((X[mask] - (W @ H)[mask]) ** 2))
        results[K] = {"mean": float(np.mean(errs)), "std": float(np.std(errs))}
    return results

@cache_data
def estimate_ne_historical(gt, snp_df, max_snp=2000):
    bp = snp_df["BP"].astype(np.float64).values; chr_arr = snp_df["CHR"].astype(str).values
    n_snp = gt.shape[1]
    if n_snp > max_snp:
        idx = np.linspace(0, n_snp - 1, max_snp).astype(int)
        gt = gt[:, idx]; bp = bp[idx]; chr_arr = chr_arr[idx]
    X = impute_mean(gt); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    r2_list, dist_list = [], []
    n = X.shape[1]
    for i in range(0, n - 1):
        for j in range(i + 1, min(i + 50, n)):
            if chr_arr[i] != chr_arr[j]: continue
            d = abs(bp[j] - bp[i]) / 1e6
            if d < 0.001 or d > 50: continue
            r = np.corrcoef(X[:, i], X[:, j])[0, 1]
            r2_list.append(r * r); dist_list.append(d)
    if not dist_list: return pd.DataFrame({"GenAgo": [], "Ne": []})
    r2 = np.array(r2_list); dist = np.array(dist_list)
    bins = np.logspace(np.log10(dist.min() + 1e-6), np.log10(dist.max()), 20)
    bc = (bins[:-1] + bins[1:]) / 2
    r2b = np.array([r2[(dist >= bins[k]) & (dist < bins[k + 1])].mean() if ((dist >= bins[k]) & (dist < bins[k + 1])).sum() > 0 else np.nan for k in range(len(bins) - 1)])
    m = np.isfinite(r2b)
    if m.sum() < 3: return pd.DataFrame({"GenAgo": [], "Ne": []})
    c_morgans = bc[m] * 0.01; r2v = r2b[m]
    ne = np.where(r2v > 1e-4, 1.0 / (4 * c_morgans * r2v), np.nan)
    df = pd.DataFrame({"GenAgo": bc[m] * 100, "Ne": ne}).dropna()
    df = df[df["Ne"] > 0].sort_values("GenAgo")
    if len(df) > 3: df["Ne"] = df["Ne"].rolling(3, center=True, min_periods=1).mean()
    return df

@cache_data
def reynolds_distance(gt, pop_labels):
    labels = np.asarray(pop_labels); pops = sorted(np.unique(labels)); K = len(pops)
    p_hat = {p: np.nanmean(gt[labels == p], 0) / 2.0 for p in pops}
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            pi, pj = p_hat[pops[i]], p_hat[pops[j]]
            num = np.nansum((pi - pj) ** 2); denom = np.nansum(pi * (1 - pj) + pj * (1 - pi))
            if denom > 0: D[i, j] = D[j, i] = num / denom
    return D, pops

def nj_tree_newick(D, labels):
    try:
        from scipy.cluster.hierarchy import linkage, to_tree
        from scipy.spatial.distance import squareform
        Dm = squareform(D, checks=False)
        Z = linkage(Dm, method="average"); tree = to_tree(Z)
        def rec(node):
            if node.is_leaf(): return f"{labels[node.id]}:{node.dist:.4f}"
            return f"({rec(node.get_left())},{rec(node.get_right())}):{node.dist:.4f}"
        return rec(tree)
    except Exception as e: return f"Erreur NJ : {e}"

@cache_data
def detect_roh(gt, snp_df_json, min_snps=30, min_kb=500.0):
    snp_df = pd.read_json(snp_df_json); chr_arr = snp_df["CHR"].astype(str).values; bp = snp_df["BP"].astype(np.int64).values
    rohs, froh_bp, genome_bp_total = [], np.zeros(gt.shape[0]), 0.0
    for chrom in pd.unique(chr_arr):
        idxs = np.where(chr_arr == chrom)[0]
        if len(idxs) < min_snps: continue
        idxs = idxs[np.argsort(bp[idxs])]; chrom_span = bp[idxs[-1]] - bp[idxs[0]]
        if chrom_span < 1: continue
        genome_bp_total += chrom_span
        for i in range(gt.shape[0]):
            col = gt[i, idxs]; start_k = None
            for k in range(len(idxs)):
                if (col[k] == 0) or (col[k] == 2):
                    if start_k is None: start_k = k
                else:
                    if start_k is not None:
                        n_run = k - start_k; length_kb = (bp[idxs[k - 1]] - bp[idxs[start_k]]) / 1000.0
                        if n_run >= min_snps and length_kb >= min_kb:
                            rohs.append({"IID_idx": i, "CHR": chrom, "start_bp": int(bp[idxs[start_k]]), "end_bp": int(bp[idxs[k - 1]]), "n_snp": int(n_run), "length_kb": float(length_kb)})
                            froh_bp[i] += bp[idxs[k - 1]] - bp[idxs[start_k]]
                        start_k = None
            if start_k is not None:
                n_run = len(idxs) - start_k; length_kb = (bp[idxs[-1]] - bp[idxs[start_k]]) / 1000.0
                if n_run >= min_snps and length_kb >= min_kb:
                    rohs.append({"IID_idx": i, "CHR": chrom, "start_bp": int(bp[idxs[start_k]]), "end_bp": int(bp[idxs[-1]]), "n_snp": int(n_run), "length_kb": float(length_kb)})
                    froh_bp[i] += bp[idxs[-1]] - bp[idxs[start_k]]
    froh = froh_bp / genome_bp_total if genome_bp_total > 0 else froh_bp
    roh_df = (pd.DataFrame(rohs) if rohs else pd.DataFrame(columns=["IID_idx", "CHR", "start_bp", "end_bp", "n_snp", "length_kb"]))
    return roh_df, froh

@cache_data
def selection_signatures(gt, snp_df, pop_labels):
    fst = fst_per_snp(gt, np.asarray(pop_labels)); hom = np.nanmean((gt == 0) | (gt == 2), axis=0)
    df = snp_df.copy(); df["FST"] = fst; df["HOM"] = hom
    hom_std = df["HOM"].std()
    df["SCORE"] = (df["FST"].fillna(0) * (df["HOM"] - df["HOM"].mean()) / (hom_std if hom_std > 1e-9 else 1.0))
    return df.sort_values("SCORE", ascending=False)

# ============================================================
# NEW : SELECTION SCAN & POST-GWAS & GBLUP
# ============================================================
@cache_data
def windowed_selection_scan(gt, snp_df, pop_labels, window_size=50):
    fst = fst_per_snp(gt, pop_labels); hom = np.nanmean((gt == 0) | (gt == 2), axis=0)
    df = snp_df.copy(); df["FST"] = fst; df["HOM"] = hom; df["SCORE"] = df["FST"].fillna(0) * df["HOM"]
    results = []
    for chrom in df["CHR"].unique():
        sub = df[df["CHR"] == chrom].sort_values("BP")
        if len(sub) < window_size: continue
        for i in range(0, len(sub) - window_size, window_size // 2):
            win = sub.iloc[i:i+window_size]
            results.append({"CHR": chrom, "START": win["BP"].min(), "END": win["BP"].max(), "MEAN_SCORE": win["SCORE"].mean(), "SNP": win.iloc[win["SCORE"].argmax()]["SNP"]})
    return pd.DataFrame(results)

def calculate_fine_mapping(res_df):
    if res_df.empty: return res_df
    df = res_df.copy().sort_values("P")
    df["ABF"] = np.exp(0.5 * (df["BETA"]/df["SE"])**2 - np.log(np.sqrt(1 + (df["SE"]**2))))
    df["PIP"] = df["ABF"] / df["ABF"].sum()
    return df.sort_values("PIP", ascending=False)

def run_gblup(gt, pheno_vec):
    valid = np.isfinite(pheno_vec)
    if valid.sum() < 10: raise ValueError("Moins de 10 phénotypes valides.")
    Y = pheno_vec[valid]; X = impute_mean(gt[valid]); p = X.mean(0) / 2.0; Z = X - 2 * p
    G = (Z @ Z.T) / (2 * np.sum(p * (1 - p))) + np.eye(len(Y)) * 1e-5
    lam = 0.7 / 0.3 
    GEBV = G @ np.linalg.inv(G + lam * np.eye(len(Y))) @ Y
    return pd.DataFrame({"IID": np.where(valid)[0], "Phenotype": Y, "GEBV": GEBV})

# ============================================================
# GWAS & LMM
# ============================================================
def load_phenotypes_file(file_bytes, filename=""):
    raw = gzip.decompress(file_bytes) if filename.endswith(".gz") else file_bytes
    text = raw.decode("utf-8", errors="replace"); sep = "\t" if text.count("\t") > text.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python"); df.columns = [str(c).strip() for c in df.columns]
    return df

def match_phenotypes(ind_df, pheno_df, iid_col=None, mapping_df=None):
    if mapping_df is not None:
        p, m = pheno_df.copy(), mapping_df.copy()
        iid_col = iid_col or next((c for c in ["Animal", "IID", "id", "ID", "sample"] if c in p.columns), None)
        p[iid_col] = p[iid_col].astype(str)
        animal_col_map = [c for c in m.columns if c != "IID"][0]
        m[animal_col_map] = m[animal_col_map].astype(str); m["IID"] = m["IID"].astype(str)
        p_merged = p.merge(m, left_on=iid_col, right_on=animal_col_map, how="left")
        ind = ind_df.copy(); ind["_key"] = ind["IID"].astype(str)
        p_merged["_key"] = p_merged["IID"].astype(str)
        return ind[["_key", "FID", "IID"]].merge(p_merged.drop(columns=["IID"], errors="ignore"), on="_key", how="left").drop(columns="_key"), iid_col
    iid_col = iid_col or next((c for c in ["IID", "id", "ID", "sample", "animal", "Animal"] if c in pheno_df.columns), None)
    p = pheno_df.copy(); p[iid_col] = p[iid_col].astype(str); ind = ind_df.copy(); ind["_key"] = ind["IID"].astype(str); p["_key"] = p[iid_col]
    return ind[["_key", "FID", "IID"]].merge(p.drop(columns=[iid_col]), on="_key", how="left").drop(columns="_key"), iid_col

def _vc_estimate_delta(Y, X0, K):
    n = len(Y); d, U = np.linalg.eigh(K); d = np.maximum(d, 0)
    Yt = U.T @ Y; X0t = U.T @ X0
    def neg_loglik(delta):
        if delta <= 0: return 1e18
        w = 1.0 / np.sqrt(delta * d + 1.0); Yw = Yt * w; Xw = X0t * w[:, None]
        try: beta, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
        except np.linalg.LinAlgError: return 1e18
        s2 = float(np.sum((Yw - Xw @ beta) ** 2) / n)
        return 1e18 if s2 <= 0 else -(-0.5 * (n * np.log(s2) + np.sum(np.log(delta * d + 1.0))))
    grid = np.logspace(-3, 3, 30); lls = np.array([neg_loglik(dl) for dl in grid])
    i_best = int(np.argmin(lls)); lo, hi = grid[max(0, i_best - 1)], grid[min(len(grid) - 1, i_best + 1)]
    try: return max(float(minimize_scalar(neg_loglik, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4}).x), 1e-6)
    except Exception: return max(float(grid[i_best]), 1e-6)

@cache_data
def run_gwas_lmm(gt, pheno_vec, K, covariates, maf_min=0.05, missing_max=0.05, use_lmm=True):
    valid_ind = np.isfinite(pheno_vec)
    if valid_ind.sum() < 10: raise ValueError("Moins de 10 individus avec phénotype valide.")
    Y, gt_v, K_v = pheno_vec[valid_ind], gt[valid_ind], np.asarray(K)[np.ix_(valid_ind, valid_ind)]
    X0 = np.ones((len(Y), 1)) if covariates is None or covariates.size == 0 else np.column_stack([np.ones(len(Y)), np.asarray(covariates)[valid_ind]])
    X0 = X0[:, np.std(X0, axis=0) > 1e-10]; n = len(Y)
    if use_lmm:
        delta = _vc_estimate_delta(Y, X0, K_v); d, U = np.linalg.eigh(K_v); d = np.maximum(d, 0)
        w = 1.0 / np.sqrt(delta * d + 1.0); Yt, X0t = (U.T @ Y) * w, (U.T @ X0) * w[:, None]
    else: Yt, X0t = Y, X0
    out = {"SNP_idx": [], "MAF": [], "BETA": [], "SE": [], "T": [], "P": []}
    for j in range(gt_v.shape[1]):
        g = gt_v[:, j].astype(float)
        if np.isnan(g).mean() > missing_max: continue
        g = np.where(np.isnan(g), np.nanmean(g), g); p = g.mean() / 2.0; maf_j = min(p, 1.0 - p)
        if maf_j < maf_min: continue
        X = np.column_stack([X0t, (U.T @ g) * w if use_lmm else g])
        beta, *_ = np.linalg.lstsq(X, Yt, rcond=None); resid = Yt - X @ beta
        s2 = float(np.sum(resid ** 2) / max(n - X.shape[1], 1))
        if s2 <= 0: continue
        se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[-1, -1]))
        if se > 0:
            t_val = beta[-1] / se; out["SNP_idx"].append(j); out["MAF"].append(maf_j); out["BETA"].append(beta[-1]); out["SE"].append(se); out["T"].append(t_val); out["P"].append(float(2 * t_dist.sf(abs(t_val), df=max(n - X.shape[1], 1))))
    return pd.DataFrame(out) if out["SNP_idx"] else pd.DataFrame(columns=["SNP_idx", "MAF", "BETA", "SE", "T", "P"])

def annotate_gwas_results(res_df, snp_df):
    if res_df.empty: return res_df
    idx = res_df["SNP_idx"].astype(int).values; res_df = res_df.copy()
    for col in ["SNP", "CHR", "BP", "CM"]: res_df[col] = snp_df[col].astype(str if col in ["SNP","CHR"] else float if col=="CM" else int).values[idx]
    return res_df[["SNP_idx", "SNP", "CHR", "BP", "CM", "MAF", "BETA", "SE", "T", "P"]]

def compute_lambda_gc(pvals):
    p = np.asarray(pvals); p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    return float(np.median(stats.chi2.isf(p, df=1)) / stats.chi2.ppf(0.5, df=1)) if len(p) >= 10 else np.nan
  # ============================================================
# IA & INTERPRÉTATION
# ============================================================
def interpret_qc(qc_stats):
    pct_ind = qc_stats["excluded_ind"] / max(qc_stats["n_ind_init"], 1) * 100
    pct_snp = qc_stats["excluded_snp"] / max(qc_stats["n_snp_init"], 1) * 100
    if pct_ind < 2 and pct_snp < 10: v, e = "🟢 EXCELLENTE", "Qualité de génotypage très élevée."
    elif pct_ind < 5 and pct_snp < 25: v, e = "🟡 ACCEPTABLE", "Qualité correcte, artéfacts éliminés normalement."
    else: v, e = "🔴 FAIBLE", "Exclusion élevée → puce/ADN problématique."
    return {"verdict": v, "explanation": e, "metrics": {"ind_exclus_%": round(pct_ind, 2), "snp_exclus_%": round(pct_snp, 2)}, "recommendation": ("Passez à l'étape suivante." if "🟢" in v else "Vérifiez la qualité des puces.")}

def interpret_maf(maf_values):
    m = np.asarray(maf_values); m = m[np.isfinite(m)]; mm = float(m.mean()); pct = float((m < 0.05).mean() * 100)
    if mm > 0.25: v, e = "🟢 Spectre riche", "Bonne diversité, variants informatifs."
    elif mm > 0.15: v, e = "🟡 Diversité modérée", "Population avec diversité moyenne."
    else: v, e = "🔴 Diversité faible", "Population consanguine ou échantillon réduit."
    return {"verdict": v, "explanation": e, "metrics": {"MAF_moyen": round(mm, 4), "pct_MAF<0.05": round(pct, 2)}, "recommendation": f"Filtre MAF 0.05 élimine {pct:.1f}% des SNPs."}

def interpret_fst(fst_values):
    m = float(np.nanmean(np.asarray(fst_values)))
    if m < 0.05: v, e = "🟢 Faible", "Fort flux génique."
    elif m < 0.15: v, e = "🟡 MODÉRÉE", "Races distinctes, échanges historiques."
    elif m < 0.25: v, e = "🟠 FORTE", "Races isolées génétiquement."
    else: v, e = "🔴 TRÈS FORTE", "Quasi-espèces distinctes."
    return {"verdict": v, "explanation": e, "metrics": {"FST_moyen": round(m, 4)}, "recommendation": "Utiliser PC1-PC10 + MDS comme covariables GWAS."}

def interpret_admixture(cv_results):
    ks = sorted(cv_results.keys()); errors = [cv_results[k]["mean"] for k in ks]
    drops = [errors[i] - errors[i + 1] for i in range(len(errors) - 1)]
    k_opt = ks[int(np.argmax(drops)) + 1] if drops else ks[0]
    return {"verdict": f"🟢 K optimal = {k_opt}", "explanation": "Nombre d'ancestralités minimisant la cross-entropy.", "metrics": {f"K={k}": round(cv_results[k]["mean"], 4) for k in ks}, "recommendation": f"Relancer l'admixture avec K={k_opt}."}

def interpret_roh(froh):
    m = float(np.nanmean(np.asarray(froh)))
    if m < 0.02: v, e = "🟢 Consanguinité faible", "Bonne diversité."
    elif m < 0.10: v, e = "🟡 Consanguinité modérée", "Niveau normal."
    else: v, e = "🔴 Consanguinité élevée", "Risque de dépression."
    return {"verdict": v, "explanation": e, "metrics": {"FROH_moyen": round(m, 4), "FROH_max": round(float(np.nanmax(froh)), 4)}, "recommendation": ("Croiser avec lignées extérieures" if m > 0.10 else "Population saine.")}

def interpret_with_llm(context, api_key=None, model="llama-3.3-70b-versatile"):
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key: return "⚠️ Pas de clé API Groq."
    try:
        import requests
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "system", "content": "Expert génétique bovine."}, {"role": "user", "content": f"Analyse ces résultats : (1) Synthèse, (2) Interprétation, (3) Recommandations.\n{json.dumps(context, indent=2, default=str)[:6000]}"}], "temperature": 0.3}, timeout=30)
        r.raise_for_status(); return r.json()["choices"][0]["message"]["content"]
    except Exception as e: return f"⚠️ Erreur LLM : {e}"

def render_ai_panel(module, data, key_suffix=""):
    st.divider(); st.markdown(f"### 🤖 Interprétation IA — {module}")
    interp_map = {"QC": interpret_qc, "MAF": interpret_maf, "FST": interpret_fst, "Admixture": interpret_admixture, "ROH": interpret_roh}
    if module in interp_map:
        result = interp_map[module](data)
        st.success(f"**{result['verdict']}** — {result['explanation']}")
        c1, c2 = st.columns(2)
        with c1: st.markdown("**📊 Métriques**"); st.json(result["metrics"])
        with c2: st.markdown("**✅ Recommandation**"); st.info(result["recommendation"])
    with st.expander("🧠 Analyse LLM approfondie (Groq gratuit)"):
        st.caption("Clé gratuite sur https://console.groq.com/keys")
        k = st.text_input("Clé API Groq", type="password", key=f"groq_{module}_{key_suffix}")
        if st.button(f"🚀 Analyser {module}", key=f"btn_llm_{module}_{key_suffix}"):
            ctx = (data if isinstance(data, dict) else {"raw": str(data)[:2000]})
            with st.spinner("Analyse LLM..."): st.markdown(interpret_with_llm(ctx, api_key=k))

# ============================================================
# EXPORT (PLINK & VCF)
# ============================================================
def _dosage_to_plink_bits(col):
    n = len(col); bits = np.zeros(n, dtype=np.uint8)
    bits[np.isnan(col)] = 0b01; bits[col == 0] = 0b11; bits[col == 1] = 0b10; bits[col == 2] = 0b00
    return bits

def build_plink_bed(gt):
    n_ind, n_snp = gt.shape; n_bytes = (n_ind + 3) // 4
    buf = bytearray([0x6C, 0x1B, 0x01])
    for j in range(n_snp):
        bits = _dosage_to_plink_bits(gt[:, j]); packed = np.zeros(n_bytes, dtype=np.uint8)
        for k in range(n_ind): packed[k // 4] |= (bits[k] & 0x03) << (2 * (k % 4))
        buf.extend(packed.tobytes())
    return bytes(buf)

def build_plink_bim(snp_df):
    lines = []
    has_a1, has_a2 = "A1" in snp_df.columns, "A2" in snp_df.columns
    for _, r in snp_df.iterrows():
        a1 = str(r["A1"]) if has_a1 else "A"; a2 = str(r["A2"]) if has_a2 else "G"
        if a1 in ("nan", ""): a1 = "A"
        if a2 in ("nan", "", a1): a2 = "G" if a1 == "A" else "A"
        lines.append(f"{r['CHR']}\t{r['SNP']}\t{r['CM']}\t{int(r['BP'])}\t{a1}\t{a2}")
    return "\n".join(lines) + "\n"

def build_plink_fam(ind_df):
    return "\n".join(f"{r['FID']}\t{r['IID']}\t0\t0\t0\t-9" for _, r in ind_df.iterrows()) + "\n"

def build_plink_zip(gt, ind_df, snp_df, prefix="bovine_qc"):
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}.bed", build_plink_bed(gt)); zf.writestr(f"{prefix}.bim", build_plink_bim(snp_df)); zf.writestr(f"{prefix}.fam", build_plink_fam(ind_df))
    return bio.getvalue()

def build_vcf_output(gt, ind_df, snp_df, project="BovineSNP"):
    n_ind, n_snp = gt.shape; has_a1, has_a2 = "A1" in snp_df.columns, "A2" in snp_df.columns
    header = ["##fileformat=VCFv4.2", f"##source=BovineSNPPlatform-{project}", f"##fileDate={datetime.now():%Y%m%d}", '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">']
    samples = ind_df["IID"].astype(str).tolist(); header.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples))
    body = []
    for j in range(n_snp):
        r = snp_df.iloc[j]; ref = str(r["A1"]) if has_a1 else "A"; alt = str(r["A2"]) if has_a2 else "G"
        if ref in ("nan", ""): ref = "A"
        if alt in ("nan", "", ref): alt = "G" if ref == "A" else "A"
        gts = ["./." if np.isnan(gt[i, j]) else "0/0" if gt[i, j] == 0 else "0/1" if gt[i, j] == 1 else "1/1" for i in range(n_ind)]
        body.append(f"{r['CHR']}\t{int(r['BP'])}\t{r['SNP']}\t{ref}\t{alt}\t.\tPASS\t.\tGT\t" + "\t".join(gts))
    return "\n".join(header + body) + "\n"

# ============================================================
# VISUALISATION (R-STYLE)
# ============================================================
def plot_hist(values, title, xlabel, color="#3498db"):
    fig = go.Figure(go.Histogram(x=values, nbinsx=80, marker_color=color))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="Fréquence", height=380, margin=dict(l=40, r=20, t=50, b=40))
    return fig

def plot_missingness_dashboard(miss_ind, miss_snp):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Missingness / individu", "Missingness / SNP"))
    fig.add_trace(go.Histogram(x=miss_ind, nbinsx=60, marker_color="skyblue"), row=1, col=1); fig.add_trace(go.Histogram(x=miss_snp, nbinsx=60, marker_color="coral"), row=1, col=2)
    fig.update_layout(height=400, showlegend=False, margin=dict(l=40, r=20, t=60, b=40))
    fig.update_xaxes(title_text="Fréquence manquante", row=1, col=1); fig.update_xaxes(title_text="Fréquence manquante", row=1, col=2); fig.update_yaxes(title_text="Nombre", row=1, col=1)
    return fig

def plot_pca(scores, var_pct, labels):
    df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1], "Population": labels})
    fig = px.scatter(df, x="PC1", y="PC2", color="Population", title=f"PCA — PC1 ({var_pct[0]:.1f}%) vs PC2 ({var_pct[1]:.1f}%)", height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white"))); return fig

def plot_mds(coords, labels):
    df = pd.DataFrame({"MDS1": coords[:, 0], "MDS2": coords[:, 1], "Population": labels})
    fig = px.scatter(df, x="MDS1", y="MDS2", color="Population", title="MDS (IBS) — Structure", height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white"))); return fig

def plot_manhattan(fst, chr_col, threshold_q=0.999):
    df = pd.DataFrame({"FST": np.asarray(fst), "CHR": np.asarray(chr_col)}).dropna(subset=["FST"]).reset_index(drop=True)
    if df.empty: return None
    df = df.assign(_k=df["CHR"].map(_chr_sort_key)).sort_values(["_k"]).reset_index(drop=True)
    fig = go.Figure()
    for i, (chrom, sub) in enumerate(df.groupby("CHR")): fig.add_trace(go.Scatter(x=sub.index, y=sub["FST"], mode="markers", marker=dict(size=5, color="#2c3e50" if i % 2 == 0 else "#7f8c8d"), showlegend=False, hoverinfo="skip"))
    q_upper = float(np.nanquantile(fst, threshold_q)); fig.add_hline(y=q_upper, line_dash="dash", line_color="red")
    fig.update_layout(title="Manhattan Plot — FST/SNP", xaxis_title="Chromosome", yaxis_title="FST", height=500); return fig

def plot_ld_decay(ld_df, bin_kb=20):
    if ld_df is None or ld_df.empty: return None
    d = ld_df.copy(); d["bin"] = (d["dist_kb"] // bin_kb) * bin_kb; agg = d.groupby("bin")["r2"].mean().reset_index()
    fig = px.line(agg, x="bin", y="r2", labels={"bin": "Distance (kb)", "r2": "r² moyen"}, title="LD decay", height=450); return fig

def plot_kinship_heatmap(G, labels):
    labels = [str(x) for x in labels]; n = len(labels); seen, uniq = {}, []
    for lab in labels: seen[lab] = seen.get(lab, 0) + 1; uniq.append(lab if seen[lab] == 1 else f"{lab}#{seen[lab]}")
    vmin, vmax = float(np.nanpercentile(G, 1)), float(np.nanpercentile(G, 99))
    if not np.isfinite(vmin) or vmin == vmax: vmin, vmax = -0.3, 0.5
    custom = np.stack([np.repeat(uniq, n).reshape(n, n), np.tile(uniq, n).reshape(n, n)], axis=-1)
    fig = go.Figure(data=go.Heatmap(z=G, colorscale="RdBu", zmid=0.0, zmin=vmin, zmax=vmax, colorbar=dict(title="Kinship"), hovertemplate="%{customdata[0]} × %{customdata[1]}<br>φ = %{z:.3f}<extra></extra>", customdata=custom))
    fig.update_layout(title="Matrice de parenté (KING robust)", height=650, xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, autorange="reversed")); return fig

def plot_admixture(Q, labels, pop_labels, K):
    df = pd.DataFrame(Q, columns=[f"K{k+1}" for k in range(K)]); df["IID"] = labels; df["Pop"] = pop_labels; df["_dom"] = Q.argmax(1)
    df["_po"] = pd.Categorical(pop_labels, categories=sorted(set(pop_labels))); df = df.sort_values(["_po", "_dom"]).reset_index(drop=True); df["x"] = np.arange(len(df))
    fig = go.Figure()
    for k in range(K): fig.add_trace(go.Bar(x=df["x"], y=df[f"K{k+1}"], name=f"Composante {k+1}", marker_color=px.colors.qualitative.Set2[k % 8]))
    fig.update_layout(barmode="stack", title=f"Proportions d'ancestralité (K={K})", xaxis_title="Individus triés", yaxis_title="Proportion", height=500); return fig, df

def plot_fst_pairwise(matrix, pops):
    fig = go.Figure(data=go.Heatmap(z=matrix, x=pops, y=pops, colorscale="Viridis", colorbar=dict(title="FST"), text=np.round(matrix, 4), texttemplate="%{text}"))
    fig.update_layout(title="FST pairwise", height=550); return fig

def plot_roh_histogram(froh, labels):
    return px.histogram(pd.DataFrame({"FROH": froh, "Population": labels}), x="FROH", color="Population", nbins=50, title="Distribution de FROH", height=450)

def plot_ne_curves(ne_dict):
    fig = go.Figure(); palette = (px.colors.qualitative.Set2 + px.colors.qualitative.Set3)
    for i, (race, df) in enumerate(ne_dict.items()):
        if df is None or df.empty: continue
        fig.add_trace(go.Scatter(x=df["GenAgo"], y=df["Ne"], mode="lines+markers", name=race, line=dict(color=palette[i % len(palette)], width=3)))
    fig.update_layout(title="Évolution historique de la taille efficace (Ne)", xaxis_title="Générations passées", yaxis_title="Ne (interprétation relative)", height=550); return fig

def plot_cv_error(cv_results):
    ks = sorted(cv_results.keys()); means = [cv_results[k]["mean"] for k in ks]; stds = [cv_results[k]["std"] for k in ks]
    fig = go.Figure(); fig.add_trace(go.Scatter(x=ks, y=means, mode="lines+markers", error_y=dict(type="data", array=stds), line=dict(color="crimson", width=3), marker=dict(size=12)))
    k_opt = ks[int(np.argmin(means))]; fig.add_vline(x=k_opt, line_dash="dash", line_color="green", annotation_text=f"K optimal = {k_opt}")
    fig.update_layout(title="Cross-Entropy Error (choix K)", xaxis_title="K (ancestralités)", yaxis_title="CV Error", height=450); return fig

def plot_manhattan_generic(df, y_col, title, threshold=None):
    df = df[np.isfinite(df[y_col])].copy(); df = df.assign(_k=df["CHR"].map(_chr_sort_key)).sort_values(["_k", "BP"]).reset_index(drop=True)
    fig = go.Figure()
    for i, (chrom, sub) in enumerate(df.groupby("CHR")):
        fig.add_trace(go.Scatter(x=sub.index, y=sub[y_col], mode="markers", marker=dict(size=5, color="#2c3e50" if i % 2 == 0 else "#7f8c8d"), text=sub["SNP"], hoverinfo="text", showlegend=False))
    if threshold: fig.add_hline(y=threshold, line_dash="dash", line_color="red")
    fig.update_layout(title=title, xaxis_title="Chromosome", yaxis_title=y_col, height=500); return fig

def plot_locuszoom(res_df, chrom, center_val, window_val, use_cm=False):
    df = res_df[res_df["CHR"].astype(str) == str(chrom)].copy()
    if use_cm: df = df[(df["CM"] >= center_val - window_val) & (df["CM"] <= center_val + window_val)]; x_vals, x_title = df["CM"], "Position (cM)"
    else: window_bp = window_val * 1000; df = df[(df["BP"] >= center_val - window_bp) & (df["BP"] <= center_val + window_bp)]; x_vals, x_title = df["BP"] / 1e6, "Position (Mb)"
    if df.empty: return None
    df["-log10P"] = -np.log10(np.clip(df["P"], 1e-300, 1)); fig = go.Figure(go.Scatter(x=x_vals, y=df["-log10P"], mode="markers", marker=dict(size=8, color=df["-log10P"], colorscale="Viridis", showscale=True), text=df["SNP"], hoverinfo="text"))
    fig.update_layout(title=f"LocusZoom — CHR {chrom}", xaxis_title=x_title, yaxis_title="-log10(P)", height=500); return fig

def plot_gblup(gebv_df):
    fig = px.scatter(gebv_df, x="Phenotype", y="GEBV", trendline="ols", title="Validation GBLUP : Phénotype vs GEBV")
    fig.update_trace(marker=dict(size=10, color="royalblue")); return fig

def plot_pip(pip_df):
    top = pip_df.head(20); fig = px.bar(top, x="SNP", y="PIP", title="Fine-Mapping : Probabilité d'Inclusion (PIP)", color="PIP", color_continuous_scale="Viridis"); return fig

# ============================================================
# STREAMLIT UI & STATE
# ============================================================
_STATE_KEYS = ["gt", "ind_df", "snp_df", "gt_filt", "ind_filt", "snp_filt", "gt_pre_outlier", "ind_pre_outlier", "snp_pre_outlier", "gt_pruned", "snp_pruned", "pca_scores", "pca_var", "king_matrix", "pheno_matched", "gwas_results", "sel_scan_df", "gebv_df", "admixture_Q", "admixture_K", "ld_df", "roh_df", "froh", "_conv_ped", "_conv_map", "cv_results", "ne_dict", "reynolds_D", "reynolds_pops", "fst", "fst_pairwise_matrix", "fst_pops", "selection_df", "qc_stats", "mds_coords", "run_requested", "_plink_zip", "_vcf_str", "_report_html"]

def init_state():
    for k in _STATE_KEYS:
        if k not in st.session_state: st.session_state[k] = None

def reset_all_derived():
    for k in ["gt_filt", "ind_filt", "snp_filt", "qc_stats", "gt_pre_outlier", "ind_pre_outlier", "snp_pre_outlier", "pca_scores", "pca_var", "mds_coords", "king_matrix", "ld_df", "fst", "fst_pairwise_matrix", "fst_pops", "admixture_Q", "admixture_K", "cv_results", "roh_df", "froh", "ne_dict", "reynolds_D", "reynolds_pops", "selection_df", "gt_pruned", "snp_pruned", "pheno_df", "pheno_matched", "gwas_results", "gwas_pheno_name", "gwas_lambda"]: st.session_state[k] = None

def main():
    init_state()
    st.title("🐄 Bovine SNP Platform v5.0 (Ultra-Professionnel)")
    st.caption("Pipeline complet — QC · LD · Structure · Admixture · SNeP · Reynolds · Sélection · GWAS · Post-GWAS · GBLUP · Export · IA.")

    with st.sidebar:
        st.header("📁 Données")
        mode = st.radio("Source :", ["Demo", "Upload PED/MAP", "Upload VCF", "🔄 Axiom → PED"], key="sb_mode_source")
        if mode == "Demo":
            c1, c2 = st.columns(2); n_ind = c1.number_input("Individus", 20, 1000, 150, 10); n_snp = c2.number_input("SNPs", 100, 10000, 800, 100); n_pop = st.slider("Populations", 2, 10, 4)
            if st.button("🎲 Générer démo", use_container_width=True):
                gt, ind_df, snp_df = generate_demo_data(int(n_ind), int(n_snp), int(n_pop)); ind_df = reconstruct_fid_from_iid(ind_df)
                st.session_state.update({"gt": gt, "ind_df": ind_df, "snp_df": snp_df}); reset_all_derived(); st.success(f"✅ {gt.shape[0]} ind × {gt.shape[1]} SNPs")
        elif mode == "Upload PED/MAP":
            ped_f = st.file_uploader("Fichier .ped", type=["ped", "gz", "txt"]); map_f = st.file_uploader("Fichier .map", type=["map", "gz", "txt"])
            if ped_f and map_f and st.button("📥 Charger"):
                map_df, _ = parse_map(map_f.read()); gt, ind_df, _ = parse_ped(ped_f.read(), len(map_df)); ind_df = reconstruct_fid_from_iid(ind_df)
                st.session_state.update({"gt": gt, "ind_df": ind_df, "snp_df": map_df}); reset_all_derived(); st.success(f"✅ {gt.shape[0]} ind × {gt.shape[1]} SNPs")
        elif mode == "Upload VCF":
            vcf_f = st.file_uploader("Fichier .vcf", type=["vcf", "gz", "txt"])
            if vcf_f and st.button("📥 Charger"):
                gt, ind_df, snp_df, _, _ = parse_vcf(vcf_f.read()); ind_df = reconstruct_fid_from_iid(ind_df)
                st.session_state.update({"gt": gt, "ind_df": ind_df, "snp_df": snp_df}); reset_all_derived(); st.success(f"✅ {gt.shape[0]} ind × {gt.shape[1]} variants")
        else:
            st.info("📄 Sélectionne le fichier brut dans la zone principale.")
            render_axiom_converter_page()

        st.divider(); st.header("⚙️ Seuils QC")
        geno = st.slider("Missingness SNP (--geno)", 0.0, 0.5, 0.05, 0.01); mind = st.slider("Missingness individu (--mind)", 0.0, 0.5, 0.05, 0.01)
        maf_thr = st.slider("MAF minimal (--maf)", 0.0, 0.5, 0.05, 0.01); hwe_thr = st.number_input("HWE p-value (--hwe)", value=1e-6, format="%.0e")
        het_sd = st.slider("Hétérozygotie ±σ intra-race", 1.0, 5.0, 3.0, 0.1)
        remove_outliers = st.checkbox("❌ Supprimer les outliers (Hétérozygotie)", value=True, help="Décochez pour garder tous les individus.")
        ld_r2 = st.slider("LD Pruning r² seuil", 0.05, 0.5, 0.2, 0.05); king_cutoff = st.slider("KING cutoff", 0.1, 0.5, 0.354, 0.01)

        st.divider()
        if st.button("🚀 Pipeline complet", type="primary", use_container_width=True):
            if st.session_state.gt is None: st.error("Chargez des données.")
            else: st.session_state.run_requested = True

    if st.session_state.get("sb_mode_source") == "🔄 Axiom → PED" and st.session_state.gt is None: return
    if st.session_state.gt is None:
        st.info("👉 Générez un jeu de démo, importez PED/MAP ou VCF, ou utilisez le convertisseur Axiom."); return

    tabs = st.tabs(["🏠 Aperçu", "🧹 QC", "✂️ LD Pruning", "🧬 Structure", "🎨 Admixture", "📈 Démographie", "🔍 Sélection", "🧪 GWAS", "🎯 Post-GWAS", "📈 GBLUP", "🌳 Phylogénie", "📤 Export", "📄 Rapport"])

    with tabs[0]:
        c1, c2, c3 = st.columns(3); c1.metric("Individus", st.session_state.gt.shape[0]); c2.metric("SNPs", st.session_state.gt.shape[1]); c3.metric("Populations (FID)", st.session_state.ind_df["FID"].nunique())
        st.dataframe(st.session_state.ind_df.head(20), use_container_width=True); st.dataframe(st.session_state.snp_df.head(5), use_container_width=True)

    with tabs[1]:
        if st.button("🧬 Filtrer autosomes 1-29"):
            gt_a, snp_a = filter_autosomes(st.session_state.gt, st.session_state.snp_df); st.session_state.update({"gt": gt_a, "snp_df": snp_a}); st.success(f"✅ Autosomes → {gt_a.shape[1]} SNPs")
        if st.button("▶ Lancer le QC complet", type="primary"):
            params = {"geno": geno, "mind": mind, "maf": maf_thr, "hwe": hwe_thr, "het_sd": het_sd}
            gt_f, ind_f, snp_f, gt_pre, ind_pre, snp_pre, qc_stats = apply_qc_filters(st.session_state.gt, st.session_state.ind_df, st.session_state.snp_df, params, remove_outliers=remove_outliers)
            st.session_state.update({"gt_filt": gt_f, "ind_filt": ind_f, "snp_filt": snp_f, "gt_pre_outlier": gt_pre, "ind_pre_outlier": ind_pre, "snp_pre_outlier": snp_pre, "qc_stats": qc_stats}); st.success("✅ QC terminé.")
        if st.session_state.gt_filt is not None:
            s = st.session_state.qc_stats; c1, c2, c3, c4 = st.columns(4)
            c1.metric("Individus finaux", s["n_ind_final"], delta=-s["excluded_ind"]); c2.metric("SNPs finaux", s["n_snp_final"], delta=-s["excluded_snp"]); c3.metric("Exclus (ind)", s["excluded_ind"]); c4.metric("Exclus (SNPs)", s["excluded_snp"])
            st.json(s.get("trace", {})); st.info(f"💾 Jeu 'Pré-Outliers' : {st.session_state.gt_pre_outlier.shape[0]} ind, {st.session_state.gt_pre_outlier.shape[1]} SNPs.")
            st.plotly_chart(plot_missingness_dashboard(missingness_per_ind(st.session_state.gt), missingness_per_snp(st.session_state.gt)), use_container_width=True)

    with tabs[2]:
        if st.session_state.gt_filt is not None and st.button("▶ Lancer le LD Pruning", type="primary"):
            keep = ld_pruning(st.session_state.gt_filt, window=50, step=5, r2_thr=ld_r2)
            st.session_state.update({"gt_pruned": st.session_state.gt_filt[:, keep], "snp_pruned": st.session_state.snp_filt[keep].reset_index(drop=True)}); st.success(f"✅ {st.session_state.gt_pruned.shape[1]} SNPs indépendants conservés.")

    with tabs[3]:
        if st.session_state.gt_filt is not None and st.button("▶ PCA + MDS + KING", type="primary"):
            gt_use = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
            s_, v_ = pca_analysis(gt_use, 10); st.session_state.update({"pca_scores": s_, "pca_var": v_, "mds_coords": mds_analysis(gt_use, 5), "king_matrix": king_kinship(gt_use)}); st.success("✅ Terminé.")
        if st.session_state.pca_scores is not None: st.plotly_chart(plot_pca(st.session_state.pca_scores, st.session_state.pca_var, st.session_state.ind_filt["FID"].values), use_container_width=True)

    with tabs[4]:
        if st.session_state.gt_filt is not None and st.button("▶ Calculer l'admixture (NMF)"):
            gt_use = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
            Q, _ = admixture_nmf(gt_use, K=4); st.session_state.update({"admixture_Q": Q, "admixture_K": 4}); st.success("✅ Admixture calculé.")
        if st.session_state.admixture_Q is not None:
            fig, _ = plot_admixture(st.session_state.admixture_Q, st.session_state.ind_filt["IID"].values, st.session_state.ind_filt["FID"].values, st.session_state.admixture_K); st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        if st.session_state.gt_filt is not None and st.button("▶ LD Decay + ROH"):
            gt_use = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
            st.session_state.ld_df = ld_decay(gt_use, st.session_state.snp_filt["BP"].values)
            roh_df, froh = detect_roh(gt_use, st.session_state.snp_filt[["CHR", "BP"]].to_json()); st.session_state.update({"roh_df": roh_df, "froh": froh}); st.success("✅ Démographie calculée.")
        if st.session_state.ld_df is not None: st.plotly_chart(plot_ld_decay(st.session_state.ld_df), use_container_width=True)

    with tabs[6]:
        if st.button("▶ Scan Sélection (Windowed)"):
            gt_use = st.session_state.gt_pre_outlier if st.session_state.gt_pre_outlier is not None else st.session_state.gt_filt
            ind_use = st.session_state.ind_pre_outlier if st.session_state.ind_pre_outlier is not None else st.session_state.ind_filt
            snp_use = st.session_state.snp_pre_outlier if st.session_state.snp_pre_outlier is not None else st.session_state.snp_filt
            st.session_state.sel_scan_df = windowed_selection_scan(gt_use, snp_use, ind_use["FID"].astype(str).values); st.success("✅ Scan terminé.")
        if st.session_state.sel_scan_df is not None: st.plotly_chart(px.scatter(st.session_state.sel_scan_df, x="START", y="MEAN_SCORE", color="CHR", title="Signatures de sélection"), use_container_width=True)

    with tabs[7]:
        mapping_file = st.file_uploader("Correspondance ID (CSV 'Animal' et 'IID')", type=["csv", "txt"]); mapping_df = pd.read_csv(mapping_file) if mapping_file else None
        pheno_file = st.file_uploader("Phénotypes", type=["csv", "txt", "gz"])
        if pheno_file and st.button("📥 Charger"):
            merged, _ = match_phenotypes(st.session_state.ind_filt, load_phenotypes_file(pheno_file.read(), pheno_file.name), mapping_df=mapping_df); st.session_state.pheno_matched = merged; st.success("✅ Chargé.")
        if st.session_state.get("pheno_matched") is not None:
            pheno_col = st.selectbox("Phénotype", [c for c in st.session_state.pheno_matched.columns if c not in ("FID", "IID", "Animal")])
            if st.button("🚀 Lancer GWAS", type="primary"):
                gt_gwas = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                snp_gwas = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                covs = st.session_state.pca_scores[:, :5] if st.session_state.pca_scores is not None else np.empty((gt_gwas.shape[0], 0))
                res = run_gwas_lmm(gt_gwas, st.session_state.pheno_matched[pheno_col].values, st.session_state.king_matrix, covs)
                st.session_state.gwas_results = annotate_gwas_results(res, snp_gwas); st.success(f"✅ {len(res)} SNPs testés.")
                st.plotly_chart(plot_manhattan_generic(st.session_state.gwas_results, "P", "GWAS Manhattan"), use_container_width=True)

    with tabs[8]:
        if st.session_state.gwas_results is not None:
            pip_df = calculate_fine_mapping(st.session_state.gwas_results); st.dataframe(pip_df[["SNP", "P", "PIP"]].head(10)); st.plotly_chart(plot_pip(pip_df), use_container_width=True)
        else: st.warning("Lancez d'abord un GWAS.")

    with tabs[9]:
        if st.session_state.get("pheno_matched") is not None:
            pheno_col = st.selectbox("Phénotype pour GBLUP", [c for c in st.session_state.pheno_matched.columns if c not in ("FID", "IID", "Animal")], key="gb_pheno")
            if st.button("▶ Calculer GEBV"):
                gt_use = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                st.session_state.gebv_df = run_gblup(gt_use, st.session_state.pheno_matched[pheno_col].values); st.success("✅ GEBV calculés.")
            if st.session_state.gebv_df is not None: st.plotly_chart(plot_gblup(st.session_state.gebv_df), use_container_width=True)
        else: st.warning("Chargez des phénotypes.")

    with tabs[10]:
        if st.button("▶ Reynolds + NJ"):
            gt_use = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
            D, pops = reynolds_distance(gt_use, st.session_state.ind_filt["FID"].values); st.session_state.reynolds_D = D; st.session_state.reynolds_pops = pops; st.success("✅ Calculé.")
        if st.session_state.reynolds_D is not None:
            st.plotly_chart(go.Figure(go.Heatmap(z=st.session_state.reynolds_D, x=st.session_state.reynolds_pops, y=st.session_state.reynolds_pops, colorscale="Blues")), use_container_width=True)
            st.code(nj_tree_newick(st.session_state.reynolds_D, st.session_state.reynolds_pops))

    with tabs[11]:
        if st.button("📦 Générer PLINK .zip"):
            gt_use = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
            snp_use = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
            st.session_state._plink_zip = build_plink_zip(gt_use, st.session_state.ind_filt, snp_use)
        if st.session_state.get("_plink_zip"): st.download_button("⬇ Télécharger .zip", st.session_state._plink_zip, "bovine_qc.zip", "application/zip")

    with tabs[12]:
        if st.button("📄 Générer rapport HTML"):
            html = f"<h1>Rapport Bovine SNP Platform</h1><p>SNPs: {st.session_state.gt_filt.shape[1]}</p>"
            st.session_state._report_html = html
        if st.session_state.get("_report_html"): st.download_button("⬇ Télécharger HTML", st.session_state._report_html.encode(), "rapport.html", "text/html")

if __name__ == "__main__":
    main()
