"""
Protein-level dosage compensation analysis.
Mirrors the RNA-level analysis but uses three protein datasets:
  1. CCLE Mass Spec Proteomics (Gygi lab, 371 cell lines, ~12K proteins)
  2. Sanger ProCan (949 cell lines, 8,498 proteins)
  3. CCLE RPPA (868 cell lines, ~200 proteins)

For sig pairs:  measure paralog_gene protein when dep_gene's arm is lost
For non-sig:    measure both genes' protein when their own arm is lost
                (exclude genes present in the sig set)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

BASE = "/Users/jonathanzhao/Desktop/Sheltzer Lab/Chromosome Compensation"
PARALOG_DIR = "/Users/jonathanzhao/Desktop/Sheltzer Lab/Paralog Difference"

# ── 1. Arm call scores ────────────────────────────────────────────────────────

arm = pd.read_csv(f"{BASE}/arm_call_scores.csv", index_col=0)

# ── 2. Cell-line ID mapping ───────────────────────────────────────────────────

model = pd.read_csv(
    "/Users/jonathanzhao/Desktop/Sheltzer Lab/Paralog Difference/data/Model.csv"
)
ccle_to_depmap = dict(zip(model["CCLEName"], model["ModelID"]))

# ── 3. Sig paralog pairs ──────────────────────────────────────────────────────

sig = pd.read_excel(f"{PARALOG_DIR}/sig_37_paralog copy.xlsx")

def map_sig_arm(pos):
    """'chr1_p' → '1p', 'chrY' → None"""
    if pd.isna(pos):
        return None
    s = str(pos).replace("chr", "").replace("_", "")
    return s if s in arm.columns else None

sig_queries = []
for _, row in sig.iterrows():
    arm_col = map_sig_arm(row["aneuploid_loss_chr"])
    if arm_col:
        sig_queries.append((row["paralog_gene"], arm_col))

sig_query_df = pd.DataFrame(sig_queries, columns=["gene", "arm"])
sig_gene_set = set(sig["dep_gene"]) | set(sig["paralog_gene"])
print(f"Sig queries: {len(sig_query_df)} (excluding chrY pairs)")

# ── 4. Non-sig paralog pairs ──────────────────────────────────────────────────

nonsig = pd.read_excel(f"{PARALOG_DIR}/non_sig_paralog copy.xlsx")

def map_nonsig_arm(pos):
    """'X19q' → '19q', 'chrY'/'chrX' → None"""
    if pd.isna(pos):
        return None
    s = str(pos).strip()
    if s.startswith("X"):
        s = s[1:]
    elif s.startswith("chr"):
        return None
    return s if s in arm.columns else None

nonsig_rows = []
for _, row in nonsig.iterrows():
    g1, g2 = row["hgnc_symbol"], row["para_gene_2"]
    arm1 = map_nonsig_arm(row["chr_position"])
    arm2 = map_nonsig_arm(row["chr_para2_position"])
    if pd.notna(g1) and arm1 and g1 not in sig_gene_set:
        nonsig_rows.append((g1, arm1))
    if pd.notna(g2) and arm2 and g2 not in sig_gene_set:
        nonsig_rows.append((g2, arm2))

nonsig_query_df = pd.DataFrame(nonsig_rows, columns=["gene", "arm"]).drop_duplicates()
print(f"Non-sig queries: {len(nonsig_query_df)} unique (gene, arm) pairs")

# ── 5. Dosage compensation function ──────────────────────────────────────────

def dosage_comp(query_df, expr_matrix):
    """
    For each (gene, arm) pair: compute median protein in chr-loss vs neutral cells.
    Returns DataFrame with ['gene','arm','median_loss','median_neutral'].
    """
    results = []
    available_genes = set(expr_matrix.columns)
    available_cells = set(expr_matrix.index)

    for _, row in query_df.iterrows():
        gene, arm_col = row["gene"], row["arm"]
        if gene not in available_genes:
            continue

        arm_series = arm[arm_col]
        loss_cells    = [c for c in arm_series[arm_series == -1].index if c in available_cells]
        neutral_cells = [c for c in arm_series[arm_series ==  0].index if c in available_cells]

        if not loss_cells or not neutral_cells:
            continue

        exp = expr_matrix[gene].astype(float)
        med_loss    = np.nanmedian(exp.loc[loss_cells].values)
        med_neutral = np.nanmedian(exp.loc[neutral_cells].values)

        if np.isnan(med_loss) or np.isnan(med_neutral):
            continue

        results.append({"gene": gene, "arm": arm_col,
                        "median_loss": med_loss, "median_neutral": med_neutral})

    return pd.DataFrame(results)

# ── 6a. Mass spec proteomics matrix ──────────────────────────────────────────

print("\nLoading mass spec proteomics...")
prot_raw = pd.read_csv(f"{BASE}/protein_quant_current_normalized.csv.gz", low_memory=False)

meta = {"Protein_Id", "Gene_Symbol", "Description", "Group_ID", "Uniprot", "Uniprot_Acc"}
expr_cols = [c for c in prot_raw.columns if c not in meta and "Peptides" not in c]

col_to_depmap = {}
for col in expr_cols:
    ccle_name = "_".join(col.split("_")[:-1])
    depmap_id = ccle_to_depmap.get(ccle_name)
    if depmap_id and depmap_id in arm.index:
        col_to_depmap[col] = depmap_id

# Convert expression columns to numeric first (before rename, to avoid dup-column issues)
prot_expr = prot_raw[["Gene_Symbol"] + list(col_to_depmap.keys())].copy()
for orig_col in col_to_depmap:
    prot_expr[orig_col] = pd.to_numeric(prot_expr[orig_col], errors="coerce")
prot_expr = prot_expr.rename(columns=col_to_depmap)

# Median across gene isoforms; groupby averages duplicate depmap_id columns automatically
depmap_id_cols = list(dict.fromkeys(col_to_depmap.values()))  # unique, order-preserved
prot_matrix = (
    prot_expr.groupby("Gene_Symbol").median().T
)
# Drop non-arm-call cell lines that may have slipped through
prot_matrix = prot_matrix[prot_matrix.index.isin(arm.index)]
print(f"  {prot_matrix.shape[0]} cell lines × {prot_matrix.shape[1]} genes")

# ── 6b. RPPA matrix with curated antibody→gene mapping ───────────────────────

print("\nLoading RPPA...")
rppa_raw = pd.read_csv(f"{BASE}/CCLE_RPPA_20181003.csv", index_col=0)

# Curated map for antibodies whose names don't match HGNC symbols
rppa_manual_map = {
    "SF2":              "SRSF2",
    "TAZ":              "WWTR1",
    "YAP_Caution":      "YAP1",
    "N-Ras":            "NRAS",
    "Akt":              "AKT1",
    "c-Myc_Caution":    "MYC",
    "c-Jun_pS73":       "JUN",
    "c-Kit":            "KIT",
    "c-Met_Caution":    "MET",
    "c-Met_pY1235":     "MET",
    "Src":              "SRC",
    "Src_pY416_Caution":"SRC",
    "Src_pY527":        "SRC",
    "Rb_Caution":       "RB1",
    "Rb_pS807_S811":    "RB1",
    "beta-Catenin":     "CTNNB1",
    "E-Cadherin":       "CDH1",
    "N-Cadherin":       "CDH2",
    "Cyclin_D1":        "CCND1",
    "Cyclin_B1":        "CCNB1",
    "Cyclin_E1":        "CCNE1",
    "Cyclin_E2_Caution":"CCNE2",
    "PTEN":             "PTEN",
    "EGFR":             "EGFR",
    "HER2":             "ERBB2",
    "HER3":             "ERBB3",
    "mTOR":             "MTOR",
    "CDK1":             "CDK1",
    "FASN":             "FASN",
    "PCNA_Caution":     "PCNA",
    "GAPDH_Caution":    "GAPDH",
    "Fibronectin":      "FN1",
    "ASNS":             "ASNS",
    "ATM":              "ATM",
    "ADAR1":            "ADAR",
    "AR":               "AR",
    "Bax":              "BAX",
    "Rad50":            "RAD50",
    "RAD51":            "RAD51",
    "MSH2":             "MSH2",
    "PRDX1":            "PRDX1",
    "TIGAR":            "TP53I3",
    "Stathmin":         "STMN1",
    "Raptor":           "RPTOR",
    "PREX1":            "PREX1",
    "G6PD":             "G6PD",
    "SCD1":             "SCD",
    "AMPK_pT172":       "PRKAA1",
    "Smad1":            "SMAD1",
    "Smad3":            "SMAD3",
    "Smad4":            "SMAD4",
    "JAK2":             "JAK2",
    "Syk":              "SYK",
    "Lck":              "LCK",
    "GATA3":            "GATA3",
    "FoxM1":            "FOXM1",
    "DJ-1":             "PARK7",
    "PAI-1":            "SERPINE1",
    "VEGFR2":           "KDR",
    "Tuberin":          "TSC2",
    "MEK1":             "MAP2K1",
    "INPP4B":           "INPP4B",
    "IRS1":             "IRS1",
    "Gab2":             "GAB2",
    "Dvl3":             "DVL3",
    "ETS-1":            "ETS1",
    "IGFBP2":           "IGFBP2",
    "RBM15":            "RBM15",
    "TFRC":             "TFRC",
    "SETD2_Caution":    "SETD2",
}

def rppa_col_to_gene(col):
    if col in rppa_manual_map:
        return rppa_manual_map[col]
    base = col.split("_")[0]
    # strip trailing common suffixes
    base = base.replace("-Caution", "").replace(" Caution", "")
    return base.upper()

rppa_gene_map = {col: rppa_col_to_gene(col) for col in rppa_raw.columns}

# Map RPPA index (CCLE names) to DepMap IDs
rppa_raw.index = [ccle_to_depmap.get(idx, idx) for idx in rppa_raw.index]
rppa_raw = rppa_raw[rppa_raw.index.isin(arm.index)]

rppa_renamed = rppa_raw.rename(columns=rppa_gene_map)
rppa_matrix = rppa_renamed.T.groupby(level=0).median().T
print(f"  {rppa_matrix.shape[0]} cell lines × {rppa_matrix.shape[1]} genes")

# ── 6c. ProCan matrix ────────────────────────────────────────────────────────

print("\nLoading ProCan...")
sidm_to_depmap = dict(zip(model["SangerModelID"], model["ModelID"]))

procan_mapping = pd.read_csv(f"{BASE}/ProCan_mapping_file_averaged.txt", sep="\t")
# Project_Identifier format: 'SIDM00018;K052' — extract SIDM ID
procan_mapping["SIDM"] = procan_mapping["Project_Identifier"].str.split(";").str[0]
procan_mapping["ModelID"] = procan_mapping["SIDM"].map(sidm_to_depmap)

# Build SIDM;CellName → DepMap ID lookup (using the index column of the matrix)
projid_to_depmap = dict(zip(procan_mapping["Project_Identifier"],
                            procan_mapping["ModelID"]))

procan_raw = pd.read_csv(f"{BASE}/ProCan_protein_matrix_8498_averaged.txt",
                         sep="\t", index_col=0)

# Parse gene symbols from column names like 'Q9Y651;SOX21_HUMAN' → 'SOX21'
def procan_col_to_gene(col):
    if ";" not in col:
        return None
    return col.split(";")[1].split("_")[0]

gene_map = {col: procan_col_to_gene(col) for col in procan_raw.columns}
procan_raw = procan_raw.rename(columns=gene_map)

# Map row index (Project_Identifier) to DepMap IDs, keep only arm-call cell lines
procan_raw.index = [projid_to_depmap.get(idx, idx) for idx in procan_raw.index]
procan_raw = procan_raw[procan_raw.index.isin(arm.index)]

# Average duplicate gene columns, convert to float
procan_matrix = procan_raw.apply(pd.to_numeric, errors="coerce")
procan_matrix = procan_matrix.T.groupby(level=0).median().T
print(f"  {procan_matrix.shape[0]} cell lines × {procan_matrix.shape[1]} genes")

# ── 7. Run analysis for each dataset ─────────────────────────────────────────

results_by_dataset = {}
for name, expr in [("Mass Spec\nProteomics", prot_matrix),
                   ("ProCan", procan_matrix),
                   ("RPPA", rppa_matrix)]:
    print(f"\nRunning dosage_comp for {name}...")
    sig_res   = dosage_comp(sig_query_df,    expr)
    nsig_res  = dosage_comp(nonsig_query_df, expr)
    print(f"  Sig: {len(sig_res)}  |  Non-sig: {len(nsig_res)}")
    if not sig_res.empty:
        sig_res["condition"] = "sig"
    if not nsig_res.empty:
        nsig_res["condition"] = "non_sig"
    combined = pd.concat([sig_res, nsig_res], ignore_index=True)
    combined["norm_exp"] = combined["median_loss"] - combined["median_neutral"]
    results_by_dataset[name] = combined
    combined.to_csv(
        f"{BASE}/protein_dosage_comp_{name.lower().replace(' ','_').replace(chr(10),'_')}.csv",
        index=False,
    )

# ── 8. Figure ─────────────────────────────────────────────────────────────────

SIG_COLOR = "#E8A76C"
PURPLE    = "#7852A9"

# Only show datasets with enough sig results to be meaningful (>= 5)
valid_datasets = {
    k: v for k, v in results_by_dataset.items()
    if (v["condition"] == "sig").sum() >= 5
    and (v["condition"] == "non_sig").sum() >= 5
}

if not valid_datasets:
    print("No datasets have sufficient sig results (n>=5). Check gene coverage.")
else:
    ncols = len(valid_datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5.5))
    if ncols == 1:
        axes = [axes]

    for ax, (dataset_name, combined) in zip(axes, valid_datasets.items()):
        sig_vals  = combined.loc[combined["condition"] == "sig",     "norm_exp"].dropna()
        nsig_vals = combined.loc[combined["condition"] == "non_sig", "norm_exp"].dropna()

        neutral_sig  = np.zeros(len(sig_vals))
        neutral_nsig = np.zeros(len(nsig_vals))

        positions = [0, 1, 2.5, 3.5]
        data      = [neutral_sig, sig_vals.values, neutral_nsig, nsig_vals.values]
        colors    = [PURPLE, SIG_COLOR, PURPLE, SIG_COLOR]

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="gray", linewidth=0.8),
            capprops=dict(color="gray",  linewidth=0.8),
            boxprops=dict(linewidth=0.8),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
            patch.set_edgecolor(color)

        ax.axhline(0, color="gray", lw=0.5, linestyle="--", alpha=0.5)

        # Bracket heights based on whisker tops
        whi_tops = [w.get_ydata()[1] for w in bp["whiskers"]]
        # whiskers come in pairs: [box0_lo, box0_hi, box1_lo, box1_hi, ...]
        sig_top  = max(whi_tops[0], whi_tops[1], 0)
        nsig_top = max(whi_tops[4], whi_tops[5], 0)  # boxes 2 and 3

        bracket_gap = 0.06

        # Wilcoxon: sig Chr loss vs 0
        if len(sig_vals) >= 5:
            _, p_sig = stats.wilcoxon(sig_vals.values, zero_method="wilcox",
                                      alternative="two-sided")
            bry_s = sig_top + bracket_gap
            ax.annotate("", xy=(1, bry_s), xytext=(0, bry_s),
                        arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
            ax.text(0.5, bry_s + 0.025, f"p = {p_sig:.2e}", ha="center", fontsize=8)

        # Wilcoxon: non-sig Chr loss vs 0
        if len(nsig_vals) >= 5:
            _, p_nsig = stats.wilcoxon(nsig_vals.values, zero_method="wilcox",
                                       alternative="two-sided")
            bry_n = nsig_top + bracket_gap
            ax.annotate("", xy=(3.5, bry_n), xytext=(2.5, bry_n),
                        arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
            ax.text(3.0, bry_n + 0.025, f"p = {p_nsig:.2e}", ha="center", fontsize=8)

        # Mann-Whitney: sig Chr loss vs non-sig Chr loss
        if len(sig_vals) >= 5 and len(nsig_vals) >= 5:
            _, p_mw = stats.mannwhitneyu(sig_vals.values, nsig_vals.values,
                                         alternative="two-sided")
            top_all = max(sig_top, nsig_top) + bracket_gap * 2.5
            ax.annotate("", xy=(3.5, top_all), xytext=(1, top_all),
                        arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
            ax.text(2.25, top_all + 0.025, f"p = {p_mw:.2e}", ha="center", fontsize=8)

        # Y limits: leave room for brackets and group labels below
        all_vals = np.concatenate([sig_vals.values, nsig_vals.values])
        y_lo = min(np.nanpercentile(all_vals, 2), -0.2) - 0.15
        ax.set_ylim(y_lo, ax.get_ylim()[1] + 0.15)

        ax.set_xticks(positions)
        ax.set_xticklabels(["Chr\nneutral", "Chr\nloss", "Chr\nneutral", "Chr\nloss"],
                           fontsize=8)
        ax.set_ylabel("Rel. protein expression", fontsize=10)
        ax.set_title(dataset_name, fontsize=10, pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Group underline labels (positioned just below x-axis tick labels)
        trans = ax.get_xaxis_transform()
        for x_lo, x_hi, label in [(0, 1, "Sig-paralog"), (2.5, 3.5, "Non-sig paralog")]:
            ax.annotate("", xy=(x_hi, -0.18), xytext=(x_lo, -0.18),
                        xycoords=trans, textcoords=trans,
                        arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
                        annotation_clip=False)
            ax.text((x_lo + x_hi) / 2, -0.24, label,
                    ha="center", va="top", fontsize=8,
                    transform=trans, clip_on=False)

        ax.text(0.98, 0.02,
                f"n sig = {len(sig_vals)}\nn non-sig = {len(nsig_vals)}",
                transform=ax.transAxes, fontsize=7, ha="right", va="bottom", color="gray")

    plt.tight_layout(pad=2.0)

    pdf_path = f"{BASE}/protein_dosage_compensation_figure.pdf"
    png_path = f"{BASE}/protein_dosage_compensation_figure.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"\nFigure saved:\n  {pdf_path}\n  {png_path}")

# ── 9. Summary ────────────────────────────────────────────────────────────────

print("\n=== Coverage summary ===")
for name, combined in results_by_dataset.items():
    if combined.empty:
        print(f"{name}: no results")
        continue
    sig_n   = (combined["condition"] == "sig").sum()
    nsig_n  = (combined["condition"] == "non_sig").sum()
    sig_v   = combined.loc[combined["condition"] == "sig",     "norm_exp"].dropna()
    nsig_v  = combined.loc[combined["condition"] == "non_sig", "norm_exp"].dropna()
    print(f"{name}:")
    print(f"  Sig:    n={sig_n},  median norm_exp = {sig_v.median():.3f}")
    print(f"  Non-sig: n={nsig_n}, median norm_exp = {nsig_v.median():.3f}")
