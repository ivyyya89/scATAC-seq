# -*- coding: utf-8 -*-
import sys
import os

os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"

import matplotlib
matplotlib.use('Agg')

import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. 配置路径与真正属于 ATAC 的 41 个亚型定义 ---
DATA_PATH     = "/data1/zihui/BA11/scATAC/BA11_Integrated_GeneActivity.h5ad"
SAVE_DIR      = "/data1/zihui/BA11/scATAC/"
CACHE_MARKERS = os.path.join(SAVE_DIR, "cache_atac_rank_genes.pkl")
CACHE_MATRIX  = os.path.join(SAVE_DIR, "cache_atac_plot_matrix_merged_41.pkl")
OUTPUT_PNG    = os.path.join(SAVE_DIR, "Perfect_ATAC_Dotplot_Merged.png")

print("正在加载 scATAC 整合数据...")
adata = sc.read_h5ad(DATA_PATH)
adata.obs.columns = adata.obs.columns.str.lower()
adata.obs['cell_subtype'] = adata.obs['cell_subtype'].astype('category')

# 【精准校准：真正的 scATAC 41 个亚型】
gaba_subtypes = [
    'InN ADARB2 FAM19A1', 'InN ADARB2 SFRP2', 'InN LAMP5 BMP7', 'InN LAMP5 LHX6 TAC1', 'InN LAMP5 RELN',
    'InN PVALB ANOS1', 'InN PVALB HPSE', 'InN PVALB PDE3A', 'InN PVALB PIEZO2', 'InN SST HS3ST5',
    'InN SST HTR2C', 'InN SST PLPP4', 'InN SST STK32A', 'InN SST THSD7B', 'InN SST TRPC7',
    'InN VIP CLSTN2', 'InN VIP EGF', 'InN VIP EXPH5', 'InN VIP HS3ST3B1', 'InN VIP SCML4'
]

glut_subtypes = [
    'L2-3 CUX2 ACVR1C THSD7A', 'L2-3 CUX2 NTNG1 COL5A2', 'L2-3 CUX2 NTNG1 PALMD', 'L2-3 CUX2 NTNG1 PLCH1',
    'L2-4 CUX2 RORB CLMN', 'L3-5 RORB GABRG1 KCNH7', 'L3-5 RORB MKX DCC', 'L3-5 RORB MKX GALR1',
    'L3-5 RORB MKX GRIN3A', 'L3-5 RORB PCBP3 IL1RAPL2', 'L3-5 RORB PCBP3 LINGO2', 'L3-5 RORB TNNT2 TSHZ2',
    'L5-6 FEZF2 NXPH2 CDH8', 'L6 FEZF2 SYT6 CDH9', 'L6 OPRK1 THEMIS RGS6'
]

non_subtypes = [
    'Astro AQP4 SLC1A2', 'Astro GFAP FABP7', 'Micro P2RY12 APBB1IP', 'OPC PDGFRA PCDH15', 'Oligo MOG OPALIN', 'PC P2RY14 GRM8'
]

# 过滤并严格对齐顺序
gaba_filtered = [c for c in gaba_subtypes if c in adata.obs['cell_subtype'].cat.categories]
glut_filtered = [c for c in glut_subtypes if c in adata.obs['cell_subtype'].cat.categories]
non_filtered  = [c for c in non_subtypes  if c in adata.obs['cell_subtype'].cat.categories]

cell_order = gaba_filtered + glut_filtered + non_filtered
print(f"最终有效 ATAC 亚型数: {len(cell_order)} (GABA: {len(gaba_filtered)}, Glut: {len(glut_filtered)}, Non: {len(non_filtered)})")

gaba_end_idx = len(gaba_filtered) - 1
glut_end_idx = len(gaba_filtered) + len(glut_filtered) - 1

# --- 2. 差异基因计算 ---
if not os.path.exists(CACHE_MARKERS):
    print("未发现差异基因存档，正在计算 Wilcoxon 检验...")
    sc.tl.rank_genes_groups(adata, groupby='cell_subtype', method='wilcoxon', pts=True, n_genes=100)
    with open(CACHE_MARKERS, 'wb') as f:
        pickle.dump(adata.uns['rank_genes_groups'], f)
else:
    print("✅ 发现差异基因存档，已直接加载。")
    with open(CACHE_MARKERS, 'rb') as f:
        adata.uns['rank_genes_groups'] = pickle.load(f)

def unique_g(l):
    seen = set()
    return [x for x in l if not (x in seen or seen.add(x)) and x in adata.var_names]

# --- 预计算 GABA / Glut 大类全局表达比例（用于跨大类排他性过滤）---
gaba_set = set(gaba_filtered)
glut_set = set(glut_filtered)

print("预计算大类表达比例（用于跨大类排他性过滤）...")
gaba_cells_mask = adata.obs['cell_subtype'].isin(gaba_set)
glut_cells_mask = adata.obs['cell_subtype'].isin(glut_set)

gaba_data = adata[gaba_cells_mask].X
glut_data = adata[glut_cells_mask].X
if sp.issparse(gaba_data): gaba_data = gaba_data.toarray()
if sp.issparse(glut_data): glut_data = glut_data.toarray()

gaba_frac_global = pd.Series(np.mean(gaba_data > 0, axis=0), index=adata.var_names)
glut_frac_global = pd.Series(np.mean(glut_data > 0, axis=0), index=adata.var_names)
del gaba_data, glut_data
print("✅ 大类表达比例计算完成。")

def get_specific_markers(group_name, n=2, max_rest_pct=0.60, max_other_class_pct=0.25):
    # 非神经元只用经典 marker，不提取差异基因
    if group_name not in gaba_set and group_name not in glut_set:
        return []

    res = sc.get.rank_genes_groups_df(adata, group=group_name)
    res = res[~res['names'].str.startswith(('MT-', 'MTRN'))]
    res = res[res['pct_nz_reference'] < max_rest_pct]

    candidate_genes = res['names'].tolist()

    # 过滤掉在对侧大类中也高表达的基因（排除如 RBFOX3 这类泛神经元基因）
    if group_name in gaba_set:
        other_frac = glut_frac_global
    else:
        other_frac = gaba_frac_global

    specific = [g for g in candidate_genes if g in other_frac.index and other_frac[g] < max_other_class_pct]
    return specific[:n]

# 提取各亚型特异 marker
extracted_gaba, extracted_glut = [], []
print("各亚型提取到的特异 marker：")
for i, group in enumerate(cell_order):
    top_genes = get_specific_markers(group, n=2)
    print(f"  {group}: {top_genes}")
    if i <= gaba_end_idx:
        extracted_gaba.extend(top_genes)
    elif gaba_end_idx < i <= glut_end_idx:
        extracted_glut.extend(top_genes)
    # 非神经元不提取，跳过

classic_gaba = ['GAD1', 'GAD2', 'SST', 'VIP', 'PVALB', 'LAMP5']
classic_glut = ['SLC17A7', 'CUX2', 'RORB']
classic_non  = ['AQP4', 'GFAP', 'P2RY12', 'PDGFRA', 'MOG']

gaba_f = unique_g(classic_gaba + extracted_gaba)
glut_f = unique_g(classic_glut + extracted_glut)
non_f  = unique_g(classic_non)  # 非神经元只用经典 marker

marker_genes_final = unique_g(gaba_f + glut_f + non_f)
print(f"\n最终 marker 基因数: {len(marker_genes_final)}")

# --- 3. 表达矩阵计算 ---
if 'log1p' not in adata.uns:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

if not os.path.exists(CACHE_MATRIX):
    print(f"正在计算表达矩阵 ({len(marker_genes_final)} 个基因)...")
    mean_expr = pd.DataFrame(index=cell_order, columns=marker_genes_final, dtype=float)
    frac_expr = pd.DataFrame(index=cell_order, columns=marker_genes_final, dtype=float)

    for ct in cell_order:
        mask = (adata.obs['cell_subtype'] == ct)
        if mask.sum() == 0: continue
        data = adata[mask, marker_genes_final].X
        if sp.issparse(data): data = data.toarray()
        mean_expr.loc[ct] = np.mean(data, axis=0)
        frac_expr.loc[ct] = np.mean(data > 0, axis=0)

    with open(CACHE_MATRIX, 'wb') as f:
        pickle.dump({'mean': mean_expr, 'frac': frac_expr}, f)
else:
    print("✅ 发现矩阵缓存，已直接加载。")
    with open(CACHE_MATRIX, 'rb') as f:
        cache = pickle.load(f)
        mean_expr, frac_expr = cache['mean'], cache['frac']

mean_scaled = mean_expr.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9), axis=0).fillna(0.0).astype(float)
frac_expr   = frac_expr.fillna(0.0).astype(float)

condition_colors = {'c': '#74C69D', 's': '#E76F51'}
sex_colors       = {'male': '#4575b4', 'female': '#d73027'}

# --- 4. 画布构建 ---
print("开始构建画布...")
fig = plt.figure(figsize=(26, 40))

gs = gridspec.GridSpec(1, 1, figure=fig, left=0.15, right=0.82, bottom=0.1, top=0.9)
inner_gs = gs[0].subgridspec(10, 1, height_ratios=[0.6, 0.1, 28, 0.4, 0.4, 0.1, 0.1, 3.5, 0.2, 1], hspace=0.04)

ax_tag     = fig.add_subplot(inner_gs[0])
ax_dot     = fig.add_subplot(inner_gs[2])
ax_cond    = fig.add_subplot(inner_gs[3])
ax_sex_bar = fig.add_subplot(inner_gs[4])
ax_xtick   = fig.add_subplot(inner_gs[5])

# A. 顶部分类条
ax_tag.set_xlim(-0.5, len(cell_order)-0.5)
ax_tag.axis('off')
ax_tag.set_title("BA11 scATAC CELL TYPE MARKERS (MERGED)", fontsize=26, fontweight='bold', pad=25)

tags = [
    ('GABAergic',    -0.4,                gaba_end_idx + 0.4,  '#3A7CA5'),
    ('Glutamatergic', gaba_end_idx + 0.6, glut_end_idx + 0.4,  '#5FA052'),
    ('Non-Neuronal',  glut_end_idx + 0.6, len(cell_order)-0.6, '#7F7F7F')
]
for name, s, e, col in tags:
    ax_tag.fill_between([s, e], 0.3, 0.8, color=col, alpha=0.8)
    ax_tag.text((s+e)/2, 0.55, name, color='white', ha='center', va='center', fontweight='bold', fontsize=15)

# B. Dotplot 主矩阵
ax_dot.set_xlim(-0.5, len(cell_order)-0.5)
ax_dot.set_ylim(-0.5, len(marker_genes_final)-0.5)
ax_dot.invert_yaxis()

for i, ct in enumerate(cell_order):
    for j, gene in enumerate(marker_genes_final):
        s_val = float(pd.Series(frac_expr.loc[ct, gene]).iloc[0])
        if s_val < 0.02: continue
        c_val = float(pd.Series(mean_scaled.loc[ct, gene]).iloc[0])
        ax_dot.scatter(i, j, s=s_val*380, c=[c_val], cmap='RdBu_r', vmin=0, vmax=1, zorder=3, edgecolors='none')

ax_dot.set_yticks(range(len(marker_genes_final)))
labels = ax_dot.set_yticklabels(marker_genes_final, fontsize=10)
for l in labels:
    gn = l.get_text()
    if gn in classic_gaba: l.set_color('#3A7CA5'); l.set_weight('bold')
    elif gn in classic_glut: l.set_color('#5FA052'); l.set_weight('bold')
    elif gn in classic_non: l.set_color('#7F7F7F'); l.set_weight('bold')
ax_dot.set_xticks([])

# C. Condition 和 Sex 比例条
def draw_bar(ax, col_name, col_map, label):
    p = pd.crosstab(adata.obs['cell_subtype'], adata.obs[col_name], normalize='index').reindex(cell_order).fillna(0.0)
    ax.set_xlim(-0.5, len(cell_order)-0.5)
    bottom = np.zeros(len(cell_order))
    for cat, col in col_map.items():
        matched_col = [c for c in p.columns if str(c).lower() == str(cat).lower()]
        if matched_col:
            c_name = matched_col[0]
            ax.bar(range(len(cell_order)), p[c_name], bottom=bottom, color=col, width=1.0, edgecolor='white', linewidth=0.1)
            bottom += p[c_name]
    ax.set_ylabel(label, rotation=0, ha='right', va='center', fontweight='bold', labelpad=25, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    [s.set_visible(False) for s in ax.spines.values()]

draw_bar(ax_cond,    'condition', condition_colors, "Condition")
draw_bar(ax_sex_bar, 'sex',       sex_colors,       "Sex")

# D. X 轴亚型标签
ax_xtick.set_xlim(-0.5, len(cell_order)-0.5)
ax_xtick.axis('off')
for i, name in enumerate(cell_order):
    ax_xtick.text(i, 0.9, name, rotation=90, ha='right', va='top', fontsize=10, fontfamily='monospace')

# --- 5. 图例 ---
legend_x = 0.85

# 颜色条
cax = fig.add_axes([legend_x, 0.65, 0.018, 0.08])
sm  = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(0, 1))
plt.colorbar(sm, cax=cax, label='Scaled Mean Expression')

# Fraction of Cells 图例（点和数字间距修复）
dot_ax = fig.add_axes([legend_x, 0.52, 0.07, 0.08])
dot_ax.set_xlim(0, 1)
dot_ax.set_ylim(0, 1)
dot_ax.axis('off')
dot_ax.text(0, 1.1, 'Fraction of Cells', fontweight='bold', fontsize=11)
for i, f in enumerate([0.2, 0.5, 0.8]):
    dot_ax.scatter(0.08, 0.8 - i*0.4, s=f*380, color='gray', zorder=3)
    dot_ax.text(0.22, 0.8 - i*0.4, f'{int(f*100)}%', va='center', fontsize=10)

# Condition / Sex 图例
def add_cat_leg(y_pos, title, cmap):
    lax = fig.add_axes([legend_x, y_pos, 0.05, 0.06])
    lax.axis('off')
    lax.text(0, 1.1, title, fontweight='bold', fontsize=11)
    for i, (k, v) in enumerate(cmap.items()):
        lax.add_patch(plt.Rectangle((0, 0.8-i*0.4), 0.2, 0.2, color=v))
        lax.text(0.3, 0.9-i*0.4, k.upper(), va='center', fontsize=10)

add_cat_leg(0.38, "Condition", condition_colors)
add_cat_leg(0.26, "Sex",       sex_colors)

# --- 6. 保存 ---
if os.path.exists(CACHE_MATRIX):
    os.remove(CACHE_MATRIX)

plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✨ 完成！输出至: {OUTPUT_PNG}")