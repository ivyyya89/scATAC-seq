# -*- coding: utf-8 -*-
import os
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["OMP_NUM_THREADS"] = "64"

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

ATAC_H5AD    = "/data1/zihui/BA11/scATAC/BA11_QC_Final_With_Scrublet.h5ad"
REFFLAT      = "/data1/zihui/ref/refFlat.txt"
OUTPUT_H5AD  = "/data1/zihui/BA11/scATAC/BA11_GeneActivity.h5ad"
PROMOTER_UP  = 2000  # 启动子上游延伸

def parse_refflat(refflat_path):
    """读取refFlat，每个基因只保留最大范围的转录本"""
    cols = ['geneName','transcriptName','chrom','strand',
            'txStart','txEnd','cdsStart','cdsEnd',
            'exonCount','exonStarts','exonEnds']
    df = pd.read_csv(refflat_path, sep='\t', header=None, names=cols)

    # 只保留标准染色体
    standard = [f'chr{i}' for i in list(range(1,23)) + ['X','Y']]
    df = df[df['chrom'].isin(standard)]

    # 每个基因取最大范围（start最小，end最大）
    gene_df = df.groupby('geneName').agg(
        chrom=('chrom', 'first'),
        start=('txStart', 'min'),
        end=('txEnd', 'max')
    ).reset_index()

    return gene_df

def compute_gene_activity(adata_atac, gene_df, promoter_up=2000):
    """将peak counts累加到基因上"""
    # 解析peak坐标
    peak_names = adata_atac.var_names
    standard = [f'chr{i}' for i in list(range(1,23)) + ['X','Y']]
    
    valid_mask = peak_names.str.split(':').str[0].isin(standard)
    peak_names = peak_names[valid_mask]
    
    peak_chrom = peak_names.str.split(':').str[0].values
    peak_coords = peak_names.str.split(':').str[1]
    peak_start = peak_coords.str.split('-').str[0].astype(int).values
    peak_end   = peak_coords.str.split('-').str[1].astype(int).values

    # 只用标准染色体的peak
    X = adata_atac.X[:, valid_mask]
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsc()  # 按列切片更快

    n_cells = adata_atac.n_obs
    n_genes = len(gene_df)
    
    rows, cols_idx, data = [], [], []

    print(f"Processing {n_genes} genes × {len(peak_names)} peaks...")
    
    for g_idx, row in enumerate(gene_df.itertuples()):
        if g_idx % 2000 == 0:
            print(f"  Gene {g_idx}/{n_genes}...")
        
        g_start = max(0, row.start - promoter_up)
        g_end   = row.end
        g_chrom = row.chrom

        # 找overlap的peaks
        overlap = np.where(
            (peak_chrom == g_chrom) &
            (peak_start < g_end) &
            (peak_end   > g_start)
        )[0]

        if len(overlap) == 0:
            continue

        # 累加该基因所有overlapping peaks的counts
        gene_counts = np.array(X[:, overlap].sum(axis=1)).flatten()
        
        nz = np.where(gene_counts > 0)[0]
        for cell_idx in nz:
            rows.append(cell_idx)
            cols_idx.append(g_idx)
            data.append(gene_counts[cell_idx])

    gene_activity = sp.csr_matrix(
        (data, (rows, cols_idx)),
        shape=(n_cells, n_genes)
    )
    return gene_activity

def main():
    print("=== Step 1: Load ATAC data ===")
    adata = sc.read_h5ad(ATAC_H5AD)
    
    # 剔除双细胞（和之前一致）
    if 'doublet_scores' in adata.obs:
        n_before = adata.n_obs
        adata = adata[adata.obs['doublet_scores'] <= 0.48].copy()
        print(f"Filtered doublets: {n_before} → {adata.n_obs} cells")

    print("=== Step 2: Parse refFlat ===")
    gene_df = parse_refflat(REFFLAT)
    print(f"Genes in refFlat: {len(gene_df)}")

    print("=== Step 3: Compute Gene Activity Matrix ===")
    gene_activity = compute_gene_activity(adata, gene_df, PROMOTER_UP)
    print(f"Gene Activity shape: {gene_activity.shape}")

    print("=== Step 4: Build AnnData & Save ===")
    adata_ga = sc.AnnData(
        X   = gene_activity,
        obs = adata.obs.copy(),
        var = pd.DataFrame(index=gene_df['geneName'].values)
    )
    # 归一化（给scANVI用）
    adata_ga.layers['counts'] = adata_ga.X.copy()# 备份原始counts到layers
    sc.pp.normalize_total(adata_ga, target_sum=1e4)
    sc.pp.log1p(adata_ga)

    adata_ga.write(OUTPUT_H5AD, compression='gzip')
    print(f"Saved: {OUTPUT_H5AD}")
    print(f"Final shape: {adata_ga.shape}")

if __name__ == "__main__":
    main()
