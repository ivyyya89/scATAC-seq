# -*- coding: utf-8 -*-
# !! 环境变量必须在所有 import 之前设置 !!
import os
os.environ["OPENBLAS_NUM_THREADS"] = "40"
os.environ["OMP_NUM_THREADS"] = "40"       # 有些库走OMP而不是OpenBLAS
os.environ["MKL_NUM_THREADS"] = "40"       # 如果用的是MKL后端

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import harmonypy as hp

INPUT_H5AD  = "/data1/zihui/BA11/scATAC/BA11_LSI_Only.h5ad"
OUTPUT_H5AD = "/data1/zihui/BA11/scATAC/BA11_Integrated_Final.h5ad"
FIGURE_DIR  = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

def main():
    print("--- Starting BA11 Integration Pipeline ---")
    adata = sc.read_h5ad(INPUT_H5AD)

    # Step 2: 剔除双细胞
    if 'doublet_scores' in adata.obs:
        n_before = adata.n_obs
        adata = adata[adata.obs['doublet_scores'] <= 0.48].copy()
        print(f"Filtered doublets. Removed: {n_before - adata.n_obs}")

    # Step 3: TF-IDF
    print("Step 3: TF-IDF Normalization...")
    n_cells  = adata.X.shape[0]
    tf       = adata.X.multiply(1.0 / adata.X.sum(axis=1))
    col_sum  = np.array(adata.X.sum(axis=0)).flatten()
    idf      = np.log1p(n_cells / (1 + col_sum))
    adata.X  = tf.multiply(idf)

    # Step 4: LSI (PCA)
    print("Step 4: Running LSI (PCA)...")
    sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
    adata.obsm['X_lsi'] = adata.obsm['X_pca'].copy()

    # Step 5: Harmony（跳过第1个LSI组分，去除测序深度影响）
    print("Step 5: Running Harmony...")
    data_mat  = adata.obsm['X_lsi'][:, 1:].copy()   # (187633, 49)
    meta_data = adata.obs[['batch']]

    ho = hp.run_harmony(data_mat, meta_data, 'batch',
                        max_iter_harmony=20,
                        theta=2.5,
                        sigma=0.1,
                        verbose=True)

    Z_corr = ho.Z_corr
    if hasattr(Z_corr, 'numpy'):
        Z_corr = Z_corr.numpy()
    elif hasattr(Z_corr, 'toarray'):
        Z_corr = Z_corr.toarray()
    Z_corr = np.array(Z_corr)
    
    print(f"Z_corr shape: {Z_corr.shape}")  # (187633, 49)
    
    # 判断是否需要转置
    if Z_corr.shape[0] == adata.n_obs:
        harmony_result = Z_corr          # 已经是 (cells, PCs)，不用转置
    elif Z_corr.shape[1] == adata.n_obs:
        harmony_result = Z_corr.T        # 需要转置
    else:
        raise ValueError(f"Z_corr shape {Z_corr.shape} doesn't match n_obs={adata.n_obs}")
    
    adata.obsm['X_harmony'] = harmony_result
    print(f"Harmony done. Result shape: {adata.obsm['X_harmony'].shape}")  # (187633, 49)
    
    # Step 6: Neighbors + UMAP
    print("Step 6: Building Neighbors and UMAP...")
    sc.pp.neighbors(adata, use_rep="X_harmony", n_neighbors=20)
    sc.tl.umap(adata, min_dist=0.3)

    # Step 7: 绘图
    print("Step 7: Plotting...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sc.pl.umap(adata, color='batch', ax=axes[0], show=False, title="Post-Harmony (Samples)")
    sc.pl.umap(adata, color='sex',   ax=axes[1], show=False, title="Post-Harmony (Sex)")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/ATAC_afterHarmony.png", dpi=300)
    print(f"Figure saved.")

    # Step 8: 保存
    print(f"Step 8: Saving to {OUTPUT_H5AD}...")
    if not isinstance(adata.X, sp.csr_matrix):
        adata.X = adata.X.tocsr()
    adata.write(OUTPUT_H5AD, compression='gzip')
    print("All Finished!")

if __name__ == "__main__":
    main()