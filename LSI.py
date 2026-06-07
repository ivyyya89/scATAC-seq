# -*- coding: utf-8 -*-
import os
os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

# ================= Configuration =================
INPUT_H5AD = "/data1/zihui/BA11/scATAC/BA11_QC_Final_With_Scrublet.h5ad"
OUTPUT_H5AD = "/data1/zihui/BA11/scATAC/BA11_LSI_Only.h5ad"
FIGURE_DIR = "figures"

# 自动创建文件夹
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)
    print(f"Created directory: {FIGURE_DIR}")
# =================================================

def main():
    print("--- Starting BA11 LSI & UMAP Pipeline (with Quality Checks) ---")
    adata = sc.read_h5ad(INPUT_H5AD)

    # 0. Filter Doublets (维持之前的 0.48 物理剔除逻辑，但我们先画图看分布)
    # 注意：为了在 UMAP 上看到双细胞分数的分布，我们先不急着剔除，
    # 或者如果你想看“剔除后”剩下的得分分布，就按下面的逻辑跑。
    
    # 1. TF-IDF Normalization
    print("Step 1: TF-IDF Normalization...")
    n_cells = adata.X.shape[0]
    tf = adata.X.multiply(1.0 / adata.X.sum(axis=1))
    col_sum = np.array(adata.X.sum(axis=0)).flatten()
    idf = np.log1p(n_cells / (1 + col_sum))
    adata.X = tf.multiply(idf)
    
    # 2. LSI Dimension Reduction
    print("Step 2: Running LSI...")
    sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
    adata.obsm['X_lsi'] = adata.obsm['X_pca'].copy()
    
    # 3. Neighbors & UMAP
    print("Step 3: Building Neighbors and UMAP (LSI 2-50)...")
    sc.pp.neighbors(adata, use_rep="X_lsi", n_neighbors=20, n_pcs=49)
    sc.tl.umap(adata, min_dist=0.3)

    # 4. Diagnostic Plots (4 Subplots)
    print("Step 4: Generating Diagnostic Plots in 'figures/'...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图 1: 样本分布
    sc.pl.umap(adata, color='batch', ax=axes[0, 0], show=False, title="Samples (Batch Check)")
    
    # 图 2: 性别分布
    sc.pl.umap(adata, color='sex', ax=axes[0, 1], show=False, title="Sex (Male vs Female)")
    
    # 图 3: 库大小 (总 Fragment 数)
    sc.pl.umap(adata, color='total_fragment_counts', ax=axes[1, 0], show=False, title="Library Size (Total Fragments)")
    
    # 图 4: 双细胞得分 (Doublet Scores)
    sc.pl.umap(adata, color='doublet_scores', ax=axes[1, 1], show=False, title="Doublet Scores (Scrublet)")
    
    plt.tight_layout()
    plot_path = os.path.join(FIGURE_DIR, "BA11_LSI_Comprehensive_Check.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved diagnostic UMAPs to {plot_path}")
    
    # 5. 物理剔除双细胞并保存 (如果你想保存干净的数据)
    #if 'predicted_doublet' in adata.obs:
        #n_before = adata.n_obs
        #adata = adata[adata.obs['doublet_scores'] <= 0.48].copy()
        #print(f"Post-processing: Filtered {n_before - adata.n_obs} doublets.")

    # 6. Final Save
    print(f"Step 5: Saving processed data to {OUTPUT_H5AD}...")
    if sp.issparse(adata.X) and not isinstance(adata.X, sp.csr_matrix):
        adata.X = adata.X.tocsr()
    adata.write(OUTPUT_H5AD, compression='gzip')
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()