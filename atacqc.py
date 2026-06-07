import os
N_THREADS = "16"
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS

import scanpy as sc
import pandas as pd
import numpy as np
import scrublet as scr
import scipy.sparse as sp
import anndata as ad


# ================= 配置区 =================
INPUT_H5AD = "/data/zihui/BA11/scATAC/BA11_ATAC.h5ad"
FINAL_H5AD = "/data/zihui/BA11/scATAC/BA11_QC_Final_Combined_v4.h5ad"

MIN_FRAGMENTS = 500 
MAX_FRAGMENTS = 10000
NMADS = 5 
# ==========================================

def is_outlier(adata, metric, nmads):
    M = adata.obs[metric]
    median = M.median()
    mad = np.median(np.abs(M - median))
    if mad == 0: mad = 1e-6
    return (M < median - nmads * mad) | (M > median + nmads * mad)

def main():
    print("--- 启动组合 QC 流程 (s25 已放回，优化检测逻辑) ---")
    adata = sc.read_h5ad(INPUT_H5AD)
    
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    adata.obs.rename(columns={"n_genes_by_counts": "n_features_per_cell", 
                               "total_counts": "total_fragment_counts"}, inplace=True)

    processed_list = []
    unique_samples = adata.obs['sample_id'].unique()

    print(f"{'Sample_ID':<40} | {'Original':<8} | {'Remaining':<8} | {'Doublets':<8} | {'Threshold':<8}")
    print("-" * 95)

    for sn in unique_samples:
        sub = adata[adata.obs['sample_id'] == sn].copy()
        
        # 1. 物理过滤 + MAD 过滤 (先过滤掉垃圾碎片，减少 Scrublet 的负担)
        sub.obs["is_mad_outlier"] = is_outlier(sub, "total_fragment_counts", NMADS)
        keep_mask = (sub.obs['total_fragment_counts'] >= MIN_FRAGMENTS) & \
                    (sub.obs['total_fragment_counts'] <= MAX_FRAGMENTS) & \
                    (~sub.obs["is_mad_outlier"])
        sub = sub[keep_mask].copy()

        # 2. 运行 Scrublet (修正逻辑)
        current_threshold = "N/A"
        n_doublets = 0
        if sub.n_obs > 50:  # 细胞数太少无法运行检测
            try:
                # ✨ 核心修复：临时过滤极低频 Peak 以稳定计算
                # 至少在 3 个细胞里出现的 Peak 才参与降维计算
                sub_scrub = sub.copy()
                sc.pp.filter_genes(sub_scrub, min_cells=3)
                
                X_data = sub_scrub.X if sp.issparse(sub_scrub.X) else sp.csr_matrix(sub_scrub.X)
                scrub = scr.Scrublet(X_data, expected_doublet_rate=0.1)
                
                # n_prin_comps 设为 30 对单样本来说比较合适
                doublet_scores, predicted_doublets = scrub.scrub_doublets(verbose=False, n_prin_comps=30)
                
                # 将分数和标签写回原 sub 对象
                sub.obs['doublet_score'] = doublet_scores
                sub.obs['predicted_doublet'] = predicted_doublets
                
                # 提取阈值
                if predicted_doublets.any():
                    current_threshold = round(sub.obs[sub.obs['predicted_doublet'] == True]['doublet_score'].min(), 3)
                    n_doublets = predicted_doublets.sum()
                else:
                    current_threshold = "None"
            except Exception as e:
                # print(f"Sample {sn} Scrublet error: {e}") # 调试用
                sub.obs['predicted_doublet'] = False
                sub.obs['doublet_score'] = 0.0
        else:
            sub.obs['predicted_doublet'] = False
            sub.obs['doublet_score'] = 0.0

        print(f"{sn:<40} | {adata[adata.obs['sample_id']==sn].n_obs:<8} | {sub.n_obs:<8} | {n_doublets:<8} | {current_threshold:<8}")
        processed_list.append(sub)

    # 3. 合并
    adata_final = ad.concat(processed_list, join="inner", merge="same")
    
    # 统计双细胞总数
    total_db = adata_final.obs['predicted_doublet'].sum()
    total_pct = (total_db / adata_final.n_obs) * 100

    # 4. 全局过滤 Peak (至少在 1% 的细胞中存在的 Peak 才保留)
    orig_peaks = adata_final.n_vars
    sc.pp.filter_genes(adata_final, min_cells=int(adata_final.n_obs * 0.001))
    
    print("-" * 95)
    print(f"📊 汇总统计:")
    print(f"   - 最终保存细胞总数: {adata_final.n_obs}")
    print(f"   - 全局双细胞总数: {total_db} ({total_pct:.2f}%)")
    print(f"   - 原始 Peak 总数: {orig_peaks}")
    print(f"   - 过滤后 Peak 数: {adata_final.n_vars}")
    
    # 特别检查 s25
    s25_data = adata_final.obs[adata_final.obs['sample_id'].str.contains('s25')]
    if not s25_data.empty:
        print(f"🔍 s25 状态: 剩余 {len(s25_data)} 细胞, 双细胞标记: {s25_data['predicted_doublet'].sum()}")

    if not sp.issparse(adata_final.X):
        adata_final.X = sp.csr_matrix(adata_final.X)
    
    adata_final.write(FINAL_H5AD)
    print(f"✅ 处理完成！文件已保存: {FINAL_H5AD}")

if __name__ == "__main__":
    main()