import scanpy as sc
import muon as mu
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import anndata as ad
import matplotlib.pyplot as plt

# ================= 配置区 =================
INPUT_H5AD = "/data/zihui/BA11/scATAC/BA11_ATAC.h5ad"
OUTPUT_DIR = "/data/zihui/BA11/scATAC/QC_Plots_PNG/"
# 如果目录不存在则创建
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# ==========================================

def main():
    print("--- 启动质控可视化程序 (PNG & 教程命名版) ---")
    
    # 1. 加载数据
    adata = sc.read_h5ad(INPUT_H5AD)
    print(f"数据读取成功，当前细胞数: {adata.n_obs}, 特征数: {adata.n_vars}")

    # 2. 计算指标并重命名 (完全遵循你提供的教程代码)
    print("正在计算质控指标...")
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    adata.obs.rename(
        columns={
            "n_genes_by_counts": "n_features_per_cell",
            "total_counts": "total_fragment_counts",
        },
        inplace=True,
    )
    # 增加 log10 转换列 (加1防止 log10(0) 报错)
    adata.obs["log_total_fragment_counts"] = np.log10(adata.obs["total_fragment_counts"] + 1)

    # 3. 分样本 Doublet 检测 (工程优化：预防内存溢出)
    print("正在分样本进行 Doublet 检测 (Scrublet)...")
    sample_list = []
    unique_samples = adata.obs['sample_id'].unique()
    
    for i, sn in enumerate(unique_samples):
        print(f"[{i+1}/{len(unique_samples)}] 正在处理样本: {sn}")
        sub = adata[adata.obs['sample_id'] == sn].copy()
        
        # Scrublet 需要样本有一定的细胞量级
        if sub.n_obs > 100:
            try:
                mu.atac.tl.scrublet(sub)
            except Exception as e:
                print(f"⚠️ 样本 {sn} 检测异常: {e}")
                sub.obs['predicted_doublet'] = False
                sub.obs['doublet_score'] = 0.0
        else:
            sub.obs['predicted_doublet'] = False
            sub.obs['doublet_score'] = 0.0
        sample_list.append(sub)
    
    # 合并包含 Doublet 信息的 adata
    adata = ad.concat(sample_list)

   # 图 1：n_features_per_cell 的直方图分布 (去除前后 2.5%)
    print("正在生成截断分布直方图...")
    
    # 计算分位数阈值
    lower_bound = np.quantile(adata.obs['n_features_per_cell'], 0.025)
    upper_bound = np.quantile(adata.obs['n_features_per_cell'], 0.975)
    
    # 提取过滤后的数据用于绘图 (不修改原始 adata)
    plot_data = adata.obs['n_features_per_cell'][
        (adata.obs['n_features_per_cell'] >= lower_bound) & 
        (adata.obs['n_features_per_cell'] <= upper_bound)
    ]

    plt.figure(figsize=(10, 6))
    plt.hist(plot_data, bins=80, color='skyblue', edgecolor='black', alpha=0.8)
    
    # 标注参考线
    plt.axvline(lower_bound, color='red', linestyle='--', label=f'Lower 2.5%: {int(lower_bound)}')
    plt.axvline(upper_bound, color='red', linestyle='--', label=f'Upper 2.5%: {int(upper_bound)}')
    
    plt.title(f'Feature Distribution (Middle 95%, n={len(plot_data)})')
    plt.xlabel('n_features_per_cell (Peaks)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(f"{OUTPUT_DIR}Histogram_Features_Filtered_95.png", dpi=300)
    plt.close()
    print(f"直方图已保存。保留区间: [{int(lower_bound)}, {int(upper_bound)}]")

    # 图 2：按 Condition (MDD/Control) 分组的小提琴图
    # 检查两组之间是否存在明显的质量偏移
    sc.pl.violin(adata, ['n_features_per_cell', 'log_total_fragment_counts'], 
                 groupby='condition', multi_panel=True, jitter=False, show=False)
    plt.savefig(f"{OUTPUT_DIR}Violin_QC_by_Condition.png", dpi=300)
    plt.close()

    # 图 3：按 Sample (sample_id) 分组的特征数小提琴图
    # 这是最关键的图，由于样本多，画布设宽一些
    fig, ax = plt.subplots(figsize=(18, 7))
    sc.pl.violin(adata, 'n_features_per_cell', groupby='batch', 
                 rotation=90, jitter=False, ax=ax, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}Violin_Features_by_Sample.png", dpi=300)
    plt.close()

    # 4. 保存包含 Doublet 结果的 h5ad 文件
    # 这样下次你可以直接读入并按照确定的百分比进行过滤
    print("正在保存 BA11_MDD_Control_with_DoubletScores.h5ad ...")
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    
    adata.write("/data/zihui/BA11/scATAC/BA11_MDD_Control_with_DoubletScores.h5ad")
    print(f"🎉 任务完成！PNG 质控图已生成，数据已保存。")

if __name__ == "__main__":
    # 强制使用无界面后端
    import matplotlib
    matplotlib.use('Agg')
    # 设置 scanpy 绘图参数
    sc.settings.set_figure_params(dpi=150, facecolor='white')
    main()
