# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')  # 后台静默画图

import os

# CPU 线程限制
os.environ["OPENBLAS_NUM_THREADS"] = "30"
os.environ["OMP_NUM_THREADS"] = "30"
os.environ["MKL_NUM_THREADS"] = "30"

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvi
import anndata
import matplotlib.pyplot as plt

# =========================
# 路径配置
# =========================

RNA_H5AD         = "/data1/zihui/sex-scRNA/Annotated.h5ad"
ATAC_GA_H5AD     = "/data1/zihui/BA11/scATAC/BA11_GeneActivity.h5ad"

OUTPUT_H5AD      = "/data1/zihui/BA11/scATAC/BA11_ATAC_Annotated_DualSCANVI.h5ad"
OUTPUT_HARMONY   = "/data1/zihui/BA11/scATAC/BA11_Integrated_Final.h5ad"
HARMONY_INPUT = "/data1/zihui/BA11/scATAC/BA11_Integrated_Final.h5ad"

MODEL_TYPE_DIR   = "/data1/zihui/BA11/scATAC/scanvi_model_type/"
MODEL_SUB_DIR    = "/data1/zihui/BA11/scATAC/scanvi_model_subtype/"

PLOT_DIR         = "/data1/zihui/BA11/scATAC/plots/"

os.makedirs(PLOT_DIR, exist_ok=True)


def main():

    # ==========================================================
    # Step 1. 读取数据
    # ==========================================================

    print("=== Step 1: Load data and filter region ===")

    rna = sc.read_h5ad(RNA_H5AD)
    atac = sc.read_h5ad(ATAC_GA_H5AD)

    # 严格限制 BA11
    rna = rna[rna.obs['region'] == 'BA11'].copy()

    print(f"scRNA (BA11 only) shape: {rna.shape}")
    print(f"scATAC Gene Activity shape: {atac.shape}")

    # 使用 raw counts
    rna.X = rna.layers['counts'].copy()
    atac.X = atac.layers['counts'].copy()

    # ==========================================================
    # Step 2. HVGs
    # ==========================================================

    print("=== Step 2: Find common HVGs ===")

    hvg = rna.var_names[rna.var['highly_variable']]
    use_genes = hvg.intersection(atac.var_names)

    print(f"HVGs ∩ ATAC genes (final selected): {len(use_genes)}")

    rna = rna[:, use_genes].copy()
    atac = atac[:, use_genes].copy()

    # sparse
    if not sp.issparse(rna.X):
        rna.X = sp.csr_matrix(rna.X)

    if not sp.issparse(atac.X):
        atac.X = sp.csr_matrix(atac.X)

    # 转 float32
    rna.X.data = np.round(rna.X.data).astype(np.float32)
    atac.X.data = np.round(atac.X.data).astype(np.float32)

    # ==========================================================
    # Step 3. 大类标签 transfer
    # ==========================================================

    print("\n" + "=" * 60)
    print("=== Step 3: Train Model 1 (scANVI for Major Type) ===")
    print("=" * 60)

    rna_t = rna.copy()
    atac_t = atac.copy()

    rna_t.obs['modality'] = 'RNA'
    atac_t.obs['modality'] = 'ATAC'

    atac_t.obs['cell_type'] = 'Unknown'

    combined_type = anndata.concat(
        [rna_t, atac_t],
        label='dataset',
        keys=['RNA', 'ATAC'],
        join='outer',
        fill_value=0
    )

    combined_type.obs_names_make_unique()

    scvi.model.SCANVI.setup_anndata(
        combined_type,
        layer=None,
        batch_key='modality',
        labels_key='cell_type',
        unlabeled_category='Unknown'
    )

    # =========================
    # 自动读取已有模型
    # =========================

    if os.path.exists(MODEL_TYPE_DIR):

        print("Loading existing Type-scANVI model...")

        model_type = scvi.model.SCANVI.load(
            MODEL_TYPE_DIR,
            adata=combined_type
        )

    else:

        model_type = scvi.model.SCANVI(
            combined_type,
            n_layers=2,
            n_latent=30,
            gene_likelihood='nb'
        )

        print("Training Type-scANVI...")

        model_type.train(
            max_epochs=80,
            early_stopping=True,
            limit_train_batches=1.0,
            datasplitter_kwargs={"num_workers": 8}
        )

        model_type.save(MODEL_TYPE_DIR, overwrite=True)

    # prediction
    combined_type.obs['cell_type_pred'] = model_type.predict()

    combined_type.obs['cell_type_prob'] = (
        model_type.predict(soft=True).max(axis=1)
    )

    atac_type_res = combined_type[
        combined_type.obs['modality'] == 'ATAC'
    ].obs[
        ['cell_type_pred', 'cell_type_prob']
    ].copy()

    # ==========================================================
    # Step 4. 亚型 transfer
    # ==========================================================

    print("\n" + "=" * 60)
    print("=== Step 4: Train Model 2 (scANVI for Fine Subtype) ===")
    print("=" * 60)

    rna_s = rna.copy()
    atac_s = atac.copy()

    rna_s.obs['modality'] = 'RNA'
    atac_s.obs['modality'] = 'ATAC'

    atac_s.obs['cell_subtype'] = 'Unknown'

    combined_sub = anndata.concat(
        [rna_s, atac_s],
        label='dataset',
        keys=['RNA', 'ATAC'],
        join='outer',
        fill_value=0
    )

    combined_sub.obs_names_make_unique()

    scvi.model.SCANVI.setup_anndata(
        combined_sub,
        layer=None,
        batch_key='modality',
        labels_key='cell_subtype',
        unlabeled_category='Unknown'
    )

    # =========================
    # 自动读取已有 subtype 模型
    # =========================

    if os.path.exists(MODEL_SUB_DIR):

        print("Loading existing Subtype-scANVI model...")

        model_sub = scvi.model.SCANVI.load(
            MODEL_SUB_DIR,
            adata=combined_sub
        )

    else:

        model_sub = scvi.model.SCANVI(
            combined_sub,
            n_layers=2,
            n_latent=30,
            gene_likelihood='nb'
        )

        print("Training Subtype-scANVI with stabilized settings...")

        model_sub.train(
            max_epochs=120,
            early_stopping=True,
            early_stopping_patience=15,
            check_val_every_n_epoch=5,
            limit_train_batches=1.0,
            gradient_clip_val=1.0,
            plan_kwargs={"lr": 5e-4},
            datasplitter_kwargs={"num_workers": 8}
        )

        model_sub.save(MODEL_SUB_DIR, overwrite=True)

    # prediction
    combined_sub.obs['cell_subtype_pred'] = model_sub.predict()

    combined_sub.obs['cell_subtype_prob'] = (
        model_sub.predict(soft=True).max(axis=1)
    )

    atac_sub_mask = combined_sub.obs['modality'] == 'ATAC'

    atac_sub_res = combined_sub[
        atac_sub_mask
    ].obs[
        ['cell_subtype_pred', 'cell_subtype_prob']
    ].copy()

    atac_obs_names = combined_sub.obs_names[atac_sub_mask]

    # ==========================================================
    # Step 5. 合并结果
    # ==========================================================

    print("\n=== Step 5: Merging Two-tier Predictions ===")

    final_atac_obs = pd.DataFrame(index=atac_obs_names)

    final_atac_obs['cell_type_pred'] = (
        atac_type_res['cell_type_pred']
    )

    final_atac_obs['cell_type_prob'] = (
        atac_type_res['cell_type_prob']
    )

    final_atac_obs['cell_subtype_pred'] = (
        atac_sub_res['cell_subtype_pred']
    )

    final_atac_obs['cell_subtype_prob'] = (
        atac_sub_res['cell_subtype_prob']
    )

    # ==========================================================
    # Step 6. 同步到 Harmony 空间
    # ==========================================================

    print("\n=== Step 6: Sync Labels to Harmony Object ===")

    adata_harmony = sc.read_h5ad(HARMONY_INPUT)

    # 使用 map 不破坏原顺序
    adata_harmony.obs['cell_type'] = (
        adata_harmony.obs.index.map(
            final_atac_obs['cell_type_pred']
        )
    )

    adata_harmony.obs['cell_type_prob'] = (
        adata_harmony.obs.index.map(
            final_atac_obs['cell_type_prob']
        )
    )

    adata_harmony.obs['cell_subtype'] = (
        adata_harmony.obs.index.map(
            final_atac_obs['cell_subtype_pred']
        )
    )

    adata_harmony.obs['cell_subtype_prob'] = (
        adata_harmony.obs.index.map(
            final_atac_obs['cell_subtype_prob']
        )
    )

# ==========================================================
    # Step 7. 在 Harmony UMAP 上可视化 (大类与亚型四合一-图例右侧居中对齐版)
    # ==========================================================

    print("\n=== Step 7: Plotting Combined 2x2 UMAP Matrix ===")

    if 'X_umap' not in adata_harmony.obsm.keys():
        raise ValueError(
            "X_umap not found in Harmony object!"
        )

    # 设置全局画图参数
    sc.settings.set_figure_params(
        dpi=150,
        facecolor='white'
    )

    # 创建大画布，适当加宽（20）和加高（14），给双列图例留出右侧空间
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # ----------------------------------------------------------
    # 第一行：Major Type (大类分布 + 大类置信度)
    # ----------------------------------------------------------
    print("  [1/2] 正在绘制第一行：Major Cell Types...")
    
    sc.pl.umap(
        adata_harmony,
        color='cell_type',
        show=False,
        ax=axes[0, 0],
        title="Major Cell Types (Harmony Space)"
    )

    sc.pl.umap(
        adata_harmony,
        color='cell_type_prob',
        show=False,
        ax=axes[0, 1],
        title="Major Type Prediction Confidence",
        cmap='viridis'
    )

    # ----------------------------------------------------------
    # 第二行：Subtype (精细亚型分布 + 亚型置信度)
    # ----------------------------------------------------------
    print("  [2/2] 正在绘制第二行：Refined Cell Subtypes...")
    
    # 【核心修改 1】：必须要把 legend_loc 设为 None，切断 scanpy 默认的流氓排版
    sc.pl.umap(
        adata_harmony,
        color='cell_subtype',
        show=False,
        ax=axes[1, 0],
        title="Refined Cell Subtypes (Harmony Space)",
        legend_loc=None
    )

    sc.pl.umap(
        adata_harmony,
        color='cell_subtype_prob',
        show=False,
        ax=axes[1, 1],
        title="Subtype Prediction Confidence",
        cmap='plasma'
    )

    # ----------------------------------------------------------
    # 极致细节微调：彻底拯救左下角图例，将其强制居中在右侧外延
    # ----------------------------------------------------------
    try:
        # 提取亚型的标签和配色
        labels = sorted(adata_harmony.obs['cell_subtype'].unique())
        colors = adata_harmony.uns['cell_subtype_colors']
        
        # 手动创建圆点句柄
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=colors[i], markersize=4, label=labels[i])
            for i in range(len(labels))
        ]
        
        # 【核心修改 2】：同时配合这三个参数，才能把双列图例死死锁在右侧中心
        legend = axes[1, 0].legend(
            handles=legend_elements,
            loc='center left',            # 锁定图例框的左边中心点
            bbox_to_anchor=(1.02, 0.5),   # X=1.02（出框右侧），Y=0.5（垂直居中）
            ncol=2,                       # 46个亚型分成2列排，高度瞬间减半，绝对不会漏下去
            fontsize=4.2,                 # 微缩字体，保证排版精致
            frameon=False,                # 去除图例边框
            labelspacing=0.2,             # 纵向行间距
            columnspacing=0.5             # 两列之间的横向间距
        )
        
        # 强迫症福音：强行把 4 个子图全部锁定为 1:1 的完美正方形比例
        axes[0, 0].set_aspect('equal', adjustable='box')
        axes[0, 1].set_aspect('equal', adjustable='box')
        axes[1, 0].set_aspect('equal', adjustable='box')
        axes[1, 1].set_aspect('equal', adjustable='box')

    except Exception as e:
        print(f"  [!] 自动微调图例时遇到异常（可忽略）: {str(e)}")

    # 整体大标题
    plt.suptitle(
        "BA11 scATAC Label Transfer & Quality Evaluation Matrix",
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    # 留出右侧 15% 的空白区（0.85）来包容双列图例，调整子图间距防重叠
    plt.tight_layout(rect=[0, 0, 0.85, 0.95], h_pad=3.0, w_pad=4.0)

    # 保存最终合体大图
    output_filename = "BA11_ATAC_Annotation_All_In_One_Matrix.png"
    plt.savefig(
        os.path.join(PLOT_DIR, output_filename),
        bbox_inches='tight'
    )
    plt.close()
    
    print(f"  [√] 终极完美版四合一矩阵大图成功输出至: {PLOT_DIR}/{output_filename}")
    # ==========================================================
    # Step 8. 保存
    # ==========================================================

    print("\n=== Step 8: Saving outputs ===")

    adata_harmony.write(
        OUTPUT_HARMONY,
        compression='gzip'
    )

    atac_final_h5ad = atac.copy()

    atac_final_h5ad.obs = final_atac_obs.copy()

    atac_final_h5ad.write(
        OUTPUT_H5AD,
        compression='gzip'
    )

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()