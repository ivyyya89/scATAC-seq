import os

# --- 1. 线程控制 (必须在 import numpy 之前) ---
N_THREADS = "16"
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS

import pandas as pd
import muon as mu
import scanpy as sc
import anndata as ad
import pyranges as pr
import numpy as np
import scipy.sparse as sp
import gc

# 抑制 MuData 警告
mu.set_options(pull_on_update=False)

# ================= 配置路径 =================
DATA_SRC = "/data/Data/BA11/scATAC-seq/GSE256207_out/"
METADATA_PATH = "/data/zihui/BA11/scATAC/SraRunTable (2).csv"
OUTPUT_FILE = "/data/zihui/BA11/scATAC/BA11_MDD_Control_Aligned.h5ad"

TARGET_CATEGORIES = ['Control', 'MajorDepression']
# ===========================================

def parse_peaks_to_pyranges(peak_index):
    """
    稳健地将 var_names (chr1:100-200 或 chr1-100-200) 转换为 PyRanges
    """
    # 转为 Series 避免 Index 对象的类型限制
    s = pd.Series(peak_index.astype(str))
    # 统一替换分隔符并分割
    df = s.str.replace(':', '-').str.split('-', expand=True).iloc[:, :3]
    df.columns = ['Chromosome', 'Start', 'End']
    
    # 转换为数值，遇到错误字符转为 NaN 并处理
    df['Start'] = pd.to_numeric(df['Start'], errors='coerce')
    df['End'] = pd.to_numeric(df['End'], errors='coerce')
    
    # 记录原始 ID 用于后续映射
    df['Original_ID'] = peak_index.values
    
    # 剔除无法解析的行（如果有）
    df = df.dropna(subset=['Start', 'End'])
    df['Start'] = df['Start'].astype(int)
    df['End'] = df['End'].astype(int)
    
    return pr.PyRanges(df)

def get_consensus_peaks(peak_list):
    """生成非重叠的共识 Peak 集"""
    print(f"正在合并 {len(peak_list)} 个样本的坐标以生成共识集...")
    combined_pr = pr.concat(peak_list)
    
    # Merge overlapping intervals
    consensus_pr = combined_pr.merge()
    
    # 生成标准 ID (Standard_ID)
    df = consensus_pr.as_df()
    df['Standard_ID'] = df['Chromosome'].astype(str) + ":" + \
                        df['Start'].astype(str) + "-" + \
                        df['End'].astype(str)
    
    return pr.PyRanges(df), df['Standard_ID'].tolist()

def main():
    print("--- 启动数据驱动的 Peak 合并程序 (稳健版) ---")
    
    # 路径检查
    if not os.path.exists(DATA_SRC) or not os.path.exists(METADATA_PATH):
        print("❌ 路径错误，请检查 DATA_SRC 或 METADATA_PATH")
        return

    # 1. 预扫描：收集所有样本的 Peak 坐标
    meta = pd.read_csv(METADATA_PATH)
    filtered_meta = meta[meta['classification'].isin(TARGET_CATEGORIES)].copy()
    all_folders = os.listdir(DATA_SRC)
    
    sample_info = [] 
    all_peak_ranges = []

    print("第一步：扫描样本并提取 Peak 坐标...")
    for _, row in filtered_meta.iterrows():
        srr_id = row['Run']
        matched = [f for f in all_folders if srr_id in f]
        if not matched: continue
        
        folder_name = matched[0]
        h5_path = os.path.join(DATA_SRC, folder_name, "outs/filtered_peak_bc_matrix.h5")
        
        if os.path.exists(h5_path):
            try:
                mdata = mu.read_10x_h5(h5_path)
                adata_atac = mdata.mod['atac']
                
                # 转换坐标
                sample_pr = parse_peaks_to_pyranges(adata_atac.var_names)
                all_peak_ranges.append(sample_pr)
                
                sample_info.append({'id': folder_name, 'path': h5_path, 'condition': row['classification']})
                print(f"  已扫描: {folder_name} (Peaks: {len(adata_atac.var_names)})")
                
                del mdata
                gc.collect()
            except Exception as e:
                print(f"  跳过 {folder_name}, 错误原因: {e}")

    if not all_peak_ranges:
        print("❌ 未发现任何有效数据。")
        return

    # 2. 生成共识 Peak 集
    ref_pr, consensus_ids = get_consensus_peaks(all_peak_ranges)
    print(f"✅ 共识集生成完毕，共 {len(consensus_ids)} 个唯一 Peak。")

    # 3. 第二次遍历：映射数据
    adatas = []
    success_ids = []

    print("\n第二步：将样本数据重采样至共识集...")
    for item in sample_info:
        try:
            mdata = mu.read_10x_h5(item['path'])
            adata = mdata.mod['atac']
            
            # 建立映射
            sample_pr = parse_peaks_to_pyranges(adata.var_names)
            overlap = sample_pr.join(ref_pr)
            
            # 建立 Original -> Standard 的 Mapping
            map_df = overlap.as_df()[['Original_ID', 'Standard_ID']].drop_duplicates('Original_ID')
            mapping = map_df.set_index('Original_ID')['Standard_ID']
            
            # 过滤掉不在共识集中的 Peak
            adata = adata[:, adata.var_names.isin(mapping.index)].copy()
            # 重命名为标准 ID
            adata.var_names = mapping.loc[adata.var_names].values
            
            # 如果多个原始 Peak 映射到了同一个合并后的共识 Peak，保留第一个
            if not adata.var_names.is_unique:
                adata = adata[:, ~adata.var_names.duplicated()].copy()

            adata.obs['sample_id'] = item['id']
            adata.obs['condition'] = item['condition']
            
            adatas.append(adata)
            success_ids.append(item['id'])
            print(f"  已处理: {item['id']}")
            
            del mdata
            gc.collect()
        except Exception as e:
            print(f"  处理 {item['id']} 时发生映射错误: {e}")

    # 4. 合并保存
    if adatas:
        print(f"\n正在执行最终合并 ({len(adatas)} 样本)...")
        # index_unique="-" 区分不同样本的相同 Barcode
        combined = ad.concat(adatas, join='outer', label="batch", keys=success_ids, index_unique="-")
        
        # 统一转为 CSR 稀疏矩阵并处理空值
        if not sp.issparse(combined.X):
            combined.X = sp.csr_matrix(np.nan_to_num(combined.X, nan=0.0))
        else:
            combined.X = combined.X.tocsr()

        print(f"正在写入文件: {OUTPUT_FILE}")
        combined.write(OUTPUT_FILE)
        print(f"🎉 任务圆满完成！最终矩阵大小: {combined.shape}")
    else:
        print("❌ 没有可合并的数据。")

if __name__ == "__main__":
    main()