# ODCR-Mamba

Optical-depth-conditioned state propagation and dual reconstruction for single-image dehazing.

本目录提供固定 ODCR-Mamba 模型的推理代码、权重及**与 V101 主文 Table I–VIII 对应的数据**。版本依据为 `ODCR_Mamba_TGRS_MainOnly_Redline_2026100111.docx`，SHA-256：`B7D5F68BF9D64527BBDE022604489FE4F26866782CE1C9DB4AB2E998EF3080EB`。这是一份研究稿件对应的代码与数据发布，不表示论文已发表或通过同行评审。

## 仅包含论文的 11 种方法

外部对比方法为 10 种，加上本文 ODCR-Mamba，共 11 种。以下引用编号对应 V101 正文；发布范围按论文名单确定，不按分数高低筛选。

| 论文名称 | 正文引用 | 记录 ID |
| --- | --- | --- |
| DCP | [3] | M01 |
| CAP | [2] | M02 |
| DehazeNet | [5] | S12 |
| AOD-Net | [7] | M07 |
| GridDehazeNet | [8] | M10 |
| WBPGNDN | [20] | M32 |
| FFA-Net | [9] | M12 |
| MB-TaylorFormer | [14] | M19 |
| PromptIR | [21] | S01 |
| MixDehazeNet | [22] | S14 |
| ODCR-Mamba | 本文方法 | ODCR-D696 |

所有跨方法对比表及逐图文件均限制在上述名单中，不发布其他方法的数据。逐图 CSV 同时给出 `method_id` 和可读的 `method` 名称。M19、S14 的归档配置名分别为 MB-TaylorFormer-B、MixDehazeNet-B，见 `results/PAPER_METHODS_2026100210.json`；这是同一批论文记录的名称对应关系，不是加入新方法或替换其输出。

Hazy/Input 是未处理输入，不是第 12 种算法；Direct、Scattering、Fused 是同一固定 ODCR-Mamba 的输出分支，不是额外对比方法。

## 数据与主文对应关系

| 路径 / 数据 | 内容 | 主文对应 |
| --- | --- | --- |
| `results/main_tables/table_*_display_2026100210.csv` | 8 张表，显示值逐单元格对应当前正文 | I–VIII |
| `results/main_tables/main_tables_fullprecision_2026100210.json` | 已保存的未舍入标量、均值及计算口径 | I–VIII |
| `results/per_image/natural_11_methods_2035_2026100210.csv` | 11 方法 × 185 对自然场景图像，2,035 条测量 | I、II、IV |
| `results/per_image/sots500_odcr_1000_2026100210.csv` | 仅本文方法，SOTS 室内、室外各 500 对 | V |
| `results/per_image/odcr_rrshid_test304_2026100210.csv` | 本文方法在 RRSHID-Test 的 304 条最终融合输出记录 | III 中 ODCR 行 |
| `results/per_image/odcr_rrshid_branches_912_2026100210.csv` | 同一 304 对图像的 3 个分支，共 912 条测量 | VIII |

自然场景共 185 对：I-HAZE 30、O-HAZE 45、Dense-Haze 55、NH-HAZE 55。11 种方法使用相同样本，因此 2,035 是测量记录数，不是独立场景数。表 I、II 的 14 个样例是这 185 对中的子集。RRSHID 的 304 条融合记录也包含在 912 条分支记录内，不能重复计数。

当前主文不包含 SOTS 200 对的额外跨方法比较，因此本版不再打包该比较文件及其 400 行便利子集；主文表 V 所需的 1,000 条本文方法记录完整保留。没有另附补充表或补充文档。

本次只整理已有测量记录；不重训、不推理、不改写分数，也不使用舍入后的论文值反推精确值。所有 8 张表的 622 个单元格与 V101 一致，602 个三位小数指标与相应标量记录一致。

## 包含内容与验证

| 路径 | 内容 |
| --- | --- |
| `src/odcr_mamba/` | 固定模型、状态扫描、重建损失、训练验证指标源码 |
| `infer.py` | 单张图像推理；保存新 PNG，不覆盖已有文件 |
| `weights/` | 仅模型张量，不含优化器和本地配置 |
| `configs/` | 结构参数及选定权重训练参数摘要 |
| `provenance/` | 模型身份、代码与发布封装验证记录 |
| `results/PAPER_METHODS_2026100210.json` | 严格限定的 11 方法及其名称映射 |
| `results/RESULTS_SCHEMA_2026100210.json` | 各表、逐图文件的用途与评估口径 |
| `PROVENANCE_RESULTS.json` | 来源哈希、数据范围及证据状态 |
| `FILE_SHA256_2026100210.json` | 所有发布文件的 SHA-256，不含自身 |

进入 `ODCR_Mamba` 目录后：

```bash
python -B verify_release.py
python -m pip install -r requirements.txt
python -B infer.py --input ../your_hazy_image.png --output ../outputs/restored.png --device cpu
```

验证器使用 Python 标准库，严格检查整个发布目录的文件哈希、未登记文件、方法名单、逐图方法对应关系及表格显示值。名单外的方法或未登记的数据文件会导致验证失败；个人输入、推理输出与虚拟环境请放在本目录外，运行时使用 `python -B` 避免写入缓存。具备兼容 CUDA 环境时，推理可改为 `--device cuda`。

## 固定模型与推理协议

- 网络类：`StandaloneODCRMamba`；参数量 2,042,064；权重张量 476。
- 选定 checkpoint 步数：60,000；所有本文方法主输出均对应这一固定模型。
- 原始完整 checkpoint SHA-256：`D696BBFA42DD1A840708B9C42E824036862ED17D538B2B47551641D8145854DB`。
- 发布张量文件 SHA-256：`7F9474B9AE6D42287EB72AE5F750AA284824EE67D14538F2C6F01DDC4F695FC2`。

发布文件去掉优化器等非模型内容，因此容器哈希不同；476 个张量与原始 checkpoint 的逐元素一致性检查见原封装验证记录。本次重打包未更改模型与推理源码、配置或权重，仅增强发布校验器。推理脚本在反序列化前检查权重哈希。哈希仅用于完整性检查，不等于数字签名。

参考封装验证环境：Python 3.8.20、PyTorch 1.10.0、NumPy 1.24.4、Pillow 9.4.0。未使用 `mamba-ssm` 或自定义 CUDA 扩展，其他版本或硬件可能存在浮点差异。此前封装一致性验证使用 CPU，并不是新的数据集实验。

输入按 RGB 解码为 `[0,1]` float32，两边至少 16 像素；保持原始尺寸，不自动裁剪、缩放、分块或测试时增强。推理采用 `eval()`、`no_grad()`，不启用混合精度。高分辨率输入需要较多内存，不会为规避内存不足自动更改协议。PNG 像素通过裁剪范围、乘 255、`np.rint` 后转换为 uint8；不同 PNG 编码器可能产生不同文件字节，文件哈希不同不必然意味着像素不同。

## 数据边界

- 主表 III 保留各项已有评估与溯源状态，不能整体称为已通过 strict-v12 完整验收。重打包不升级原有证据等级。
- 表 I、II 是 14 个展示样例的比较，表 IV 是四个数据集各自的均值；不能相互代替。表 V 只有本文方法，不应解读为外部方法的完整 SOTS 排名。
- RRSHID 与自然场景/SOTS 的 SSIM 协议不同，分别为 Uniform11 与 Gaussian11。`training_metrics.py` 不能代替所有表格的正式评估程序；LPIPS、DISTS 的完整历史评估环境未打包。
- 外部方法采用的预训练权重与训练条件不完全相同，训练图像交叠尚未全部独立确认。空白 checkpoint 哈希表示来源记录未提供，不能把名称映射当成权重身份验证。
- 表 VIII 是同一固定模型的分支输出分析，不是移除模块后重新训练的消融实验。
- 表 VII 的 SIFT 统计是四个数据集等权平均。表 VI 的 GMAE/RGBMAE 使用 ×1,000 显示尺度，JSON 保存未缩放值。GMAE 为 RGB `[0,1]` 图像水平、垂直一阶差分绝对误差的等权均值；RGBMAE 为全图 RGB 绝对误差均值。
- 包内未提供外部方法 RRSHID 的全部逐图记录，也未提供表 VI/VII 的逐图文件；对应正文均值和可用完整精度值已提供。不宣称这是完整训练或全部原始数据复现包。
- 原始数据集、原始/处理后图像、其他方法源码和权重、私人路径、论文文档、历史模型不在发布包内。项目相对来源路径仅用于追溯，不表示整份历史结果矩阵已发布。原始图像像素与外部模型未在本次重新验证。

## 权利说明

未擅自新增 MIT、Apache 等许可证。源码和权重的具体许可由所有者确认；没有许可证不等于公有领域。第三方依赖与数据仍遵循其各自许可。

生成时间（北京时间，到小时）：2026-10-02 10时。
