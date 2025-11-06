# 软件描述
**靶点信息查询和分析。可基于单个靶点，调研Human Protein Altas (HPA), Uniprot, GTEx, DepMap, TCGA, MET500数据库靶点信息，明确该靶点在癌症细胞系，不同正常组织，不同癌症组织的表达分布，及转移灶的表达分布，同时进行该靶点的信息注释**

# 环境配置

**python环境包**
```txt
collections  
scipy  
typing  
argparse  
dask  
json  
logging  
numpy  
os  
pandas  
requests  
subprocess  
sys  
time  
yaml  
```

**R 环境包（r4.3）**
```txt
jsonlite  
purrr  
rstatix  
scales  
tidyr  
dplyr  
ggplot2  
ggpmisc  
ggpubr  
```
**需要安装DeepTMHMM:A Deep Learning Model for Transmembrane Topology Prediction and Classification，并配置到环境变量**

DeepTMHMM软件安装可参考https://dtu.biolib.com/DeepTMHMM/


# 使用说明
```bash
不同来源生信数据集解析工具

options:
  -h, --help            show this help message and exit

可用解析器:
  选择要使用的数据解析器

  {all,hpa,uniprot,gtex,depmap,tcga,met500}
    all                 批量执行所有可用的解析器
    hpa                 HPA数据库解析
    uniprot             UniProt数据库解析
    gtex                GTEx数据库解析
    depmap              DepMap数据库解析
    tcga                TCGA数据库解析
    met500              MET500数据库解析

example: python target_query.main.py all --gene-anno HGNC.gene.anno.tsv --gene-symbol CA9 --config-file config.yaml --output-dir ./
```
## config示例（yaml）
```txt
common_params:
  r_software_path: "./software/annoconda/bin/envs/r4.3/bin/Rscript"

GTExdata:
    sample_expr_file: ./GTEx/V10/Bulk_tissue_expression/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct
    tissue_expr_file: ./GTEx/V10/Bulk_tissue_expression/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_median_tpm.tau.bytissue.gct
    metadata_file: ./GTEx/V10/Metadata/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt
DepMapdata:
    sample_expr_file: ./DepMap/25Q2/Expression/OmicsExpressionProteinCodingGenesTPMLogp1.csv
    metadata_file: ./DepMap/25Q2/Model_Condition_Mapping/Model.csv
TCGAdata:
    sample_expr_file: ./xena/tcga_RSEM_gene_tpm.reverselog.tsv
    metadata_file: ./xena/all_clinial/TCGA.allcancer.sample_type.tsv
    clinical_file: ./TCGA/cancer_samples_data.json
MET500data:
    sample_expr_file: ./xena/MET500/MET500.expression.fpkm.tsv
    metadata_file: ./xena/MET500/MET500.data_clinical_sample.txt
```

# 版本信息
v1.0: 仅可获取不同数据库分析后的json文件
v1.1: 在v1.0基础上添加了数据可视化功能，生成可视化结果文件
v1.2: 在v1.1基础上添加了DeepTMHMM蛋白结构域预测功能


# 运行示例
## 调研CA9在所有数据库的信息
```
python target_query.main.v1.2.py all --gene-anno HGNC.gene.anno.tsv --gene-symbol CA9 --config-file config.yaml --output-dir ./
```

## 仅调研TCGA数据集信息
```
python target_query.main.v1.2.py tcga --gene-anno HGNC.gene.anno.tsv --gene-symbol CA9 --config-file config.yaml --output-dir ./
```

## 仅调研GTEx和TCGA数据集信息
```
python target_query.main.v1.2.py all --gene-anno HGNC.gene.anno.tsv --gene-symbol CA9 --config-file config.yaml --output-dir ./  --skip-parsers hpa,uniprot,depmap,met500 --only-parsers  gtex,tcga
```
