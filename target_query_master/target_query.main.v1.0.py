import argparse
import os
import sys
import yaml
from py_parser.HPAParser import HPAParser
from py_parser.UniProtParser import UniProtParser
from py_parser.GTExParser import GTExParser
from py_parser.DepMapParser import DepMapParser
from py_parser.TCGAParser import TCGAParser
from py_parser.MET500Parser import MET500Parser

class ParserManager:
    """管理所有数据解析器的统一接口"""
    
    def __init__(self):
        self.parsers = {
            'hpa': HPAParser,
            'uniprot': UniProtParser, 
            'gtex': GTExParser,
            'depmap': DepMapParser,
            'tcga': TCGAParser,
            'met500': MET500Parser
        }
        self.common_files = {}
        self.config_data = {}  # 存储解析后的配置数据
    
    def load_config_file(self, config_file_path):
        """加载并解析配置文件"""
        if not config_file_path:
            return {}
        
        try:
            self.validate_file(config_file_path, "配置文件")
            with open(config_file_path, 'r') as f:
                config_data = yaml.safe_load(f)
                self.config_data = config_data if config_data else {}
                print(f"✅ 配置文件加载成功: {config_file_path}")
                return self.config_data
        except Exception as e:
            print(f"⚠️ 配置文件加载失败: {e}，将使用默认配置")
            return {}
    
    def get_parser_config(self, parser_name):
        """获取特定解析器的配置"""
        config_key_map = {
            'gtex': 'GTExdata',
            'depmap': 'DepMapdata', 
            'tcga': 'TCGAdata',
            'met500': 'MET500data'
        }
        
        config_key = config_key_map.get(parser_name)
        if config_key and config_key in self.config_data:
            return self.config_data[config_key]
        return {}

    def get_gene_info(self, gene_anno_file, gene_symbol):
        """根据gene_symbol在gene_anno文件中搜索基因信息"""
        try:
            self.validate_file(gene_anno_file, "基因注释文件")
            
            # 读取基因注释文件
            import pandas as pd
            df = pd.read_csv(gene_anno_file, sep='\t')  # 假设是制表符分隔
            
            # 1. 首先搜索Approved Symbol列
            approved_match = df[df['Approved Symbol'] == gene_symbol]
            
            if not approved_match.empty:
                ensembl = approved_match['Ensembl'].values[0]
                geneid = approved_match['GeneID'].values[0]
                uniprotid = approved_match['Uniprot Entry'].values[0]
                return {
                    'ensembl': ensembl,
                    'geneid': geneid,
                    'uniprotid': uniprotid,
                    'gene_symbol': gene_symbol,
                    'found_in': 'Approved Symbol'
                }
            
            # 2. 如果Approved Symbol中没找到，搜索Synonyms列
            synonyms_match = df[df['Synonyms'].notna()]  # 排除NaN值
            
            for idx, row in synonyms_match.iterrows():
                synonyms = str(row['Synonyms']).split(',')
                # 清理每个同义词的空格并转为大写比较
                cleaned_synonyms = [syn.strip() for syn in synonyms]
                
                if gene_symbol in cleaned_synonyms:
                    ensembl = row['Ensembl']
                    geneid = row['GeneID']
                    uniprotid = row['Uniprot Entry']
                    return {
                        'ensembl': ensembl,
                        'geneid': geneid,
                        'uniprotid': uniprotid,
                        'gene_symbol': gene_symbol,
                        'found_in': 'Synonyms'
                    }
            
            # 3. 如果都没找到，返回None
            return None
            
        except Exception as e:
            raise ValueError(f"基因信息检索错误: {e}")
    
    def validate_gene_info(self, gene_anno_file, gene_symbol):
        """验证基因信息并返回结果，如果找不到则退出程序"""
        gene_info = self.get_gene_info(gene_anno_file, gene_symbol)
        
        if gene_info is None:
            print(f"❌ 未找到基因符号 '{gene_symbol}'")
            print("3. 请检查gene symbol名称，修改后重新输入")
            sys.exit(1)
        
        print(f"✅ 基因信息验证成功:")
        print(f"   基因符号: {gene_info['gene_symbol']}")
        print(f"   Ensembl ID: {gene_info['ensembl']}")
        print(f"   Gene ID: {gene_info['geneid']}")
        print(f"   Uniprot ID: {gene_info['uniprotid']}")
        print(f"   查找位置: {gene_info['found_in']}")
        
        return gene_info

    def validate_common_files(self, args):
        """验证所有解析器共用的文件，并验证基因信息"""
        common_files_info = {}
        
        if hasattr(args, 'gene_anno') and args.gene_anno:
            self.validate_file(args.gene_anno, "基因注释文件")
            common_files_info['gene_anno'] = args.gene_anno
        else:
            raise ValueError("基因注释文件是必需参数，请使用--gene-anno指定")
        
        if hasattr(args, 'gene_symbol') and args.gene_symbol:
            # 验证基因信息
            gene_info = self.validate_gene_info(args.gene_anno, args.gene_symbol)
            common_files_info['gene_info'] = gene_info
        else:
            raise ValueError("基因符号是必需参数，请使用--gene-symbol指定")
        
        # 加载配置文件
        if hasattr(args, 'config_file') and args.config_file:
            self.load_config_file(args.config_file)
            common_files_info['config_file'] = args.config_file
            common_files_info['config_data'] = self.config_data
        
        if hasattr(args, 'output_dir') and args.output_dir:
            self.ensure_output_dir(args.output_dir)
            common_files_info['output_dir'] = args.output_dir
        
        self.common_files = common_files_info
        return common_files_info
    
    def validate_files(self, file_paths, parser_name, min_files=1):
        """验证多个文件是否存在且可读"""
        if not file_paths:
            raise ValueError(f"{parser_name}需要指定文件路径列表")
        
        if len(file_paths) < min_files:
            raise ValueError(f"{parser_name}至少需要{min_files}个文件，但只提供了{len(file_paths)}个")
        
        for file_path in file_paths:
            self.validate_file(file_path, parser_name)
        
        return True
    
    def validate_optional_file(self, file_path, parser_name):
        """验证可选文件（如果提供则验证，不提供则跳过）"""
        if file_path and file_path.strip():
            return self.validate_file(file_path, parser_name)
        return True
    
    def validate_file(self, file_path, file_description):
        """验证单个文件是否存在且可读"""
        if not file_path:
            raise ValueError(f"{file_description}路径不能为空")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_description}不存在: {file_path}")
        
        if not os.path.isfile(file_path):
            raise ValueError(f"{file_description}路径不是文件: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"无法读取{file_description}: {file_path}")
        
        return True
    
    def ensure_output_dir(self, output_dir):
        """确保输出目录存在且可写"""
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                raise OSError(f"无法创建输出目录 {output_dir}: {e}")
        
        if not os.path.isdir(output_dir):
            raise ValueError(f"输出路径不是目录: {output_dir}")
        
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"输出目录不可写: {output_dir}")
        
        return True

def setup_argparse():
    """设置argparse子命令系统，包含通用参数和all命令"""
    
    # 创建主解析器
    parser = argparse.ArgumentParser(
        description='不同来源生信数据集解析工具',
        epilog='example: python target_query.main.py all --gene-anno HGNC.gene.anno.tsv --gene-symbol CA9 --config-file config.yaml'
    )
    
    # 添加子命令解析器
    subparsers = parser.add_subparsers(
        title='可用解析器',
        description='选择要使用的数据解析器',
        dest='command',
        required=True
    )
    
    # 定义通用参数添加函数
    def add_common_arguments(subparser):
        """为所有子命令添加通用参数"""
        subparser.add_argument('--gene-anno', required=True,
                             help='基因注释文件路径（所有解析器都需要）', dest='gene_anno')
        subparser.add_argument('--gene-symbol', required=True,
                             help='基因名称', dest='gene_symbol')
        subparser.add_argument('--config-file', 
                             help='配置文件路径（YAML格式，可选）', dest='config_file')
        subparser.add_argument('--output-dir', default='./results',
                             help='输出目录路径（默认: ./results）', dest='output_dir')
        return subparser
    
    # 添加all子命令 - 批量执行所有解析器
    all_parser = subparsers.add_parser('all', help='批量执行所有可用的解析器')
    all_parser = add_common_arguments(all_parser)
    all_parser.add_argument('--skip-parsers', nargs='+', 
                          choices=['hpa', 'uniprot', 'gtex', 'depmap', 'tcga', 'met500'],
                          help='要跳过的解析器列表', dest='skip_parsers', default=[])
    all_parser.add_argument('--only-parsers', nargs='+',
                          choices=['hpa', 'uniprot', 'gtex', 'depmap', 'tcga', 'met500'],
                          help='只运行指定的解析器（与skip-parsers互斥）', dest='only_parsers')
    all_parser.set_defaults(func=run_all_parsers)

    # HPA解析器子命令
    hpa_parser = subparsers.add_parser('hpa', help='HPA数据库解析')
    hpa_parser = add_common_arguments(hpa_parser)
    hpa_parser.set_defaults(func=run_hpa_parser)

    # Uniprot解析器子命令
    uniprot_parser = subparsers.add_parser('uniprot', help='UniProt数据库解析')
    uniprot_parser = add_common_arguments(uniprot_parser)
    uniprot_parser.set_defaults(func=run_uniprot_parser)

    # GTEx解析器子命令
    gtex_parser = subparsers.add_parser('gtex', help='GTEx数据库解析')
    gtex_parser = add_common_arguments(gtex_parser)
    gtex_parser.add_argument('--sample_expr_file',
                            help='GTEx表达数据集，前两行为数据描述，行为基因，第一二列为基因描述，其余列为样本，值为TPM，（优先于配置文件）', dest='sample_expr_file')
    gtex_parser.add_argument('--tissue_expr_file',
                            help='GTEx表达数据集，基于单个组织的中位值计算的表达值，行为基因，第一二列为基因描述，其余列为样本，值为TPM，（优先于配置文件）', dest='tissue_expr_file')
    gtex_parser.add_argument('--metadata_file',
                            help='GTEx样本信息文件（优先于配置文件）', dest='metadata_file')                        
    gtex_parser.set_defaults(func=run_gtex_parser)

    # DepMap解析器子命令
    depmap_parser = subparsers.add_parser('depmap', help='DepMap数据库解析')
    depmap_parser = add_common_arguments(depmap_parser)
    depmap_parser.add_argument('--sample_expr_file',
                              help='DepMap表达数据集，行为样本，列为基因，值为log2(TPM+1)（优先于配置文件）', dest='sample_expr_file')
    depmap_parser.add_argument('--metadata_file',
                              help='细胞系样本信息文件（可选，优先于配置文件）', dest='metadata_file')
    depmap_parser.set_defaults(func=run_depmap_parser)

    # TCGA解析器子命令
    tcga_parser = subparsers.add_parser('tcga', help='TCGA数据库解析')
    tcga_parser = add_common_arguments(tcga_parser)
    tcga_parser.add_argument('--sample_expr_file', 
                            help='TCGA表达数据集，行为基因，列为样本，值为TPM（可选，优先于配置文件）', dest='sample_expr_file')
    tcga_parser.add_argument('--metadata-file', 
                            help='TCGA元数据文件路径（可选，优先于配置文件）', dest='metadata_file')
    tcga_parser.add_argument('--clinical-file',
                            help='TCGA临床数据文件路径，json格式，（可选，优先于配置文件）', dest='clinical_file')
    tcga_parser.set_defaults(func=run_tcga_parser)    

    # MET500解析器子命令
    met500_parser = subparsers.add_parser('met500', help='MET500数据库解析')
    met500_parser = add_common_arguments(met500_parser)
    met500_parser.add_argument('--sample_expr_file',
                              help='MET500表达数据集，行为样本，列为基因，值为log2(TPM+1)（优先于配置文件）', dest='sample_expr_file')
    met500_parser.add_argument('--metadata_file',
                              help='细胞系样本信息文件（可选，优先于配置文件）', dest='metadata_file')
    met500_parser.set_defaults(func=run_met500_parser)

    return parser

def run_hpa_parser(args):
    """执行HPA解析器 - 只需要基因信息"""
    manager = ParserManager()
    try:
        # 1. 验证通用文件并加载配置（包含基因验证）
        common_files = manager.validate_common_files(args)
        gene_info = common_files['gene_info']
        odir = common_files['output_dir']
        print("✅ 通用文件验证通过")
        
        # 2. 初始化解析器并传递配置
        gene = gene_info['gene_symbol']
        ensembl_id = gene_info['ensembl']
        hpa_parser = HPAParser(ensembl_id)
        hpa_parsed_data = hpa_parser.parse()
        hpa_parser.save_to_file(f"{odir}/{gene}.{ensembl_id}.HPA_data.json")
        print("✅ HPA数据解析完成")
        print(f"   目标基因: {gene} Ensembl: {ensembl_id}")
    except (ValueError, PermissionError) as e:
        print(f"❌ HPA解析错误: {e}")
        sys.exit(1)

def run_uniprot_parser(args):
    """执行Uniprot解析器 - 只需要基因信息"""
    manager = ParserManager()
    try:
        # 1. 验证通用文件并加载配置（包含基因验证）
        common_files = manager.validate_common_files(args)
        gene_info = common_files['gene_info']
        odir = common_files['output_dir']
        print("✅ 通用文件验证通过")
        
        # 2. 初始化解析器并传递配置
        gene = gene_info['gene_symbol']
        uniprot_id = gene_info['uniprotid'] 
        uniprot_parser = UniProtParser(uniprot_id)
        uniprot_parsed_data = uniprot_parser.parse()
        uniprot_parser.save_to_file(f"{odir}/{gene}.{uniprot_id}.UniProt_data.json")
        print("✅ UniProt数据解析完成")
        print(f"   目标基因: {gene} UniProt ID: {uniprot_id}")
    except (ValueError, PermissionError) as e:
        print(f"❌ UniProt解析错误: {e}")
        sys.exit(1)

def run_gtex_parser(args):
    """执行GTEx解析器 - 支持配置文件"""
    manager = ParserManager()
    try:
        # 1. 验证通用文件并加载配置
        common_files = manager.validate_common_files(args)
        gene_info = common_files['gene_info']
        odir = common_files['output_dir']        
        print("✅ 通用文件验证通过")
        
        # 2. 获取GTEx配置
        gtex_config = manager.get_parser_config('gtex')
        print(f"📋 GTEx配置加载: {len(gtex_config)} 个配置项")
        
        # 3. 合并文件路径：命令行参数优先，其次配置文件
        sample_expr_file = (getattr(args, 'sample_expr_file', None) or 
                    gtex_config.get('sample_expr_file'))
        tissue_expr_file = (getattr(args, 'tissue_expr_file', None) or 
                    gtex_config.get('tissue_expr_file'))
        metadata_file = (getattr(args, 'metadata_file', None) or 
                    gtex_config.get('metadata_file'))

        # 4. 验证文件
        if not sample_expr_file:
            raise ValueError("GTEx表达数据集未指定，请通过--sample_expr_file或配置文件指定")
        if not tissue_expr_file:
            raise ValueError("GTEx组织水平表达数据集未指定，请通过--tissue_expr_file或配置文件指定")
        if not metadata_file:
            raise ValueError("GTEx临床数据未指定，请通过--metadata_file或配置文件指定")
        
        manager.validate_file(sample_expr_file, "GTEx数据文件")
        manager.validate_file(tissue_expr_file, "GTEx数据文件")
        manager.validate_file(metadata_file, "GTEx数据文件")

        # 5. 初始化解析器并传递完整配置
        gene = gene_info['gene_symbol']
        ensembl_id = gene_info['ensembl']
        gtex_parser = GTExParser(
                ensembl_id = ensembl_id,
                sample_expr_file = sample_expr_file,
                tissue_expr_file = tissue_expr_file,
                metadata_file = metadata_file            
        )
        result_json = gtex_parser.parse()
        gtex_parser.save_to_file(f"{odir}/{gene}.{ensembl_id}.GTEx_data.json") 
        print("✅ GTEx数据解析完成")
        print(f"   目标基因: {gene} Ensembl: {ensembl_id}") 
    except (FileNotFoundError, ValueError, PermissionError) as e:
        print(f"❌ GTEx解析错误: {e}")
        sys.exit(1)

def run_depmap_parser(args):
    """执行DepMap解析器 - 支持配置文件"""
    manager = ParserManager()
    try:
        # 1. 验证通用文件并加载配置
        common_files = manager.validate_common_files(args)
        gene_info = common_files['gene_info']
        odir = common_files['output_dir']
        print("✅ 通用文件验证通过")
        
        # 2. 获取DepMap配置
        depmap_config = manager.get_parser_config('depmap')
        print(f"📋 DepMap配置加载: {len(depmap_config)} 个配置项")
        
        # 3. 合并文件路径：命令行参数优先，其次配置文件
        sample_expr_file = (getattr(args, 'sample_expr_file', None) or 
                    depmap_config.get('sample_expr_file'))
        metadata_file = (getattr(args, 'metadata_file', None) or 
                    depmap_config.get('metadata_file'))

        # 4. 验证文件
        if not sample_expr_file:
            raise ValueError("DepMap表达数据集未指定，请通过--sample_expr_file或配置文件指定")
        if not metadata_file:
            raise ValueError("DepMap临床数据未指定，请通过--metadata_file或配置文件指定")
        
        manager.validate_file(sample_expr_file, "DepMap数据文件")
        manager.validate_file(metadata_file, "DepMap数据文件")

        # 5. 初始化解析器并传递完整配置
        gene = gene_info['gene_symbol']
        ensembl_id = gene_info['ensembl']
        depmap_parser = DepMapParser(
                genename = gene,
                sample_expr_file = sample_expr_file,
                metadata_file = metadata_file, 
        )
        result_json = depmap_parser.parse()        
        depmap_parser.save_to_file(f"{odir}/{gene}.{ensembl_id}.DepMap_data.json")       
        print(f"✅ DepMap数据解析完成")
        print(f"   目标基因: {gene} Ensembl: {ensembl_id}")
    except (FileNotFoundError, ValueError, PermissionError) as e:
        print(f"❌ DepMap解析错误: {e}")
        sys.exit(1)

def run_tcga_parser(args):
    """执行TCGA解析器 - 支持配置文件"""
    manager = ParserManager()
    try:
        # 1. 验证通用文件并加载配置（包含基因验证）
        common_files = manager.validate_common_files(args)
        gene_info = common_files['gene_info']
        odir = common_files['output_dir']
        print("✅ 通用文件验证通过")
        
        # 2. 获取TCGA配置
        tcga_config = manager.get_parser_config('tcga')
        print(f"📋 TCGA配置加载: {len(tcga_config)} 个配置项")
        
        # 3. 合并文件路径：命令行参数优先，其次配置文件
        sample_expr_file = (getattr(args, 'sample_expr_file', None) or 
                        tcga_config.get('sample_expr_file'))

        metadata_file = (getattr(args, 'metadata_file', None) or 
                        tcga_config.get('metadata_file'))
        
        clinical_file = (getattr(args, 'clinical_file', None) or 
                        tcga_config.get('clinical_file'))

        # 4. 验证文件
        if not sample_expr_file:
            raise ValueError("TCGA表达数据集未指定，请通过--sample_expr_file或配置文件指定")
        if not metadata_file:
            raise ValueError("TCGA元数据文件未指定，请通过--metadata-file或配置文件指定")
        if not clinical_file:
            raise ValueError("TCGA临床数据未指定，请通过--clinical_file或配置文件指定")

        manager.validate_files([sample_expr_file], "TCGA表达数据文件", min_files=1)
        manager.validate_files([metadata_file], "TCGA元数据文件", min_files=1)
        manager.validate_files([clinical_file], "TCGA临床数据文件", min_files=1)

        # 5. 初始化解析器并传递完整配置
        tcga_parser = TCGAParser(
            ensembl_id = gene_info['ensembl'],
            genename = gene_info['gene_symbol'],
            sample_expr_file = sample_expr_file,
            metadata_file = metadata_file, 
            metadata_all_json_file = clinical_file
        )
        result_json = tcga_parser.parse()
        tcga_parser.save_to_file(f"{odir}/{gene_info['gene_symbol']}.{gene_info['ensembl']}.TCGA_data.json")
        
        print(f"✅ TCGA数据解析完成")
        print(f"   目标基因: {gene_info['gene_symbol']} Ensembl: {gene_info['ensembl']}")             
    except (FileNotFoundError, ValueError, PermissionError) as e:
        print(f"❌ TCGA解析错误: {e}")
        sys.exit(1)

def run_met500_parser(args):
    """执行MET500解析器 - 支持配置文件"""
    manager = ParserManager()
    try:
        # 1. 验证通用文件并加载配置
        common_files = manager.validate_common_files(args)
        gene_info = common_files['gene_info']
        odir = common_files['output_dir']        
        print("✅ 通用文件验证通过")
        
        # 2. 获取MET500配置
        met500_config = manager.get_parser_config('met500')
        print(f"📋 MET500配置加载: {len(met500_config)} 个配置项")
        
        # 3. 合并文件路径
        sample_expr_file = (getattr(args, 'sample_expr_file', None) or 
                    met500_config.get('sample_expr_file'))
        metadata_file = (getattr(args, 'metadata_file', None) or 
                    met500_config.get('metadata_file'))

        # 4. 验证文件
        if not sample_expr_file:
            raise ValueError("MET500表达数据集未指定，请通过--sample_expr_file或配置文件指定")
        if not metadata_file:
            raise ValueError("MET500临床数据未指定，请通过--metadata_file或配置文件指定")
        
        manager.validate_file(sample_expr_file, "MET500数据文件")
        manager.validate_file(metadata_file, "MET500数据文件")

        # 5. 初始化解析器并传递完整配置
        gene = gene_info['gene_symbol']
        ensembl_id = gene_info['ensembl']
        met500_parser = MET500Parser(
            ensembl_id = ensembl_id,
            genename = gene,
            sample_expr_file = sample_expr_file, 
            metadata_file = metadata_file
        )

        result_json = met500_parser.parse()    
        met500_parser.save_to_file(f"{odir}/{gene}.{ensembl_id}.MET500_data.json")  
        print("✅ MET500数据解析完成")
        print(f"   目标基因: {gene} Ensembl: {ensembl_id}")
    except (FileNotFoundError, ValueError, PermissionError) as e:
        print(f"❌ MET500解析错误: {e}")
        sys.exit(1)

# 批量执行函数 (执行所有解析器)
def run_all_parsers(args):
    """批量执行所有可用的解析器"""
    manager = ParserManager()
   
    try:
        # 验证通用文件
        common_files = manager.validate_common_files(args)
        print("✅ 通用文件验证通过")
        
        # 确定要运行的解析器列表
        parsers_to_run = determine_parsers_to_run(args)
        
        if not parsers_to_run:
            print("⚠️ 没有可运行的解析器，请检查--skip-parsers和--only-parsers参数")

        print(f"🎯 准备运行 {len(parsers_to_run)} 个解析器: {', '.join(parsers_to_run)}")
        
        # 为每个解析器创建模拟参数对象
        for parser_name in parsers_to_run:
            print(f"\n{'='*50}")
            print(f"🚀 开始执行 {parser_name} 解析器")
            print(f"{'='*50}")
            
            try:
                # 创建该解析器所需的模拟参数
                mock_args = create_mock_args_for_parser(parser_name, args)
                
                # 执行对应的解析器
                if parser_name == 'tcga':
                    run_tcga_parser(mock_args)
                elif parser_name == 'depmap':
                    run_depmap_parser(mock_args)
                elif parser_name == 'hpa':
                    run_hpa_parser(mock_args)
                elif parser_name == 'gtex':
                    run_gtex_parser(mock_args)
                elif parser_name == 'met500':
                    run_met500_parser(mock_args)
                elif parser_name == 'uniprot':
                    run_uniprot_parser(mock_args)
                    
                print(f"✅ {parser_name} 解析完成")
            except Exception as e:
                print(f"❌ {parser_name.upper()} 解析失败: {e}")
                continue
    except Exception as e:
        print(f"💥 批量执行错误: {e}")
        sys.exit(1)

# 其他辅助函数保持不变...
def determine_parsers_to_run(args):
    """确定要运行哪些解析器"""
    all_parsers = ['hpa', 'uniprot', 'gtex', 'depmap', 'tcga', 'met500']
    
    if hasattr(args, 'only_parsers') and args.only_parsers:
        parsers_to_run = [p for p in args.only_parsers if p in all_parsers]
    else:
        skip_parsers = getattr(args, 'skip_parsers', [])
        parsers_to_run = [p for p in all_parsers if p not in skip_parsers]
    
    return parsers_to_run

def create_mock_args_for_parser(parser_name, original_args):
    """为特定解析器创建模拟参数对象"""
    class MockArgs:
        def __init__(self, original_args, parser_name):
            # 复制通用参数
            self.gene_anno = original_args.gene_anno
            self.gene_symbol = original_args.gene_symbol  # 添加gene_symbol
            self.config_file = getattr(original_args, 'config_file', None)
            self.output_dir = getattr(original_args, 'output_dir', './results')
            
            # 为特定解析器设置文件参数
            if parser_name == 'tcga':
                self.sample_expr_file = getattr(original_args, 'sample_expr_file', None)
                self.metadata_file = getattr(original_args, 'metadata_file', None)
                self.clinical_file = getattr(original_args, 'clinical_file', None)
            elif parser_name == 'depmap':
                self.sample_expr_file = getattr(original_args, 'sample_expr_file', None)
                self.metadata_file = getattr(original_args, 'metadata_file', None)
            elif parser_name == 'gtex':
                self.gtex_file = getattr(original_args, 'gtex_file', None)
                self.tissue_expr_file = getattr(original_args, 'tissue_expr_file', None)
                self.metadata_file = getattr(original_args, 'metadata_file', None)
            elif parser_name == 'met500':
                self.sample_expr_file = getattr(original_args, 'sample_expr_file', None)
                self.metadata_file = getattr(original_args, 'metadata_file', None)
    return MockArgs(original_args, parser_name)

import logging
import time

def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('parser_tool.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """主函数"""
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        parser = setup_argparse()
        args = parser.parse_args()
        
        # 参数验证
        if not validate_required_args(args):
            sys.exit(1)
        
        logger.info(f"开始执行 {args.command} 命令")
        
        # 执行功能
        start_time = time.time()
        args.func(args)
        end_time = time.time()
        logger.info(f"任务完成，耗时: {end_time - start_time:.2f}秒")
    except Exception as e:
        logger.error(f"程序执行错误: {e}", exc_info=True)
        sys.exit(1)

def validate_required_args(args):
    """验证必需参数"""
    required_params = [
        ('gene_symbol', '--gene-symbol'),
        ('gene_anno', '--gene-anno')
    ]
    
    for attr, param_name in required_params:
        if not hasattr(args, attr) or not getattr(args, attr):
            print(f"❌ 错误: 必须指定 {param_name} 参数")
            return False
    return True

if __name__ == "__main__":
    main()
