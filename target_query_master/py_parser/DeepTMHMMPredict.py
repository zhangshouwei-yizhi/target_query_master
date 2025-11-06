import json
import re
import os
import shutil
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import subprocess

class ProteinTopologyPredictor:
    """
    蛋白质拓扑结构预测与数据管理类
    基于UniProt解析的蛋白序列、使用DeepTMHMM预测，后转换为结构化的json文件
    """
    
    # 蛋白质类型描述字典
    PROTEIN_DESCRIPTIONS = {
        "TM": "transmembrane proteins without a signal peptide",
        "SP+TM": "transmembrane proteins with signal peptide", 
        "SP": "signal peptide",
        "GLOB": "globular proteins without a signal peptide",
        "SP+GLOB": "globular proteins with a signal peptide"
    }
    
    def __init__(self, output_dir: str = "./results"):
        """
        初始化预测器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_data = None
    
    def safe_load_json(self, file_path: str) -> Optional[Dict]:
        """
        安全加载JSON文件，包含完整的错误处理
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            JSON数据字典或None（如果出错）
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"错误：文件 {file_path} 未找到")
        except json.JSONDecodeError as e:
            print(f"错误：JSON格式无效 - {e}")
        except Exception as e:
            print(f"错误：{e}")
        return None
    
    def create_fasta_from_uniprot(self, uniprot_json_path: str, gene_name: str) -> str:
        """
        从UniProt JSON文件创建FASTA序列文件
        
        Args:
            uniprot_json_path: UniProt JSON文件路径
            gene_name: 基因名称
            
        Returns:
            生成的FASTA文件路径
        """
        # 验证输入文件存在
        if not os.path.exists(uniprot_json_path):
            raise FileNotFoundError(f"UniProt JSON文件不存在: {uniprot_json_path}")
        
        # 创建基因专属目录
        gene_dir = self.output_dir / gene_name
        gene_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载UniProt JSON数据
        uniprot_data = self.safe_load_json(uniprot_json_path)
        if uniprot_data is None:
            raise ValueError(f"无法解析UniProt JSON文件: {uniprot_json_path}")
        
        # 验证必需的序列数据
        if 'Sequence' not in uniprot_data:
            raise KeyError(f"UniProt JSON文件中缺少'Sequence'字段")
        
        # 从文件名解析UniProt ID
        filename = os.path.basename(uniprot_json_path)
        parts = filename.split('.')
        uniprot_id = parts[1] if len(parts) > 1 else "unknown"
        
        # 生成输出FASTA文件路径
        output_fasta_path = gene_dir / f"{gene_name}.fasta"
        
        # 准备序列数据
        sequence_data = {
            'header': f"{gene_name} {uniprot_id}",
            'sequence': uniprot_data['Sequence']
        }
        
        # 创建FASTA文件
        self._create_single_fasta(sequence_data, output_fasta_path)
        
        print(f"已生成FASTA文件: {output_fasta_path}")
        return str(output_fasta_path)
    
    def _create_single_fasta(self, sequence_data: Dict, output_path: Path) -> None:
        """
        创建单序列FASTA文件（内部方法）
        
        Args:
            sequence_data: 包含header和sequence的字典
            output_path: 输出文件路径
        """
        # 参数验证
        if not isinstance(sequence_data, dict):
            raise ValueError("序列数据必须是字典类型")
        
        if 'header' not in sequence_data or 'sequence' not in sequence_data:
            raise ValueError("序列数据必须包含'header'和'sequence'键")
        
        if not sequence_data['header'].strip():
            raise ValueError("header不能为空")
        
        if not sequence_data['sequence'].strip():
            raise ValueError("sequence不能为空")
        
        # 创建序列记录
        try:
            seq_obj = Seq(sequence_data['sequence'])
            header_parts = sequence_data['header'].split(maxsplit=1)
            seq_id = header_parts[0]
            description = header_parts[1] if len(header_parts) > 1 else ""
            
            seq_record = SeqRecord(seq_obj, id=seq_id, description=description)
            
            # 写入文件
            with open(output_path, "w") as output_handle:
                SeqIO.write(seq_record, output_handle, "fasta")
                
        except Exception as e:
            raise ValueError(f"创建FASTA文件时出错: {str(e)}")
    
    def run_deeptmhmm_prediction(self, fasta_path: str) -> str:
        """
        运行DeepTMHMM预测
        
        Args:
            fasta_path: 输入FASTA文件路径
            
        Returns:
            预测结果目录路径（biolib_results所在目录）
        """
        fasta_path = os.path.abspath(fasta_path)
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA文件不存在: {fasta_path}")
        
        # 获取FASTA文件所在目录（即基因专属目录）
        gene_dir = Path(fasta_path).parent
        
        print(f"运行DeepTMHMM预测...")
        print(f"输入FASTA: {fasta_path}")
        print(f"工作目录: {gene_dir}")
        
        try:
            # 在基因目录下运行DeepTMHMM
            result = subprocess.run([
                'biolib', 'run', 'DTU/DeepTMHMM', 
                '--fasta', fasta_path
            ], cwd=gene_dir, capture_output=True, text=True, check=True)
            
            # DeepTMHMM会在当前目录生成biolib_results文件夹
            biolib_results_dir = gene_dir / "biolib_results"
            
            if not biolib_results_dir.exists():
                raise FileNotFoundError(f"DeepTMHMM未生成结果目录: {biolib_results_dir}")
            
            print("DeepTMHMM预测完成")
            return str(gene_dir)  # 返回基因目录路径，因为biolib_results在其中
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"DeepTMHMM运行失败: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"运行DeepTMHMM时出错: {str(e)}")
    
    def parse_deeptmhmm_results(self, deeptmhmm_output_dir: str, gene_name: str) -> Dict[str, Any]:
        """
        解析DeepTMHMM预测结果并返回结构化数据
        
        Args:
            deeptmhmm_output_dir: DeepTMHMM输出目录（基因目录）
            gene_name: 基因名
            
        Returns:
            结构化的预测数据
        """
        deeptmhmm_dir = Path(deeptmhmm_output_dir)
        biolib_results_dir = deeptmhmm_dir / "biolib_results"
        
        # 检查必要的输出文件
        gff_file = biolib_results_dir / "TMRs.gff3"
        fasta_file = biolib_results_dir / "predicted_topologies.3line"
        
        if not gff_file.exists():
            raise FileNotFoundError(f"GFF文件不存在: {gff_file}")
        if not fasta_file.exists():
            raise FileNotFoundError(f"拓扑结构文件不存在: {fasta_file}")
        
        # 解析结果
        genes_info = self._parse_gff_content(str(gff_file))
        genes_seq_info = self._parse_fasta_like_content(str(fasta_file))
        
        if not genes_info:
            raise ValueError("无法从GFF文件中解析到基因信息")
        
        # 生成JSON数据
        json_data = self._generate_json_structure(genes_info, genes_seq_info, gene_name)
        self.parsed_data = json_data
        
        return json_data
    
    def _parse_gff_content(self, gff_file: str) -> Dict[str, Dict]:
        """
        解析GFF文件内容（内部方法）
        """
        genes = {}
        
        try:
            with open(gff_file, 'r', encoding='utf-8') as file:
                gff_text = file.read()
        except Exception as e:
            print(f"读取GFF文件时出错: {e}")
            return {}
        
        lines = gff_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line == '//':
                continue
                
            if line.startswith('#'):
                if 'Length:' in line:
                    match = re.search(r'#\s*(\w+)\s*Length:\s*(\d+)', line)
                    if match:
                        gene_name = match.group(1)
                        if gene_name not in genes:
                            genes[gene_name] = {'length': 0, 'num_tmrs': 0, 'regions': []}
                        genes[gene_name]['length'] = int(match.group(2))
                elif 'Number of predicted TMRs:' in line:
                    match = re.search(r'#\s*(\w+)\s*Number of predicted TMRs:\s*(\d+)', line)
                    if match:
                        gene_name = match.group(1)
                        if gene_name not in genes:
                            genes[gene_name] = {'length': 0, 'num_tmrs': 0, 'regions': []}
                        genes[gene_name]['num_tmrs'] = int(match.group(2))
            else:
                parts = line.split()
                if len(parts) >= 4:
                    gene_name = parts[0]
                    region_type = parts[1]
                    start, end = int(parts[2]), int(parts[3])
                    
                    if gene_name not in genes:
                        genes[gene_name] = {'length': 0, 'num_tmrs': 0, 'regions': []}
                    
                    genes[gene_name]['regions'].append({
                        'predicted region': region_type,
                        'start': start,
                        'end': end
                    })
        
        return genes
    
    def _parse_fasta_like_content(self, fasta_file: str) -> Dict[str, Dict]:
        """
        解析类似FASTA格式的文件（内部方法）
        """
        try:
            with open(fasta_file, 'r', encoding='utf-8') as file:
                fasta_text = file.read()
        except Exception as e:
            print(f"读取FASTA文件时出错: {e}")
            return {}
        
        clean_text = re.sub(r'<[^>]+>', '', fasta_text)
        lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
        
        genes_seq_info = {}
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            if line.startswith('>'):
                header_match = re.match(r'>(\w+)\s*\|\s*(\w+)', line)
                if header_match:
                    gene_name = header_match.group(1)
                    protein_type = header_match.group(2)
                    
                    if i + 2 < len(lines):
                        amino_acids = lines[i + 1].strip()
                        topology = lines[i + 2].strip()
                        
                        genes_seq_info[gene_name] = {
                            'predicted protein type': protein_type,
                            'amino acids': amino_acids,
                            'predicted amino acid topology type': topology
                        }
                        
                        i += 3
                        continue
            i += 1
        
        return genes_seq_info
    
    def _generate_json_structure(self, genes_info: Dict, genes_seq_info: Dict, 
                               gene_name: str) -> Dict[str, Any]:
        """
        生成JSON数据结构（内部方法）
        """
        gene_data = genes_info.get(gene_name, {})
        
        return {
            "gene_information": {
                "gene_name": gene_name
            },
            "DeepTMHMM": {
                "Region": gene_data.get('regions', []),
                "Protein Length": gene_data.get('length', np.nan),
                "Number of predicted TMRs": gene_data.get('num_tmrs', np.nan),
                "Sequence": genes_seq_info.get(gene_name, {
                    "predicted protein type": "",
                    "amino acids": "", 
                    "predicted amino acid topology type": ""
                })
            },
            "Protein description": self.PROTEIN_DESCRIPTIONS,
            "data_metadata": {
                "data_source": "DeepTMHMM for Transmembrane Topology Prediction and Classification",
                "processing_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            }
        }
    
    def save_to_file(self, filename: str, gene_name: Optional[str] = None) -> None:
        """
        将解析结果保存到文件
        
        Args:
            filename (str): 输出文件名
            gene_name (str, optional): 基因名，如果不提供则使用已解析的数据
        """
        if not self.parsed_data:
            if gene_name:
                # 如果没有解析数据但有基因名，可以尝试运行完整流程
                raise ValueError("没有解析数据，请先运行完整流程或提供基因名和UniProt JSON文件路径")
            else:
                raise ValueError("没有解析数据且未提供基因名")
        
        # 确保输出目录存在
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用json.dump保存数据[1](@ref)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.parsed_data, f, indent=2, ensure_ascii=False)
            print(f"💾 数据已保存到: {filename}")
        except Exception as e:
            raise IOError(f"保存文件时出错: {str(e)}")
    
    def run_complete_pipeline(self, uniprot_json_path: str, 
                            cleanup: bool = True) -> str:
        """
        运行完整的处理流程
        
        Args:
            uniprot_json_path: UniProt JSON文件路径
            cleanup: 是否清理临时文件
            
        Returns:
            最终生成的JSON文件路径
        """
        print("=" * 50)
        print("开始蛋白质拓扑结构预测流程")
        print("=" * 50)
        
        try:
            # 从文件名解析基因名
            gene_name = Path(uniprot_json_path).stem.split('.')[0]
            
            # 步骤1: 从UniProt JSON生成FASTA文件
            print(f"\n1. 从UniProt JSON生成FASTA序列（基因: {gene_name}）...")
            fasta_path = self.create_fasta_from_uniprot(uniprot_json_path, gene_name)
            
            # 步骤2: 运行DeepTMHMM预测
            print("\n2. 运行DeepTMHMM预测...")
            deeptmhmm_result_dir = self.run_deeptmhmm_prediction(fasta_path)
            
            # 步骤3: 解析结果生成结构化JSON
            print("\n3. 解析预测结果生成JSON数据...")
            json_data = self.parse_deeptmhmm_results(deeptmhmm_result_dir, gene_name)
            
            # 步骤4: 保存JSON文件到输出目录
            json_output_path = self.output_dir / f"{gene_name}.DeepTMHMM_data.json"
            self.save_to_file(str(json_output_path))
            
            # 步骤5: 清理临时文件（删除基因专属目录）
            if cleanup:
                print("\n4. 清理临时文件...")
                gene_dir = self.output_dir / gene_name
                if gene_dir.exists():
                    shutil.rmtree(gene_dir)
                    print(f"已清理临时目录: {gene_dir}")
            
            print("\n" + "=" * 50)
            print("流程完成!")
            print(f"最终结果: {json_output_path}")
            print("=" * 50)
            
            return json_data
            
        except Exception as e:
            print(f"\n流程执行失败: {str(e)}")
            raise
    
    def batch_process(self, uniprot_json_files: List[str], 
                    cleanup: bool = True) -> List[str]:
        """
        批量处理多个UniProt JSON文件
        
        Args:
            uniprot_json_files: UniProt JSON文件路径列表
            cleanup: 是否清理临时文件
            
        Returns:
            生成的JSON文件路径列表
        """
        results = []
        
        for i, json_file in enumerate(uniprot_json_files, 1):
            print(f"\n处理文件 {i}/{len(uniprot_json_files)}: {json_file}")
            try:
                result_path = self.run_complete_pipeline(json_file, cleanup)
                results.append(result_path)
            except Exception as e:
                print(f"处理文件失败 {json_file}: {e}")
                continue
        
        return results
