"""
Spreadsheet compression module implementing the four techniques from SPREADSHEETLLM paper.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from collections import defaultdict
from utils import (
    load_spreadsheet, detect_data_type, find_structural_anchors,
    create_address_map, format_cell_for_markdown, get_cell_address
)


class SpreadsheetCompressor:
    """
    Implements the three compression techniques from SPREADSHEETLLM paper:
    1. Structural-anchor-based Extraction
    2. Inverted-index Translation
    3. Data-format-aware Aggregation
    """
    
    def __init__(self):
        self.compression_stats = {}
    
    def compress_spreadsheet(self, file_path: str) -> Dict[str, Any]:
        """
        Apply all four compression techniques to the spreadsheet.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary containing compressed representations and metadata
        """
        # Load spreadsheet
        df, formatting_info = load_spreadsheet(file_path)
        
        # Apply compression techniques
        structural_anchors = self._structural_anchor_extraction(df)
        inverted_index = self._inverted_index_translation(df, formatting_info)
        data_aggregation = self._data_format_aggregation(df, formatting_info)
        
        # Calculate compression ratios
        original_size = len(df) * len(df.columns)
        self.compression_stats = {
            'original_size': original_size,
            'structural_compression': len(structural_anchors['skeleton']) / original_size,
            'inverted_compression': len(inverted_index) / original_size,
            'aggregation_compression': len(data_aggregation) / original_size
        }
        
        return {
            'structural_anchors': structural_anchors,
            'inverted_index': inverted_index,
            'data_aggregation': data_aggregation,
            'compression_stats': self.compression_stats,
            'original_shape': df.shape,
            'formatting_info': formatting_info
        }
    

    
    def _structural_anchor_extraction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Technique 1: Structural-anchor-based Extraction.
        
        Identifies heterogeneous rows/columns as anchors and creates skeleton.
        Uses k-neighborhood filtering to discard rows/columns more than k cells away from any anchor.
        """
        # Find structural anchors with k-neighborhood filtering
        anchor_rows, anchor_columns = find_structural_anchors(df, k=3)
        
        # Create address mapping for the filtered anchors
        address_map = create_address_map(anchor_rows, anchor_columns)
        
        # Extract skeleton (filtered anchor cells only)
        skeleton = []
        for i, row_idx in enumerate(anchor_rows):
            skeleton_row = []
            for j, col_idx in enumerate(anchor_columns):
                cell_value = df.iloc[row_idx, col_idx]
                cell_address = get_cell_address(row_idx, col_idx)
                skeleton_row.append({
                    'address': cell_address,
                    'value': cell_value,
                    'data_type': detect_data_type(cell_value)
                })
            skeleton.append(skeleton_row)
        
        # Calculate compression ratio based on k-neighborhood filtering
        total_cells = len(df) * len(df.columns)
        filtered_cells = len(anchor_rows) * len(anchor_columns)
        compression_ratio = filtered_cells / total_cells if total_cells > 0 else 0
        
        return {
            'anchor_rows': anchor_rows,
            'anchor_columns': anchor_columns,
            'address_map': address_map,
            'skeleton': skeleton,
            'compression_ratio': compression_ratio,
            'k_neighborhood': 3,
            'total_cells': total_cells,
            'filtered_cells': filtered_cells
        }
    
    def _inverted_index_translation(self, df: pd.DataFrame, formatting_info: Dict) -> Dict[str, List[str]]:
        """
        Technique 3: Inverted-index Translation.
        
        Tt = invert(T) := {Value : Address or Address_Region, ...}
        """
        inverted_index = defaultdict(list)
        
        for i in range(len(df)):
            for j in range(len(df.columns)):
                cell_value = df.iloc[i, j]
                cell_address = get_cell_address(i, j)
                
                # Skip empty cells
                if cell_value is None or pd.isna(cell_value):
                    continue
                
                # Convert value to string for dictionary key
                value_key = str(cell_value)
                
                # Add address to the list for this value
                inverted_index[value_key].append(cell_address)
        
        # Convert defaultdict to regular dict
        return dict(inverted_index)
    
    def _data_format_aggregation(self, df: pd.DataFrame, formatting_info: Dict) -> Dict[str, Any]:
        """
        Technique 4: Data-format-aware Aggregation.
        
        Groups cells by data type using Number Format Strings (NFS) and rule-based detection.
        """
        # Extract NFS from formatting info
        nfs_by_cell = {}
        for address, format_data in formatting_info.items():
            nfs_by_cell[address] = format_data.get('number_format', 'General')
        
        # Group cells by data type
        type_groups = defaultdict(list)
        
        for i in range(len(df)):
            for j in range(len(df.columns)):
                cell_value = df.iloc[i, j]
                cell_address = get_cell_address(i, j)
                
                # Get NFS for this cell
                nfs = nfs_by_cell.get(cell_address, 'General')
                
                # Detect data type
                data_type = detect_data_type(cell_value, nfs)
                
                # Add to type group
                type_groups[data_type].append({
                    'address': cell_address,
                    'value': cell_value,
                    'nfs': nfs
                })
        
        # Calculate aggregation statistics
        aggregation_stats = {}
        for data_type, cells in type_groups.items():
            aggregation_stats[data_type] = {
                'count': len(cells),
                'unique_values': len(set(str(cell['value']) for cell in cells)),
                'sample_values': [cell['value'] for cell in cells[:5]]  # First 5 values as sample
            }
        
        return {
            'type_groups': dict(type_groups),
            'aggregation_stats': aggregation_stats,
            'total_cells': sum(len(cells) for cells in type_groups.values())
        }
    
    def get_compression_summary(self) -> str:
        """
        Generate a summary of compression results.
        """
        if not self.compression_stats:
            return "No compression data available."
        
        summary = "Compression Summary:\n"
        summary += f"Original size: {self.compression_stats['original_size']} cells\n"
        summary += f"Structural anchor compression: {self.compression_stats['structural_compression']:.2%}\n"
        summary += f"Inverted index compression: {self.compression_stats['inverted_compression']:.2%}\n"
        summary += f"Data aggregation compression: {self.compression_stats['aggregation_compression']:.2%}\n"
        
        return summary
    
    def create_llm_prompt(self, compressed_data: Dict[str, Any]) -> str:
        """
        Create a prompt for the LLM based on compressed spreadsheet data.
        """
        prompt = "SPREADSHEET COMPRESSION DATA:\n\n"
        
        # Add structural information
        prompt += "STRUCTURAL ANCHORS (k-neighborhood filtered):\n"
        anchors = compressed_data['structural_anchors']
        prompt += f"Anchor rows: {anchors['anchor_rows']}\n"
        prompt += f"Anchor columns: {anchors['anchor_columns']}\n"
        prompt += f"K-neighborhood: {anchors.get('k_neighborhood', 3)}\n"
        prompt += f"Total cells: {anchors.get('total_cells', 0)}\n"
        prompt += f"Filtered cells: {anchors.get('filtered_cells', 0)}\n"
        prompt += f"Compression ratio: {anchors['compression_ratio']:.2%}\n\n"
        
        # Add data type aggregation
        prompt += "DATA TYPE AGGREGATION:\n"
        agg_data = compressed_data['data_aggregation']
        for data_type, stats in agg_data['aggregation_stats'].items():
            prompt += f"{data_type}: {stats['count']} cells, {stats['unique_values']} unique values\n"
            prompt += f"Sample values: {stats['sample_values'][:3]}\n"
        prompt += "\n"
        
        # Add inverted index (limited to avoid token overflow)
        prompt += "INVERTED INDEX (Top 10 most common values):\n"
        inverted = compressed_data['inverted_index']
        sorted_values = sorted(inverted.items(), key=lambda x: len(x[1]), reverse=True)
        for value, addresses in sorted_values[:10]:
            prompt += f"'{value}': {len(addresses)} occurrences at {addresses[:3]}{'...' if len(addresses) > 3 else ''}\n"
        
        return prompt 