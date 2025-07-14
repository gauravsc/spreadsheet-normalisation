"""
Spreadsheet compression module implementing the four techniques from SPREADSHEETLLM paper.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
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
        Apply compression techniques sequentially as described in the paper:
        1. Structural-anchor-based Extraction (creates smaller 24×8 sheet)
        2. Inverted-index Translation (removes empty cells)
        3. Data-format-aware Aggregation (achieves compact representation)
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary containing compressed representations and metadata
        """
        # Load spreadsheet
        df, formatting_info = load_spreadsheet(file_path)
        original_shape = df.shape
        
        # Step 1: Structural-anchor-based Extraction
        # Creates a smaller sheet by extracting only anchor cells
        structural_result = self._structural_anchor_extraction(df)
        anchor_df = self._create_anchor_dataframe(df, structural_result)
        
        # Step 2: Inverted-index Translation on anchor data
        # Removes empty cells and creates value-to-address mapping
        inverted_result = self._inverted_index_translation_sequential(anchor_df, structural_result)
        
        # Step 3: Data-format-aware Aggregation on inverted data
        # Groups remaining cells by data format for compact representation
        aggregation_result = self._data_format_aggregation_sequential(inverted_result, formatting_info)
        
        # Calculate sequential compression ratios
        original_size = original_shape[0] * original_shape[1]
        anchor_size = structural_result['non_empty_cells_count']  # Use non-empty cells count
        inverted_size = len(inverted_result)
        final_size = len(aggregation_result['type_groups'])
        
        self.compression_stats = {
            'original_size': original_size,
            'original_shape': original_shape,
            'step1_anchor_size': anchor_size,
            'step1_compression': anchor_size / original_size if original_size > 0 else 0,
            'step2_inverted_size': inverted_size,
            'step2_compression': inverted_size / anchor_size if anchor_size > 0 else 0,
            'step3_final_size': final_size,
            'step3_compression': final_size / inverted_size if inverted_size > 0 else 0,
            'total_compression': final_size / original_size if original_size > 0 else 0
        }
        
        return {
            'structural_anchors': structural_result,
            'anchor_dataframe': anchor_df,
            'inverted_index': inverted_result,
            'data_aggregation': aggregation_result,
            'compression_stats': self.compression_stats,
            'original_shape': original_shape,
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
        print(f"Anchor rows: {anchor_rows}")
        print(f"Anchor columns: {anchor_columns}")
        
        # Create address mapping for the filtered anchors
        address_map = create_address_map(anchor_rows, anchor_columns)
        
        # Extract skeleton (filtered anchor cells only, excluding empty cells)
        skeleton = []
        non_empty_cells = []
        for i, row_idx in enumerate(anchor_rows):
            skeleton_row = []
            for j, col_idx in enumerate(anchor_columns):
                cell_value = df.iloc[row_idx, col_idx]
                cell_address = get_cell_address(row_idx, col_idx)
                
                # Skip empty cells in skeleton
                if (cell_value is None or 
                    pd.isna(cell_value) or 
                    (isinstance(cell_value, np.floating) and np.isnan(cell_value)) or
                    str(cell_value).strip() == ''):
                    continue
                
                cell_info = {
                    'address': cell_address,
                    'value': cell_value,
                    'data_type': detect_data_type(cell_value)
                }
                skeleton_row.append(cell_info)
                non_empty_cells.append(cell_info)
            
            if skeleton_row:  # Only add rows that have non-empty cells
                skeleton.append(skeleton_row)
        
        # Calculate compression ratio based on k-neighborhood filtering and empty cell removal
        total_cells = len(df) * len(df.columns)
        anchor_cells = len(anchor_rows) * len(anchor_columns)
        non_empty_cells_count = len(non_empty_cells)
        compression_ratio = non_empty_cells_count / total_cells if total_cells > 0 else 0
        
        return {
            'anchor_rows': anchor_rows,
            'anchor_columns': anchor_columns,
            'address_map': address_map,
            'skeleton': skeleton,
            'non_empty_cells': non_empty_cells,
            'compression_ratio': compression_ratio,
            'k_neighborhood': 3,
            'total_cells': total_cells,
            'anchor_cells': anchor_cells,
            'non_empty_cells_count': non_empty_cells_count
        }
    
    def _create_anchor_dataframe(self, df: pd.DataFrame, structural_result: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a smaller DataFrame containing only the non-empty anchor cells.
        This represents the "24×8 sheet" mentioned in the paper, with empty cells removed.
        """
        # Use the non-empty cells data from structural result
        non_empty_cells = structural_result.get('non_empty_cells', [])
        
        if not non_empty_cells:
            # Return empty DataFrame if no non-empty cells
            return pd.DataFrame()
        
        # Create DataFrame from non-empty cells
        data = []
        for cell in non_empty_cells:
            data.append({
                'address': cell['address'],
                'value': cell['value'],
                'data_type': cell['data_type']
            })
        
        return pd.DataFrame(data)
    
    def _inverted_index_translation_sequential(self, anchor_df: pd.DataFrame, structural_result: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Technique 2: Inverted-index Translation applied to anchor data.
        
        Creates value-to-address mapping and removes empty cells.
        """
        inverted_index = defaultdict(list)
        
        # Use the non-empty cells data directly
        non_empty_cells = structural_result.get('non_empty_cells', [])
        
        for cell in non_empty_cells:
            cell_value = cell['value']
            cell_address = cell['address']
            
            # Convert value to string for dictionary key
            value_key = str(cell_value)
            
            # Add address to the list for this value
            inverted_index[value_key].append(cell_address)
        
        return dict(inverted_index)
    
    def _data_format_aggregation_sequential(self, inverted_result: Dict[str, List[str]], formatting_info: Dict) -> Dict[str, Any]:
        """
        Technique 3: Data-format-aware Aggregation applied to inverted index data.
        
        Groups remaining cells by data type for compact representation.
        """
        # Extract NFS from formatting info
        nfs_by_cell = {}
        for address, format_data in formatting_info.items():
            nfs_by_cell[address] = format_data.get('number_format', 'General')
        
        # Group cells by data type from inverted index
        type_groups = defaultdict(list)
        
        for value, addresses in inverted_result.items():
            # Use the first address to get formatting info (assuming same format for same value)
            if addresses:
                sample_address = addresses[0]
                nfs = nfs_by_cell.get(sample_address, 'General')
                
                # Detect data type
                data_type = detect_data_type(value, nfs)
                
                # Add to type group
                type_groups[data_type].append({
                    'value': value,
                    'addresses': addresses,
                    'count': len(addresses),
                    'nfs': nfs
                })
        
        # Calculate aggregation statistics
        aggregation_stats = {}
        for data_type, cells in type_groups.items():
            total_cells = sum(cell['count'] for cell in cells)
            unique_values = len(cells)
            sample_values = [cell['value'] for cell in cells[:5]]  # First 5 values as sample
            
            aggregation_stats[data_type] = {
                'count': total_cells,
                'unique_values': unique_values,
                'sample_values': sample_values
            }
        
        return {
            'type_groups': dict(type_groups),
            'aggregation_stats': aggregation_stats,
            'total_cells': sum(len(cells) for cells in type_groups.values())
        }
    
    def get_compression_summary(self) -> str:
        """
        Generate a summary of sequential compression results.
        """
        if not self.compression_stats:
            return "No compression data available."
        
        summary = "Sequential Compression Summary:\n"
        summary += f"Original size: {self.compression_stats['original_size']} cells ({self.compression_stats['original_shape'][0]}×{self.compression_stats['original_shape'][1]})\n"
        summary += f"Step 1 - Anchor extraction: {self.compression_stats['step1_anchor_size']} cells ({self.compression_stats['step1_compression']:.2%} of original)\n"
        summary += f"Step 2 - Inverted index: {self.compression_stats['step2_inverted_size']} unique values ({self.compression_stats['step2_compression']:.2%} of anchor cells)\n"
        summary += f"Step 3 - Data aggregation: {self.compression_stats['step3_final_size']} data type groups ({self.compression_stats['step3_compression']:.2%} of inverted data)\n"
        summary += f"Total compression: {self.compression_stats['total_compression']:.2%} of original size\n"
        
        return summary
    
    def create_llm_prompt(self, compressed_data: Dict[str, Any]) -> str:
        """
        Create a prompt for the LLM based on sequentially compressed spreadsheet data.
        """
        prompt = "SEQUENTIAL SPREADSHEET COMPRESSION DATA:\n\n"
        
        # Add original shape information
        original_shape = compressed_data['original_shape']
        prompt += f"ORIGINAL SPREADSHEET: {original_shape[0]}×{original_shape[1]} cells\n\n"
        
        # Step 1: Structural anchors
        prompt += "STEP 1 - STRUCTURAL ANCHOR EXTRACTION:\n"
        anchors = compressed_data['structural_anchors']
        anchor_df = compressed_data['anchor_dataframe']
        prompt += f"Extracted anchor cells: {len(anchors['anchor_rows'])}×{len(anchors['anchor_columns'])} = {anchors['anchor_cells']} cells\n"
        prompt += f"Non-empty cells after filtering: {anchors['non_empty_cells_count']} cells\n"
        prompt += f"Anchor rows: {anchors['anchor_rows'][:10]}{'...' if len(anchors['anchor_rows']) > 10 else ''}\n"
        prompt += f"Anchor columns: {anchors['anchor_columns'][:10]}{'...' if len(anchors['anchor_columns']) > 10 else ''}\n"
        prompt += f"K-neighborhood filtering: {anchors.get('k_neighborhood', 3)}\n"
        prompt += f"Compression ratio: {anchors['compression_ratio']:.2%}\n\n"
        
        # Step 2: Inverted index
        prompt += "STEP 2 - INVERTED INDEX TRANSLATION:\n"
        inverted = compressed_data['inverted_index']
        prompt += f"Unique values after removing empty cells: {len(inverted)}\n"
        # Show top values by frequency
        sorted_values = sorted(inverted.items(), key=lambda x: len(x[1]), reverse=True)
        prompt += "Most frequent values:\n"
        for value, addresses in sorted_values[:5]:
            prompt += f"  '{value}': {len(addresses)} occurrences\n"
        prompt += "\n"
        
        # Step 3: Data aggregation
        prompt += "STEP 3 - DATA FORMAT AGGREGATION:\n"
        agg_data = compressed_data['data_aggregation']
        prompt += f"Data type groups: {len(agg_data['type_groups'])}\n"
        for data_type, stats in agg_data['aggregation_stats'].items():
            prompt += f"{data_type}: {stats['count']} total cells, {stats['unique_values']} unique values\n"
            prompt += f"  Sample values: {stats['sample_values'][:3]}\n"
        
        # Add compression statistics
        stats = compressed_data['compression_stats']
        prompt += f"\nFINAL COMPRESSION: {stats['total_compression']:.2%} of original size\n"
        
        return prompt 