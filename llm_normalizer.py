"""
LLM-based spreadsheet normalizer that generates transformation instructions.
"""

import os
import json
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from utils import load_spreadsheet, save_normalized_spreadsheet

# Load environment variables
load_dotenv()


class LLMNormalizer:
    """
    Handles LLM interaction for spreadsheet normalization.
    Generates transformation instructions based on compressed spreadsheet data.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM normalizer.
        
        Args:
            api_key: OpenAI API key (optional, will use environment variable if not provided)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4"  # Using GPT-4 for better code generation
    
    def generate_normalization_instructions(self, compressed_data: Dict[str, Any]) -> str:
        """
        Generate normalization instructions using LLM.
        
        Args:
            compressed_data: Compressed spreadsheet data from SpreadsheetCompressor
            
        Returns:
            Generated transformation instructions/code
        """
        # Create prompt for the LLM
        prompt = self._create_normalization_prompt(compressed_data)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert data analyst and Python programmer specializing in spreadsheet normalization. 
                        Your task is to analyze compressed spreadsheet data and generate Python code that transforms the original spreadsheet 
                        into a normalized, standardized format. The code should:
                        1. Clean and standardize data formats
                        2. Handle missing values appropriately
                        3. Normalize column names and data types
                        4. Remove duplicates and inconsistencies
                        5. Apply best practices for data structure
                        
                        Return ONLY the Python code that performs the transformation. The code should be complete and executable."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.1  # Low temperature for consistent code generation
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"Error generating normalization instructions: {str(e)}")
    
    def _create_normalization_prompt(self, compressed_data: Dict[str, Any]) -> str:
        """
        Create a comprehensive prompt for the LLM based on compressed data.
        """
        prompt = "ANALYZE THE FOLLOWING COMPRESSED SPREADSHEET DATA AND GENERATE NORMALIZATION CODE:\n\n"
        
        # Add original shape information
        prompt += f"ORIGINAL SPREADSHEET SHAPE: {compressed_data['original_shape']}\n\n"
        
        # Add structural anchor information
        anchors = compressed_data['structural_anchors']
        prompt += "STRUCTURAL ANALYSIS:\n"
        prompt += f"- Anchor rows (heterogeneous): {anchors['anchor_rows']}\n"
        prompt += f"- Anchor columns (heterogeneous): {anchors['anchor_columns']}\n"
        prompt += f"- Compression ratio: {anchors['compression_ratio']:.2%}\n\n"
        
        # Add data type aggregation
        agg_data = compressed_data['data_aggregation']
        prompt += "DATA TYPE ANALYSIS:\n"
        for data_type, stats in agg_data['aggregation_stats'].items():
            prompt += f"- {data_type}: {stats['count']} cells, {stats['unique_values']} unique values\n"
            prompt += f"  Sample values: {stats['sample_values'][:3]}\n"
        prompt += "\n"
        
        # Add inverted index for value patterns
        prompt += "VALUE PATTERNS (Most common values):\n"
        inverted = compressed_data['inverted_index']
        sorted_values = sorted(inverted.items(), key=lambda x: len(x[1]), reverse=True)
        for value, addresses in sorted_values[:15]:  # Top 15 most common values
            prompt += f"- '{value}': appears {len(addresses)} times\n"
        prompt += "\n"
        
        # Add specific normalization requirements
        prompt += "NORMALIZATION REQUIREMENTS:\n"
        prompt += "1. Standardize column names (remove spaces, special characters, make lowercase)\n"
        prompt += "2. Convert data types appropriately (dates, numbers, text)\n"
        prompt += "3. Handle missing values (fill with appropriate defaults or remove rows)\n"
        prompt += "4. Remove duplicate rows\n"
        prompt += "5. Normalize text data (trim whitespace, standardize case)\n"
        prompt += "6. Ensure consistent date formats\n"
        prompt += "7. Convert currency values to numeric format\n"
        prompt += "8. Create a clean, structured output\n\n"
        
        prompt += "GENERATE COMPLETE PYTHON CODE THAT:\n"
        prompt += "- Takes a pandas DataFrame as input\n"
        prompt += "- Applies all necessary transformations\n"
        prompt += "- Returns a normalized DataFrame\n"
        prompt += "- Includes error handling and logging\n"
        prompt += "- Uses pandas and numpy for data manipulation\n"
        
        return prompt
    
    def execute_normalization(self, input_file: str, output_file: str, compressed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the normalization process using LLM-generated instructions.
        
        Args:
            input_file: Path to input Excel file
            output_file: Path to output Excel file
            compressed_data: Compressed spreadsheet data
            
        Returns:
            Dictionary with execution results and metadata
        """
        try:
            # Generate normalization instructions
            instructions = self.generate_normalization_instructions(compressed_data)
            
            # Load original spreadsheet
            df, formatting_info = load_spreadsheet(input_file)
            
            # Execute the generated code
            normalized_df = self._execute_normalization_code(instructions, df)
            
            # Save normalized spreadsheet
            save_normalized_spreadsheet(normalized_df, output_file, formatting_info)
            
            return {
                'success': True,
                'instructions': instructions,
                'original_shape': df.shape,
                'normalized_shape': normalized_df.shape,
                'output_file': output_file
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'instructions': instructions if 'instructions' in locals() else None
            }
    
    def _execute_normalization_code(self, code: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the generated normalization code safely.
        
        Args:
            code: Generated Python code
            df: Input DataFrame
            
        Returns:
            Normalized DataFrame
        """
        # Create a safe execution environment
        local_vars = {
            'df': df.copy(),
            'pd': pd,
            'numpy': __import__('numpy'),
            'np': __import__('numpy'),
            're': __import__('re'),
            'datetime': __import__('datetime'),
            'logging': __import__('logging')
        }
        
        # Add safety wrapper
        safe_code = f"""
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    {code}
    
    # Ensure we return a DataFrame
    if 'normalized_df' in locals():
        result = normalized_df
    elif 'df' in locals():
        result = df
    else:
        raise ValueError("No DataFrame variable found in generated code")
        
    logger.info(f"Normalization completed. Shape: {{result.shape}}")
    
except Exception as e:
    logger.error(f"Error in normalization: {{e}}")
    raise
"""
        
        # Execute the code
        exec(safe_code, globals(), local_vars)
        
        return local_vars['result']
    
    def create_normalization_report(self, compressed_data: Dict[str, Any], execution_result: Dict[str, Any]) -> str:
        """
        Create a comprehensive report of the normalization process.
        """
        report = "SPREADSHEET NORMALIZATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Compression statistics
        report += "COMPRESSION STATISTICS:\n"
        stats = compressed_data['compression_stats']
        report += f"Original size: {stats['original_size']} cells\n"
        report += f"Structural compression: {stats['structural_compression']:.2%}\n"
        report += f"Data aggregation compression: {stats['aggregation_compression']:.2%}\n\n"
        
        # Data type analysis
        report += "DATA TYPE ANALYSIS:\n"
        agg_data = compressed_data['data_aggregation']
        for data_type, stats in agg_data['aggregation_stats'].items():
            report += f"- {data_type}: {stats['count']} cells\n"
        report += "\n"
        
        # Execution results
        report += "NORMALIZATION RESULTS:\n"
        if execution_result['success']:
            report += f"Status: SUCCESS\n"
            report += f"Original shape: {execution_result['original_shape']}\n"
            report += f"Normalized shape: {execution_result['normalized_shape']}\n"
            report += f"Output file: {execution_result['output_file']}\n"
        else:
            report += f"Status: FAILED\n"
            report += f"Error: {execution_result['error']}\n"
        
        return report
