#!/usr/bin/env python3
"""
Demo script for the Spreadsheet Normalization Tool.
Creates sample data and demonstrates the compression and normalization process.
"""

import os
import sys
from spreadsheet_compressor import SpreadsheetCompressor
from llm_normalizer import LLMNormalizer
from utils import load_spreadsheet, save_normalized_spreadsheet
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def create_demo_data():
    """Create a simple demo spreadsheet for testing."""
    
    # Create sample data with various issues
    data = {
        'customer_id': [f"CUST-{i:03d}" for i in range(1, 21)],
        'customer_name': [f"Customer {i}" for i in range(1, 21)],
        'email': [f"customer{i}@example.com" for i in range(1, 21)],
        'phone': [f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}" for _ in range(20)],
        'registration_date': [(datetime.now() - timedelta(days=random.randint(1, 365))) for _ in range(20)],
        'total_spent': [round(random.uniform(100, 5000), 2) for _ in range(20)],
        'status': random.choices(['Active', 'Inactive', 'Pending'], k=20)
    }
    
    df = pd.DataFrame(data)
    
    # Add some formatting issues
    df.loc[5, 'customer_name'] = '  customer 5  '  # Extra spaces
    df.loc[10, 'email'] = 'customer10@EXAMPLE.COM'  # Mixed case
    df.loc[15, 'phone'] = '(555) 123-4567'  # Different format
    df.loc[3, 'total_spent'] = '$1,234.56'  # Currency format
    
    # Add some missing values
    df.loc[7, 'email'] = None
    df.loc[12, 'phone'] = ''
    
    # Save to Excel
    df.to_excel('demo_data.xlsx', index=False)
    print("Demo data created: demo_data.xlsx")
    return 'demo_data.xlsx'


def run_demo():
    """Run the complete demo of compression and normalization."""
    
    print("=== SPREADSHEET NORMALIZATION TOOL DEMO ===\n")
    
    # Step 1: Create demo data
    print("1. Creating demo spreadsheet...")
    input_file = create_demo_data()
    
    # Step 2: Compress the spreadsheet
    print("\n2. Compressing spreadsheet using SPREADSHEETLLM techniques...")
    compressor = SpreadsheetCompressor()
    compressed_data = compressor.compress_spreadsheet(input_file)
    
    # Display compression results
    print("\nCompression Results:")
    print(compressor.get_compression_summary())
    
    # Step 3: Generate LLM prompt
    print("\n3. Creating LLM prompt from compressed data...")
    prompt = compressor.create_llm_prompt(compressed_data)
    print("Prompt created successfully!")
    print(f"Prompt length: {len(prompt)} characters")
    
    # Step 4: Test LLM normalization (if API key available)
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("\n4. Testing LLM normalization...")
        try:
            normalizer = LLMNormalizer(api_key=api_key)
            
            # Generate instructions
            instructions = normalizer.generate_normalization_instructions(compressed_data)
            print("✓ Normalization instructions generated!")
            print(f"Instructions length: {len(instructions)} characters")
            
            # Execute normalization
            result = normalizer.execute_normalization(input_file, 'normalized_demo.xlsx', compressed_data)
            
            if result['success']:
                print("✓ Normalization executed successfully!")
                print(f"Original shape: {result['original_shape']}")
                print(f"Normalized shape: {result['normalized_shape']}")
                print(f"Output file: {result['output_file']}")
            else:
                print(f"✗ Normalization failed: {result['error']}")
                
        except Exception as e:
            print(f"✗ LLM normalization error: {str(e)}")
            print("Note: Make sure you have a valid OpenAI API key set in OPENAI_API_KEY environment variable")
    else:
        print("\n4. Skipping LLM normalization (no API key found)")
        print("To test LLM normalization, set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
    
    # Step 5: Show data analysis
    print("\n5. Data Analysis:")
    df, _ = load_spreadsheet(input_file)
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types: {df.dtypes.to_dict()}")
    
    # Show sample of compressed data
    print("\n6. Compressed Data Sample:")
    print("Structural anchors:")
    anchors = compressed_data['structural_anchors']
    print(f"  - Anchor rows: {anchors['anchor_rows']}")
    print(f"  - Anchor columns: {anchors['anchor_columns']}")
    
    print("\nData type aggregation:")
    agg_data = compressed_data['data_aggregation']
    for data_type, stats in agg_data['aggregation_stats'].items():
        print(f"  - {data_type}: {stats['count']} cells")
    
    print("\nMost common values:")
    inverted = compressed_data['inverted_index']
    sorted_values = sorted(inverted.items(), key=lambda x: len(x[1]), reverse=True)
    for value, addresses in sorted_values[:5]:
        print(f"  - '{value}': {len(addresses)} occurrences")
    
    print("\n=== DEMO COMPLETED ===")
    print("Files created:")
    print(f"  - {input_file} (original data)")
    if api_key and 'result' in locals() and result.get('success'):
        print(f"  - {result['output_file']} (normalized data)")


if __name__ == "__main__":
    run_demo() 