#!/usr/bin/env python3
"""
Test script for the Spreadsheet Normalization Tool.
Tests all components without requiring external dependencies.
"""

import os
import sys
import tempfile
import pandas as pd
from datetime import datetime, timedelta
import random


def create_test_data():
    """Create a simple test DataFrame."""
    data = {
        'id': [i for i in range(1, 11)],
        'name': [f"Test {i}" for i in range(1, 11)],
        'email': [f"test{i}@example.com" for i in range(1, 11)],
        'amount': [round(random.uniform(10, 1000), 2) for _ in range(10)],
        'date': [(datetime.now() - timedelta(days=random.randint(1, 30))) for _ in range(10)],
        'status': random.choices(['Active', 'Inactive'], k=10)
    }
    
    df = pd.DataFrame(data)
    
    # Add some issues
    df.loc[2, 'name'] = '  test 2  '  # Extra spaces
    df.loc[5, 'email'] = 'TEST5@EXAMPLE.COM'  # Mixed case
    df.loc[7, 'amount'] = '$123.45'  # Currency format
    
    return df


def test_utils():
    """Test utility functions."""
    print("Testing utility functions...")
    
    # Create test data
    df = create_test_data()
    
    # Test data type detection
    from utils import detect_data_type
    
    test_cases = [
        (123, 'Integer'),
        (123.45, 'Float'),
        ('test@example.com', 'Email'),
        ('2023-01-01', 'Date'),
        ('$123.45', 'Currency'),
        ('15%', 'Percentage'),
        ('test', 'Text'),
        (None, 'Empty')
    ]
    
    for value, expected in test_cases:
        result = detect_data_type(value)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {value} -> {result} (expected: {expected})")
    
    # Test cell address conversion
    from utils import get_cell_address
    
    test_addresses = [
        (0, 0, 'A1'),
        (0, 25, 'Z1'),
        (0, 26, 'AA1'),
        (9, 0, 'A10')
    ]
    
    for row, col, expected in test_addresses:
        result = get_cell_address(row, col)
        status = "✓" if result == expected else "✗"
        print(f"  {status} ({row}, {col}) -> {result} (expected: {expected})")
    
    print("✓ Utility functions test completed\n")


def test_compression():
    """Test compression techniques."""
    print("Testing compression techniques...")
    
    # Create test file
    import os
    # df = create_test_data()
    test_file = os.path.join(os.path.dirname(__file__), "pnl.xlsx")
    # df.to_excel(test_file, index=False)
    
    try:
        from spreadsheet_compressor import SpreadsheetCompressor
        
        compressor = SpreadsheetCompressor()
        compressed_data = compressor.compress_spreadsheet(test_file)
        print("Compressed data output:")
        import pprint
        pprint.pprint(compressed_data)
        
        # Check compression results
        required_keys = ['structural_anchors', 'inverted_index', 'data_aggregation']
        for key in required_keys:
            if key in compressed_data:
                print(f"  ✓ {key} compression successful")
            else:
                print(f"  ✗ {key} compression missing")
        
        # Check compression stats
        stats = compressed_data.get('compression_stats', {})
        if stats:
            print(f"  ✓ Compression stats calculated: {len(stats)} metrics")
        else:
            print("  ✗ Compression stats missing")
        
        # Test prompt generation
        prompt = compressor.create_llm_prompt(compressed_data)
        if len(prompt) > 100:
            print(f"  ✓ LLM prompt generated: {len(prompt)} characters")
        else:
            print("  ✗ LLM prompt too short")
        
        print("✓ Compression test completed\n")
        
    except Exception as e:
        print(f"  ✗ Compression test failed: {str(e)}\n")
    
    finally:
        pass
        # Cleanup
        # if os.path.exists(test_file):
        #     os.remove(test_file)


def test_llm_normalizer():
    """Test LLM normalizer (without API calls)."""
    print("Testing LLM normalizer...")
    
    # Create mock compressed data
    mock_compressed_data = {
        'original_shape': (10, 6),
        'structural_anchors': {
            'anchor_rows': [0, 9],
            'anchor_columns': [0, 1, 5],
            'compression_ratio': 0.3
        },
        'data_aggregation': {
            'aggregation_stats': {
                'Text': {'count': 30, 'unique_values': 15},
                'Integer': {'count': 10, 'unique_values': 10},
                'Float': {'count': 10, 'unique_values': 10},
                'Date': {'count': 10, 'unique_values': 8}
            }
        },
        'inverted_index': {
            'test1': ['A1', 'B1'],
            'test2': ['A2', 'B2'],
            'Active': ['F1', 'F2', 'F3']
        }
    }
    
    try:
        from llm_normalizer import LLMNormalizer
        
        # Test without API key
        try:
            normalizer = LLMNormalizer()
            print("  ✗ Should have failed without API key")
        except ValueError:
            print("  ✓ Correctly failed without API key")
        
        # Test prompt creation
        normalizer = LLMNormalizer(api_key="dummy_key")
        prompt = normalizer._create_normalization_prompt(mock_compressed_data)
        
        if len(prompt) > 200:
            print("  ✓ Normalization prompt created successfully")
        else:
            print("  ✗ Normalization prompt too short")
        
        print("✓ LLM normalizer test completed\n")
        
    except Exception as e:
        print(f"  ✗ LLM normalizer test failed: {str(e)}\n")


def test_integration():
    """Test the complete integration workflow."""
    print("Testing complete integration workflow...")
    
    # Create test data
    df = create_test_data()
    test_file = "integration_test.xlsx"
    df.to_excel(test_file, index=False)
    
    try:
        from spreadsheet_compressor import SpreadsheetCompressor
        from utils import load_spreadsheet
        
        # Test complete workflow
        compressor = SpreadsheetCompressor()
        compressed_data = compressor.compress_spreadsheet(test_file)
        
        # Verify data loading
        loaded_df, formatting_info = load_spreadsheet(test_file)
        if loaded_df.shape == df.shape:
            print("  ✓ Data loading successful")
        else:
            print("  ✗ Data loading failed")
        
        # Verify compression
        if compressed_data['original_shape'] == df.shape:
            print("  ✓ Compression successful")
        else:
            print("  ✗ Compression failed")
        
        # Test with mock LLM (simulate successful normalization)
        mock_result = {
            'success': True,
            'original_shape': df.shape,
            'normalized_shape': df.shape,
            'output_file': 'test_output.xlsx'
        }
        
        from llm_normalizer import LLMNormalizer
        normalizer = LLMNormalizer(api_key="dummy_key")
        report = normalizer.create_normalization_report(compressed_data, mock_result)
        
        if len(report) > 100:
            print("  ✓ Report generation successful")
        else:
            print("  ✗ Report generation failed")
        
        print("✓ Integration test completed\n")
        
    except Exception as e:
        print(f"  ✗ Integration test failed: {str(e)}\n")
    
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists('test_output.xlsx'):
            os.remove('test_output.xlsx')


def run_all_tests():
    """Run all tests."""
    print("=== SPREADSHEET NORMALIZATION TOOL - SYSTEM TESTS ===\n")
    
    test_utils()
    test_compression()
    test_llm_normalizer()
    test_integration()
    
    print("=== ALL TESTS COMPLETED ===")
    print("If all tests passed, the system is ready for use!")
    print("\nTo test with real LLM functionality:")
    print("1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
    print("2. Run: python demo.py")
    print("3. Or run: python main.py normalize --input your_file.xlsx --output normalized.xlsx")


if __name__ == "__main__":
    run_all_tests() 