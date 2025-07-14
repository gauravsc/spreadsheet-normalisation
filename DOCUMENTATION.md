# Spreadsheet Normalization Tool Documentation

## Overview

The Spreadsheet Normalization Tool is a Python-based system that implements the compression techniques from the SPREADSHEETLLM paper to normalize Excel spreadsheets using LLM-powered analysis. The tool consists of two main parts:

1. **Spreadsheet Compression**: Implements four compression techniques to reduce spreadsheet size while preserving structure
2. **LLM-based Normalization**: Uses compressed data to generate transformation instructions for creating normalized spreadsheets

## Architecture

### Core Components

- **`utils.py`**: Utility functions for spreadsheet processing, data type detection, and format recognition
- **`spreadsheet_compressor.py`**: Implements the four compression techniques from SPREADSHEETLLM
- **`llm_normalizer.py`**: Handles LLM interaction and code generation for normalization
- **`main.py`**: CLI interface for orchestrating the compression and normalization process

### Compression Techniques

The tool implements the three compression techniques described in the SPREADSHEETLLM paper:

#### 1. Structural-anchor-based Extraction
- **Purpose**: Identifies heterogeneous rows/columns as structural anchors
- **Method**: Finds rows/columns with mixed data types and applies k-neighborhood filtering
- **K-neighborhood**: Discards rows/columns more than k cells away from any anchor point
- **Implementation**: `_structural_anchor_extraction()` method

#### 2. Inverted-index Translation
- **Purpose**: Converts matrix to dictionary format for lossless compression
- **Format**: `{Value: [Address1, Address2, ...]}`
- **Implementation**: `_inverted_index_translation()` method

#### 3. Data-format-aware Aggregation
- **Purpose**: Groups cells by data type using Number Format Strings (NFS)
- **Types**: Integer, Float, Date, Time, Currency, Email, Percentage, Text
- **Implementation**: `_data_format_aggregation()` method

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (for LLM functionality)

### Quick Setup
```bash
# Clone or download the project
cd spreadsheet-normalisation

# Run setup script
python setup.py

# Or install manually
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Command Line Interface

#### Analyze Spreadsheet (No API Key Required)
```bash
python main.py analyze --input spreadsheet.xlsx --output analysis.txt
```

#### Normalize Spreadsheet (Requires API Key)
```bash
python main.py normalize --input spreadsheet.xlsx --output normalized.xlsx --verbose
```

#### Available Options
- `--input, -i`: Input Excel file path (required)
- `--output, -o`: Output file path (required)
- `--api-key`: OpenAI API key (optional, can use env var)
- `--verbose, -v`: Enable verbose output
- `--report, -r`: Save detailed report to file

### Demo and Testing

#### Run Demo
```bash
python demo.py
```

#### Run System Tests
```bash
python test_system.py
```

#### Create Sample Data
```bash
python create_sample_data.py
```

## API Reference

### SpreadsheetCompressor

Main class for implementing compression techniques.

#### Methods

- `compress_spreadsheet(file_path)`: Apply all four compression techniques
- `get_compression_summary()`: Generate compression statistics
- `create_llm_prompt(compressed_data)`: Create LLM prompt from compressed data

### LLMNormalizer

Handles LLM interaction for spreadsheet normalization.

#### Methods

- `generate_normalization_instructions(compressed_data)`: Generate transformation code
- `execute_normalization(input_file, output_file, compressed_data)`: Execute normalization
- `create_normalization_report(compressed_data, execution_result)`: Generate report

### Utility Functions

#### Data Type Detection
```python
from utils import detect_data_type

# Detect data type of a value
data_type = detect_data_type(value, number_format="General")
```

#### Spreadsheet Loading
```python
from utils import load_spreadsheet

# Load spreadsheet with formatting information
df, formatting_info = load_spreadsheet("file.xlsx")
```

## Data Types Supported

The tool recognizes and handles the following data types:

- **Integer**: Whole numbers
- **Float**: Decimal numbers
- **Date**: Date values (various formats)
- **Time**: Time values
- **Currency**: Monetary values with currency symbols
- **Percentage**: Percentage values
- **Email**: Email addresses
- **Text**: General text data
- **Empty**: Null or empty values

## Compression Statistics

The tool provides detailed compression statistics:

- **Original size**: Total number of cells
- **Structural compression**: Ratio of anchor cells to total cells
- **Data aggregation compression**: Compression based on data type grouping
- **Inverted index compression**: Compression from value deduplication

## Normalization Process

### Step 1: Compression
1. Load spreadsheet with formatting information
2. Identify structural anchors (heterogeneous rows/columns)
3. Create inverted index for value deduplication
4. Aggregate cells by data type

### Step 2: LLM Analysis
1. Create comprehensive prompt from compressed data
2. Send to LLM for analysis
3. Generate Python transformation code
4. Execute code safely in controlled environment

### Step 3: Output
1. Apply transformations to original data
2. Save normalized spreadsheet
3. Generate detailed report

## Error Handling

The tool includes comprehensive error handling:

- **File validation**: Checks for valid Excel files
- **API key validation**: Ensures OpenAI API key is provided
- **Safe code execution**: Executes LLM-generated code in controlled environment
- **Graceful failures**: Provides detailed error messages

## Performance Considerations

### Compression Efficiency
- **Small spreadsheets**: All techniques applied
- **Large spreadsheets**: Focus on structural anchors and data aggregation
- **Memory usage**: Optimized for large datasets

### LLM Token Usage
- **Prompt optimization**: Limits compressed data to avoid token overflow
- **Batch processing**: Processes large spreadsheets in chunks
- **Cost management**: Provides token usage estimates

## Troubleshooting

### Common Issues

#### Missing Dependencies
```bash
pip install -r requirements.txt
```

#### API Key Issues
```bash
export OPENAI_API_KEY="your-key-here"
```

#### File Format Issues
- Ensure input files are valid Excel (.xlsx) format
- Check file permissions
- Verify file is not corrupted

#### Memory Issues
- For large spreadsheets, consider processing in chunks
- Monitor system memory usage
- Use analyze mode first to understand data structure

### Debug Mode
Enable verbose output for detailed debugging:
```bash
python main.py normalize --input file.xlsx --output output.xlsx --verbose
```

## Examples

### Basic Usage
```python
from spreadsheet_compressor import SpreadsheetCompressor
from llm_normalizer import LLMNormalizer

# Compress spreadsheet
compressor = SpreadsheetCompressor()
compressed_data = compressor.compress_spreadsheet("input.xlsx")

# Normalize with LLM
normalizer = LLMNormalizer(api_key="your-key")
result = normalizer.execute_normalization("input.xlsx", "output.xlsx", compressed_data)
```

### Custom Data Type Detection
```python
from utils import detect_data_type

# Custom data type detection
def custom_detect_type(value, format_string):
    # Add custom logic here
    return detect_data_type(value, format_string)
```

## Contributing

### Development Setup
1. Fork the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Run tests: `python test_system.py`

### Code Style
- Follow PEP 8 guidelines
- Add type hints for all functions
- Include docstrings for all classes and methods
- Write tests for new functionality

## License

This project implements techniques from the SPREADSHEETLLM paper. Please ensure compliance with the original paper's licensing terms.

## References

- SPREADSHEETLLM Paper: [Paper Reference]
- OpenAI API Documentation: https://platform.openai.com/docs
- Pandas Documentation: https://pandas.pydata.org/docs/
- OpenPyXL Documentation: https://openpyxl.readthedocs.io/ 