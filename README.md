# Spreadsheet Normalization Tool

A Python-based tool for normalizing Excel spreadsheets using LLM-powered compression and transformation techniques, based on the SPREADSHEETLLM paper.

## Features

### Part 1: Spreadsheet Compression
Implements three compression techniques from SPREADSHEETLLM:

1. **Structural-anchor-based Extraction**: Identifies heterogeneous rows/columns as anchors to preserve structure
2. **Inverted-index Translation**: Converts matrix to dictionary format for lossless compression
3. **Data-format-aware Aggregation**: Groups cells by data type using Number Format Strings (NFS)

### Part 2: LLM-based Normalization
- Feeds compressed spreadsheet data to LLM
- Generates transformation instructions/code
- Creates normalized output spreadsheet

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --input spreadsheet.xlsx --output normalized.xlsx
```

## Environment Setup

Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

- `spreadsheet_compressor.py`: Implements the four compression techniques
- `llm_normalizer.py`: Handles LLM interaction and code generation
- `main.py`: Main CLI interface
- `utils.py`: Utility functions for spreadsheet processing 