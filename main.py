#!/usr/bin/env python3
"""
Main CLI interface for the Spreadsheet Normalization Tool.
Orchestrates the compression and LLM-based normalization process.
"""

import os
import sys
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text
from typing import Optional

from spreadsheet_compressor import SpreadsheetCompressor
from llm_normalizer import LLMNormalizer
from utils import load_spreadsheet

console = Console()


@click.command()
@click.option('--input', '-i', 'input_file', required=True, 
              help='Input Excel file path')
@click.option('--output', '-o', 'output_file', required=True,
              help='Output Excel file path')
@click.option('--api-key', 'api_key', envvar='OPENAI_API_KEY',
              help='OpenAI API key (can also be set via OPENAI_API_KEY env var)')
@click.option('--verbose', '-v', is_flag=True, default=False,
              help='Enable verbose output')
@click.option('--report', '-r', 'report_file', default=None,
              help='Save detailed report to file')
def main(input_file: str, output_file: str, api_key: Optional[str], 
         verbose: bool, report_file: Optional[str]):
    """
    Spreadsheet Normalization Tool
    
    Compresses spreadsheet data using SPREADSHEETLLM techniques and generates
    normalization instructions using LLM.
    """
    
    if verbose:
        console.print(f"[bold blue]Starting Spreadsheet Normalization[/bold blue]")
        console.print(f"Input: {input_file}")
        console.print(f"Output: {output_file}")
    
    # Validate input file
    if not os.path.exists(input_file):
        console.print(f"[bold red]Error: Input file '{input_file}' not found[/bold red]")
        sys.exit(1)
    
    try:
        # Step 1: Compress the spreadsheet
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Compressing spreadsheet...", total=None)
            
            compressor = SpreadsheetCompressor()
            compressed_data = compressor.compress_spreadsheet(input_file)
            
            progress.update(task, description="Compression completed!")
        
        if verbose:
            console.print(Panel(compressor.get_compression_summary(), 
                              title="Compression Results", border_style="green"))
        
        # Step 2: Generate and execute normalization
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating normalization instructions...", total=None)
            
            normalizer = LLMNormalizer(api_key=api_key)
            execution_result = normalizer.execute_normalization(
                input_file, output_file, compressed_data
            )
            
            progress.update(task, description="Normalization completed!")
        
        # Display results
        if execution_result['success']:
            console.print(f"[bold green]✓ Normalization successful![/bold green]")
            console.print(f"Output saved to: {output_file}")
            
            # Create results table
            table = Table(title="Normalization Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Original Shape", str(execution_result['original_shape']))
            table.add_row("Normalized Shape", str(execution_result['normalized_shape']))
            table.add_row("Status", "SUCCESS")
            
            console.print(table)
            
            # Show compression stats
            stats = compressed_data['compression_stats']
            compression_table = Table(title="Compression Statistics")
            compression_table.add_column("Technique", style="cyan")
            compression_table.add_column("Compression Ratio", style="yellow")
            
            compression_table.add_row("Structural Anchor", f"{stats['structural_compression']:.2%}")
            compression_table.add_row("Data Aggregation", f"{stats['aggregation_compression']:.2%}")
            compression_table.add_row("Inverted Index", f"{stats['inverted_compression']:.2%}")
            
            console.print(compression_table)
            
        else:
            console.print(f"[bold red]✗ Normalization failed![/bold red]")
            console.print(f"Error: {execution_result['error']}")
            sys.exit(1)
        
        # Generate and save report if requested
        if report_file:
            report = normalizer.create_normalization_report(compressed_data, execution_result)
            with open(report_file, 'w') as f:
                f.write(report)
            console.print(f"[green]Detailed report saved to: {report_file}[/green]")
        
        # Show sample of generated instructions if verbose
        if verbose and execution_result.get('instructions'):
            console.print(Panel(
                Text(execution_result['instructions'][:500] + "...", 
                     style="dim"),
                title="Generated Instructions (Sample)",
                border_style="blue"
            ))
    
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@click.command()
@click.option('--input', '-i', 'input_file', required=True,
              help='Input Excel file path')
@click.option('--output', '-o', 'output_file', default='compression_analysis.txt',
              help='Output analysis file path')
def analyze(input_file: str, output_file: str):
    """
    Analyze spreadsheet structure and compression potential without normalization.
    """
    
    if not os.path.exists(input_file):
        console.print(f"[bold red]Error: Input file '{input_file}' not found[/bold red]")
        sys.exit(1)
    
    try:
        console.print(f"[bold blue]Analyzing spreadsheet: {input_file}[/bold blue]")
        
        # Load and analyze spreadsheet
        df, formatting_info = load_spreadsheet(input_file)
        compressor = SpreadsheetCompressor()
        compressed_data = compressor.compress_spreadsheet(input_file)
        
        # Create detailed analysis
        analysis = f"SPREADSHEET ANALYSIS REPORT\n"
        analysis += "=" * 50 + "\n\n"
        
        analysis += f"FILE: {input_file}\n"
        analysis += f"SHAPE: {df.shape}\n"
        analysis += f"TOTAL CELLS: {df.shape[0] * df.shape[1]}\n\n"
        
        # Compression statistics
        stats = compressed_data['compression_stats']
        analysis += "COMPRESSION STATISTICS:\n"
        analysis += f"- Original size: {stats['original_size']} cells\n"
        analysis += f"- Structural compression: {stats['structural_compression']:.2%}\n"
        analysis += f"- Data aggregation compression: {stats['aggregation_compression']:.2%}\n"
        analysis += f"- Inverted index compression: {stats['inverted_compression']:.2%}\n\n"
        
        # Data type analysis
        agg_data = compressed_data['data_aggregation']
        analysis += "DATA TYPE DISTRIBUTION:\n"
        for data_type, type_stats in agg_data['aggregation_stats'].items():
            analysis += f"- {data_type}: {type_stats['count']} cells ({type_stats['count']/stats['original_size']:.1%})\n"
        analysis += "\n"
        
        # Structural anchors
        anchors = compressed_data['structural_anchors']
        analysis += "STRUCTURAL ANCHORS:\n"
        analysis += f"- Anchor rows: {anchors['anchor_rows']}\n"
        analysis += f"- Anchor columns: {anchors['anchor_columns']}\n"
        analysis += f"- Compression ratio: {anchors['compression_ratio']:.2%}\n\n"
        
        # Value patterns
        inverted = compressed_data['inverted_index']
        analysis += "MOST COMMON VALUES:\n"
        sorted_values = sorted(inverted.items(), key=lambda x: len(x[1]), reverse=True)
        for value, addresses in sorted_values[:10]:
            analysis += f"- '{value}': {len(addresses)} occurrences\n"
        
        # Save analysis
        with open(output_file, 'w') as f:
            f.write(analysis)
        
        console.print(f"[green]Analysis saved to: {output_file}[/green]")
        
        # Display summary
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"- Shape: {df.shape}")
        console.print(f"- Data types: {len(agg_data['aggregation_stats'])}")
        console.print(f"- Structural compression: {stats['structural_compression']:.1%}")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)


@click.group()
def cli():
    """Spreadsheet Normalization Tool - LLM-powered spreadsheet compression and normalization."""
    pass


cli.add_command(main, name="normalize")
cli.add_command(analyze, name="analyze")


if __name__ == '__main__':
    cli() 