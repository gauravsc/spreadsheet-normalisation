#!/usr/bin/env python3
"""
Setup script for the Spreadsheet Normalization Tool.
Installs dependencies and creates necessary files.
"""

import os
import sys
import subprocess
import shutil


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def create_env_file():
    """Create .env file template."""
    env_file = ".env"
    if not os.path.exists(env_file):
        with open(env_file, "w") as f:
            f.write("# OpenAI API Key\n")
            f.write("# Get your API key from: https://platform.openai.com/api-keys\n")
            f.write("OPENAI_API_KEY=your_api_key_here\n")
        print("✅ Created .env file template")
        print("   Please add your OpenAI API key to the .env file")
    else:
        print("✅ .env file already exists")


def run_tests():
    """Run system tests."""
    print("\n🧪 Running system tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ System tests passed")
            return True
        else:
            print("❌ System tests failed")
            print(result.stdout)
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("❌ test_system.py not found")
        return False


def create_sample_data():
    """Create sample data for testing."""
    print("\n📊 Creating sample data...")
    
    try:
        result = subprocess.run([sys.executable, "create_sample_data.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Sample data created successfully")
            return True
        else:
            print("❌ Failed to create sample data")
            print(result.stdout)
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("❌ create_sample_data.py not found")
        return False


def show_usage_examples():
    """Show usage examples."""
    print("\n📖 Usage Examples:")
    print("=" * 50)
    
    print("\n1. Analyze a spreadsheet (no API key required):")
    print("   python main.py analyze --input sample_data.xlsx --output analysis.txt")
    
    print("\n2. Normalize a spreadsheet (requires OpenAI API key):")
    print("   python main.py normalize --input sample_data.xlsx --output normalized.xlsx --verbose")
    
    print("\n3. Run the demo:")
    print("   python demo.py")
    
    print("\n4. Run system tests:")
    print("   python test_system.py")
    
    print("\n📝 Available commands:")
    print("   - analyze: Analyze spreadsheet structure and compression")
    print("   - normalize: Compress and normalize spreadsheet using LLM")
    print("   - demo: Run complete demo with sample data")


def main():
    """Main setup function."""
    print("🚀 Setting up Spreadsheet Normalization Tool")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed. Please install dependencies manually:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    # Run tests
    if not run_tests():
        print("\n⚠️  Some tests failed, but the tool may still work")
    
    # Create sample data
    create_sample_data()
    
    # Show usage
    show_usage_examples()
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Add your OpenAI API key to the .env file")
    print("2. Test the tool with: python demo.py")
    print("3. Process your own spreadsheets with the normalize command")


if __name__ == "__main__":
    main() 