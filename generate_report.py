"""
Generate a comprehensive codebase report and save it to a file.
"""

import os
import sys
from pathlib import Path

# Add project root to path if needed
sys.path.append(str(Path(__file__).parent))

# Import required components
from core.code_analyzer import CodeAnalyzer
from agents.tools.codebase_reporter import CodebaseReportGenerator
from config.settings import get_settings

def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Generate a codebase report")
    parser.add_argument("--type", choices=["comprehensive", "summary", "dependencies", "knowledge_graph"], 
                        default="comprehensive", help="Type of report to generate")
    parser.add_argument("--output", default="report.md", help="Output file path")
    parser.add_argument("--directory", default=None, help="Codebase directory to analyze")
    args = parser.parse_args()
    
    # Get directory from command line or use default
    directory = args.directory or str(Path(__file__).parent)
    report_type = args.type
    output_file = args.output
    
    print(f"Analyzing codebase in: {directory}")
    print(f"Generating {report_type} report...")
    
    # Initialize the code analyzer and parse the directory
    analyzer = CodeAnalyzer()
    modules = analyzer.parse_directory(directory)
    print(f"Found {len(analyzer.modules_by_name)} modules")
    
    # Initialize the reporter
    try:
        # Use settings for Neo4j configuration if available
        settings = get_settings()
        reporter = CodebaseReportGenerator()
        
        # Generate the report
        report = reporter.run(
            code_analyzer=analyzer,
            report_type=report_type,
            output_format="markdown",
            include_visualization=True
        )
        
        # Save the report to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report["content"])
        
        print(f"Report successfully generated and saved to: {output_file}")
        
        # Clean up resources
        reporter.close()
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
