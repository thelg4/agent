'''
Code Analysis for python projects.

This module provides functionality for analyzing multiple Python files in a directory,
aggregating results, and calculating metrics
'''

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .ast_parser import ASTParser
from .schema import ModuleInfo

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    '''
    High-level code analyzer that processes multiple python files in a directory,

    This wraps the ASTParser to provide directory-level analysis and metrics
    '''

    def __init__(self):
        '''
        Inits the analyzer
        '''
        self.modules_by_name: Dict[str, ModuleInfo] = {}
        self.skipped_files: List[str] = []

    def parse_directory(self, directory: Union[str, Path]) -> dict[str, ModuleInfo]:
        '''
        Recursively analyze all python files in the given directory.

        Args:
            directory: path to the directory containing the python files

        Returns:
            Dict mapping module names to their parsed ModuleInfo objects
        '''

        directory = Path(directory)
        logger.info(f'Analyzing Python files in: {directory}')

        self.modules_by_name.clear()
        self.skipped_files.clear()

        # Recursively look for *.py files
        for py_file in directory.rglob("*.py"):
            logger.debug(f"Parsing file: {py_file}")
            parser = ASTParser(py_file)
            module_info = parser.parse()
            
            # If parse() returns None (e.g., empty file), skip
            if module_info is not None:
                # Validate the module info
                try:
                    self._validate_module_info(module_info)
                    # Store by module name for easy lookup
                    self.modules_by_name[module_info.name] = module_info
                except AssertionError as e:
                    logger.warning(f"Invalid module info for {py_file}: {e}")
                    self.skipped_files.append(str(py_file))
            else:
                self.skipped_files.append(str(py_file))
                
        logger.info(f"Analyzed {len(self.modules_by_name)} modules, skipped {len(self.skipped_files)} files")
        return self.modules_by_name
    
    def _validate_module_info(self, module_info: ModuleInfo) -> None:
        """
        Validate the module info to ensure it has the required fields.
        
        Args:
            module_info: The ModuleInfo object to validate
            
        Raises:
            AssertionError: If validation fails
        """
        assert isinstance(module_info.name, str), "Module name must be a string."
        assert isinstance(module_info.source_file, str), "Source file must be a string."

        for func in module_info.functions:
            assert isinstance(func.end_line_number, int), f"Function '{func.name}' has invalid end_line_number."
            assert isinstance(func.line_number, int), f"Function '{func.name}' has invalid line_number."

        for cls in module_info.classes:
            assert isinstance(cls.end_line_number, int), f"Class '{cls.name}' has invalid end_line_number."
            assert isinstance(cls.line_number, int), f"Class '{cls.name}' has invalid line_number."

    def get_code_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return metrics about the analyzed codebase.
        
        Returns:
            Dict containing metrics like total lines, class count, etc.
        """
        total_lines = 0
        total_functions = 0
        total_classes = 0
        total_methods = 0
        total_variables = 0
        
        for module in self.modules_by_name.values():
            # For accurate line counting, we'd need to read the file
            with open(module.source_file, 'r', encoding='utf-8') as f:
                total_lines += sum(1 for _ in f)
                
            total_functions += len(module.functions)
            total_classes += len(module.classes)
            total_variables += len(module.variables)
            
            for cls in module.classes:
                total_methods += len(cls.methods)
        
        return {
            "total_modules": len(self.modules_by_name),
            "total_lines": total_lines,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "total_methods": total_methods,
            "total_variables": total_variables,
            "skipped_files": len(self.skipped_files),
        }
    
    def find_module(self, module_name: str) -> Optional[ModuleInfo]:
        """
        Find a module by name.
        
        Args:
            module_name: Name of the module to find
            
        Returns:
            ModuleInfo for the module, or None if not found
        """
        return self.modules_by_name.get(module_name)
    
    def find_class(self, class_name: str) -> List[Dict[str, Any]]:
        """
        Find classes by name across all modules.
        
        Args:
            class_name: Name of the class to find
            
        Returns:
            List of dictionaries with module and class information
        """
        results = []
        
        for module_name, module in self.modules_by_name.items():
            for cls in module.classes:
                if cls.name == class_name:
                    results.append({
                        "module": module_name,
                        "class": cls
                    })
                    
        return results
    
    def find_function(self, function_name: str) -> List[Dict[str, Any]]:
        """
        Find functions by name across all modules.
        
        Args:
            function_name: Name of the function to find
            
        Returns:
            List of dictionaries with module and function information
        """
        results = []
        
        for module_name, module in self.modules_by_name.items():
            # Check module-level functions
            for func in module.functions:
                if func.name == function_name:
                    results.append({
                        "module": module_name,
                        "function": func,
                        "type": "function"
                    })
            
            # Check class methods
            for cls in module.classes:
                for method in cls.methods:
                    if method.name == function_name:
                        results.append({
                            "module": module_name,
                            "class": cls.name,
                            "function": method,
                            "type": "method"
                        })
                    
                    
        return results