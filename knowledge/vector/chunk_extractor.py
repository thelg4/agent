"""
Code chunk extractor.

This module extracts code chunks from parsed module information for embedding
and vector search.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from ...core.schema import ModuleInfo, ClassInfo, FunctionInfo, CodeChunk

logger = logging.getLogger(__name__)


class ChunkExtractor:
    """
    Extracts code chunks from parsed module information.
    
    This class is responsible for extracting code chunks for embedding based on
    line numbers and code structure.
    """
    
    def __init__(self, min_chunk_size: int = 3):
        """
        Initialize the chunk extractor.
        
        Args:
            min_chunk_size: Minimum number of lines for a chunk to be extracted
        """
        self.min_chunk_size = min_chunk_size
        
    def extract_chunks(self, module_info: ModuleInfo) -> List[CodeChunk]:
        """
        Extract code chunks from a module.
        
        Args:
            module_info: Parsed information about a module
            
        Returns:
            List of extracted code chunks
        """
        chunks = []
        module_name = module_info.name
        source_file = module_info.source_file
        
        try:
            with open(source_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            # Extract class chunks
            for cls in module_info.classes:
                if cls.end_line_number - cls.line_number + 1 >= self.min_chunk_size:
                    code = ''.join(lines[cls.line_number - 1 : cls.end_line_number])
                    chunks.append(CodeChunk(
                        name=cls.name,
                        type='class',
                        code=code,
                        module=module_name,
                        start_line=cls.line_number,
                        end_line=cls.end_line_number,
                        metadata={
                            "type": "class",
                            "name": cls.name,
                            "module": module_name,
                            "docstring": cls.docstring,
                            "bases": cls.bases,
                            "methods": [m.name for m in cls.methods],
                            "start_line": cls.line_number,
                            "end_line": cls.end_line_number
                        }
                    ))
                    
                # Also extract method chunks
                for method in cls.methods:
                    if method.end_line_number - method.line_number + 1 >= self.min_chunk_size:
                        method_code = ''.join(lines[method.line_number - 1 : method.end_line_number])
                        method_chunk = CodeChunk(
                            name=f"{cls.name}.{method.name}",
                            type='method',
                            code=method_code,
                            module=module_name,
                            start_line=method.line_number,
                            end_line=method.end_line_number,
                            metadata={
                                "type": "method",
                                "name": method.name,
                                "class": cls.name,
                                "module": module_name,
                                "docstring": method.docstring,
                                "returns": method.returns,
                                "is_async": method.is_async,
                                "args": [arg["name"] for arg in method.args],
                                "start_line": method.line_number,
                                "end_line": method.end_line_number
                            }
                        )
                        chunks.append(method_chunk)
            
            # Extract function chunks
            for func in module_info.functions:
                if func.end_line_number - func.line_number + 1 >= self.min_chunk_size:
                    code = ''.join(lines[func.line_number - 1 : func.end_line_number])
                    chunks.append(CodeChunk(
                        name=func.name,
                        type='function',
                        code=code,
                        module=module_name,
                        start_line=func.line_number,
                        end_line=func.end_line_number,
                        metadata={
                            "type": "function",
                            "name": func.name,
                            "module": module_name,
                            "docstring": func.docstring,
                            "returns": func.returns,
                            "is_async": func.is_async,
                            "args": [arg["name"] for arg in func.args],
                            "start_line": func.line_number,
                            "end_line": func.end_line_number
                        }
                    ))
                    
            # Extract the module docstring as a chunk if it exists
            if module_info.docstring and len(module_info.docstring.strip()) > 0:
                chunks.append(CodeChunk(
                    name=f"{module_name}_docstring",
                    type='module_docstring',
                    code=module_info.docstring,
                    module=module_name,
                    start_line=1,  # Assuming docstring is at the top
                    end_line=1,    # Placeholder
                    metadata={
                        "type": "module_docstring",
                        "name": f"{module_name}_docstring",
                        "module": module_name
                    }
                ))
                
            logger.info(f"Extracted {len(chunks)} chunks from {module_name}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting chunks from {source_file}: {str(e)}")
            return []
            
    def extract_chunks_from_modules(
        self, 
        modules: Dict[str, ModuleInfo]
    ) -> List[CodeChunk]:
        """
        Extract code chunks from multiple modules.
        
        Args:
            modules: Dictionary mapping module names to their ModuleInfo objects
            
        Returns:
            List of all extracted code chunks
        """
        all_chunks = []
        
        for module_name, module_info in modules.items():
            chunks = self.extract_chunks(module_info)
            all_chunks.extend(chunks)
            
        logger.info(f"Extracted total of {len(all_chunks)} chunks from {len(modules)} modules")
        return all_chunks
        
    def extract_custom_chunks(
        self, 
        source_file: Union[str, Path], 
        chunk_size: int = 50,
        overlap: int = 10
    ) -> List[CodeChunk]:
        """
        Extract chunks from a source file using a sliding window approach.
        
        This method is useful when you don't have parsed module information
        and just want to chunk a file by a fixed window size.
        
        Args:
            source_file: Path to the source file
            chunk_size: Number of lines per chunk
            overlap: Number of lines to overlap between chunks
            
        Returns:
            List of code chunks
        """
        source_file = Path(source_file)
        module_name = source_file.stem
        chunks = []
        
        try:
            with open(source_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            total_lines = len(lines)
            
            if total_lines <= chunk_size:
                # File is small enough to be a single chunk
                chunks.append(CodeChunk(
                    name=f"{module_name}_full",
                    type='file',
                    code=''.join(lines),
                    module=module_name,
                    start_line=1,
                    end_line=total_lines,
                    metadata={
                        "type": "file",
                        "name": module_name,
                        "source_file": str(source_file),
                        "start_line": 1,
                        "end_line": total_lines
                    }
                ))
            else:
                # Create overlapping chunks
                for start_line in range(1, total_lines, chunk_size - overlap):
                    end_line = min(start_line + chunk_size - 1, total_lines)
                    
                    # Adjust indices (lines are 1-indexed, list is 0-indexed)
                    chunk_text = ''.join(lines[start_line - 1:end_line])
                    
                    chunks.append(CodeChunk(
                        name=f"{module_name}_{start_line}_{end_line}",
                        type='sliding_window',
                        code=chunk_text,
                        module=module_name,
                        start_line=start_line,
                        end_line=end_line,
                        metadata={
                            "type": "sliding_window",
                            "name": f"{module_name}_{start_line}_{end_line}",
                            "source_file": str(source_file),
                            "start_line": start_line,
                            "end_line": end_line
                        }
                    ))
                    
                    # If we've reached the end of the file, stop
                    if end_line >= total_lines:
                        break
                        
            logger.info(f"Extracted {len(chunks)} custom chunks from {source_file}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting custom chunks from {source_file}: {str(e)}")
            return []