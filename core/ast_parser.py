'''
Python AST (Abstract Syntax Tree) parser module

This module provides tools to analyze Python source files and extract
structural information about the code through AST parsing.
'''

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any

from .schema import (
    ModuleInfo,
    ClassInfo, 
    FunctionInfo,
    ImportInfo,
    VariableInfo
)

# setup logging
logger = logging.getLogger(__name__)

class ASTParser(ast.NodeVisitor):
    '''
    A python ast parser that analyzes python source files and extracts structural information.

    Inherits from ast.NodeVisitor to traverse the ast nodes. Uses 'generic_visit' to ensure it visits
    all child nodes, including async functions, classes, imports, etc.

    Attributes:
        source_file (str): Path to the python file being parsed
        current_class (Optional[ClassInfo]): Current class being processed
        current_function (Optional[FunctionInfo]): Current function being processed
        module_info (Optional[ModuleInfo]): Resulting module info after parsing
    '''

    def __init__(self, source_file: Union[str, Path]):
        '''
        Initialize the parser with a source file path.
        '''

        self.source_file = str(source_file)

        # scope tracking
        self.current_class: Optional[ClassInfo] = None
        self.current_function: Optional[FunctionInfo] = None

        # Final result: one ModuleInfo per file
        self.module_info: Optional[ModuleInfo] = None

        # Temp storages
        self._current_functions: List[FunctionInfo] = []
        self._current_classes: List[ClassInfo] = []
        self._current_imports: List[ImportInfo] = []
        self._current_assignments: List[Dict[str, str]] = []
        self._current_variables: List[VariableInfo] = []

    def parse(self) -> Optional[ModuleInfo]:
        '''
        Parse the given python file and return the populated ModuleInfo

        Returns:
            Optional[ModuleInfo]: Structured info about the module, or None
        '''

        try:
            with open(self.source_file, 'r', encoding='utf-8') as f:
                content = f.read()

                # Create the AST
                tree = ast.parse(content)

                # kick off the NodeVisitor traversal
                self.visit(tree)
                return self.module_info
            
        except Exception as e:
            logging.error(f'Error parsing {self.source_file}: {str(e)}')
            return None
        
    #####################################################################################
    # Overridden visit methods
    #####################################################################################

    def visit_Module(self, node: ast.Module):
        '''
        Process a module node. 
        Uses generic_visit to traverse all child nodes
        '''
        module_name = Path(self.source_file).stem
        docstring = ast.get_docstring(node)

        # clear temp storages in case parse() is called repeatedly
        self._current_functions.clear()
        self._current_classes.clear()
        self._current_imports.clear()
        self._current_assignments.clear()
        self._current_variables.clear()

        # traverse all children
        self.generic_visit(node)

        # build the final ModuleInfo object
        self.module_info = ModuleInfo(
            name=module_name,
            docstring=docstring,
            functions=self._current_functions,
            classes=self._current_classes,
            imports=self._current_imports,
            global_assignments=self._current_assignments,
            source_file=self.source_file,
            variables=self._current_variables,
        )

    def visit_ClassDef(self, node: ast.ClassDef):
        '''
        Process a class definition. 
        Temporarily set a self.current_class to capture methods/variables
        '''

        # create a new ClassInfo
        end_line = self._get_end_lineno(node)

        class_info = ClassInfo(
            name=node.name,
            bases=[ast.unparse(base) for base in node.bases],
            docstring=ast.get_docstring(node),
            methods=[],
            class_variables=[],
            decorators=[ast.unparse(dec) for dec in node.decorator_list],
            line_number=node.lineno,
            end_line_number=end_line,  # capture the end_line
        )

        logger.debug(f"Class '{class_info.name}' ends at line {class_info.end_line_number}")

        # Remember previous class (in case of nested classes)
        prev_class = self.current_class
        self.current_class = class_info

        # Visit children so any FunctionDefs become methods,
        # Assigns become class vars, etc.
        self.generic_visit(node)

        # Attach the fully built class to our list
        self._current_classes.append(class_info)

        # Restore
        self.current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        '''
        Process a normal (sync) function definition
        '''
        self._handle_function_def(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        '''
        Process an async function definition
        '''
        self._handle_function_def(node, is_async=True)

    def visit_Import(self, node: ast.Import):
        '''
        Process a standard import statement
        '''
        self._current_imports.append(
            ImportInfo(
                module = None, 
                names=[
                    {"name": name.name, "alias": name.asname} for name in node.names
                ],
                line_number=node.lineno,
            )
        )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        '''
        processes an import-from statement
        '''
        self._current_imports.append(
            ImportInfo(
                module=node.module,
                names=[
                    {"name": name.name, "alias": name.asname} for name in node.names
                ],
                line_number=node.lineno,
            )
        )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        '''
        Process an assignemnt statement and categorize it based on current scope:
        - If not inside a function or class, it's a 'global' assignment
        - If inside a class but not a function, it's a 'class' assignment
        - If inside a function, it's a 'local' assignment
        '''
        scope = 'global'
        context = None

        if self.current_class and not self.current_function:
            scope = 'class'
            context = self.current_class.name
        elif self.current_function:
            scope = 'local'
            context = self.current_function.name

        # for high-level assignment record (like global_assignments)
        if (
            scope == 'global'
            and self.current_function is None
            and self.current_class is None
        ):
            self._current_assignments.extend(self._process_assignment(node))

        # for variable analysis
        self._analyze_variable(node, context=context, scope=scope)

        self.generic_visit(node)


    #####################################################################################
    # Helpers
    #####################################################################################

    def _handle_function_def(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool
    ):
        '''
        Shared logic for sync/async function defs
        '''
        func_name = f'async {node.name}' if is_async else node.name

        end_line = self._get_end_lineno(node)

        function_info = FunctionInfo(
            name=func_name,
            args=self._process_arguments(node.args),
            returns=self._get_annotation(node.returns),
            docstring=ast.get_docstring(node),
            calls=[],
            assignments=[],
            decorators=[ast.unparse(dec) for dec in node.decorator_list],
            is_method=self.current_class is not None,
            line_number=node.lineno,
            end_line_number=end_line,
            is_async=is_async,
            variables=[]
        )

        logger.debug(f"Function '{function_info.name}' ends at line {function_info.end_line_number}")

        # Track previous function context (in case of nested function defs)
        prev_function = self.current_function
        self.current_function = function_info

        # walk the function body to detect calls and assignment types
        for child in ast.walk(node):
            # capture the function calls
            if isinstance(child, ast.Call):
                call_name = self._get_call_name(child)

                # avoid duplicates
                if call_name not in function_info.calls:
                    function_info.calls.append(call_name)
            
            # track direct assignments
            elif isinstance(child, ast.Assign) and child in node.body:
                function_info.assignments.extend(self._process_assignment(child))

        # now that the function is fully populated, store in correct scope
        if self.current_class:
            # if a method
            self.current_class.methods.append(function_info)
        else:
            # module level function
            self._current_functions.append(function_info)

        # restore prev function context
        self.current_function = prev_function

        # visit children so if there are nested classes or functions, we catch them
        self.generic_visit(node)

    def _analyze_variable(
        self, node: ast.Assign, context: Optional[str] = None, scope: str = 'local'
    ):
        '''
        Analyze assigned variables for name, value, and optional type inference
        '''

        for target in node.targets:
            if isinstance(target, ast.Name):
                variable_info = VariableInfo(
                    name=target.id,
                    value=ast.unparse(node.value),
                    line_number=node.lineno,
                    context=context,
                    scope=scope,
                )

                # attempt type inference
                try:
                    evaluated_value = ast.literal_eval(node.value)
                    variable_info.inferred_type = type(evaluated_value).__name__
                except (ValueError, SyntaxError):
                    variable_info.inferred_type = self._infer_type_from_ast(node.value)

                # add to current function's variables if in a function, otherwise to module
                if self.current_function and scope == 'local':
                    self.current_function.variables.append(variable_info)
                else:
                    self._current_variables.append(variable_info)
            
            elif isinstance(target, (ast.Tuple, ast.List)):
                # handle multiple assignment
                for idx, elt in enumerate(target.elts):
                    if isinstance(elt, ast.Name):
                        variable_info = VariableInfo(
                            name=elt.id,
                            value=f"part of multiple assignment at index {idx}",
                            line_number=node.lineno,
                            context=context,
                            scope=scope,
                            inferred_type="multiple_assignment",
                        )

                        # add to current function's variables if in a function, otherwise to module
                        if self.current_function and scope == 'local':
                            self.current_function.variables.append(variable_info)
                        else:
                            self._current_variables.append(variable_info)
    
    def _infer_type_from_ast(self, node: ast.AST) -> Optional[str]:
        '''
        Fallback type inference for non-literal assignments
        '''
        if isinstance(node, ast.List):
            return "list"
        elif isinstance(node, ast.Dict):
            return "dict"
        elif isinstance(node, ast.Tuple):
            return "tuple"
        elif isinstance(node, ast.Call):
            # best-effort guess: return the name of the function being called
            return self._get_call_name(node)
        return None
    
    def _process_assignment(self, node: ast.Assign) -> List[Dict[str, str]]:
        """Returns a simple list of {'target': ..., 'value': ...} for the assignment."""
        assignments = []
        for target in node.targets:
            assignments.append(
                {"target": ast.unparse(target), "value": ast.unparse(node.value)}
            )
        return assignments

    def _process_arguments(self, args: ast.arguments) -> List[Dict[str, Optional[str]]]:
        """Build a list of function argument metadata, including *args and **kwargs."""
        results = []
        # Positional
        for arg in args.args:
            results.append(
                {"name": arg.arg, "annotation": self._get_annotation(arg.annotation)}
            )

        # *args
        if args.vararg:
            results.append(
                {
                    "name": f"*{args.vararg.arg}",
                    "annotation": self._get_annotation(args.vararg.annotation),
                }
            )

        # Keyword-only
        for arg in args.kwonlyargs:
            results.append(
                {"name": arg.arg, "annotation": self._get_annotation(arg.annotation)}
            )

        # **kwargs
        if args.kwarg:
            results.append(
                {
                    "name": f"**{args.kwarg.arg}",
                    "annotation": self._get_annotation(args.kwarg.annotation),
                }
            )
        return results

    def _get_annotation(self, node: Optional[ast.AST]) -> Optional[str]:
        """Convert a type annotation AST node into a string."""
        if node is None:
            return None
        return ast.unparse(node)

    def _get_call_name(self, node: ast.Call) -> str:
        """Extract the function name (or attribute chain) from a Call node."""
        if isinstance(node.func, ast.Attribute):
            base = ast.unparse(node.func.value)
            attr = node.func.attr
            return f"{base}.{attr}"
        elif isinstance(node.func, ast.Name):
            return node.func.id
        return ast.unparse(node.func)
    
    def _get_end_lineno(self, node: ast.AST) -> int:
        """
        Recursively find the maximum line number in the AST node.
        This ensures that end_line_number is always set as an integer.
        """
        max_lineno = getattr(node, 'lineno', 0)
        for child in ast.iter_child_nodes(node):
            child_end_lineno = getattr(child, 'end_lineno', None)
            if child_end_lineno:
                max_lineno = max(max_lineno, child_end_lineno)
            else:
                max_lineno = max(max_lineno, self._get_end_lineno(child))
        return max_lineno