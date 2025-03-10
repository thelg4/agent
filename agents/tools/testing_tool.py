import logging
import os
import sys
import tempfile
import unittest
import importlib.util
import inspect
import json
import re
import traceback
import shutil
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pathlib import Path

from ..base_agent import AgentTool
from ...core.code_analyzer import CodeAnalyzer

logger = logging.getLogger(__name__)

class FunctionTestingTool(AgentTool):
    """
    Tool for testing functions in the codebase.
    
    This tool analyzes a function, generates test cases,
    and executes tests to verify the function's behavior.
    """
    
    def __init__(self, llm_client=None):
        """Initialize the function testing tool."""
        print(f"Loaded FunctionTestingTool from {__file__}")
        super().__init__(
            name="function_tester",
            description="Tests functions by generating and running test cases"
        )
        self.llm_client = llm_client
        
    def run(self, function_name: str, code_analyzer=None, test_type: str = "unit", 
            num_tests: int = 5, include_edge_cases: bool = True) -> Dict[str, Any]:
        """
        Test a specific function in the codebase.
        
        Args:
            function_name: Name of the function to test
            code_analyzer: CodeAnalyzer instance to locate the function
            test_type: Type of tests to generate ('unit', 'property', 'integration')
            num_tests: Number of test cases to generate
            include_edge_cases: Whether to include edge case testing
            
        Returns:
            Dictionary with test cases and results
        """
        # Force reload to ensure latest code
        if 'codeassistant.agents.tools.testing_tool' in sys.modules:
            del sys.modules['codeassistant.agents.tools.testing_tool']
        logger.debug("Reloaded FunctionTestingTool to ensure latest version")
        
        if not code_analyzer:
            return {"error": "Code analyzer required to find the function"}
        
        functions = code_analyzer.find_function(function_name)
        if not functions:
            # Try fuzzy matching if exact match fails
            all_functions = code_analyzer.get_all_functions()
            possible_matches = [f for f in all_functions 
                                if function_name.lower() in f['name'].lower()]
            if possible_matches:
                functions = [possible_matches[0]]
                logger.info(f"Exact match not found for '{function_name}', using closest match: {functions[0]['name']}")
            else:
                return {"error": f"Function '{function_name}' not found in the codebase"}
        
        function_info = functions[0]
        function_info["code_analyzer"] = code_analyzer
        
        # Get the source file
        source_file = self._get_source_file(function_info, code_analyzer)
        if not source_file:
            return {"error": f"Could not locate source file for function '{function_name}'"}
        
        # Extract function and dependencies
        func_def, dependencies, required_modules = self._extract_function(source_file, function_info)
        if not func_def:
            return {"error": f"Could not extract definition for function '{function_name}'"}
        
        # Generate test cases based on test_type
        test_cases = self._generate_test_cases(
            function_name=function_name, 
            func_def=func_def, 
            dependencies=dependencies,
            test_type=test_type,
            num_tests=num_tests,
            include_edge_cases=include_edge_cases
        )
        
        # Run the tests
        test_results = self._run_tests(
            function_name=function_name, 
            func_def=func_def, 
            dependencies=dependencies, 
            required_modules=required_modules,
            test_cases=test_cases
        )
        
        return {
            "function_name": function_name,
            "test_cases": test_cases,
            "results": test_results,
            "source_file": source_file
        }
    
    def _get_source_file(self, function_info: Dict[str, Any], code_analyzer: Any) -> Optional[str]:
        """Get the source file for a function based on its type."""
        source_file = None
        
        if function_info["type"] == "function":
            module_name = function_info["module"]
            module_info = code_analyzer.find_module(module_name)
            if module_info:
                source_file = module_info.source_file
        elif function_info["type"] == "method":
            module_name = function_info["module"]
            module_info = code_analyzer.find_module(module_name)
            if module_info:
                source_file = module_info.source_file
        elif function_info["type"] == "class":
            # Handle class methods
            module_name = function_info["module"]
            module_info = code_analyzer.find_module(module_name)
            if module_info:
                source_file = module_info.source_file
                
        return source_file
        
    def _extract_function(self, source_file: str, function_info: Dict[str, Any]) -> Tuple[Optional[str], List[str], List[str]]:
        """
        Extract function definitions and required dependencies.
        
        Returns:
            Tuple of (function definition, import statements, required module names)
        """
        try:
            with open(source_file, "r", encoding="utf-8") as f:
                source_lines = f.readlines()
            
            # Use the CodeAnalyzer's pre-parsed module data
            module_name = function_info.get("module")
            if not module_name:
                logger.error(f"Module name not found in function_info: {function_info}")
                return None, [], []
                
            code_analyzer = function_info.get("code_analyzer")
            module_info = code_analyzer.find_module(module_name)
            if not module_info:
                logger.error(f"Module '{module_name}' not found in CodeAnalyzer data")
                return None, [], []
            
            # Parse imports to identify required modules
            required_modules = []
            dependencies = []
            for i, line in enumerate(source_lines):
                line = line.strip()
                if line.startswith("import "):
                    module = line[7:].strip().split(" as ")[0].split(",")[0].strip()
                    required_modules.append(module)
                    dependencies.append(line + "\n")
                elif line.startswith("from "):
                    module = line.split("import")[0][5:].strip()
                    if not module.startswith("app") and not module.startswith("."):
                        required_modules.append(module)
                        dependencies.append(line + "\n")
            
            # Handle class method vs. standalone function
            target_function = function_info.get("name")
            if not target_function:
                # Check alternative keys or use a fallback name
                if "function_name" in function_info:
                    target_function = function_info["function_name"]
                else:
                    logger.error(f"Function name not found in function_info: {function_info}")
                    # Try to use a function from the module as fallback
                    if module_info.functions:
                        target_function = module_info.functions[0].name
                        logger.warning(f"Using fallback function name: {target_function}")
                    else:
                        return None, [], []
            
            # If it's a method, we need to extract the class as well
            if function_info.get("type") == "method":
                # Find the class definition
                class_name = function_info.get("class_name")
                if class_name:
                    class_info = None
                    for cls in module_info.classes:
                        if cls.name == class_name:
                            class_info = cls
                            break
                    
                    if class_info:
                        class_start = class_info.line_number - 1
                        class_end = class_info.end_line_number
                        class_def = "".join(source_lines[class_start:class_end])
                        return class_def, dependencies, required_modules
            
            # For standalone functions, collect all related function definitions
            func_defs = []
            target_function_found = False
            
            # Sort functions by line number to maintain order
            sorted_functions = sorted(module_info.functions, key=lambda func_obj: getattr(func_obj, 'line_number', 0))
            
            for func_obj in sorted_functions:
                # Check if function object has required attributes
                if not hasattr(func_obj, 'line_number') or not hasattr(func_obj, 'end_line_number'):
                    logger.error(f"Function object missing required attributes: {func_obj}")
                    continue
                    
                f_start = func_obj.line_number - 1  # 0-based index
                f_end = func_obj.end_line_number
                
                if f_start < 0 or f_end > len(source_lines) or f_start >= f_end:
                    logger.error(f"Invalid line range for '{func_obj.name}': {f_start+1} to f_end")
                    continue
                
                # Always include the target function
                if func_obj.name == target_function:
                    target_function_found = True
                    func_defs.append("".join(source_lines[f_start:f_end]))
                    
                # Include helper functions called by the target function
                elif target_function_found:
                    func_content = "".join(source_lines[f_start:f_end])
                    # Check if this function is used by the target function
                    if any(func_def for func_def in func_defs if func_obj.name in func_def):
                        func_defs.append(func_content)
            
            if not func_defs:
                logger.error(f"Function '{target_function}' definition not found in module '{module_name}'")
                return None, [], []
            
            # Combine all function definitions
            combined_func_def = "\n\n".join(func_defs)
            
            return combined_func_def, dependencies, required_modules
    
        except Exception as e:
            logger.error(f"Error extracting functions: {str(e)}\n{traceback.format_exc()}")
            return None, [], []
            
    def _generate_test_cases(
        self, 
        function_name: str, 
        func_def: str, 
        dependencies: List[str],
        test_type: str = "unit",
        num_tests: int = 5,
        include_edge_cases: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate test cases for the function using LLM.
        
        Args:
            function_name: Name of the function to test
            func_def: Function definition code
            dependencies: Import statements
            test_type: Type of tests to generate
            num_tests: Number of test cases to generate
            include_edge_cases: Whether to include edge case testing
            
        Returns:
            List of test case dictionaries
        """
        if not self.llm_client:
            return [{"error": "LLM client required to generate test cases"}]
        
        logger.debug(f"Generating {test_type} test cases for function:\n{func_def}")
        
        # Extract function signature to better inform test generation
        signature = self._extract_function_signature(func_def, function_name)
        
        # Create prompt based on test type
        if test_type == "property":
            prompt = self._create_property_test_prompt(
                function_name, func_def, dependencies, signature, num_tests, include_edge_cases
            )
        elif test_type == "integration":
            prompt = self._create_integration_test_prompt(
                function_name, func_def, dependencies, signature, num_tests, include_edge_cases
            )
        else:  # Default to unit tests
            prompt = self._create_unit_test_prompt(
                function_name, func_def, dependencies, signature, num_tests, include_edge_cases
            )
        
        try:
            response = self.llm_client.get_completion(prompt)
            logger.debug(f"Raw LLM response for test cases: {response}")
            
            # Extract JSON from response
            json_match = re.search(r'\[\s*{.*}\s*\]', response, re.DOTALL)
            if json_match:  # Fixed: removed unnecessary parentheses
                json_str = json_match.group(0)
            else:
                json_str = response.strip()
            
            # Clean and parse JSON
            json_str = self._clean_json_string(json_str)
            test_cases = json.loads(json_str)
            
            if not isinstance(test_cases, list):
                raise ValueError("LLM response must be a JSON array")
            
            # Validate and fix test cases
            test_cases = [self._validate_test_case(tc, signature) for tc in test_cases]
            return test_cases
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)} - Response: {response}")
            test_cases = self._attempt_recovery_from_invalid_json(response)
            if test_cases:
                return test_cases
            return [{"error": f"Invalid JSON from LLM: {str(e)}", "raw_response": response}]
        except Exception as e:
            logger.error(f"Error generating test cases: {str(e)}\n{traceback.format_exc()}")
            return [{"error": f"Error generating test cases: {str(e)}"}]

    def _clean_json_string(self, json_str: str) -> str:
        """Clean and format JSON string for parsing."""
        # Fix common issues with LLM-generated JSON
        json_str = re.sub(r'`+json\s*|\s*`+', '', json_str)  # Remove code block markers
        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas in objects
        json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
        
        # Fix Python-style literals
        json_str = json_str.replace("'", '"').replace("None", "null")
        json_str = re.sub(r'b"([^"]*)"', r'"\1"', json_str)  # Convert b"xyz" to "xyz"
        
        # Fix tuple representation
        json_str = re.sub(r'\(([^)]*)\)', r'[\1]', json_str)  # Convert (x,y) to [x,y]
        
        return json_str

    def _attempt_recovery_from_invalid_json(self, response: str) -> List[Dict[str, Any]]:
        """Attempt to recover test cases from invalid JSON."""
        # Look for patterns that might be test cases
        test_cases = []
        
        # Try to find individual test cases by looking for description/input/expected patterns
        case_patterns = re.finditer(r'"description"\s*:\s*"([^"]*)"\s*,\s*"input"\s*:\s*({[^}]*}|\[[^\]]*\])\s*,\s*"expected"\s*:', response)
        
        for match in case_patterns:
            try:
                # Extract the test case snippet and try to parse it
                start_idx = match.start()
                # Find the closing brace/bracket
                level = 0
                end_idx = start_idx
                in_string = False
                escape_next = False
                
                for i in range(start_idx, len(response)):
                    char = response[i]
                    
                    if escape_next:
                        escape_next = False
                        continue
                        
                    if char == '\\':
                        escape_next = True
                    elif char == '"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if char in '{[':
                            level += 1
                        elif char in '}]':
                            level -= 1
                            if level == 0:
                                end_idx = i + 1
                                break
                
                if end_idx > start_idx:
                    case_str = '{' + response[start_idx:end_idx] + '}'
                    case_str = self._clean_json_string(case_str)
                    test_case = json.loads(case_str)
                    test_cases.append(test_case)
            except Exception as e:
                logger.error(f"Failed to recover test case: {str(e)}")
                continue
        
        return test_cases

    def _extract_function_signature(self, func_def: str, function_name: str) -> Dict[str, Any]:
        """
        Extract function signature from definition.
        
        Returns:
            Dictionary with function signature information
        """
        signature = {
            "name": function_name,
            "params": [],
            "return_type": None,
            "is_async": False
        }
        
        # Check if function is async
        if re.search(r'async\s+def\s+' + re.escape(function_name), func_def):
            signature["is_async"] = True
        
        # Extract parameters and return type
        pattern = r'def\s+' + re.escape(function_name) + r'\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?\s*:'
        match = re.search(pattern, func_def)
        
        if match:
            params_str = match.group(1).strip()
            return_type_str = match.group(2).strip() if match.group(2) else None
            
            # Parse parameters
            if params_str:
                # Handle self/cls for methods
                params = []
                for param in params_str.split(','):
                    param = param.strip()
                    if param in ('self', 'cls'):
                        continue
                        
                    # Extract parameter name and type
                    param_match = re.match(r'(\w+)(?:\s*:\s*([^=]+))?(?:\s*=\s*(.+))?', param)
                    if param_match:
                        param_name = param_match.group(1)
                        param_type = param_match.group(2).strip() if param_match.group(2) else None
                        default_value = param_match.group(3) if param_match.group(3) else None
                        
                        params.append({
                            "name": param_name,
                            "type": param_type,
                            "has_default": default_value is not None,
                            "default": default_value
                        })
                
                signature["params"] = params
            
            # Parse return type
            if return_type_str:
                signature["return_type"] = return_type_str
        
        return signature

    def _create_unit_test_prompt(
        self, 
        function_name: str, 
        func_def: str, 
        dependencies: List[str],
        signature: Dict[str, Any],
        num_tests: int,
        include_edge_cases: bool
    ) -> str:
        """Create a prompt for generating unit tests."""
        edge_case_text = "Include edge cases like empty inputs, null values, boundary conditions, etc." if include_edge_cases else ""
        
        params_desc = ""
        for param in signature.get("params", []):
            param_type = f": {param['type']}" if param.get('type') else ""
            default = f" = {param['default']}" if param.get('has_default') and param.get('default') else ""
            params_desc += f"- {param['name']}{param_type}{default}\n"
        
        return_type = f" -> {signature.get('return_type')}" if signature.get('return_type') else ""
        
        prompt = f"""
        Generate {num_tests} unit test cases for the following function. Return *only* a valid JSON array of test case objects.
        
        Function Signature:
        ```python
        def {function_name}({', '.join(p['name'] for p in signature.get('params', []))}) {return_type}
        ```
        
        Parameters:
        {params_desc}
        
        Dependencies:
        ```python
        {"".join(dependencies)}
        ```
        
        Function Definition:
        ```python
        {func_def}
        ```
        
        {edge_case_text}
        
        Each test case must include:
        - "description": string describing the test
        - "input": dict matching the function's parameters (use strings for bytes, e.g., "Hello" for b"Hello")
        - "expected": string or array of strings (use arrays for tuple outputs)
        
        Use JSON-compatible types only (strings, numbers, booleans, arrays, objects).
        For exceptions, use the exception name as a string in "expected" (e.g., "TypeError").
        
        Example format:
        [
            {{"description": "Normal input", "input": {{"param1": "value1"}}, "expected": "expected_result"}},
            {{"description": "Edge case", "input": {{"param1": ""}}, "expected": "ValueError"}}
        ]
        """
        
        return prompt

    def _create_property_test_prompt(
        self, 
        function_name: str, 
        func_def: str, 
        dependencies: List[str],
        signature: Dict[str, Any],
        num_tests: int,
        include_edge_cases: bool
    ) -> str:
        """Create a prompt for generating property-based tests."""
        prompt = f"""
        Generate {num_tests} property-based test cases for the following function. These tests should verify properties 
        that always hold true for the function regardless of specific inputs. Return *only* a valid JSON array of test case objects.
        
        Function Signature:
        ```python
        def {function_name}({', '.join(p['name'] for p in signature.get('params', []))}) {signature.get('return_type', '')}
        ```
        
        Dependencies:
        ```python
        {"".join(dependencies)}
        ```
        
        Function Definition:
        ```python
        {func_def}
        ```
        
        Each test case must include:
        - "description": string describing the property being tested
        - "property_type": one of "invariant", "inverse", "idempotence", "commutativity", "associativity"
        - "input": dict with input parameters that demonstrate the property
        - "property_check": string describing how to verify the property (e.g., "result == input" for idempotence)
        
        Example format:
        [
            {{
                "description": "Sorting is idempotent - sorting an already sorted list gives the same result",
                "property_type": "idempotence",
                "input": {{"items": [1, 2, 3, 4, 5]}},
                "property_check": "sort(sort(items)) == sort(items)"
            }},
            {{
                "description": "Length of concatenated lists equals sum of individual lengths",
                "property_type": "invariant",
                "input": {{"list1": [1, 2], "list2": [3, 4]}},
                "property_check": "len(list1 + list2) == len(list1) + len(list2)"
            }}
        ]
        """
        
        return prompt

    def _create_integration_test_prompt(
        self, 
        function_name: str, 
        func_def: str, 
        dependencies: List[str],
        signature: Dict[str, Any],
        num_tests: int,
        include_edge_cases: bool
    ) -> str:
        """Create a prompt for generating integration tests."""
        prompt = f"""
        Generate {num_tests} integration test cases for the following function. These tests should focus on how this function 
        interacts with other components. Return *only* a valid JSON array of test case objects.
        
        Function Signature:
        ```python
        def {function_name}({', '.join(p['name'] for p in signature.get('params', []))}) {signature.get('return_type', '')}
        ```
        
        Dependencies:
        ```python
        {"".join(dependencies)}
        ```
        
        Function Definition:
        ```python
        {func_def}
        ```
        
        Each test case must include:
        - "description": string describing the integration scenario
        - "setup": list of steps to prepare the test environment
        - "input": dict matching the function's parameters
        - "expected": expected output or behavior
        - "cleanup": list of steps to clean up after the test (optional)
        
        Example format:
        [
            {{
                "description": "Test integration with database",
                "setup": ["Create test database", "Insert test data"],
                "input": {{"user_id": 123}},
                "expected": {{"name": "Test User", "email": "test@example.com"}},
                "cleanup": ["Delete test data", "Close database connection"]
            }}
        ]
        """
        
        return prompt

    def _validate_test_case(self, test_case: Dict[str, Any], signature: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix test case if needed."""
        if "error" in test_case:
            return test_case
            
        # Ensure all required fields are present
        required_fields = ["description", "input", "expected"]
        for field in required_fields:
            if field not in test_case:
                test_case[field] = None
        
        # Validate input against function signature
        if test_case["input"] is not None and isinstance(test_case["input"], dict):
            # Check all required parameters are present
            for param in signature.get("params", []):
                if not param.get("has_default") and param["name"] not in test_case["input"]:
                    # Add a placeholder for missing required parameter
                    test_case["input"][param["name"]] = None
        
        return test_case

    def _execute_tests(self, temp_path: str, required_modules: List[str]) -> Dict[str, Any]:
        """Execute the test file and collect results."""
        from io import StringIO
        import importlib.util
        import sys
        import unittest
        import json
        
        # Redirect stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        captured_stdout = StringIO()
        captured_stderr = StringIO()
        
        sys.stdout = captured_stdout
        sys.stderr = captured_stderr
        
        test_results = {
            "passed": [],
            "failed": [],
            "errors": [],
            "passed_count": 0,
            "total": 0
        }
        
        try:
            # Dynamically import the test module
            module_name = f"test_module_{hash(temp_path)}"
            spec = importlib.util.spec_from_file_location(module_name, temp_path)
            test_module = importlib.util.module_from_spec(spec)
            
            # Add module to sys.modules
            sys.modules[module_name] = test_module
            
            logger.debug(f"Executing test module from {temp_path}")
            spec.loader.exec_module(test_module)
            
            # Collect and run the tests
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            
            # Process test results
            test_results["passed_count"] = result.testsRun - len(result.errors) - len(result.failures)
            test_results["total"] = result.testsRun
            
            # Collect detailed test results from test class instance
            test_class_name = next(
                (name for name in dir(test_module) if name.startswith('Test')), 
                None
            )
            
            if test_class_name:
                test_class = getattr(test_module, test_class_name)
                
                # Use the results dictionary from the last test instance
                test_instances = [t for t in suite._tests if isinstance(t, unittest.TestSuite)]
                if test_instances:
                    for test_case in test_instances[0]._tests:
                        if hasattr(test_case, 'results'):
                            for test_id, result_info in test_case.results.items():
                                if result_info.get("status") == "pass":
                                    test_results["passed"].append({
                                        "name": f"test_{test_id}",
                                        "result": result_info.get("result", ""),
                                        "exception": result_info.get("exception", "")
                                    })
                                else:
                                    test_results["failed"].append({
                                        "name": f"test_{test_id}",
                                        "message": result_info.get("message", ""),
                                        "exception": result_info.get("exception", ""),
                                        "traceback": result_info.get("traceback", "")
                                    })
            
            # Process failures
            for failure in result.failures:
                test_name = failure[0].id().split('.')[-1]
                error_msg = failure[1]
                
                if test_name not in [f["name"] for f in test_results["failed"]]:
                    test_results["failed"].append({
                        "name": test_name,
                        "message": error_msg
                    })
            
            # Process errors
            for error in result.errors:
                test_name = error[0].id().split('.')[-1]
                error_msg = error[1]
                
                test_results["errors"].append({
                    "name": test_name,
                    "message": error_msg
                })
                
        except Exception as e:
            error_msg = f"Error running tests: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            test_results["errors"].append({
                "name": "test_execution",
                "message": error_msg
            })
        
        finally:
            # Restore stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
            # Capture output
            test_results["output"] = captured_stdout.getvalue()
            test_results["error_output"] = captured_stderr.getvalue()
            
            # Remove the module from sys.modules to avoid conflicts
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        return test_results

    def _run_tests(
        self, 
        function_name: str, 
        func_def: str, 
        dependencies: List[str],
        required_modules: List[str],
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run the tests for the function."""
        if not test_cases or "error" in test_cases[0]:
            return {"error": test_cases[0].get("error", "Invalid test cases")}
        
        logger.debug("Starting test code construction")
        
        # Create a temporary directory for test files
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"test_{function_name}.py")
        
        try:
            # Create test code
            test_code = self._generate_test_code(function_name, func_def, dependencies, test_cases)
            
            with open(temp_path, "w", encoding="utf-8") as temp_file:
                temp_file.write(test_code)
                logger.info(f"Temporary test file saved at {temp_path} for inspection")
            
            # Run the tests in a controlled environment
            return self._execute_tests(temp_path, required_modules)
            
        except Exception as e:
            error_msg = f"Error setting up tests: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {"error": error_msg}
        finally:
            # Keep temp files for inspection if DEBUG is enabled
            if os.environ.get("DEBUG"):
                logger.info(f"Test files preserved at {temp_dir} for debugging")
            else:
                # Clean up temp files safely
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp files: {str(e)}")

    # def _generate_test_code(
    #     self, 
    #     function_name: str, 
    #     func_def: str, 
    #     dependencies: List[str], 
    #     test_cases: List[Dict[str, Any]]
    # ) -> str:
    #         """Generate test code for the given function and test cases."""
    #         import json
            
    #         test_code = "import unittest\n"
    #         test_code += "import json\n"
    #         test_code += "import sys\n"
    #         test_code += "import traceback\n"
            
    #         # Add dependencies (imports)
    #         test_code += "".join(dependencies) + "\n\n"
            
    #         # Add the function definition
    #         test_code += func_def + "\n\n"
            
    #         # Add test class
    #         test_code += f"class Test{function_name.capitalize()}(unittest.TestCase):\n"
            
    #         # Add setup and teardown methods
    #         test_code += """    def setUp(self):
    #         # Called before each test
    #         self.results = {}
        
    #     def tearDown(self):
    #         # Called after each test
    #         pass
            
    # """
            
    #         # Add helper methods with properly escaped docstrings
    #         test_code += """    def _convert_to_bytes(self, value):
    #         \"\"\"Convert string values to bytes for testing.\"\"\"
    #         if isinstance(value, str):
    #             return value.encode('utf-8')
    #         elif isinstance(value, dict):
    #             return {k: self._convert_to_bytes(v) for k, v in value.items()}
    #         elif isinstance(value, list):
    #             return [self._convert_to_bytes(v) for v in value]
    #         else:
    #             return value
                
    #     def _convert_from_bytes(self, value):
    #         \"\"\"Convert bytes back to strings for comparison.\"\"\"
    #         if isinstance(value, bytes):
    #             return value.decode('utf-8')
    #         elif isinstance(value, dict):
    #             return {k: self._convert_from_bytes(v) for k, v in value.items()}
    #         elif isinstance(value, list) or isinstance(value, tuple):
    #             return [self._convert_from_bytes(v) for v in value]
    #         else:
    #             return value
                
    # """
            
    #         # Add test methods for each test case
    #         for i, test_case in enumerate(test_cases):
    #             if "error" in test_case:
    #                 continue
                    
    #             description = test_case.get("description", f"Test case {i+1}")
    #             # Escape both single and double quotes in the description
    #             description = description.replace("'", "\\'").replace('"', '\\"')
                
    #             input_data = test_case.get("input", {})
    #             expected = test_case.get("expected", None)
                
    #             # Use json.dumps with proper escaping, then escape again for string embedding
    #             input_str = json.dumps(input_data).replace("'", "\\'").replace('"', '\\"')
    #             expected_str = json.dumps(expected).replace("'", "\\'").replace('"', '\\"')
                
    #             # Build test method with consistent indentation and safe JSON parsing
    #             test_method = f"    def test_{i+1}(self):\n"
    #             test_method += f'        \"\"\"Test: {description}\"\"\"\n'
    #             test_method += "        try:\n"
    #             test_method += f'            inputs = json.loads("{input_str}")\n'
    #             test_method += f'            expected = json.loads("{expected_str}")\n'
    #             test_method += "\n"
    #             test_method += "            # Handle different input types\n"
    #             test_method += "            if isinstance(inputs, dict):\n"
    #             test_method += "                # Convert string to bytes for byte inputs\n"
    #             test_method += "                inputs = self._convert_to_bytes(inputs)\n"
    #             test_method += f"                result = {function_name}(**inputs)\n"
    #             test_method += "            elif isinstance(inputs, list):\n"
    #             test_method += "                inputs = self._convert_to_bytes(inputs)\n"
    #             test_method += f"                result = {function_name}(*inputs)\n"
    #             test_method += "            else:\n"
    #             test_method += "                if isinstance(inputs, str):\n"
    #             test_method += "                    inputs = inputs.encode('utf-8')\n"
    #             test_method += f"                result = {function_name}(inputs)\n"
    #             test_method += "\n"
    #             test_method += "            # Convert result bytes to strings for comparison\n"
    #             test_method += "            result = self._convert_from_bytes(result)\n"
    #             test_method += "\n"
    #             test_method += "            # Handle expected exceptions\n"
    #             test_method += '            if isinstance(expected, str) and expected in ["TypeError", "ValueError", "KeyError", "AttributeError"]:\n'
    #             test_method += '                self.fail(f"Expected {expected} but got result: {result}")\n'
    #             test_method += "            else:\n"
    #             test_method += "                self.assertEqual(result, expected)\n"
    #             test_method += "                \n"
    #             test_method += f"            self.results['{i+1}'] = {{\"status\": \"pass\", \"result\": result}}\n"
    #             test_method += "            \n"
    #             test_method += "        except Exception as e:\n"
    #             test_method += "            exc_type = type(e).__name__\n"
    #             test_method += "            exc_message = str(e)\n"
    #             test_method += "            \n"
    #             test_method += "            # Check if this was an expected exception\n"
    #             test_method += "            if isinstance(expected, str) and exc_type == expected:\n"
    #             test_method += f"                self.results['{i+1}'] = {{\"status\": \"pass\", \"exception\": exc_type}}\n"
    #             test_method += "            else:\n"
    #             test_method += f"                self.results['{i+1}'] = {{\n"
    #             test_method += '                    "status": "fail", \n'
    #             test_method += '                    "exception": exc_type,\n'
    #             test_method += '                    "message": exc_message,\n'
    #             test_method += '                    "traceback": traceback.format_exc()\n'
    #             test_method += "                }}\n"
    #             test_method += '                self.fail(f"Test failed with exception: {exc_type}: {exc_message}")\n'
                
    #             test_code += test_method + "\n"
            
    #         # Add main test runner
    #         test_code += "\n"
    #         test_code += "if __name__ == '__main__':\n"
    #         test_code += "    unittest.main()\n"
            
    #         return test_code
    def _generate_test_code(
    self, 
    function_name: str, 
    func_def: str, 
    dependencies: List[str], 
    test_cases: List[Dict[str, Any]]
) -> str:
        """Generate test code for the given function and test cases."""
        import json
        
        test_code = "import unittest\n"
        test_code += "import json\n"
        test_code += "import sys\n"
        test_code += "import traceback\n"
        
        # Add dependencies (imports)
        test_code += "".join(dependencies) + "\n\n"
        
        # Add the function definition
        test_code += func_def + "\n\n"
        
        # Add test class
        test_code += f"class Test{function_name.capitalize()}(unittest.TestCase):\n"
        
        # Add setup and teardown methods - FIXED: Fixed indentation
        test_code += """    def setUp(self):
        # Called before each test
        self.results = {}
    
    def tearDown(self):
        # Called after each test
        pass
        
"""
        
        # Add helper methods with properly escaped docstrings
        test_code += """    def _convert_to_bytes(self, value):
        \"\"\"Convert string values to bytes for testing.\"\"\"
        if isinstance(value, str):
            return value.encode('utf-8')
        elif isinstance(value, dict):
            return {k: self._convert_to_bytes(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._convert_to_bytes(v) for v in value]
        else:
            return value
            
    def _convert_from_bytes(self, value):
        \"\"\"Convert bytes back to strings for comparison.\"\"\"
        if isinstance(value, bytes):
            return value.decode('utf-8')
        elif isinstance(value, dict):
            return {k: self._convert_from_bytes(v) for k, v in value.items()}
        elif isinstance(value, list) or isinstance(value, tuple):
            return [self._convert_from_bytes(v) for v in value]
        else:
            return value
            
"""
        
        # Add test methods for each test case
        for i, test_case in enumerate(test_cases):
            if "error" in test_case:
                continue
                
            description = test_case.get("description", f"Test case {i+1}")
            # Escape both single and double quotes in the description
            description = description.replace("'", "\\'").replace('"', '\\"')
            
            input_data = test_case.get("input", {})
            expected = test_case.get("expected", None)
            
            # Use json.dumps with proper escaping, then escape again for string embedding
            input_str = json.dumps(input_data).replace("'", "\\'").replace('"', '\\"')
            expected_str = json.dumps(expected).replace("'", "\\'").replace('"', '\\"')
            
            # Build test method with consistent indentation and safe JSON parsing
            test_method = f"    def test_{i+1}(self):\n"
            test_method += f'        \"\"\"Test: {description}\"\"\"\n'
            test_method += "        try:\n"
            test_method += f'            inputs = json.loads("{input_str}")\n'
            test_method += f'            expected = json.loads("{expected_str}")\n'
            test_method += "\n"
            test_method += "            # Handle different input types\n"
            test_method += "            if isinstance(inputs, dict):\n"
            test_method += "                # Convert string to bytes for byte inputs\n"
            test_method += "                inputs = self._convert_to_bytes(inputs)\n"
            test_method += f"                result = {function_name}(**inputs)\n"
            test_method += "            elif isinstance(inputs, list):\n"
            test_method += "                inputs = self._convert_to_bytes(inputs)\n"
            test_method += f"                result = {function_name}(*inputs)\n"
            test_method += "            else:\n"
            test_method += "                if isinstance(inputs, str):\n"
            test_method += "                    inputs = inputs.encode('utf-8')\n"
            test_method += f"                result = {function_name}(inputs)\n"
            test_method += "\n"
            test_method += "            # Convert result bytes to strings for comparison\n"
            test_method += "            result = self._convert_from_bytes(result)\n"
            test_method += "\n"
            test_method += "            # Handle expected exceptions\n"
            test_method += '            if isinstance(expected, str) and expected in ["TypeError", "ValueError", "KeyError", "AttributeError"]:\n'
            test_method += '                self.fail(f"Expected {expected} but got result: {result}")\n'
            test_method += "            else:\n"
            test_method += "                self.assertEqual(result, expected)\n"
            test_method += "                \n"
            test_method += f"            self.results['{i+1}'] = {{\"status\": \"pass\", \"result\": result}}\n"
            test_method += "            \n"
            test_method += "        except Exception as e:\n"
            test_method += "            exc_type = type(e).__name__\n"
            test_method += "            exc_message = str(e)\n"
            test_method += "            \n"
            test_method += "            # Check if this was an expected exception\n"
            test_method += "            if isinstance(expected, str) and exc_type == expected:\n"
            test_method += f"                self.results['{i+1}'] = {{\"status\": \"pass\", \"exception\": exc_type}}\n"
            test_method += "            else:\n"
            test_method += f"                self.results['{i+1}'] = {{\n"
            test_method += '                    "status": "fail", \n'
            test_method += '                    "exception": exc_type,\n'
            test_method += '                    "message": exc_message,\n'
            test_method += '                    "traceback": traceback.format_exc()\n'
            test_method += "                }}\n"
            test_method += '                self.fail(f"Test failed with exception: {exc_type}: {exc_message}")\n'
            
            test_code += test_method + "\n"
        
        # Add main test runner
        test_code += "\n"
        test_code += "if __name__ == '__main__':\n"
        test_code += "    unittest.main()\n"
        
        return test_code
    
    def format_results(self, results: Dict[str, Any], question: str, llm_client: Any = None) -> str:
        """Format test results into a readable answer."""
        if "error" in results:
            return f"Error testing function: {results['error']}"
            
        function_name = results.get("function_name", "the function")
        test_cases = results.get("test_cases", [])
        test_results = results.get("results", {})
        
        passed_count = test_results.get("passed_count", 0)
        total = test_results.get("total", 0)
        failed = test_results.get("failed", [])
        errors = test_results.get("errors", [])
        
        # Create a summary
        summary = f"# Testing Results for `{function_name}`\n\n"
        
        if total > 0:
            summary += f"✅ **{passed_count}/{total}** tests passed\n\n"
        
        if failed:
            summary += "## Failed Tests\n\n"
            for fail in failed:
                summary += f"- **{fail.get('name', 'Unknown')}**: {fail.get('message', 'No message')}\n\n"
                if fail.get('traceback'):
                    summary += f"```\n{fail.get('traceback')}\n```\n\n"
                
        if errors:
            summary += "## Errors\n\n"
            for error in errors:
                summary += f"- **{error.get('name', 'Unknown')}**: {error.get('message', 'No message')}\n\n"
                
        summary += "## Test Cases\n\n"
        for i, test in enumerate(test_cases):
            if "error" in test:
                continue
                
            # Check if this test passed by comparing with passed test names
            passed_names = [p.get("name", "") for p in test_results.get("passed", [])]
            is_passed = f"test_{i+1}" in passed_names
            
            status = "✅" if is_passed else "❌"
            summary += f"{status} **Test {i+1}**: {test.get('description', 'No description')}\n"
            summary += f"  - Input: `{test.get('input', '')}`\n"
            summary += f"  - Expected: `{test.get('expected', '')}`\n\n"
            
        # Use LLM to provide additional analysis if available
        if llm_client and (failed or errors):
            analysis_prompt = f"""
            Based on these test results for the function '{function_name}':
            
            Passed: {passed_count}/{total} tests
            
            Failed tests: {failed}
            
            Errors: {errors}
            
            Please provide a brief analysis of what might be wrong with the function and how to fix it.
            Be specific and focus on the likely issues based on the test failures.
            """
            
            try:
                analysis = llm_client.get_completion(analysis_prompt)
                summary += "## Analysis\n\n" + analysis
            except Exception as e:
                logger.error(f"Error generating analysis: {str(e)}")
                
        return summary