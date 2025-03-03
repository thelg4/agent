# codeassistant/agents/tools/testing_tool.py
import logging
import os
import sys
import tempfile
import unittest
import importlib.util
import inspect
import json
import re
from typing import Dict, List, Any, Optional


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
        
    def run(self, function_name: str, code_analyzer=None) -> Dict[str, Any]:
        """Test a specific function in the codebase."""
        # Force reload to ensure latest code
        if 'codeassistant.agents.tools.testing_tool' in sys.modules:
            del sys.modules['codeassistant.agents.tools.testing_tool']
        logger.debug("Reloaded FunctionTestingTool to ensure latest version")
        
        if not code_analyzer:
            return {"error": "Code analyzer required to find the function"}
        
        functions = code_analyzer.find_function(function_name)
        if not functions:
            return {"error": f"Function '{function_name}' not found in the codebase"}
        
        function_info = functions[0]
        function_info["code_analyzer"] = code_analyzer
        
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
                
        if not source_file:
            return {"error": f"Could not locate source file for function '{function_name}'"}
        
        func_def, dependencies = self._extract_function(source_file, function_info)
        if not func_def:
            return {"error": f"Could not extract definition for function '{function_name}'"}
        
        test_cases = self._generate_test_cases(function_name, func_def, dependencies)
        test_results = self._run_tests(function_name, func_def, dependencies, test_cases)
        
        return {
            "function_name": function_name,
            "test_cases": test_cases,
            "results": test_results
        }
        
    def _extract_function(self, source_file: str, function_info: Dict[str, Any]) -> tuple:
        """Extract all function definitions from the module containing the target function."""
        try:
            with open(source_file, "r", encoding="utf-8") as f:
                source_lines = f.readlines()
            
            # Use the CodeAnalyzer's pre-parsed module data
            module_name = function_info["module"]
            module_info = function_info.get("code_analyzer").find_module(module_name)
            if not module_info:
                logger.error(f"Module '{module_name}' not found in CodeAnalyzer data")
                return None, []
            
            # Collect all function definitions from the module
            func_defs = []
            for f in module_info.functions:
                f_start = f.line_number - 1  # 0-based index
                f_end = f.end_line_number
                if f_start < 0 or f_end > len(source_lines) or f_start >= f_end:
                    logger.error(f"Invalid line range for '{f.name}': {f_start+1} to {f_end}")
                    continue
                func_defs.append("".join(source_lines[f_start:f_end]))
            
            if not func_defs:
                logger.error(f"No function definitions found in module '{module_name}'")
                return None, []
            
            # Combine all function definitions
            combined_func_def = "\n\n".join(func_defs)
            
            # Extract dependencies (imports) up to the first function
            dependencies = []
            first_func_start = min(f.line_number - 1 for f in module_info.functions if f.line_number > 0)
            for i, line in enumerate(source_lines[:first_func_start]):
                line = line.strip()
                if line.startswith("import ") or line.startswith("from "):
                    # Filter out app.* imports to avoid external module errors
                    if not line.startswith("from app") and not line.startswith("import app"):
                        dependencies.append(line + "\n")
            
            return combined_func_def, dependencies
    
        except Exception as e:
            logger.error(f"Error extracting functions: {str(e)}")
            return None, []
            
    def _generate_test_cases(self, function_name: str, func_def: str, dependencies: List[str]) -> List[Dict[str, Any]]:
        """Generate test cases for the function using LLM."""
        if not self.llm_client:
            return [{"error": "LLM client required to generate test cases"}]
        
        logger.debug(f"Generating test cases for function:\n{func_def}")
        prompt = f"""
        Generate 3-5 test cases for the following function. Return *only* a valid JSON array of test case objects, with no additional text outside the JSON and no trailing comma after the last item. Each test case must include:
        - "description": string describing the test
        - "input": dict matching the function's parameters (use strings for bytes, e.g., "Hello" for b"Hello")
        - "expected": string or array of strings (use arrays for tuple outputs, e.g., ["Hello", " World"])
        
        Use JSON-compatible types only (strings, numbers, booleans, arrays, objects). For bytes inputs, use plain strings and escape special characters properly (e.g., "\\x00" for null bytes). For tuple outputs, use arrays. For exceptions, use the exception name as a string.
        
        Dependencies:
        {"".join(dependencies)}
        
        Function Definition:
        {func_def}
        
        Example output:
        [
            {{"description": "Normal input", "input": {{"buffer": "Hello World"}}, "expected": ["Hello", " World"]}},
            {{"description": "Empty input", "input": {{"buffer": ""}}, "expected": ["", ""]}},
            {{"description": "Invalid input", "input": {{"buffer": null}}, "expected": "TypeError"}}
        ]
        """
        
        try:
            response = self.llm_client.get_completion(prompt)
            logger.debug(f"Raw LLM response for test cases: {response}")
            
            test_cases = json.loads(response.strip())
            if not isinstance(test_cases, list):
                raise ValueError("LLM response must be a JSON array")
            
            return test_cases
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)} - Response: {response}")
            # Fix common issues: quotes, tuples, and escape sequences
            fixed_response = response.replace("b'", '"').replace("'", '"').replace("(", "[").replace(")", "]")
            # Escape invalid \x sequences (e.g., \x00 -> \\x00)
            fixed_response = re.sub(r'\\x([0-9a-fA-F]{2})', r'\\\\x\1', fixed_response)
            fixed_response = re.sub(r',\s*]', ']', fixed_response.strip())
            try:
                test_cases = json.loads(fixed_response)
                logger.info("Fixed malformed JSON from LLM response")
                return test_cases
            except json.JSONDecodeError as e2:
                logger.error(f"Failed to fix JSON: {str(e2)}")
                return [{"error": f"Invalid JSON from LLM: {str(e)}"}]
        except Exception as e:
            logger.error(f"Error generating test cases: {str(e)}")
            return [{"error": f"Error generating test cases: {str(e)}"}]

    def _run_tests(
        self, 
        function_name: str, 
        func_def: str, 
        dependencies: List[str], 
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run the tests for the function."""
        import json
        logger.info("Entering _run_tests from /Users/larrygunteriv/github/agent/agents/tools/testing_tool.py")
        if not test_cases or "error" in test_cases[0]:
            return {"error": test_cases[0].get("error", "Invalid test cases")}
        
        logger.debug("Starting test code construction")
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_path = temp_file.name
            
            test_code = "import unittest\n"
            test_code += "import json\n"
            logger.debug("Added imports to test_code")
            
            # Add dependencies (imports like 'from ... import ...')
            test_code += "".join(dependencies) + "\n\n"
            logger.debug("Added dependencies to test_code")
            
            # Add the main function definition
            test_code += func_def + "\n\n"
            logger.debug("Added function definition to test_code")
            
            # Add find_frame_end definition (assuming it's in the same module)
            # We'll need to extract it dynamically; for now, assume a simple version
            # If complex, we'll need to enhance _extract_function
            test_code += (
                "def find_frame_end(buffer: bytes) -> int:\n"
                "    # Simplified mock for testing; replace with actual extraction if needed\n"
                "    return buffer.find(b' ') if b' ' in buffer else len(buffer)\n\n"
            )
            logger.debug("Added find_frame_end definition to test_code")
            
            test_code += f"class Test{function_name.capitalize()}(unittest.TestCase):\n"
            logger.debug("Added class definition to test_code")
            
            for i, test_case in enumerate(test_cases):
                if "error" in test_case:
                    continue
                    
                description = test_case.get("description", f"Test case {i+1}")
                input_data = test_case.get("input", {})
                expected = test_case.get("expected", None)
                
                logger.debug(f"Building test case {i+1}: {description}")
                input_str = json.dumps(input_data)
                expected_str = json.dumps(expected)
                
                test_method = (
                    f"    def test_{i+1}(self):\n"
                    f"        \"\"\"Test: {description}\"\"\"\n"
                    f"        inputs = json.loads('''{input_str}''')\n"
                    f"        expected = json.loads('''{expected_str}''')\n"
                    f"        \n"
                    f"        # Convert inputs to bytes\n"
                    f"        if isinstance(inputs, dict):\n"
                    f"            for key in inputs:\n"
                    f"                if isinstance(inputs[key], str):\n"
                    f"                    inputs[key] = inputs[key].encode('utf-8')\n"
                    f"            result = {function_name}(**inputs)\n"
                    f"        elif isinstance(inputs, list):\n"
                    f"            inputs = [x.encode('utf-8') if isinstance(x, str) else x for x in inputs]\n"
                    f"            result = {function_name}(*inputs)\n"
                    f"        else:\n"
                    f"            if isinstance(inputs, str):\n"
                    f"                inputs = inputs.encode('utf-8')\n"
                    f"            result = {function_name}(inputs)\n"
                    f"        \n"
                    f"        # Convert result bytes to strings for comparison if expected is a list\n"
                    f"        if isinstance(expected, list):\n"
                    f"            result = [r.decode('utf-8') if isinstance(r, bytes) else r for r in result]\n"
                    f"        elif isinstance(expected, str) and expected not in [\"TypeError\", \"ValueError\"]:\n"
                    f"            if isinstance(result, bytes):\n"
                    f"                result = result.decode('utf-8')\n"
                    f"            \n"
                    f"        self.assertEqual(result, expected)\n"
                )
                test_code += test_method
                logger.debug(f"Added test case {i+1} to test_code")
            
            test_code += "\n"
            test_code += "if __name__ == '__main__':\n"
            test_code += "    unittest.main()\n"
            logger.debug(f"Completed test_code construction:\n{test_code}")
            temp_file.write(test_code.encode('utf-8'))
            logger.info(f"Temporary test file saved at {temp_path} for inspection")
        
        try:
            from io import StringIO
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            
            test_results = {
                "passed": [],
                "failed": [],
                "errors": []
            }
            
            try:
                loader = unittest.TestLoader()
                spec = importlib.util.spec_from_file_location("test_module", temp_path)
                test_module = importlib.util.module_from_spec(spec)
                logger.debug(f"Executing test module from {temp_path}")
                spec.loader.exec_module(test_module)
                
                suite = loader.loadTestsFromModule(test_module)
                result = unittest.TextTestRunner(verbosity=2).run(suite)
                # ... (rest of result processing unchanged)
            except Exception as e:
                test_results["errors"].append({
                    "name": "test_execution",
                    "message": str(e)
                })
                logger.error(f"Test execution failed: {str(e)}")
                
            test_output = sys.stdout.getvalue()
            test_errors = sys.stderr.getvalue()
            
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
            test_results["output"] = test_output
            test_results["error_output"] = test_errors
            
            return test_results
        
        finally:
            pass  # Keep temp file for inspection
                
    def format_results(self, results: Dict[str, Any], question: str, llm_client: Any) -> str:
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
                summary += f"- **{fail['name']}**: {fail['message']}\n\n"
                
        if errors:
            summary += "## Errors\n\n"
            for error in errors:
                summary += f"- **{error['name']}**: {error['message']}\n\n"
                
        summary += "## Test Cases\n\n"
        for i, test in enumerate(test_cases):
            if "error" in test:
                continue
                
            status = "✅" if i < passed_count else "❌"
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