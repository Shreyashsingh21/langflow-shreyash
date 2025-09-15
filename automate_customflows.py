#!/usr/bin/env python3
"""
Automation script to integrate custom components from customflows/ folder into Langflow.

This script automatically:
1. Scans the customflows/ directory for Python components
2. Copies them to the Langflow components directory under 'custom_flows' category
3. Updates the necessary __init__.py files
4. Validates component structure before integration
"""

import ast
import shutil
import importlib.util
import os
import sys
import sysconfig
from pathlib import Path
from typing import List, Dict, Any, Optional


class ComponentValidator:
    """Validates that Python files are proper Langflow components."""
    
    @staticmethod
    def is_langflow_component(file_path: Path) -> Dict[str, Any]:
        """Check if a Python file contains a valid Langflow component."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST to find component classes
            tree = ast.parse(content)
            
            component_info = {
                'is_valid': False,
                'component_classes': [],
                'imports_langflow': False,
                'error': None
            }
            
            # Check for Langflow imports
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and 'langflow' in node.module:
                        component_info['imports_langflow'] = True
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if 'langflow' in alias.name:
                            component_info['imports_langflow'] = True
            
            # Find component classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Validate class name (no hyphens, valid Python identifier)
                    if not node.name.isidentifier() or '-' in node.name:
                        component_info['error'] = f"Invalid class name '{node.name}' - must be valid Python identifier"
                        return component_info
                    
                    # Check if class inherits from Langflow component base classes
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            if base.id in ['LCModelComponent', 'Component', 'CustomComponent', 'ToolCallingAgentComponent', 'LCToolsAgentComponent']:
                                component_info['component_classes'].append({
                                    'name': node.name,
                                    'base_class': base.id
                                })
                        elif isinstance(base, ast.Attribute):
                            if base.attr in ['LCModelComponent', 'Component', 'CustomComponent', 'ToolCallingAgentComponent', 'LCToolsAgentComponent']:
                                component_info['component_classes'].append({
                                    'name': node.name,
                                    'base_class': base.attr
                                })
            
            if component_info['imports_langflow'] and component_info['component_classes']:
                component_info['is_valid'] = True
            
            return component_info
            
        except Exception as e:
            return {
                'is_valid': False,
                'component_classes': [],
                'imports_langflow': False,
                'error': str(e)
            }


class CustomflowsAutomation:
    """Main automation class for integrating customflows into Langflow."""
    
    def __init__(self, workspace_path: Optional[str] = None):
        # Resolve workspace path; default to the directory containing this script
        base_path = Path(workspace_path) if workspace_path else Path(__file__).resolve().parent
        self.workspace_path = base_path
        self.customflows_dir = self.workspace_path / "customflows"
        # Resolve Langflow components path dynamically (active venv / environment)
        self.langflow_components_path = self._resolve_langflow_components_path()
        # Category folder name shown in Langflow sidebar (no spaces). Override with CUSTOM_CATEGORY_NAME env.
        self.custom_category_name = os.environ.get("CUSTOM_CATEGORY_NAME", "SelfDeployedModels")
        self.validator = ComponentValidator()

    def _resolve_langflow_components_path(self) -> Path:
        """Resolve the langflow/components directory in the active environment.

        Resolution order:
        1) ENV override: LANGFLOW_COMPONENTS_PATH
        2) importlib: path from installed langflow module
        3) sysconfig: purelib site-packages + /langflow/components
        4) Workspace-local venv guesses: .venv, venv (all pythonX.Y variants)
        """
        # 1) Environment override
        env_override = os.environ.get("LANGFLOW_COMPONENTS_PATH")
        if env_override:
            path = Path(env_override).expanduser().resolve()
            return path

        # 2) From installed module
        try:
            spec = importlib.util.find_spec("langflow")
            if spec and spec.origin:
                langflow_pkg = Path(spec.origin).resolve().parent  # .../site-packages/langflow
                components_path = langflow_pkg / "components"
                if components_path.exists():
                    return components_path
        except Exception:
            pass

        # 3) sysconfig purelib
        try:
            purelib = sysconfig.get_paths().get("purelib")
            if purelib:
                components_path = Path(purelib) / "langflow" / "components"
                if components_path.exists():
                    return components_path
        except Exception:
            pass

        # 4) Guess common local venv layouts inside workspace
        venv_roots = [self.workspace_path / ".venv", self.workspace_path / "venv"]
        candidates = []
        for root in venv_roots:
            if root.exists():
                # Support multiple python versions
                lib_dir = root / "lib"
                if lib_dir.exists():
                    for py_dir in lib_dir.glob("python*"):
                        candidates.append(py_dir / "site-packages" / "langflow" / "components")
                # Some distros use site-packages directly under venv
                candidates.append(root / "site-packages" / "langflow" / "components")

        for c in candidates:
            if c.exists():
                return c

        # As a last resort, return the most likely sysconfig path even if not exists
        fallback_purelib = sysconfig.get_paths().get("purelib", "")
        if fallback_purelib:
            return Path(fallback_purelib) / "langflow" / "components"

        # Ultimate fallback to workspace .venv structure (may not exist)
        return self.workspace_path / ".venv/lib/python3/site-packages/langflow/components"
        
    def scan_customflows_directory(self) -> List[Path]:
        """Scan the customflows directory for Python component files."""
        if not self.customflows_dir.exists():
            print(f"customflows directory not found: {self.customflows_dir}")
            return []
        
        python_files = []
        for file_path in self.customflows_dir.rglob("*.py"):
            if file_path.name != "__init__.py":
                python_files.append(file_path)
        
        print(f"Found {len(python_files)} Python files in customflows")
        return python_files
    
    def validate_components(self, python_files: List[Path]) -> List[Dict[str, Any]]:
        """Validate that Python files are proper Langflow components."""
        valid_components = []
        
        for file_path in python_files:
            print(f"Validating: {file_path.name}")
            validation_result = self.validator.is_langflow_component(file_path)
            
            if validation_result['is_valid']:
                component_info = {
                    'file_path': file_path,
                    'file_name': file_path.name,
                    'component_classes': validation_result['component_classes']
                }
                valid_components.append(component_info)
                print(f"  Valid component with classes: {[c['name'] for c in validation_result['component_classes']]}")
            else:
                if validation_result['error']:
                    print(f"  Error parsing {file_path.name}: {validation_result['error']}")
                elif not validation_result['imports_langflow']:
                    print(f"  {file_path.name} doesn't import langflow - skipping")
                elif not validation_result['component_classes']:
                    print(f"  {file_path.name} doesn't contain component classes - skipping")
                else:
                    print(f"  {file_path.name} is not a valid Langflow component")
        
        return valid_components
    
    def create_custom_category_directory(self) -> Path:
        """Create the custom_flows category directory structure."""
        custom_flows_dir = self.langflow_components_path / self.custom_category_name
        custom_flows_dir.mkdir(exist_ok=True)
        
        # Create __init__.py for the category
        init_file = custom_flows_dir / "__init__.py"
        if not init_file.exists():
            init_content = f'"""Custom flows components from customflows/ directory."""\n'
            init_file.write_text(init_content)
            print(f"Created {self.custom_category_name} category directory")
        else:
            print(f"{self.custom_category_name} category directory already exists")
        
        return custom_flows_dir
    
    def copy_components_to_langflow(self, valid_components: List[Dict[str, Any]], target_dir: Path) -> List[str]:
        """Copy valid components to the Langflow components directory."""
        copied_files = []
        
        for component_info in valid_components:
            source_file = component_info['file_path']
            target_file = target_dir / component_info['file_name']
            
            try:
                # Copy the component file
                shutil.copy2(source_file, target_file)
                copied_files.append(component_info['file_name'])
                print(f"Copied: {component_info['file_name']}")
                
                # Log component classes found
                for cls_info in component_info['component_classes']:
                    print(f"  Component: {cls_info['name']} (extends {cls_info['base_class']})")
                    
            except Exception as e:
                print(f"Failed to copy {source_file.name}: {str(e)}")
        
        return copied_files
    
    def update_category_init_file(self, target_dir: Path, copied_files: List[str]):
        """Update the category's __init__.py file to import the copied components."""
        init_file = target_dir / "__init__.py"
        
        # Read current content
        if init_file.exists():
            current_content = init_file.read_text()
        else:
            current_content = f'"""Custom flows components from customflows/ directory."""\n\n'
        
        # Add imports for each copied component
        imports_to_add = []
        for file_name in copied_files:
            module_name = file_name.replace('.py', '')
            import_line = f"from .{module_name} import *\n"
            
            if import_line not in current_content:
                imports_to_add.append(import_line)
        
        if imports_to_add:
            if current_content and not current_content.endswith('\n'):
                current_content += '\n'
            current_content += ''.join(imports_to_add)
            init_file.write_text(current_content)
            print(f"Updated {self.custom_category_name}/__init__.py with {len(imports_to_add)} new imports")
        else:
            print(f"{self.custom_category_name}/__init__.py already up to date")
    
    def update_main_components_init_file(self):
        """Update the main components __init__.py to include the custom category."""
        components_init_path = self.langflow_components_path / "__init__.py"
        
        # Read the current content
        if components_init_path.exists():
            current_content = components_init_path.read_text()
        else:
            current_content = ""
        
        # Add import for our custom category if not already present
        new_import = f"from . import {self.custom_category_name}\n"
        
        if new_import not in current_content:
            if current_content and not current_content.endswith('\n'):
                current_content += '\n'
            current_content += new_import
            components_init_path.write_text(current_content)
            print("Updated main components __init__.py")
        else:
            print("Main components __init__.py already updated")
    
    def create_backup(self):
        """Create a backup of the current Langflow components directory."""
        backup_dir = self.workspace_path / "langflow_components_backup"
        if not backup_dir.exists():
            try:
                shutil.copytree(self.langflow_components_path, backup_dir)
                print(f"Created backup at: {backup_dir}")
            except Exception as e:
                print(f"Could not create backup: {str(e)}")
    
    def run_automation(self, create_backup: bool = True):
        """Run the complete automation process."""
        print("Starting customflows automation")
        print(f"Source: {self.customflows_dir}")
        print(f"Target: {self.langflow_components_path / self.custom_category_name}")
        print("-" * 60)

        # Sanity check for target parent directory existence
        if not (self.langflow_components_path.parent).exists():
            print("Could not locate Langflow installation in the active environment.")
            print("Hints:")
            print(" - Ensure your virtual environment is activated before running this script.")
            print(" - Install langflow in the active environment: pip install langflow")
            print(" - Or set LANGFLOW_COMPONENTS_PATH to the correct components directory.")
            print(f" - Checked path: {self.langflow_components_path}")
            return
        
        # Create backup if requested
        if create_backup:
            self.create_backup()
        
        # Step 1: Scan for Python files
        python_files = self.scan_customflows_directory()
        if not python_files:
            print("No Python files found in customflows directory")
            return
        
        # Step 2: Validate components
        valid_components = self.validate_components(python_files)
        if not valid_components:
            print("No valid Langflow components found")
            return
        
        print(f"\nFound {len(valid_components)} valid component(s)")
        
        # Step 3: Create custom category directory
        target_dir = self.create_custom_category_directory()
        
        # Step 4: Copy components
        copied_files = self.copy_components_to_langflow(valid_components, target_dir)
        
        if copied_files:
            # Step 5: Update __init__.py files
            self.update_category_init_file(target_dir, copied_files)
            self.update_main_components_init_file()
            
            print(f"\nAutomation completed successfully")
            print(f"Integrated {len(copied_files)} component(s):")
            for file_name in copied_files:
                print(f"   - {file_name}")
            print(f"\nRestart Langflow to see your custom components in the '{self.custom_category_name}' section")
        else:
            print("No components were successfully copied")


def main():
    """Main function to run the automation."""
    automation = CustomflowsAutomation()
    automation.run_automation(create_backup=True)


if __name__ == "__main__":
    main()
