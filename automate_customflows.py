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
from tqdm import tqdm
import time


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
    """Main automation class for integrating customflows and customtools into Langflow."""
    
    def __init__(self, workspace_path: Optional[str] = None):
        # Resolve workspace path; default to the directory containing this script
        base_path = Path(workspace_path) if workspace_path else Path(__file__).resolve().parent
        self.workspace_path = base_path
        self.custommodels_dir = self.workspace_path / "custommodels"
        self.customtools_dir = self.workspace_path / "customtools"
        # Resolve Langflow components path dynamically (active venv / environment)
        self.langflow_components_path = self._resolve_langflow_components_path()
        # Category folder names shown in Langflow sidebar (no spaces). Override with env vars.
        self.models_category_name = os.environ.get("MODELS_CATEGORY_NAME", "OwnDeployedModels")
        self.tools_category_name = os.environ.get("TOOLS_CATEGORY_NAME", "OwnDeployedTools")
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
        venv_roots = [
            self.workspace_path / ".langflow-uat-env",
            self.workspace_path / "langflow-uat-env",
            self.workspace_path / ".venv",
            self.workspace_path / "venv",
        ]
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
        
    def scan_directory(self, directory: Path, directory_name: str) -> List[Path]:
        """Scan a directory for Python component files."""
        if not directory.exists():
            print(f"{directory_name} directory not found: {directory}")
            return []
        
        python_files = []
        for file_path in directory.rglob("*.py"):
            if file_path.name != "__init__.py":
                python_files.append(file_path)
        
        print(f"Found {len(python_files)} Python files in {directory_name}")
        return python_files
    
    def scan_all_directories(self) -> Dict[str, List[Path]]:
        """Scan both custommodels and customtools directories."""
        all_files = {}
        
        # Scan custommodels directory
        models_files = self.scan_directory(self.custommodels_dir, "custommodels")
        if models_files:
            all_files["models"] = models_files
        
        # Scan customtools directory
        tools_files = self.scan_directory(self.customtools_dir, "customtools")
        if tools_files:
            all_files["tools"] = tools_files
        
        return all_files
    
    def validate_components(self, python_files: List[Path], progress_desc: Optional[str] = None) -> List[Dict[str, Any]]:
        """Validate that Python files are proper Langflow components.

        When progress_desc is provided, shows a tqdm progress bar with that description.
        """
        valid_components = []

        iterable = python_files
        progress = None
        if progress_desc is not None:
            progress = tqdm(
                total=len(python_files),
                desc=progress_desc,
                unit="file",
                position=1,
                leave=False,
                dynamic_ncols=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

        try:
            for file_path in iterable:
                msg_prefix = f"Validating: {file_path.name}"
                if progress is None:
                    print(msg_prefix)

                validation_result = self.validator.is_langflow_component(file_path)

                if validation_result['is_valid']:
                    component_info = {
                        'file_path': file_path,
                        'file_name': file_path.name,
                        'component_classes': validation_result['component_classes']
                    }
                    valid_components.append(component_info)
                    valid_msg = f"  Valid component with classes: {[c['name'] for c in validation_result['component_classes']]}"
                    if progress is None:
                        print(valid_msg)
                else:
                    if validation_result['error']:
                        err_msg = f"  Error parsing {file_path.name}: {validation_result['error']}"
                    elif not validation_result['imports_langflow']:
                        err_msg = f"  {file_path.name} doesn't import langflow - skipping"
                    elif not validation_result['component_classes']:
                        err_msg = f"  {file_path.name} doesn't contain component classes - skipping"
                    else:
                        err_msg = f"  {file_path.name} is not a valid Langflow component"
                    if progress is None:
                        print(err_msg)

                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        return valid_components
    
    def create_category_directory(self, category_name: str, description: str) -> Path:
        """Create a category directory structure."""
        category_dir = self.langflow_components_path / category_name
        category_dir.mkdir(exist_ok=True)
        
        # Create __init__.py for the category
        init_file = category_dir / "__init__.py"
        if not init_file.exists():
            init_content = f'"""{description}"""\n'
            init_file.write_text(init_content)
            print(f"Created {category_name} category directory")
        else:
            print(f"{category_name} category directory already exists")
        
        return category_dir
    
    def copy_components_to_langflow(self, valid_components: List[Dict[str, Any]], target_dir: Path, progress_desc: Optional[str] = None) -> List[str]:
        """Copy valid components to the Langflow components directory.

        When progress_desc is provided, shows a tqdm progress bar with that description.
        """
        copied_files = []

        progress = None
        if progress_desc is not None:
            progress = tqdm(
                total=len(valid_components),
                desc=progress_desc,
                unit="file",
                position=1,
                leave=False,
                dynamic_ncols=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

        try:
            for component_info in valid_components:
                source_file = component_info['file_path']
                target_file = target_dir / component_info['file_name']

                try:
                    # Copy the component file
                    shutil.copy2(source_file, target_file)
                    copied_files.append(component_info['file_name'])
                    msg = f"Copied: {component_info['file_name']}"
                    if progress is None:
                        print(msg)

                    # Log component classes found
                    for cls_info in component_info['component_classes']:
                        cls_msg = f"  Component: {cls_info['name']} (extends {cls_info['base_class']})"
                        if progress is None:
                            print(cls_msg)

                except Exception as e:
                    err = f"Failed to copy {source_file.name}: {str(e)}"
                    if progress is None:
                        print(err)

                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        return copied_files
    
    def update_category_init_file(self, target_dir: Path, copied_files: List[str], category_name: str):
        """Update the category's __init__.py file to import the copied components."""
        init_file = target_dir / "__init__.py"
        
        # Read current content
        if init_file.exists():
            current_content = init_file.read_text()
        else:
            current_content = f'"""Custom components from {category_name.lower()}/ directory."""\n\n'
        
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
            print(f"Updated {category_name}/__init__.py with {len(imports_to_add)} new imports")
        else:
            print(f"{category_name}/__init__.py already up to date")
    
    def update_main_components_init_file(self, categories: List[str]):
        """Update the main components __init__.py to include the custom categories."""
        components_init_path = self.langflow_components_path / "__init__.py"
        
        # Read the current content
        if components_init_path.exists():
            current_content = components_init_path.read_text()
        else:
            current_content = ""
        
        # Add imports for our custom categories if not already present
        imports_to_add = []
        for category in categories:
            new_import = f"from . import {category}\n"
            if new_import not in current_content:
                imports_to_add.append(new_import)
        
        if imports_to_add:
            if current_content and not current_content.endswith('\n'):
                current_content += '\n'
            current_content += ''.join(imports_to_add)
            components_init_path.write_text(current_content)
            print(f"Updated main components __init__.py with {len(imports_to_add)} new imports")
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
        """Run the complete automation process for both models and tools."""
        print("Starting customflows and customtools automation")
        print(f"Models Source: {self.custommodels_dir}")
        print(f"Tools Source: {self.customtools_dir}")
        print(f"Target: {self.langflow_components_path}")
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
            print("Creating backup...")
            self.create_backup()
        
        # Step 1: Scan for Python files in both directories
        all_files = self.scan_all_directories()
        if not all_files:
            print("No Python files found in custommodels or customtools directories")
            return
        
        total_components = 0
        created_categories = []
        
        # Process each directory type
        for directory_type, python_files in all_files.items():
            if not python_files:
                continue
                
            print(f"\n--- Processing {directory_type.upper()} ---")
            
            # Step 2: Validate components
            valid_progress_desc = f"Validating {directory_type}"
            valid_components = self.validate_components(python_files, progress_desc=valid_progress_desc)
            if not valid_components:
                print(f"No valid Langflow components found in {directory_type}")
                continue
            
            print(f"Found {len(valid_components)} valid {directory_type} component(s)")
            
            # Step 3: Create category directory
            if directory_type == "models":
                category_name = self.models_category_name
                description = "Custom models components from custommodels/ directory."
            else:  # tools
                category_name = self.tools_category_name
                description = "Custom tools components from customtools/ directory."
            
            target_dir = self.create_category_directory(category_name, description)
            created_categories.append(category_name)
            
            # Step 4: Copy components
            copy_progress_desc = f"Copying {directory_type}"
            copied_files = self.copy_components_to_langflow(valid_components, target_dir, progress_desc=copy_progress_desc)
            
            if copied_files:
                # Step 5: Update category __init__.py file
                self.update_category_init_file(target_dir, copied_files, category_name)
                total_components += len(copied_files)
                
                print(f"Integrated {len(copied_files)} {directory_type} component(s):")
                for file_name in copied_files:
                    print(f"   - {file_name}")
            else:
                print(f"No {directory_type} components were successfully copied")
        
        # Step 6: Update main components __init__.py
        if created_categories:
            self.update_main_components_init_file(created_categories)
            
            print(f"\nAutomation completed successfully")
            print(f"Total integrated components: {total_components}")
            print(f"Created categories: {', '.join(created_categories)}")
            print(f"\nRestart Langflow to see your custom components in the following sections:")
            for category in created_categories:
                print(f"   - {category}")
        else:
            print("No components were successfully integrated")


def main():
    """Main function to run the automation."""
    automation = CustomflowsAutomation()
    automation.run_automation(create_backup=True)


if __name__ == "__main__":
    main()
