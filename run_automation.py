#!/usr/bin/env python3
"""
Simple runner script for the customflows automation.
Run this whenever you add new components to the customflows/ folder.
"""

from automate_customflows import CustomflowsAutomation

def main():
    print("CustomFlows to Langflow Automation")
    print("=" * 50)
    
    automation = CustomflowsAutomation()
    automation.run_automation(create_backup=False)
    
    print("\n" + "=" * 50)
    print("Usage Instructions:")
    print("1. Add your custom model components to the 'custommodels/' folder")
    print("2. Add your custom tool components to the 'customtools/' folder")
    print("3. Run this script: python run_automation.py")
    print("4. Restart Langflow to see your components in separate sections:")
    print("   - Models will appear in 'OwnDeployedModels' section")
    print("   - Tools will appear in 'OwnDeployedTools' section")
    print("\nTips:")
    print("- Components must inherit from LCModelComponent, Component, or CustomComponent")
    print("- Components must import from langflow")
    print("- A backup is automatically created before any changes")
    print("- You can customize section names using environment variables:")
    print("  - MODELS_CATEGORY_NAME for models section name")
    print("  - TOOLS_CATEGORY_NAME for tools section name")

if __name__ == "__main__":
    main()
