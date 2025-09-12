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
    print("1. Add your custom Langflow components to the 'customflows/' folder")
    print("2. Run this script: python run_automation.py")
    print("3. Restart Langflow to see your components in the 'Custom Flows' section")
    print("\nTips:")
    print("- Components must inherit from LCModelComponent, Component, or CustomComponent")
    print("- Components must import from langflow")
    print("- A backup is automatically created before any changes")

if __name__ == "__main__":
    main()
