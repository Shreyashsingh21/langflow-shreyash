#!/usr/bin/env python3
"""
Simple runner script for both customflows automation and frontend branding.
Run this to apply both custom components and frontend branding.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_customflows_automation():
    """Run the customflows automation."""
    print("CustomFlows to Langflow Automation")
    print("=" * 50)
    
    try:
        from automate_customflows import CustomflowsAutomation
        automation = CustomflowsAutomation()
        automation.run_automation(create_backup=False)
        print("✅ CustomFlows automation completed successfully")
        return True
    except Exception as e:
        print(f"❌ CustomFlows automation failed: {str(e)}")
        return False

def run_frontend_branding(brand="TrinkaAI", color="#ffffff", margin_left=2, top=1, height=32, font_size=14):
    """Run the frontend branding automation."""
    print("\nFrontend Branding Automation")
    print("=" * 50)
    
    try:
        # Import and run the branding script
        from brand_frontend import apply_branding, resolve_frontend_paths, backup_file
        from pathlib import Path
        
        # Resolve frontend path
        frontend_dir = resolve_frontend_paths(None, None)
        index_path = frontend_dir / "index.html"
        
        print(f"Using frontend: {index_path}")
        
        # Create file backup
        if not index_path.with_suffix(index_path.suffix + ".bak").exists():
            backup = backup_file(index_path)
            print(f"File backup created: {backup}")
        
        # Apply branding
        apply_branding(
            index_path=index_path,
            brand=brand,
            title_prefix="Langflow",
            x_str="x",
            color=color,
            margin_left=margin_left,
            top=top,
            height=height,
            font_size=font_size,
        )
        print("✅ Frontend branding applied successfully")
        return True
    except Exception as e:
        print(f"❌ Frontend branding failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Langflow automation scripts")
    parser.add_argument("--customflows-only", action="store_true", help="Run only customflows automation")
    parser.add_argument("--frontend-only", action="store_true", help="Run only frontend branding")
    parser.add_argument("--brand", default="TrinkaAI", help="Brand name for frontend")
    parser.add_argument("--color", default="#ffffff", help="Text color for frontend")
    parser.add_argument("--margin-left", type=int, default=2, help="Left margin for frontend")
    parser.add_argument("--top", type=int, default=1, help="Top position for frontend")
    parser.add_argument("--height", type=int, default=32, help="Height for frontend")
    parser.add_argument("--font-size", type=int, default=14, help="Font size for frontend")
    args = parser.parse_args()
    
    print("Langflow Automation Suite")
    print("=" * 60)
    
    success_count = 0
    total_operations = 0
    
    # Run customflows automation
    if not args.frontend_only:
        total_operations += 1
        if run_customflows_automation():
            success_count += 1
    
    # Run frontend branding
    if not args.customflows_only:
        total_operations += 1
        if run_frontend_branding(
            brand=args.brand,
            color=args.color,
            margin_left=args.margin_left,
            top=args.top,
            height=args.height,
            font_size=args.font_size
        ):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Automation Summary: {success_count}/{total_operations} operations completed successfully")
    
    if success_count == total_operations:
        print("🎉 All automations completed successfully!")
        print("\nNext steps:")
        print("1. Restart Langflow to see your custom components")
        print("2. Hard refresh your browser (Ctrl+Shift+R) to see the branding")
        print("3. Your custom components will appear in:")
        print("   - Models: 'OwnDeployedModels' section")
        print("   - Tools: 'OwnDeployedTools' section")
        print("4. Frontend will show 'Langflow x TrinkaAI' branding")
    else:
        print("⚠️  Some operations failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()