#!/usr/bin/env python3
"""
Simple runner script for both customflows automation and frontend branding.
Run this to apply both custom components and frontend branding.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import time

def run_customflows_automation():
    """Run the customflows automation."""
    print("Running CustomFlows automation...")
    
    try:
        from automate_customflows import CustomflowsAutomation
        automation = CustomflowsAutomation()
        automation.run_automation(create_backup=False)
        print("✓ CustomFlows done")
        return True
    except Exception as e:
        print(f"✗ CustomFlows failed: {str(e)}")
        return False

def run_frontend_branding(brand="TrinkaAI", color="#ffffff", margin_left=2, top=1, height=32, font_size=14):
    """Run the frontend branding automation."""
    print("Running frontend branding...")
    
    try:
        # Import and run the branding script
        from brand_frontend import apply_branding, resolve_frontend_paths, backup_file
        from pathlib import Path
        
        # Resolve frontend path
        frontend_dir = resolve_frontend_paths(None, None)
        index_path = frontend_dir / "index.html"
        
        # Create file backup
        if not index_path.with_suffix(index_path.suffix + ".bak").exists():
            backup = backup_file(index_path)
        
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
        print("✓ Frontend branding done")
        return True
    except Exception as e:
        print(f"✗ Frontend branding failed: {str(e)}")
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
    
    print("Langflow Automation")
    
    # Determine which operations to run
    operations = []
    if not args.frontend_only:
        operations.append(("CustomFlows Automation", run_customflows_automation))
    if not args.customflows_only:
        operations.append(("Frontend Branding", lambda: run_frontend_branding(
            brand=args.brand,
            color=args.color,
            margin_left=args.margin_left,
            top=args.top,
            height=args.height,
            font_size=args.font_size
        )))
    
    if not operations:
        print("No operations to run")
        return
    
    # Create progress bar for overall operations
    with tqdm(total=len(operations), desc="Overall Progress", unit="operation", 
             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        success_count = 0
        
        for operation_name, operation_func in operations:
            pbar.set_description(f"Running {operation_name}")
            
            try:
                if operation_func():
                    success_count += 1
                    pbar.set_postfix(status="✓ Success")
                else:
                    pbar.set_postfix(status="✗ Failed")
            except Exception as e:
                pbar.set_postfix(status=f"✗ Error: {str(e)}")
            
            pbar.update(1)
            time.sleep(0.2)  # Small delay for visual feedback
    
    # Summary
    if success_count == len(operations):
        print(f"\n✓ All done ({success_count}/{len(operations)})")
        print("Restart Langflow and refresh browser to see changes")
    else:
        print(f"\n✗ Failed ({success_count}/{len(operations)})")
        sys.exit(1)

if __name__ == "__main__":
    main()