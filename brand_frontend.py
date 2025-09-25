#!/usr/bin/env python3
"""
Automation script to apply frontend branding to Langflow.

This script automatically:
1. Locates the Langflow frontend directory
2. Creates a backup of the original index.html
3. Applies branding with configurable options
4. Ensures the branding appears on first load and persists across navigation

Use --revert to restore the backup.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import sysconfig
import importlib.util
from pathlib import Path


def resolve_frontend_paths(conda_prefix: str | None, venv_prefix: str | None) -> Path:
    """Resolve the langflow/frontend directory in the active environment.

    Resolution order:
    1) ENV override: LANGFLOW_FRONTEND_PATH
    2) importlib: path from installed langflow module
    3) sysconfig: purelib site-packages + /langflow/frontend
    4) Workspace-local venv guesses: .venv, venv (all pythonX.Y variants)
    5) Conda/Anaconda paths
    """
    # 1) Environment override
    env_override = os.environ.get("LANGFLOW_FRONTEND_PATH")
    if env_override:
        path = Path(env_override).expanduser().resolve()
        if path.exists():
            return path

    # 2) From installed module
    try:
        import importlib.util
        spec = importlib.util.find_spec("langflow")
        if spec and spec.origin:
            langflow_pkg = Path(spec.origin).resolve().parent  # .../site-packages/langflow
            frontend_path = langflow_pkg / "frontend"
            if frontend_path.exists():
                return frontend_path
    except Exception:
        pass

    # 3) sysconfig purelib
    try:
        import sysconfig
        purelib = sysconfig.get_paths().get("purelib")
        if purelib:
            frontend_path = Path(purelib) / "langflow" / "frontend"
            if frontend_path.exists():
                return frontend_path
    except Exception:
        pass

    # 4) Guess common local venv layouts inside workspace
    workspace_path = Path.cwd()
    venv_roots = [
        workspace_path / ".langflow-uat-env",
        workspace_path / "langflow-uat-env", 
        workspace_path / ".venv",
        workspace_path / "venv",
    ]
    candidates = []
    for root in venv_roots:
        if root.exists():
            # Support multiple python versions
            lib_dir = root / "lib"
            if lib_dir.exists():
                for py_dir in lib_dir.glob("python*"):
                    candidates.append(py_dir / "site-packages" / "langflow" / "frontend")
            # Some distros use site-packages directly under venv
            candidates.append(root / "site-packages" / "langflow" / "frontend")

    # 5) Conda/Anaconda paths
    if not conda_prefix:
        conda_prefix = os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_DEFAULT_ENV")
        # If CONDA_DEFAULT_ENV is a name, try guessing default path
        if conda_prefix and not os.path.isabs(conda_prefix):
            guessed = Path.home() / "anaconda3"
            if guessed.exists():
                conda_prefix = str(guessed)
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib/python3.12/site-packages/langflow/frontend")

    # 6) If venv prefix provided or detected
    if not venv_prefix:
        venv_prefix = os.environ.get("VIRTUAL_ENV") or str(Path.cwd() / "venv")
    if venv_prefix:
        candidates.append(Path(venv_prefix) / "lib/python3.12/site-packages/langflow/frontend")
        candidates.append(Path(venv_prefix) / "lib64/python3.12/site-packages/langflow/frontend")

    for c in candidates:
        if c.exists():
            return c

    # As a last resort, return the most likely sysconfig path even if not exists
    try:
        import sysconfig
        fallback_purelib = sysconfig.get_paths().get("purelib", "")
        if fallback_purelib:
            return Path(fallback_purelib) / "langflow" / "frontend"
    except Exception:
        pass

    # Ultimate fallback to workspace .venv structure (may not exist)
    return workspace_path / ".venv/lib/python3/site-packages/langflow/frontend"


def backup_file(file_path: Path) -> Path:
    backup = file_path.with_suffix(file_path.suffix + ".bak")
    shutil.copy2(file_path, backup)
    return backup


def revert_backup(file_path: Path) -> bool:
    backup = file_path.with_suffix(file_path.suffix + ".bak")
    if backup.exists():
        shutil.copy2(backup, file_path)
        return True
    return False


def apply_branding(index_path: Path, brand: str, title_prefix: str, x_str: str,
                   color: str, margin_left: int, top: int, height: int, font_size: int) -> None:
    # Use the complete branded HTML template from FrontendTrinkaAI/index.html
    desired_title = f"{title_prefix} {x_str} {brand}"
    
    # Complete branded HTML template with ULTRA AGGRESSIVE first-load detection
    branded_html = f"""<!doctype html>
<html lang="en">
  <head>
    <base href="/" />
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="icon" href="./favicon.ico" />
    <link rel="manifest" href="./manifest.json" />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      href="https://fonts.googleapis.com/css2?family=Chivo:ital,wght@0,100..900;1,100..900&family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&display=swap"
      rel="stylesheet"
    />
    <title>{desired_title}</title>
    <script type="module" crossorigin src="./assets/index-B6y615tP.js"></script>
    <link rel="stylesheet" crossorigin href="./assets/index-CYIVEhXL.css">
  </head>
  <body id="body" class="dark" style="width: 100%; height: 100%">
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div style="width: 100vw; height: 100vh" id="root"></div>
    <script>
      (function () {{
        var desiredTitle = '{desired_title}';
        function setDesiredTitle(){{ if(document.title!==desiredTitle) document.title=desiredTitle; }}
        setDesiredTitle();
        new MutationObserver(setDesiredTitle).observe(document.querySelector('title')||document.head,{{subtree:true,childList:true,characterData:true}});

        function addLabel(){{
          var left=document.querySelector('[data-testid="header_left_section_wrapper"]');
          if(!left) return false;
          if(left.querySelector('#branding-badge')) return true;
          var s=document.createElement('span');
          s.id='branding-badge'; s.textContent='{brand}';
          s.style.marginLeft='{margin_left}px';
          s.style.fontFamily='Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
          s.style.fontWeight='700';
          s.style.color='{color}';
          s.style.fontSize='{font_size}px';
          s.style.display='inline-flex'; s.style.alignItems='center';
          s.style.height='{height}px'; s.style.lineHeight='{height}px'; s.style.verticalAlign='left';
          s.style.position='relative'; s.style.top='{top}px';
          left.insertBefore(s,left.firstChild && left.firstChild.nextSibling ? left.firstChild.nextSibling : null);
          return true;
        }}
        function ensureLabel(){{ addLabel(); }}
        
        // IMMEDIATE execution - try right now
        ensureLabel();
        
        // Multiple immediate attempts with different timings
        setTimeout(ensureLabel, 0);
        setTimeout(ensureLabel, 1);
        setTimeout(ensureLabel, 5);
        setTimeout(ensureLabel, 10);
        setTimeout(ensureLabel, 25);
        setTimeout(ensureLabel, 50);
        setTimeout(ensureLabel, 100);
        setTimeout(ensureLabel, 200);
        setTimeout(ensureLabel, 500);
        setTimeout(ensureLabel, 1000);
        setTimeout(ensureLabel, 2000);
        setTimeout(ensureLabel, 3000);
        setTimeout(ensureLabel, 5000);
        
        // Event listeners for all possible scenarios
        if(document.readyState==='complete' || document.readyState==='interactive'){{ 
          ensureLabel(); 
        }} else {{ 
          window.addEventListener('DOMContentLoaded', ensureLabel); 
        }}
        window.addEventListener('load', ensureLabel);
        document.addEventListener('DOMContentLoaded', ensureLabel);
        document.addEventListener('readystatechange', function(){{
          if(document.readyState==='complete' || document.readyState==='interactive'){{
            ensureLabel();
          }}
        }});
        
        // Ultra aggressive retry mechanism
        var retryCount = 0;
        var maxRetries = 200;
        var retryInterval = setInterval(function(){{
          if(ensureLabel() || retryCount >= maxRetries){{
            clearInterval(retryInterval);
          }}
          retryCount++;
        }}, 10);
        
        // Multiple observers for different scenarios
        new MutationObserver(function(){{ ensureLabel(); }}).observe(document.documentElement,{{childList:true,subtree:true}});
        new MutationObserver(function(){{ ensureLabel(); }}).observe(document.body,{{childList:true,subtree:true}});
        new MutationObserver(function(){{ ensureLabel(); }}).observe(document.head,{{childList:true,subtree:true}});
        
        // Additional observers for specific elements
        var rootObserver = new MutationObserver(function(){{ ensureLabel(); }});
        rootObserver.observe(document.getElementById('root'), {{childList:true,subtree:true}});
        
        // Handle SPA navigation
        var originalPushState = history.pushState;
        var originalReplaceState = history.replaceState;
        history.pushState = function(){{ originalPushState.apply(history, arguments); setTimeout(ensureLabel, 1); }};
        history.replaceState = function(){{ originalReplaceState.apply(history, arguments); setTimeout(ensureLabel, 1); }};
        window.addEventListener('popstate', function(){{ setTimeout(ensureLabel, 1); }});
        window.addEventListener('hashchange', function(){{ setTimeout(ensureLabel, 1); }});
        
        // Additional SPA detection
        window.addEventListener('beforeunload', function(){{ ensureLabel(); }});
        window.addEventListener('pageshow', function(){{ ensureLabel(); }});
        window.addEventListener('pagehide', function(){{ ensureLabel(); }});
        
        // React/Vue specific detection
        if(window.React){{
          var originalReactCreateElement = window.React.createElement;
          window.React.createElement = function(){{
            var result = originalReactCreateElement.apply(this, arguments);
            setTimeout(ensureLabel, 1);
            return result;
          }};
        }}
        
        // Additional framework detection
        if(window.Vue){{
          var originalVueCreateElement = window.Vue.createElement;
          window.Vue.createElement = function(){{
            var result = originalVueCreateElement.apply(this, arguments);
            setTimeout(ensureLabel, 1);
            return result;
          }};
        }}
        
        // Force execution on any DOM change
        var forceObserver = new MutationObserver(function(){{
          setTimeout(ensureLabel, 1);
        }});
        forceObserver.observe(document, {{childList:true,subtree:true,attributes:true,characterData:true}});
        
        // Additional immediate execution strategies
        requestAnimationFrame(ensureLabel);
        requestAnimationFrame(function(){{ requestAnimationFrame(ensureLabel); }});
        
        // Webpack/Module loading detection
        if(window.webpackChunkName){{
          setTimeout(ensureLabel, 100);
        }}
        
        // Service Worker detection
        if('serviceWorker' in navigator){{
          navigator.serviceWorker.addEventListener('message', function(){{ ensureLabel(); }});
        }}
        
        // Additional immediate execution
        if(window.requestIdleCallback){{
          requestIdleCallback(ensureLabel);
        }} else {{
          setTimeout(ensureLabel, 1);
        }}
      }})();
    </script>
  </body>
</html>"""
    
    # Replace the entire file with the branded version
    index_path.write_text(branded_html, encoding="utf-8")


def create_backup(frontend_dir: Path) -> bool:
    """Create a backup of the current frontend directory."""
    workspace_path = Path.cwd()
    backup_dir = workspace_path / "langflow_frontend_backup"
    if not backup_dir.exists():
        try:
            shutil.copytree(frontend_dir, backup_dir)
            print(f"Created backup at: {backup_dir}")
            return True
        except Exception as e:
            print(f"Could not create backup: {str(e)}")
            return False
    else:
        print(f"Backup already exists at: {backup_dir}")
        return True

def main() -> int:
    p = argparse.ArgumentParser(description="Apply or revert Langflow frontend branding.")
    p.add_argument("--brand", default="TrinkaAI")
    p.add_argument("--title-prefix", default="Langflow")
    p.add_argument("--x", default="x")
    p.add_argument("--color", default="#ffffff")
    p.add_argument("--margin-left", type=int, default=2)
    p.add_argument("--top", type=int, default=1)
    p.add_argument("--height", type=int, default=32)
    p.add_argument("--font-size", type=int, default=14)
    p.add_argument("--conda-prefix")
    p.add_argument("--venv-prefix")
    p.add_argument("--revert", action="store_true")
    p.add_argument("--backup", action="store_true", help="Create backup before applying branding")
    args = p.parse_args()

    print("Starting Langflow frontend branding automation")
    print("-" * 60)

    try:
        frontend_dir = resolve_frontend_paths(args.conda_prefix, args.venv_prefix)
    except Exception as e:
        print(f"Could not locate Langflow frontend directory: {str(e)}")
        print("Hints:")
        print(" - Ensure your virtual environment is activated before running this script.")
        print(" - Install langflow in the active environment: pip install langflow")
        print(" - Or set LANGFLOW_FRONTEND_PATH to the correct frontend directory.")
        print(" - Check that langflow is properly installed and accessible.")
        return 1

    index_path = frontend_dir / "index.html"
    print(f"Using frontend: {index_path}")

    # Sanity check for target file existence
    if not index_path.exists():
        print(f"Could not locate index.html at: {index_path}")
        print("This may not be a valid Langflow frontend directory.")
        return 1

    if args.revert:
        if revert_backup(index_path):
            print("Reverted index.html from backup.")
            return 0
        print("No backup found to revert.")
        return 1

    # Create backup only if requested
    if args.backup:
        if not create_backup(frontend_dir):
            print("Warning: Could not create backup. Proceeding anyway...")

        # Create individual file backup
        if not index_path.with_suffix(index_path.suffix + ".bak").exists():
            backup = backup_file(index_path)
            print(f"File backup created: {backup}")
        else:
            print("File backup already exists; proceeding with update.")
    else:
        print("Skipping backup creation (use --backup to enable)")

    try:
        apply_branding(
            index_path=index_path,
            brand=args.brand,
            title_prefix=args.title_prefix,
            x_str=args.x,
            color=args.color,
            margin_left=args.margin_left,
            top=args.top,
            height=args.height,
            font_size=args.font_size,
        )
        print("Branding applied successfully.")
        print("Tip: hard refresh the browser (Ctrl+Shift+R). Use --revert to undo.")
        return 0
    except Exception as e:
        print(f"Failed to apply branding: {str(e)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())