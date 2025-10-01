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
    # Update the existing index.html in-place: preserve assets, change title, inject script once
    desired_title = f"{title_prefix} {x_str} {brand}"
    
    html = index_path.read_text(encoding="utf-8")

    # Cleanup from any previous bad regex replacement that left literal \1 ... \3 text in the HTML
    # Remove occurrences like: \1...\3 (non-greedy)
    html = re.sub(r"\\1.*?\\3", "", html, flags=re.DOTALL)

    # 1) Update <title> while preserving the rest
    title_re = re.compile(r"(<title>)(.*?)(</title>)", re.IGNORECASE | re.DOTALL)
    if title_re.search(html):
        html = title_re.sub(rf"\\1{re.escape(desired_title)}\\3", html)
    else:
        head_close_re = re.compile(r"</head>", re.IGNORECASE)
        if head_close_re.search(html):
            html = head_close_re.sub(f"  <title>{desired_title}</title>\n  </head>", html)
        else:
            html = f"<head>\n  <title>{desired_title}</title>\n</head>\n" + html

    # 1.1) Inject CSS-based fallback label via ::after on header container
    style_marker = "BRANDING_STYLE_INJECTED"
    if style_marker not in html:
        css = (
            "<style id=\"branding-style\">/* "+style_marker+" */\n"
            ":root { --branding-margin-left: "+str(margin_left)+"px; }\n"
            "header [data-testid=\"header_left_section_wrapper\"], [data-testid=\"header_left_section_wrapper\"] { position: relative; }\n"
            "header [data-testid=\"header_left_section_wrapper\"]::after, [data-testid=\"header_left_section_wrapper\"]::after {\n"
            "  content: '"+brand+"';\n"
            "  display: inline-flex; align-items: center; font-weight: 700; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;\n"
            "  color: inherit; margin-left: var(--branding-margin-left); font-size: "+str(font_size)+"px;\n"
            "}\n"
            "</style>"
        )
        head_close_re2 = re.compile(r"</head>", re.IGNORECASE)
        if head_close_re2.search(html):
            html = head_close_re2.sub(css + "\n</head>", html)
        else:
            html = css + "\n" + html

    # 2) Inject branding script idempotently before </body>
    marker_comment = "BRANDING_INJECTED"
    branding_marker_id = "branding-badge"
    if marker_comment not in html and branding_marker_id not in html:
        script_template = """<script>/* BRANDING_INJECTED */(function(){
  var desiredTitle='__TITLE__'.trim();
  function setDesiredTitle(){ if(document.title!==desiredTitle) document.title=desiredTitle; }
  function qs(sel){ try{ return document.querySelector(sel); }catch(e){ return null; } }
  function findHeaderHost(){
    var selectors = [
      '[data-testid="header_left_section_wrapper"]',
      '[data-testid="header_left_section"]',
      '[data-testid="header"]',
      '[data-testid^="header_"]',
      'header [data-testid="header_left_section_wrapper"]',
      'header [data-testid="header_left_section"]',
      'header [data-testid]',
      'header .mantine-Group-root',
      'header .mantine-Group-inner',
      'header [class*="Group-root"]',
      'header [class*="Group-inner"]',
      'header [class*="Flex-root"]',
      'header [class*="Stack-root"]',
      'header',
      '[role="banner"]'
    ];
    for(var i=0;i<selectors.length;i++){
      var el=qs(selectors[i]);
      if(el) return el;
    }
    return null;
  }
  function getLogo(host){ return (host && (host.querySelector('svg, img') || host.firstElementChild)) || null; }
  function ensureHostPositioning(host){
    try{
      var cs = window.getComputedStyle(host);
      if(cs && cs.position === 'static'){ host.style.position='relative'; }
    }catch(e){}
  }
  function makeSpan(){
    var s=document.createElement('span');
    s.id='branding-badge'; s.textContent='__BRAND__'; s.setAttribute('aria-label','__BRAND__');
    s.style.fontFamily='Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
    s.style.fontWeight='700'; s.style.color='inherit'; s.style.fontSize='__FONTSIZE__px'; s.style.whiteSpace='nowrap';
    return s;
  }
  function addLabelInHost(){
    var host=findHeaderHost();
    if(!host) return false;
    ensureHostPositioning(host);
    var old=document.getElementById('branding-badge'); if(old&&old.parentNode) old.parentNode.removeChild(old);
    var s=makeSpan();
    // Inline element placed right after the logo – avoids layout shifts
    s.style.position='static'; s.style.display='inline-flex'; s.style.alignItems='center'; s.style.verticalAlign='middle'; s.style.marginLeft='__MARGIN__px';
    function sync(){
      try{
        var hostRect=host.getBoundingClientRect(); var h=Math.max(24, Math.round(hostRect.height));
        s.style.height=h+'px'; s.style.lineHeight=h+'px';
      }catch(e){}
    }
    var logo=getLogo(host);
    if(logo && logo.parentNode){ logo.parentNode.insertBefore(s, logo.nextSibling); } else { host.appendChild(s); }
    sync();
    window.addEventListener('resize', sync, {passive:true});
    try{ new ResizeObserver(sync).observe(host); }catch(e){}
    return true;
  }
  function addLabelFixed(){
    var header = qs('header') || qs('[role="banner"]') || document.body;
    var old=document.getElementById('branding-badge'); if(old&&old.parentNode) old.parentNode.removeChild(old);
    var s=makeSpan();
    s.style.position='fixed'; s.style.pointerEvents='none'; s.style.zIndex='9999';
    s.style.color='#ffffff'; s.style.background='rgba(0,0,0,0.35)'; s.style.backdropFilter='blur(2px)';
    s.style.borderRadius='6px'; s.style.padding='2px 6px';
    function sync(){
      try{
        var r = header.getBoundingClientRect();
        var top = r.top + r.height/2; var left = r.left + 56 + __MARGIN__;
        s.style.left=left+'px'; s.style.top=top+'px'; s.style.transform='translateY(-50%)';
      }catch(e){}
    }
    document.body.appendChild(s); sync(); window.addEventListener('resize', sync, {passive:true});
    return true;
  }
  function addLabel(){ return addLabelInHost() || addLabelFixed(); }
  function ensure(){ setDesiredTitle(); addLabel(); }
  ensure(); setTimeout(ensure,1); setTimeout(ensure,50); setTimeout(ensure,200); setTimeout(ensure,500);
  var tries=0; var maxTries=300; var timer=setInterval(function(){ if(addLabel()||tries++>=maxTries){ clearInterval(timer);} },50);
  try{ new MutationObserver(function(){ addLabel(); }).observe(document.body,{childList:true,subtree:true}); }catch(e){}
  window.addEventListener('load',ensure); document.addEventListener('DOMContentLoaded',ensure);
})();</script>"""
        script = (script_template
                  .replace('__TITLE__', desired_title)
                  .replace('__BRAND__', brand)
                  .replace('__MARGIN__', str(margin_left))
                  .replace('__FONTSIZE__', str(font_size))
                  .replace('__HEIGHT__', str(height))
                  )

        body_html_close_re = re.compile(r"</body>\s*</html>\s*$", re.IGNORECASE)
        if body_html_close_re.search(html):
            html = body_html_close_re.sub(script + "\n</body>\n</html>", html)
        else:
            body_close_re = re.compile(r"</body>", re.IGNORECASE)
            if body_close_re.search(html):
                html = body_close_re.sub(script + "\n</body>", html)
            else:
                html = html + "\n" + script

    index_path.write_text(html, encoding="utf-8")


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