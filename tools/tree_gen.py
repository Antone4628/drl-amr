#!/usr/bin/env python3
"""
Generate project tree structure for session context sharing.

Default: Shows directory structure + .py files only.
Use CLI options to include additional file types as needed.

Examples:
    # Default (directories + .py files)
    python tools/tree_gen.py
    
    # Include SLURM and YAML files
    python tools/tree_gen.py -i slurm -i yaml
    
    # Include everything (original behavior)
    python tools/tree_gen.py --include-all
    
    # Expand data directories that are normally collapsed
    python tools/tree_gen.py --expand-data
"""

import argparse
from datetime import datetime
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Always ignore these (system/meta directories and files)
ALWAYS_IGNORE = {
    '__pycache__', '.git', '.pytest_cache', '.vscode', '.idea',
    'node_modules', '.env', 'venv', 'env', '.conda', '.DS_Store',
    '.gitignore', '.gitattributes',
}

# Directories to collapse by default (show existence but don't recurse)
# These are "data dump" directories that clutter the tree
COLLAPSE_DIRS = {
    'results', 'logs', 'data', 'animations', 'mesh_evolution',
    'debug_output', 'tensorboard', 'checkpoints', 'exports',
    'processed', 'raw', 'figures', 'reports', 'thesis_assets',
    'best_model', 'eval_results', 'models',
}

# Directory name patterns to collapse (startswith match)
COLLAPSE_PATTERNS = [
    'training_',
    'run_',
    'session',
]

# Extension aliases for convenience
EXTENSION_ALIASES = {
    'slurm': ['.slurm'],
    'yaml': ['.yaml', '.yml'],
    'config': ['.yaml', '.yml', '.json', '.toml'],
    'md': ['.md'],
    'txt': ['.txt'],
    'png': ['.png'],
    'pdf': ['.pdf'],
    'csv': ['.csv'],
    'json': ['.json'],
    'sh': ['.sh'],
    'rtf': ['.rtf'],
}


# =============================================================================
# Core Functions
# =============================================================================

def should_always_ignore(name: str) -> bool:
    """Check if item should always be ignored."""
    if name in ALWAYS_IGNORE:
        return True
    if name.startswith('.'):
        return True
    # Ignore SLURM output files in wrong locations
    if name.endswith('.out') or name.endswith('.err'):
        return True
    return False


def should_collapse_dir(name: str) -> bool:
    """Check if directory should be collapsed (shown but not recursed)."""
    if name.lower() in COLLAPSE_DIRS:
        return True
    for pattern in COLLAPSE_PATTERNS:
        if name.lower().startswith(pattern):
            return True
    return False


def count_items_in_dir(path: Path) -> tuple:
    """Count subdirectories and files in a directory."""
    try:
        items = list(path.iterdir())
        dirs = sum(1 for i in items if i.is_dir() and not should_always_ignore(i.name))
        files = sum(1 for i in items if i.is_file() and not should_always_ignore(i.name))
        return dirs, files
    except PermissionError:
        return 0, 0


def generate_tree(
    root_path: Path,
    extensions: set,
    expand_data: bool = False,
    max_depth: int = 4,
    prefix: str = "",
    current_depth: int = 0,
) -> list:
    """Generate tree structure recursively."""
    
    if current_depth >= max_depth:
        return []
    
    lines = []
    
    try:
        all_items = list(root_path.iterdir())
    except PermissionError:
        return [f"{prefix}[Permission Denied]"]
    
    # Separate and filter
    dirs = []
    files = []
    
    for item in all_items:
        if should_always_ignore(item.name):
            continue
        if item.is_dir():
            dirs.append(item)
        elif item.is_file():
            if file_matches_extensions(item.name, extensions):
                files.append(item)
    
    # Sort alphabetically (case-insensitive)
    dirs.sort(key=lambda x: x.name.lower())
    files.sort(key=lambda x: x.name.lower())
    
    # Combine: directories first, then files
    all_sorted = dirs + files
    
    for i, item in enumerate(all_sorted):
        is_last = i == len(all_sorted) - 1
        current_prefix = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "
        
        if item.is_dir():
            # Check if we should collapse this directory
            if should_collapse_dir(item.name) and not expand_data:
                n_dirs, n_files = count_items_in_dir(item)
                summary = f"{n_dirs} dirs, {n_files} files"
                lines.append(f"{prefix}{current_prefix}{item.name}/ [{summary}]")
            else:
                lines.append(f"{prefix}{current_prefix}{item.name}/")
                sub_lines = generate_tree(
                    item,
                    extensions,
                    expand_data,
                    max_depth,
                    prefix + next_prefix,
                    current_depth + 1,
                )
                lines.extend(sub_lines)
        else:
            lines.append(f"{prefix}{current_prefix}{item.name}")
    
    return lines


def resolve_extensions(include_args: list, include_all: bool) -> set:
    """Convert CLI include arguments to a set of extensions."""
    if include_all:
        # Return a special marker that matches everything
        return {'__ALL__'}
    
    extensions = {'.py'}  # Always include Python files
    
    if not include_args:
        return extensions
    
    for arg in include_args:
        arg_lower = arg.lower().lstrip('.')
        if arg_lower in EXTENSION_ALIASES:
            extensions.update(EXTENSION_ALIASES[arg_lower])
        else:
            # Treat as literal extension
            ext = f".{arg_lower}" if not arg_lower.startswith('.') else arg_lower
            extensions.add(ext)
    
    return extensions


def file_matches_extensions(filename: str, extensions: set) -> bool:
    """Check if file matches any of the allowed extensions."""
    if '__ALL__' in extensions:
        return True
    return any(filename.lower().endswith(ext) for ext in extensions)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate project tree structure for session context.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      # Default: directories + .py files
  %(prog)s -i slurm -i yaml     # Add SLURM and YAML files
  %(prog)s -i config            # Add config files (.yaml, .yml, .json, .toml)
  %(prog)s --include-all        # Include all files
  %(prog)s --expand-data        # Show contents of data directories

Extension aliases:
  slurm   → .slurm
  yaml    → .yaml, .yml
  config  → .yaml, .yml, .json, .toml
  md      → .md
  png     → .png
  pdf     → .pdf
  csv     → .csv
  sh      → .sh
        """
    )
    
    parser.add_argument(
        '-i', '--include',
        action='append',
        dest='include',
        metavar='EXT',
        help='Include file extension (repeatable). Use aliases or literal extensions.'
    )
    
    parser.add_argument(
        '--include-all',
        action='store_true',
        help='Include all file types (overrides -i flags)'
    )
    
    parser.add_argument(
        '--expand-data',
        action='store_true',
        help='Expand data directories that are normally collapsed'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        default=4,
        metavar='N',
        help='Maximum directory depth (default: 4)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='PROJECT_TREE.md',
        metavar='FILE',
        help='Output filename (default: PROJECT_TREE.md)'
    )
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path.cwd()
    project_name = project_root.name
    
    # Resolve extensions
    extensions = resolve_extensions(args.include or [], args.include_all)
    
    # Build description of what's included
    if '__ALL__' in extensions:
        ext_desc = "all files"
    else:
        ext_desc = ", ".join(sorted(extensions))
    
    print(f"Generating tree for {project_name}...")
    print(f"Including: {ext_desc}")
    if not args.expand_data:
        print("Data directories: collapsed (use --expand-data to expand)")
    
    # Generate tree
    tree_lines = [f"{project_name}/"]
    tree_lines.extend(generate_tree(
        project_root,
        extensions,
        expand_data=args.expand_data,
        max_depth=args.max_depth,
    ))
    
    # Build output
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    header = [
        "# Project Tree",
        "",
        f"Generated: {timestamp}",
        f"Project: {project_name}",
        f"Included: {ext_desc}",
        f"Data dirs: {'expanded' if args.expand_data else 'collapsed'}",
        "",
        "```",
    ]
    
    footer = [
        "```",
        "",
        "---",
        "*Generated by tools/tree_gen.py*",
    ]
    
    # Write output
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        f.write('\n'.join(header + tree_lines + footer))
    
    print(f"\nOutput written to: {output_path}")
    print(f"Total lines: {len(tree_lines)}")


if __name__ == "__main__":
    main()