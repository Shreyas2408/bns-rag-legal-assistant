#!/usr/bin/env python3
"""
Fix unescaped quotes in JSON file with proper handling of BOM and string values.
"""

import json
import re
from pathlib import Path

def fix_json_unescaped_quotes(input_filepath):
    """
    Read JSON file and fix unescaped quotes within string values.
    Returns: (success, replacement_count, error_message)
    """
    
    print("=" * 70)
    print("JSON Unescaped Quotes Fixer")
    print("=" * 70)
    
    # Step 1: Read file with BOM handling
    print("\n[1/5] Reading file...")
    try:
        with open(input_filepath, 'r', encoding='utf-8-sig', errors='replace') as f:
            content = f.read()
        print(f"  ✓ File read successfully ({len(content)} characters)")
    except Exception as e:
        return False, 0, f"Failed to read file: {e}"
    
    # Step 2: Try to parse to identify issues
    print("\n[2/5] Validating original JSON...")
    try:
        json.loads(content)
        print("  ✓ JSON is already valid - no fixes needed!")
        return True, 0, None
    except json.JSONDecodeError as e:
        print(f"  ! JSON Error found at line {e.lineno}: {e.msg}")
        print(f"    Beyond this point: {e.doc[max(0,e.pos-50):e.pos+50]}")
    
    # Step 3: Identify and fix problematic quotes
    print("\n[3/5] Analyzing content for unescaped quotes...")
    
    # Strategy: Parse line by line and look for unescaped quotes in content values
    # A quote that needs escaping is one that:
    # 1. Is inside a JSON string value (after "content": ")
    # 2. Is NOT already escaped (not preceded by \)
    # 3. Is NOT a delimiter quote
    
    lines = content.split('\n')
    fixed_lines = []
    replacement_count = 0
    problematic_lines = []
    
    in_content_value = False
    quote_depth = 0
    
    for line_num, line in enumerate(lines, 1):
        original_line = line
        
        # Check if we're starting a content field
        if '"content": "' in line:
            in_content_value = True
            quote_depth = 0
        
        if in_content_value:
            # Process the line to escape unescaped quotes
            # Use regex to find quotes not preceded by backslash within line context
            
            # Find the part after "content": "
            if '"content": "' in line:
                # Split: before and after content starts
                before_part = line[:line.index('"content": "') + len('"content": "')]
                content_part = line[line.index('"content": "') + len('"content": "'):]
                
                # In content_part, find unescaped quotes that are not line-ending quotes
                # Pattern: (?<!\\)" followed by word characters and then (?<!\\)"
                # More specifically: (?<!\\)"([^"]*)"
                
                # But we need to handle the case where content ends on this line
                # Look for pattern: "word" where word can have spaces and hyphens
                
                def escape_inner_quotes(text):
                    """Escape quoted words in text"""
                    count = 0
                    # Find patterns like "word" or "phrase with spaces" that are not escaped
                    # Pattern matches: (?<!\\)"((?:[^"\\]|\\.)*)"
                    # This matches: quote, then any chars (escaped or not), then quote
                    
                    # For this specific legal text, we're looking for quoted terms like:
                    # "act", "animal", "child", "offence", etc.
                    
                    # Use a more careful approach: find [not \]" followed by [not "]+ followed by [not \]"
                    result = text
                    
                    # Match: quote not preceded by \, then word chars/spaces/hyphens, then quote not preceded by \
                    pattern = r'(?<!\\)"([a-zA-Z\s\-\']+)"'
                    
                    def replacer(match):
                        nonlocal count
                        count += 1
                        return '\\"' + match.group(1) + '\\"'
                    
                    result = re.sub(pattern, replacer, result)
                    return result, count
                
                escaped_content, escape_count = escape_inner_quotes(content_part)
                replacement_count += escape_count
                
                if escape_count > 0:
                    line = before_part + escaped_content
                    problematic_lines.append((line_num, original_line[:80], escape_count))
            
            # Check if content value ends on this line
            # Look for the closing quote followed by } or ,
            if line.rstrip().endswith('}') or line.rstrip().endswith(',') or line.rstrip().endswith('"'):
                # Content value likely ends
                if line.count('"') % 2 == 1:  # Odd number of unescaped quotes found
                    in_content_value = False
        
        fixed_lines.append(line)
    
    fixed_content = '\n'.join(fixed_lines)
    
    print(f"  ✓ Analyzed {len(lines)} lines")
    print(f"  ✓ Found and fixed {replacement_count} unescaped quotes")
    
    if problematic_lines:
        print(f"\n  Fixed quotes in {len(problematic_lines)} lines:")
        for line_num, preview, count in problematic_lines[:5]:
            print(f"    - Line {line_num} ({count} quotes): {preview}...")
        if len(problematic_lines) > 5:
            print(f"    ... and {len(problematic_lines) - 5} more lines")
    
    # Step 4: Validate the fixed JSON
    print("\n[4/5] Validating fixed JSON...")
    try:
        parsed = json.loads(fixed_content)
        print(f"  ✓ JSON is now VALID!")
        print(f"  ✓ Successfully parsed {len(parsed)} sections")
        is_valid = True
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON still has errors at line {e.lineno}: {e.msg}")
        is_valid = False
        # Try to show context
        error_lines = fixed_content.split('\n')
        if e.lineno <= len(error_lines):
            print(f"    {error_lines[e.lineno-1][:100]}")
    
    # Step 5: Save if valid
    print("\n[5/5] Saving file...")
    if is_valid:
        try:
            # Write with UTF-8 (no BOM by default)
            with open(input_filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"  ✓ File saved successfully")
            return True, replacement_count, None
        except Exception as e:
            return False, replacement_count, f"Failed to save file: {e}"
    else:
        return False, replacement_count, "JSON validation failed - not saving"

# Main execution
if __name__ == '__main__':
    filepath = Path('data/bns_sections.json')
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        exit(1)
    
    success, count, error = fix_json_unescaped_quotes(filepath)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Status: {'SUCCESS ✓' if success else 'FAILED ✗'}")
    print(f"Replacements made: {count}")
    if error:
        print(f"Error: {error}")
    print("=" * 70)
    
    exit(0 if success else 1)
