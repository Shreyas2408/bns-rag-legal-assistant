#!/usr/bin/env python3
"""
Fix unescaped quotes in JSON by processing the raw file content carefully.
"""

import json
import re

def fix_json_quotes():
    """Fix unescaped quotes in the JSON file"""
    
    # Read file with BOM handling
    with open('data/bns_sections.json', 'r', encoding='utf-8-sig') as f:
        content = f.read()
    
    original_content = content
    print("=" * 70)
    print("JSON Quote Fixer - Comprehensive Pass")
    print("=" * 70)
    
    fix_count = 0
    
    # Pass 1: Fix double quotes (""word"" → "word")
    print("\n[Pass 1] Fixing double quotes (""word"" pattern)...")
    doubleQuote_pattern = r'""([^"]*?)""'
    matches_pass1 = len(re.findall(doubleQuote_pattern, content))
    if matches_pass1 > 0:
        content = re.sub(doubleQuote_pattern, r'"\1"', content)
        print(f"  ✓ Fixed {matches_pass1} double quote instances")
        fix_count += matches_pass1
    else:
        print("  - No double quotes found")
    
    # Pass 2: Fix unescaped single quotes
    print("\n[Pass 2] Fixing unescaped quoted words...")
    
    lines = content.split('\n')
    fixed_lines = []
    pass2_count = 0
    
    for i, line in enumerate(lines):
        original_line = line
        
        # Only process lines with content values
        if '"content": "' in line:
            # Split the line at the content start
            content_start_idx = line.index('"content": "') + len('"content": "')
            before_content = line[:content_start_idx]
            after_content = line[content_start_idx:]
            
            # In the content part, find quoted words not preceded by backslash
            # Pattern: (?<!\\)"([a-zA-Z\s\-\']+?)"(?!:)
            pattern = r'(?<!\\)"([a-zA-Z\s\-\']+?)"'
            
            def replacer(m):
                # Check if already has escape
                match_text = m.group(0)
                if match_text.startswith('\\'):
                    return match_text
                # Escape the quotes
                return '\\"' + m.group(1) + '\\"'
            
            fixed_after_content = re.sub(pattern, replacer, after_content)
            
            if fixed_after_content != after_content:
                count_changes = len(re.findall(pattern, after_content))
                pass2_count += count_changes
                line = before_content + fixed_after_content
        
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    if pass2_count > 0:
        print(f"  ✓ Fixed {pass2_count} unescaped quoted words")
        fix_count += pass2_count
    else:
        print("  - No unescaped quoted words found")
    
    # Validate the result
    print("\n[Validate] Testing JSON validity...")
    try:
        parsed_json = json.loads(content)
        print(f"  ✓ JSON is VALID!")
        print(f"  ✓ Successfully validated {len(parsed_json)} sections")
        
        # Save the fixed file
        print("\n[Save] Writing fixed file...")
        with open('data/bns_sections.json', 'w', encoding='utf-8') as f:
            f.write(content)
        print("  ✓ File saved successfully")
        
        print("\n" + "=" * 70)
        print("SUCCESS SUMMARY")
        print("=" * 70)
        print(f"Total replacements made: {fix_count}")
        print(f"File status: VALID ✓")
        print(f"Sections processed: {len(parsed_json)}")
        print("=" * 70)
        
        return True, fix_count
        
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON validation FAILED")
        print(f"  Error: {e.msg} at line {e.lineno}")
        
        # Show the problematic area
        error_lines = content.split('\n')
        if e.lineno <= len(error_lines):
            print(f"\n  Context (line {e.lineno}):")
            print(f"  {error_lines[e.lineno-1][:120]}")
        
        print(f"\n  Made {fix_count} replacement attempts before failure")
        print("  File NOT saved due to validation error")
        
        return False, fix_count

# Main execution
if __name__ == '__main__':
    success, count = fix_json_quotes()
    exit(0 if success else 1)
