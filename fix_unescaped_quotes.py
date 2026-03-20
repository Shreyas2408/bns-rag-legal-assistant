import re
import json
from pathlib import Path

def fix_unescaped_quotes_in_json(filepath):
    """
    Fix unescaped double quotes within JSON string values.
    Identifies quoted terms like "word" that should be \"word\"
    """
    
    # Read the raw file content
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern to find unescaped quotes within content values:
    # This matches: "text": "...content with unescaped quoted terms..."
    # We need to be careful to only escape quotes that are inside JSON string values,
    # not the quotes that delimit the JSON strings themselves.
    
    # Strategy: Find all content fields and process them
    # Match pattern: "content": "..." but handle newlines and capture the full content
    
    # Let's use a different approach - find quoted words inside content strings
    # that are not already escaped (don't have \ before them)
    
    replacement_count = 0
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        if '"content":' in line or (i > 0 and '"content":' in lines[i-1]):
            # This line is part of a content field
            
            # Find all instances of unescaped quotes followed by word patterns
            # Pattern: when we have a quote that's not preceded by a backslash
            # We need to find :" or " that denotes a quoted term
            
            # Use regex to find quoted words that aren't escaped
            # Look for patterns like: [space or start]"[word]" but not \"[word]\"
            
            # This regex finds quotes that should be escaped
            # It looks for a quote character that:
            # 1. Is not preceded by a backslash
            # 2. Is inside a JSON string value (surrounded by other quotes)
            
            # More sophisticated: Find content value and process it
            if '"content": "' in line:
                # Example: "content": "...text..."
                # We need to escape inner quotes
                
                # Split at "content": "
                before = line[:line.find('"content": "') + len('"content": "')]
                after_start = line[line.find('"content": "') + len('"content": "'):]
                
                # Find where the content value ends (next unescaped "}
                # This is complex because of line breaks. Skip this approach.
                pass
            
            # Alternative simpler approach: 
            # Use regex to find [not \]"[non-quote characters]"
            # and replace with [not \]\"[non-quote characters]\"
            
            # Pattern to find: "word" that's not escaped
            # Using negative lookbehind to ensure \ is not before the quote
            original_line = line
            
            # Find all quoted terms that aren't escaped
            # Pattern: (?<!\\)"([a-zA-Z\s\-]+)"
            pattern = r'(?<!\\)"([a-zA-Z\s\-]+)"'
            
            def replace_func(match):
                nonlocal replacement_count
                replacement_count += 1
                return '\\"' + match.group(1) + '\\"'
            
            line = re.sub(pattern, replace_func, line)
            
            if line != original_line:
                print(f"Line {i+1}: Modified content")
        
        fixed_lines.append(line)
    
    fixed_content = '\n'.join(fixed_lines)
    
    # Verify the fixed JSON is valid
    try:
        json.loads(fixed_content)
        print(f"\n✓ JSON is now VALID after fixing {replacement_count} quotes")
        return fixed_content, replacement_count, True
    except json.JSONDecodeError as e:
        print(f"\n✗ JSON validation failed: {e}")
        return fixed_content, replacement_count, False

# Main execution
filepath = Path('data/bns_sections.json')
if filepath.exists():
    fixed_content, count, is_valid = fix_unescaped_quotes_in_json(filepath)
    
    if is_valid:
        # Save the fixed file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        print(f"\n✓ File saved successfully with {count} replacements made")
    else:
        print(f"\n✗ Could not save file - JSON validation failed")
        print(f"\nPlease review the replacement logic. Made {count} replacement attempts.")
else:
    print(f"File not found: {filepath}")
