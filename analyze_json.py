import re

# Read the file
with open('data/bns_sections.json', 'r', encoding='utf-8', errors='replace') as f:
    raw_content = f.read()

# Find problematic patterns - unescaped quotes within content
# Look for patterns of: "word" that are NOT preceded by backslash

# Let's try to parse as JSON to see where it fails
import json

try:
    data = json.loads(raw_content)
    print("JSON is already valid!")
except json.JSONDecodeError as e:
    print(f"JSON Error: {e}")
    print(f"Error at line {e.lineno}, column {e.colno}")
    
    # Extract context around error
    lines = raw_content.split('\n')
    if e.lineno <= len(lines):
        print(f"\nContext around error:")
        start = max(0, e.lineno - 3)
        end = min(len(lines), e.lineno + 2)
        for i in range(start, end):
            prefix = ">>> " if i == e.lineno - 1 else "    "
            print(f"{prefix}{i+1}: {lines[i][:100]}")

# Find all unescaped quotes in content fields
print("\n\nSearching for unescaped quotes in content...")

# Use raw string processing to find quotes that are NOT escaped
# Pattern: (?<!\\)" - quote not preceded by backslash

matches = []
for i, line in enumerate(raw_content.split('\n')):
    # Find quotes not preceded by backslash
    # But exclude the quotes that delimit the JSON strings
    
    # Look for specific pattern: text with unquoted words inside quotes
    # e.g., (1) "act" denotes - the "act" needs escaping
    
    # Find all " that are not preceded by \ or \\ (escaped)
    for match in re.finditer(r'(?<!\\)"', line):
        pos = match.start()
        # Check context - is this inside a content string value?
        # Count quotes before this position in the line
        quotes_before = len(re.findall(r'(?<!\\)"', line[:pos]))
        
        # If odd number of unescaped quotes before, we're inside a value
        if quotes_before % 2 == 1:
            # Extract word after quote
            context = line[max(0, pos-30):min(len(line), pos+30)]
            matches.append({
                'line': i + 1,
                'column': pos + 1,
                'context': context,
                'char_at_pos': line[pos] if pos < len(line) else 'EOF'
            })

print(f"\nFound {len(matches)} potentially problematic quote locations")
if matches:
    print("\nFirst 10 matches:")
    for m in matches[:10]:
        print(f"  Line {m['line']}, Col {m['column']}: ...{m['context']}...")
