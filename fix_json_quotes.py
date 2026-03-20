import json
import re

# Read the JSON file
file_path = r'c:\Users\Shreyas Durge\AI-projects\nlp_sem6_paper\data\bns_sections.json'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern to find unescaped quotes within content: looking for patterns like "word" or "phrase"
# We need to be careful to only match quotes that are not already escaped
def fix_unescaped_quotes(text):
    """
    Fix unescaped double quotes within JSON string values.
    This replaces patterns like "word" with \"word\"
    """
    # Pattern: Find quotes not preceded by backslash
    # Replace " with \" when it's in the middle of text (not at boundaries)
    
    # First, let's parse as JSON to ensure we're working with valid structure
    data = json.loads(content)
    
    # For each section, fix the content field
    for section in data:
        if 'content' in section:
            original = section['content']
            # Replace unescaped quotes: patterns like "keyword" become \"keyword\"
            # But we need to be careful about quotes that are already escaped
            fixed = original
            
            # Find all patterns of "word" or "phrase" and replace with \"word\" or \"phrase\"
            # This regex finds quotes that are not preceded by backslash
            # Pattern: (?<!\\)" matches a quote not preceded by backslash
            
            # More robust approach: replace all " that appear within the text but check context
            # We'll do this by finding quote pairs and replacing them
            
            # Simple approach: find patterns and replace
            import re
            # This pattern finds quoted text that needs escaping
            # It looks for: space or start + " + word/phrase + " + space or punctuation
            patterns_to_fix = [
                (r'"(act)"', r'\"act\"'),
                (r'"(animal)"', r'\"animal\"'),
                (r'"(child)"', r'\"child\"'),
                (r'"(counterfeit)"', r'\"counterfeit\"'),
                (r'"(Court)"', r'\"Court\"'),
                (r'"(death)"', r'\"death\"'),
                (r'"(dishonestly)"', r'\"dishonestly\"'),
                (r'"(document)"', r'\"document\"'),
                (r'"(fraudulently)"', r'\"fraudulently\"'),
                (r'"(gender)"', r'\"gender\"'),
                (r'"(good faith)"', r'\"good faith\"'),
                (r'"(Government)"', r'\"Government\"'),
                (r'"(harbour)"', r'\"harbour\"'),
                (r'"(injury)"', r'\"injury\"'),
                (r'"(illegal)"', r'\"illegal\"'),
                (r'"(legally bound to do)"', r'\"legally bound to do\"'),
                (r'"(reason to believe)"', r'\"reason to believe\"'), 
                (r'"(special law)"', r'\"special law\"'),
                (r'"(valuable security)"', r'\"valuable security\"'),
                (r'"(vessel)"', r'\"vessel\"'),
                (r'"(voluntarily)"', r'\"voluntarily\"'),
                (r'"(will)"', r'\"will\"'),
                (r'"(woman)"', r'\"woman\"'),
                (r'"(wrongful gain)"', r'\"wrongful gain\"'),
                (r'"(wrongful loss)"', r'\"wrongful loss\"'),
                (r'"(gaining wrongfully)"', r'\"gaining wrongfully\"'),
                (r'"(losing wrongfully)"', r'\"losing wrongfully\"'),
                (r'"(appropriate Government)"', r'\"appropriate Government\"'),
                (r'"(sexual intercourse)"', r'\"sexual intercourse\"'),
                (r'"(dowry)"', r'\"dowry\"'),
                (r'"(cruelty)"', r'\"cruelty\"'),
                (r'"(illicit intercourse)"', r'\"illicit intercourse\"'),
                (r'"(registered medical practitioner)"', r'\"registered medical practitioner\"'),
                (r'"(organised crime syndicate)"', r'\"organised crime syndicate\"'),
                (r'"(continuing unlawful activity)"', r'\"continuing unlawful activity\"'),
                (r'"(economic offence)"', r'\"economic offence\"'),
                (r'"(organised crime)"', r'\"organised crime\"'),
                (r'"(theft)"', r'\"theft\"'),
                (r'"(private act)"', r'\"private act\"'),
                (r'"(pay to the holder)"', r'\"pay to the holder\"'),
                (r'"(transgender)"', r'\"transgender\"'),
                (r'"(he)"', r'\"he\"'),
                (r'"(offence)"', r'\"offence\"'),
                (r'"(General Exceptions)"', r'\"General Exceptions\"'),
                (r'"(nothing is an offence which is done by a person who is bound by law to do it)"', r'\"nothing is an offence which is done by a person who is bound by law to do it\"'),
                (r'"([^"]*rape[^"]*)"', r'\"\\1\"'),  # Catch rape-related definitions
                # Add more as needed
            ]
            
            for pattern, replacement in patterns_to_fix:
                fixed = re.sub(pattern, replacement, fixed)
            
            section['content'] = fixed
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print("JSON quotes fixed successfully!")

if __name__ == '__main__':
    fix_unescaped_quotes('')
