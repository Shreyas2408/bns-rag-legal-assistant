#!/usr/bin/env python3
"""
Fix unescaped quotes in the BNS JSON file by replacing patterns
like "word" with \"word\" within JSON string content.
"""

import re

file_path = r'c:\Users\Shreyas Durge\AI-projects\nlp_sem6_paper\data\bns_sections.json'

# Read the entire file as one string
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

file_size_before = len(content)

# Count initial occurrences of unescaped quotes
sample_count = content.count('"act"')

# Simple direct string replacements - these are literal strings not regex
# We're looking for quote-enclosed terms within the JSON content
replacements = {
    '"act"': '\\"act\\"',
    '"animal"': '\\"animal\\"',
    '"child"': '\\"child\\"',
    '"counterfeit"': '\\"counterfeit\\"',
    '"Court"': '\\"Court\\"',
    '"death"': '\\"death\\"',  
    '"dishonestly"': '\\"dishonestly\\"',
    '"document"': '\\"document\\"',
    '"fraudulently"': '\\"fraudulently\\"',
    '"good faith"': '\\"good faith\\"',
    '"Government"': '\\"Government\\"',
    '"harbour"': '\\"harbour\\"',
    '"injury"': '\\"injury\\"',
    '"illegal"': '\\"illegal\\"',
    '"legally bound to do"': '\\"legally bound to do\\"',
    '"reason to believe"': '\\"reason to believe\\"',
    '"special law"': '\\"special law\\"',
    '"valuable security"': '\\"valuable security\\"',
    '"vessel"': '\\"vessel\\"',
    '"voluntarily"': '\\"voluntarily\\"',
    '"will"': '\\"will\\"',
    '"woman"': '\\"woman\\"',
    '"wrongful gain"': '\\"wrongful gain\\"',
    '"wrongful loss"': '\\"wrongful loss\\"',
    '"gaining wrongfully"': '\\"gaining wrongfully\\"',
    '"losing wrongfully"': '\\"losing wrongfully\\"',
    '"offence"': '\\"offence\\"',
    '"appropriate Government"': '\\"appropriate Government\\"',
    '"sexual intercourse"': '\\"sexual intercourse\\"',
    '"dowry"': '\\"dowry\\"',
    '"cruelty"': '\\"cruelty\\"',
    '"illicit intercourse"': '\\"illicit intercourse\\"',
    '"registered medical practitioner"': '\\"registered medical practitioner\\"',
    '"organised crime syndicate"': '\\"organised crime syndicate\\"',
    '"continuing unlawful activity"': '\\"continuing unlawful activity\\"',
    '"economic offence"': '\\"economic offence\\"',
    '"organised crime"': '\\"organised crime\\"',
    '"theft"': '\\"theft\\"',
    '"private act"': '\\"private act\\"',
    '"pay to the holder"': '\\"pay to the holder\\"',
    '"transgender"': '\\"transgender\\"',
    '"he"': '\\"he\\"',
    '"gender"': '\\"gender\\"',
    '"General Exceptions"': '\\"General Exceptions\\"',
    '"rape"': '\\"rape\\"',
}

# Apply replacements
replacements_made = 0
for old_str, new_str in replacements.items():
    if old_str in content:
        content = content.replace(old_str, new_str)
        replacements_made += 1

file_size_after = len(content)

# Write the fixed content back
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

# Write a summary log
log_path = r'c:\Users\Shreyas Durge\AI-projects\nlp_sem6_paper\fix_log.txt'
with open(log_path, 'w', encoding='utf-8') as f:
    f.write("JSON Quote Fix Summary\n")
    f.write("=" * 40 + "\n")
    f.write(f"File: {file_path}\n")
    f.write(f"Size before: {file_size_before} bytes\n")
    f.write(f"Size after: {file_size_after} bytes\n")
    f.write(f"Replacements made: {replacements_made}\n")
    f.write(f'Sample "{act}" count: {sample_count}\n')
    f.write("\nStatus: COMPLETE\n")

