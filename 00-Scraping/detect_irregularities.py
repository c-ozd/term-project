"""
Scans dota2_vector_chunks.jsonl for unresolved wiki template irregularities.
Outputs a JSON report file with all findings.
"""

import json
import re
import os

INPUT_FILE = os.path.join(os.path.dirname(__file__), "dota2_vector_chunks.jsonl")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "irregularities_report.json")

# Patterns to detect unresolved wiki markup / template artifacts
PATTERNS = {
    "expression_tag": re.compile(r'\[EXPR:[^\]]+\]'),  # [EXPR: ... ]
    "double_curly_braces_template": re.compile(r'\{\{[^}]{2,}\}\}'),  # {{show|...|...}} etc.
    "question_mark_pipe_values": re.compile(r'\?\|[a-zA-Z0-9_]+='),  # ?|v1=30/45/60
    "unresolved_wiki_link": re.compile(r'\[\[[^\]]*\]\]'),  # [[wiki links]]
    "html_tags": re.compile(r'<(?!br\s*/?\s*>)[a-zA-Z][a-zA-Z0-9]*[^>]*>'),  # HTML tags (excluding <br>)
    "pipe_separated_template_fragment": re.compile(r'(?<!\|)\|[a-zA-Z0-9_]+=(?:[^|"\\]|\\.)*(?:\|[a-zA-Z0-9_]+=)'),  # |param=val|param2=val
    "unresolved_magic_words": re.compile(r'__[A-Z]+__'),  # __NOTOC__ etc.
    "parser_functions": re.compile(r'\{\{#(if|switch|ifeq|expr|rel2abs|titleparts):[^}]+\}\}'), # {{#if: ... }}
    "wiki_table_remnants": re.compile(r'(?:^|\n)(?:\{\||\|\}|\|\+|\|-|!|\|)'), # {|, |}, |-, etc at start of lines
    "category_tags": re.compile(r'\[\[Category:[^\]]*\]\]', re.IGNORECASE),  # [[Category:...]]
    "ref_tags": re.compile(r'<ref[^>]*>.*?</ref>', re.IGNORECASE | re.DOTALL),  # <ref>...</ref>
    "nowiki_tags": re.compile(r'<nowiki[^>]*>.*?</nowiki>', re.IGNORECASE | re.DOTALL),  # <nowiki>
    "gallery_tags": re.compile(r'<gallery[^>]*>.*?</gallery>', re.IGNORECASE | re.DOTALL),  # <gallery>
    "triple_apostrophe_bold": re.compile(r"'''[^']+'''"),  # '''bold text'''
    "double_apostrophe_italic": re.compile(r"''[^']+''"),  # ''italic text''
    "equals_heading": re.compile(r'(?:^|\n)={2,}[^=]+=+'),  # == Heading ==
    "template_parameter_default": re.compile(r'\{\{\{[^}]+\}\}\}'),  # {{{param|default}}}
    "unresolved_file_image": re.compile(r'\[\[(?:File|Image):[^\]]*\]\]', re.IGNORECASE),  # [[File:...]] / [[Image:...]]
    "html_entities": re.compile(r'&[a-z0-9#]+;'), # &nbsp;, &#160; etc.
    "unicode_escape_sequence": re.compile(r'\\u[0-9a-fA-F]{4}')
}

def scan_file():
    results = []
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return results

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                results.append({
                    "item_number": line_num,
                    "type": "json_parse_error",
                    "detail": str(e),
                    "snippet": line[:200]
                })
                continue
            
            content = item.get("page_content", "")
            metadata = item.get("metadata", {})
            item_name = metadata.get("item", "unknown")
            path = metadata.get("full_path", "unknown")
            
            for pattern_name, pattern in PATTERNS.items():
                matches = pattern.findall(content)
                if matches:
                    filtered = []
                    for m in matches:
                        # Logic to filter false positives
                        if pattern_name == "double_curly_braces_template":
                            if m.startswith('{{') and ('|' in m or any(keyword in m for keyword in ['{{show', '{{Gold', '{{Rune', '{{Ability', '{{Item', '{{Hero'])):
                                filtered.append(m)
                            elif m.startswith('{{') and not any(c in m for c in ['":', '": ']):
                                filtered.append(m)
                        elif pattern_name == "double_apostrophe_italic":
                            if len(m) > 4 and not m.startswith("''s"):
                                filtered.append(m)
                        elif pattern_name == "html_tags":
                            tag_lower = m.lower()
                            if not any(t in tag_lower for t in ['<br', '<p>', '</p>', '<strong', '</strong']):
                                filtered.append(m)
                        else:
                            filtered.append(m)
                    
                    if filtered:
                        results.append({
                            "item_number": line_num,
                            "item": item_name,
                            "path": path,
                            "irregularity_type": pattern_name,
                            "matches": filtered[:10],
                            "match_count": len(filtered),
                            "snippet": content[:300] if len(content) > 300 else content
                        })
    
    return results

def main():
    print(f"Scanning {INPUT_FILE}...")
    results = scan_file()
    
    type_counts = {}
    for r in results:
        t = r.get("irregularity_type", r.get("type", "unknown"))
        type_counts[t] = type_counts.get(t, 0) + 1
    
    report = {
        "scan_date": "2026-02-22",
        "file_scanned": os.path.basename(INPUT_FILE),
        "total_irregularities_found": len(results),
        "summary_by_type": type_counts,
        "findings": results
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nScan complete!")
    print(f"Total irregularities found: {len(results)}")
    print(f"\nBreakdown by type:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    print(f"\nReport saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()