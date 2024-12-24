import re
from typing import List, Tuple

def clean_query(query: str) -> str:
    """Remove special characters and punctuation from query."""
    query = query.rstrip('?.!,;:')
    return query

def escape_regex_chars(text: str) -> str:
    """Escape special regex characters in text."""
    special_chars = r'[]{()}*+?.|^$\\'
    return ''.join('\\' + char if char in special_chars else char for char in text)

def format_output(docs_with_score: List[Tuple], query: str) -> str:
    query = clean_query(query)
    
    stop_words = {'what', 'is', 'the', 'between', 'and', 'how', 'why', 'when', 'where', 'which'}
    keywords = [word.lower() for word in query.split() if word.lower() not in stop_words]
    
    formatted_output = []
    
    for doc, score in docs_with_score:
        content = doc.page_content
        
        # Replace section titles and numbers without bolding, but keep them formatted
        content = re.sub(
            r'(Section\s+\d+\.\d+(?:\.\d+)?)',
            r'<div class="section-name">\1</div>',
            content
        )
        
        # Replace standalone section numbers without bolding
        content = re.sub(
            r'(?<![.\d])(\d+\.\d+(?:\.\d+)?)(?![.\d])',
            r'\1',  # Just keep the section number without <b> tag
            content
        )
        
        # Highlight keywords (still highlighting without bold)
        for keyword in keywords:
            escaped_keyword = escape_regex_chars(keyword)
            try:
                pattern = re.compile(f'({escaped_keyword})', re.IGNORECASE)
                content = pattern.sub(r'<span class="highlighted-text">\1</span>', content)
            except re.error:
                continue
        
        # Split paragraphs by `▶` and format with bullet points
        paragraphs = content.split('▶')
        formatted_paragraphs = []
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if para:
                if i == 0:
                    formatted_paragraphs.append(f"<div class='section-name'>{para}</div>")
                else:
                    formatted_paragraphs.append(f"<div class='bullet-point section-content'>{para}</div>")
        
        formatted_output.append('\n'.join(formatted_paragraphs))
    
    return '\n\n'.join(formatted_output)
