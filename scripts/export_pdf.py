#!/usr/bin/env python3
"""
scripts/export_pdf.py - Export report_1.md to a clean, publication-quality report_final.pdf
Uses Python Markdown with MathJax and Mermaid.js, rendered via headless Brave/Chromium.
"""

import os
import re
import subprocess
import markdown

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MD_FILE = os.path.join(PROJECT_DIR, "report_3.md")
HTML_FILE = os.path.join(PROJECT_DIR, "report_temp.html")
PDF_FILE = os.path.join(PROJECT_DIR, "report_final.pdf")

def main():
    print(f"Reading {MD_FILE}...")
    with open(MD_FILE, "r", encoding="utf-8") as f:
        md_text = f.read()

    # 1. Protect Mermaid blocks
    mermaid_blocks = []
    def mermaid_repl(match):
        mermaid_blocks.append(match.group(1))
        return f"<!--MERMAID_PLACEHOLDER_{len(mermaid_blocks)-1}-->"
    
    md_text = re.sub(r'```mermaid\n(.*?)\n```', mermaid_repl, md_text, flags=re.DOTALL)

    # 2. Protect Display Math ($$...$$)
    display_math = []
    def display_math_repl(match):
        display_math.append(match.group(1))
        return f"@@DISPLAY_MATH_{len(display_math)-1}@@"

    md_text = re.sub(r'\$\$(.*?)\$\$', display_math_repl, md_text, flags=re.DOTALL)

    # 3. Protect Inline Math ($...$)
    inline_math = []
    def inline_math_repl(match):
        inline_math.append(match.group(1))
        return f"@@INLINE_MATH_{len(inline_math)-1}@@"

    md_text = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', inline_math_repl, md_text)

    # 4. Ensure blank line before tables and convert Markdown to HTML
    md_text = re.sub(r'([^\n|])\n+(\|.*?\|)', r'\1\n\n\2', md_text)
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])

    # 5. Restore Inline Math
    for idx, math in enumerate(inline_math):
        # Keep escaped or clean for MathJax
        html_body = html_body.replace(f"@@INLINE_MATH_{idx}@@", f"\\({math}\\)")

    # 6. Restore Display Math
    for idx, math in enumerate(display_math):
        html_body = html_body.replace(f"@@DISPLAY_MATH_{idx}@@", f"\\[{math}\\]")

    # 7. Restore Mermaid
    for idx, code in enumerate(mermaid_blocks):
        html_body = html_body.replace(f"&lt;!--MERMAID_PLACEHOLDER_{idx}--&gt;", f'<div class="mermaid">\n{code}\n</div>')
        html_body = html_body.replace(f"<!--MERMAID_PLACEHOLDER_{idx}-->", f'<div class="mermaid">\n{code}\n</div>')

    # 8. Fix image paths to absolute file:// URLs
    artifacts_dir = os.path.join(PROJECT_DIR, "artifacts")
    figures_dir = os.path.join(PROJECT_DIR, "figures")
    html_body = re.sub(r'src=["\']artifacts/([^"\']+)["\']', f'src="file://{artifacts_dir}/\\1"', html_body)
    html_body = re.sub(r'src=["\']figures/([^"\']+)["\']', f'src="file://{figures_dir}/\\1"', html_body)

    # 9. Build complete HTML template
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CS328 Project Report: Molecular Similarity Search Using Graph Neural Networks and Approximate Nearest Neighbors</title>
  
  <!-- MathJax Configuration -->
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['\\\\(', '\\\\)'], ['$', '$']],
        displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']],
        processEscapes: true
      }},
      options: {{
        renderActions: {{
          findScript: [10, function (doc) {{
            for (const node of document.querySelectorAll('script[type^="math/tex"]')) {{
              const display = !!node.type.match(/; *mode=display/);
              const math = new doc.options.MathItem(node.textContent, doc.inputJax[0], display);
              const text = document.createTextNode('');
              node.parentNode.replaceChild(text, node);
              math.start = {{node: text, delim: '', n: 0}};
              math.end = {{node: text, delim: '', n: 0}};
              doc.math.push(math);
            }}
          }}, '']
        }}
      }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

  <!-- Mermaid Configuration -->
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true, theme: 'neutral', securityLevel: 'loose' }});
  </script>

  <style>
    @page {{
      size: A4;
      margin: 13.5mm 12mm 10.5mm 12mm;
    }}
    
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      font-size: 8.5pt;
      line-height: 1.32;
      color: #212529;
      max-width: 100%;
      margin: 0 auto;
    }}

    h1 {{
      font-size: 13.8pt;
      font-weight: 700;
      color: #0b1d3a;
      line-height: 1.22;
      margin-top: 0;
      margin-bottom: 3.5pt;
      border-bottom: 2px solid #0b1d3a;
      padding-bottom: 2pt;
    }}

    h2 {{
      font-size: 10.5pt;
      font-weight: 600;
      color: #1a365d;
      margin-top: 6pt;
      margin-bottom: 2.2pt;
      border-bottom: 1px solid #cbd5e1;
      padding-bottom: 1.5pt;
      page-break-after: avoid;
    }}

    h3 {{
      font-size: 9.1pt;
      font-weight: 600;
      color: #2b6cb0;
      margin-top: 4pt;
      margin-bottom: 1.8pt;
      page-break-after: avoid;
    }}

    h4 {{
      font-size: 8.4pt;
      font-weight: 600;
      color: #4a5568;
      margin-top: 3pt;
      margin-bottom: 1.5pt;
      page-break-after: avoid;
    }}

    p {{
      margin-top: 1.5pt;
      margin-bottom: 3.0pt;
      text-align: justify;
    }}

    hr {{
      border: 0;
      height: 1px;
      background: #e2e8f0;
      margin: 4.5pt 0;
    }}

    p:has(+ table) {{
      margin-bottom: 2pt;
      font-weight: 600;
      color: #1e293b;
      page-break-after: avoid;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 3pt 0 5pt 0;
      font-size: 7.8pt;
      page-break-inside: avoid;
    }}

    th, td {{
      border: 1px solid #cbd5e1;
      padding: 2.8pt 4.5pt;
      text-align: left;
    }}

    th {{
      background-color: #f1f5f9;
      font-weight: 600;
      color: #0f172a;
    }}

    tr:nth-child(even) {{
      background-color: #f8fafc;
    }}

    img {{
      max-width: 98%;
      max-height: 255pt;
      width: auto;
      height: auto;
      object-fit: contain;
      display: block;
      margin: 3.5pt auto;
      page-break-inside: avoid;
      border-radius: 4px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}

    ol, ul {{
      margin-top: 1.5pt;
      margin-bottom: 2.5pt;
      padding-left: 15pt;
    }}

    li {{
      margin-bottom: 1.2pt;
      line-height: 1.25;
      font-size: 8.2pt;
    }}

    blockquote {{
      margin: 4pt 0;
      padding: 4pt 7pt;
      background-color: #f8fafc;
      border-left: 3.5px solid #3182ce;
      color: #334155;
      font-size: 8.0pt;
      page-break-inside: avoid;
    }}

    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 7.7pt;
      background-color: #f1f5f9;
      padding: 1pt 2.5pt;
      border-radius: 2.5pt;
      color: #0f172a;
    }}

    pre {{
      background-color: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 4pt;
      padding: 4.5pt 6.5pt;
      overflow-x: auto;
      page-break-inside: avoid;
      font-size: 7.2pt;
      line-height: 1.2;
    }}

    pre code {{
      background: none;
      padding: 0;
      font-size: 7.2pt;
    }}

    .mermaid {{
      display: flex;
      justify-content: center;
      margin: 16pt 0;
      page-break-inside: avoid;
    }}

    ul, ol {{
      margin-top: 4pt;
      margin-bottom: 8pt;
      padding-left: 18pt;
    }}

    li {{
      margin-bottom: 3pt;
    }}
  </style>
</head>
<body>
{html_body}
</body>
</html>
"""

    print(f"Writing temporary HTML to {HTML_FILE}...")
    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(full_html)

    # 10. Run Brave in headless mode with virtual time to allow MathJax & Mermaid to render
    print("Launching Brave headless PDF generation...")
    cmd = [
        "brave-browser",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=10000",
        f"--print-to-pdf={PDF_FILE}",
        f"file://{HTML_FILE}"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode == 0:
        pdf_size = os.path.getsize(PDF_FILE)
        print(f"SUCCESS: Generated {PDF_FILE} ({pdf_size / 1024:.1f} KB)")
    else:
        print(f"ERROR generating PDF: {res.stderr}")
        return False

    # Cleanup temporary HTML
    if os.path.exists(HTML_FILE):
        os.remove(HTML_FILE)
    return True

if __name__ == "__main__":
    main()
