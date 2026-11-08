import os
import re

md_path = "CAPSTONE_PROJECT_REPORT.md"
html_path = "CAPSTONE_PROJECT_REPORT.html"

with open(md_path, "r", encoding="utf-8") as f:
    md_content = f.read()

# Convert markdown tables to HTML tables
def format_md_to_html(text):
    # Basic markdown conversions
    lines = text.split("\n")
    html_lines = []
    in_code_block = False
    in_table = False
    table_header = True

    for line in lines:
        if line.startswith("```"):
            if in_code_block:
                html_lines.append("</pre>")
                in_code_block = False
            else:
                lang = line[3:].strip()
                html_lines.append(f"<pre class='code-block {lang}'>")
                in_code_block = True
            continue
        
        if in_code_block:
            html_lines.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            continue

        # Tables
        if "|" in line:
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if not in_table:
                in_table = True
                table_header = True
                html_lines.append("<div class='table-container'><table>")
                html_lines.append("<thead><tr>" + "".join([f"<th>{p}</th>" for p in parts]) + "</tr></thead><tbody>")
                continue
            elif all(set(p).issubset({"-", ":", " "}) for p in parts):
                continue
            else:
                html_lines.append("<tr>" + "".join([f"<td>{p}</td>" for p in parts]) + "</tr>")
                continue
        elif in_table:
            html_lines.append("</tbody></table></div>")
            in_table = False

        # Headers
        if line.startswith("# "):
            html_lines.append(f"<h1 class='title-1'>{line[2:].strip()}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2 class='title-2'>{line[3:].strip()}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3 class='title-3'>{line[4:].strip()}</h3>")
        elif line.startswith("#### "):
            html_lines.append(f"<h4 class='title-4'>{line[5:].strip()}</h4>")
        elif line.strip() == "---":
            html_lines.append("<hr class='divider'/>")
        elif line.strip() == '<div style="page-break-after: always;"></div>':
            html_lines.append("<div class='page-break'></div>")
        elif line.strip() == '<div align="center">' or line.strip() == '</div>':
            html_lines.append(line)
        elif line.startswith("- "):
            html_lines.append(f"<li>{line[2:].strip()}</li>")
        elif line.strip() != "":
            html_lines.append(f"<p>{line.strip()}</p>")

    if in_table:
        html_lines.append("</tbody></table></div>")

    return "\n".join(html_lines)

body_html = format_md_to_html(md_content)

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lovely Professional University - Capstone Project Report (Paras Rajput - 12304811)</title>
<style>
    @media print {{
        .page-break {{ page-break-after: always; }}
        body {{ margin: 0; padding: 20mm; }}
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.65;
        color: #1e293b;
        max-width: 900px;
        margin: 40px auto;
        padding: 0 25px;
        background-color: #ffffff;
    }}
    h1, h2, h3, h4 {{
        color: #0f172a;
        font-weight: 700;
        margin-top: 1.8em;
        margin-bottom: 0.6em;
    }}
    .title-1 {{
        font-size: 24pt;
        letter-spacing: -0.5px;
        text-align: center;
        margin-top: 25px;
    }}
    .title-2 {{
        font-size: 16pt;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
        margin-top: 40px;
    }}
    .title-3 {{ font-size: 13pt; color: #1e3a8a; }}
    .title-4 {{ font-size: 11pt; }}
    p {{ margin-bottom: 1em; text-align: justify; }}
    li {{ margin-bottom: 0.35em; }}
    .table-container {{
        overflow-x: auto;
        margin: 20px 0;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 10pt;
    }}
    th, td {{
        border: 1px solid #cbd5e1;
        padding: 10px 12px;
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
    pre.code-block {{
        background: #0f172a;
        color: #f8fafc;
        padding: 16px;
        border-radius: 6px;
        font-family: Consolas, "Courier New", monospace;
        font-size: 9.5pt;
        overflow-x: auto;
        line-height: 1.45;
        margin: 18px 0;
    }}
    .divider {{
        border: 0;
        height: 1px;
        background: #e2e8f0;
        margin: 35px 0;
    }}
    .page-break {{
        page-break-after: always;
        height: 25px;
    }}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(full_html)

print("Saved beautiful HTML report to:", html_path)
