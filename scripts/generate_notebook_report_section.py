import json
import re
import os

with open("Financial_Transaction_Fraud_Detection_PySpark.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def clean_ansi(text):
    return ansi_escape.sub('', text)

html_cells = []
md_cells = []

cell_counter = 1

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        md_text = "".join(cell["source"]).strip()
        md_cells.append(f"\n{md_text}\n")
        html_cells.append(f"<div class='nb-markdown'>{md_text}</div>")
    elif cell["cell_type"] == "code":
        code_text = "".join(cell["source"]).strip()
        md_cells.append(f"**In [{cell_counter}]:**\n```python\n{code_text}\n```")
        
        # HTML for code input
        code_html = f"""<div class='nb-cell'>
    <div class='nb-input-label'>In [{cell_counter}]:</div>
    <div class='nb-input-code'><pre class='code-snippet'><code>{code_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')}</code></pre></div>
</div>"""
        
        output_blocks_md = []
        output_blocks_html = []
        
        for out in cell.get("outputs", []):
            otype = out.get("output_type")
            if otype == "stream":
                txt = clean_ansi("".join(out.get("text", []))).strip()
                if txt:
                    output_blocks_md.append(f"```text\n{txt}\n```")
                    output_blocks_html.append(f"<pre class='nb-output-stream'>{txt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')}</pre>")
            elif otype == "display_data":
                data = out.get("data", {})
                if "text/html" in data:
                    table_html = "".join(data["text/html"]).strip()
                    output_blocks_html.append(f"<div class='nb-output-table'>{table_html}</div>")
                    output_blocks_md.append(f"*(Interactive DataFrame/Table displayed)*")
                elif "text/plain" in data:
                    plain_txt = clean_ansi("".join(data["text/plain"])).strip()
                    if "<IPython.core.display.Image" not in plain_txt:
                        output_blocks_md.append(f"```text\n{plain_txt}\n```")
                        output_blocks_html.append(f"<pre class='nb-output-stream'>{plain_txt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')}</pre>")
            elif otype == "execute_result":
                data = out.get("data", {})
                if "text/html" in data:
                    table_html = "".join(data["text/html"]).strip()
                    output_blocks_html.append(f"<div class='nb-output-table'>{table_html}</div>")
                elif "text/plain" in data:
                    plain_txt = clean_ansi("".join(data["text/plain"])).strip()
                    output_blocks_md.append(f"```text\n{plain_txt}\n```")
                    output_blocks_html.append(f"<pre class='nb-output-stream'>{plain_txt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')}</pre>")

        if output_blocks_md:
            md_cells.append(f"**Out [{cell_counter}]:**\n" + "\n".join(output_blocks_md))
        
        if output_blocks_html:
            code_html += f"""<div class='nb-output'>
    <div class='nb-output-label'>Out [{cell_counter}]:</div>
    <div class='nb-output-data'>{''.join(output_blocks_html)}</div>
</div>"""
        
        html_cells.append(code_html)
        cell_counter += 1

with open("scripts/notebook_section.html", "w", encoding="utf-8") as f:
    f.write("\n".join(html_cells))

with open("scripts/notebook_section.md", "w", encoding="utf-8") as f:
    f.write("\n\n".join(md_cells))

print("Successfully extracted notebook section to scripts/notebook_section.html and .md")
