import os
import json
from fpdf import FPDF

UPLOAD_DIR = "uploads_stress"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 1. TRICKY PDF: Multi-column, fragmented lines, and headers
def create_layout_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Simulating a header that appears on every page
    pdf.cell(200, 10, txt="CONFIDENTIAL - INTERNAL USE ONLY", ln=1, align='C')

    # Content with forced line breaks (the "fragmentation" issue)
    text_lines = [
        "1. PRO-", "JECT ALPHA",  # Hyphenation split
        "The project Alpha deadline is set for", "December 31st, 2025."
    ]
    for line in text_lines:
        pdf.cell(0, 10, txt=line, ln=1)

    pdf.output(os.path.join(UPLOAD_DIR, "project_alpha.pdf"))
    print(" -> Created project_alpha.pdf (Tests PDF fragmentation)")

# 2. MESSY HTML: Div soup and hidden text
def create_messy_html():
    html_content = """
    <html>
    <body>
        <div class=\"header\"><h1>IGNORE THIS HEADER</h1></div>
        <div class=\"content\">
            <p>The <b>Q3 Financial Report</b> indicates a <u>20% growth</u> in revenue.</p>
            <span style=\"display:none\">Ignore this hidden text.</span>
            <p>Key driver: <i>AI adoption</i> across the enterprise.</p>
        </div>
        <div class=\"footer\">Page 1 of 1</div>
    </body>
    </html>
    """
    with open(os.path.join(UPLOAD_DIR, "q3_financials.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    print(" -> Created q3_financials.html (Tests HTML cleaning)")

# 3. STRUCTURED DATA (JSON): Often misunderstood as text
def create_json_record():
    data = {
        "employee_id": "E-999",
        "name": "Robert 'Bobby' Tables",
        "department": "Security",
        "access_level": "Level 5",
        "notes": "Authorized for server room access."
    }
    with open(os.path.join(UPLOAD_DIR, "employee_record.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(" -> Created employee_record.json (Tests structured data parsing)")

# 4. CONTRADICTION TRAP: Two files, same topic, different facts
def create_contradictions():
    # File A
    with open(os.path.join(UPLOAD_DIR, "security_policy_v1.txt"), "w", encoding="utf-8") as f:
        f.write("PASSWORD POLICY V1\nPasswords must be at least 8 characters long.")

    # File B (Newer)
    with open(os.path.join(UPLOAD_DIR, "security_policy_v2_2025.txt"), "w", encoding="utf-8") as f:
        f.write("PASSWORD POLICY V2 (2025)\nPasswords must be at least 14 characters long and include a symbol.")

    print(" -> Created contradiction files (Tests temporal reasoning)")

if __name__ == "__main__":
    create_layout_pdf()
    create_messy_html()
    create_json_record()
    create_contradictions()
