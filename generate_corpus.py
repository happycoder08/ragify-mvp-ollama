import os

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

policy_text = """
HR POLICY MANUAL - UPDATED JAN 2025

1. REMOTE WORK
Effective 2025, employees are permitted to work remotely 3 days per week (Tuesday, Wednesday, Thursday). 
Monday and Friday are mandatory in-office days.

2. MEAL ALLOWANCE
The travel meal allowance has been increased to $75 per day. Receipts are required for all expenses over $25.

3. CONFLICT OF INTEREST
Employees may not hold outside employment that competes with RAGify.
"""

old_policy_text = """
[ARCHIVED] HR POLICY 2023

1. REMOTE WORK
Employees may work remotely 5 days a week.

2. MEAL ALLOWANCE
The meal allowance is $40 per day.
"""

tech_manual = """
RAGIFY SERVER DEPLOYMENT GUIDE (v2.0)

ERROR CODES:
- E-101: Connection Timeout. Fix: Restart the Redis service.
- E-102: Auth Failure. Fix: Rotate the JWT secret.
- E-500: Memory Overflow. Fix: Increase CONTEXT_BUDGET_CHARS.

DEPLOYMENT STEPS:
1. Install dependencies using 'pip install -r requirements.txt'.
2. Set the RAGIFY_MODE environment variable.
3. Run migrations via 'alembic upgrade head'.
4. Start the server using 'uvicorn main:app'.
"""

meeting_notes = """
Lunch Order - Friday
- Sarah: Salad
- Mike: Burger (No onions)
- Admin: Please remind everyone to submit timesheets.
- Note: The 'Remote Work' discussion has been postponed to next quarter.
"""

def generate():
    files = {
        "policy_2025.txt": policy_text,
        "policy_2023_archived.txt": old_policy_text,
        "deployment_guide.txt": tech_manual,
        "friday_lunch_notes.txt": meeting_notes
    }

    print(f"Generating corpus in {UPLOAD_DIR}...")
    for filename, content in files.items():
        path = os.path.join(UPLOAD_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f" -> Created {filename}")

if __name__ == "__main__":
    generate()
