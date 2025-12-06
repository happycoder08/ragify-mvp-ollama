#!/usr/bin/env python3
"""
One-click demo setup for RAGify.
Pre-populates with sample documents for fast, impressive demos.
"""

import os
from pathlib import Path

# Sample documents to pre-load
SAMPLE_DOCS = {
    "late_fee_policy.txt": """
Late Fee Policy - Effective January 1, 2024

1. STANDARD LATE FEES
   - 1-10 days late: $25 fee applied
   - 11-30 days late: $50 fee applied
   - 31+ days late: $100 fee + interest at 8% annual rate

2. GRACE PERIODS
   - Automatic 5-day grace period for first-time late payers
   - Military personnel: 30-day grace period
   - Hardship cases: Contact customer service for options

3. PAYMENT ARRANGEMENTS
   - Late payers can set up payment plans with 0% interest
   - Plans available for 30, 60, or 90 days
   - Must be initiated before day 60 of delinquency

4. HARDSHIP EXCEPTIONS
   - Medical hardship: Up to 60-day extension available
   - Job loss: 90-day forbearance program
   - Natural disaster: Fees waived for affected areas

5. APPEAL PROCESS
   - Appeals must be submitted within 30 days of fee assessment
   - Valid reasons: Billing error, system error, documented hardship
   - Average appeal resolution time: 5-7 business days
""",
    
    "customer_service_procedures.txt": """
Customer Service Response Time Standards

1. CONTACT CHANNELS & RESPONSE TIMES
   - Phone: Answer within 2 minutes or offer callback
   - Email: Respond within 4 business hours
   - Chat: Respond within 30 seconds
   - Social media: Respond within 2 hours

2. ESCALATION PROCEDURES
   - Level 1: Standard rep (can authorize up to $250 credit)
   - Level 2: Supervisor (can authorize up to $1,000 credit)
   - Level 3: Manager (can authorize up to $5,000 credit)
   - Level 4: Director (unlimited authority for settlement)

3. COMPLAINT HANDLING
   - Acknowledge complaint within 1 hour
   - Investigate within 24 hours
   - Provide resolution within 5 business days
   - Follow up on customer satisfaction

4. QUALITY ASSURANCE
   - 10% of interactions monitored monthly
   - Customer satisfaction surveys after each interaction
   - Target: 90% satisfaction rate
   - Root cause analysis for complaints

5. DOCUMENTATION REQUIREMENTS
   - Every interaction logged with timestamp
   - Resolution documented with reason code
   - Customer consent recorded for policy exceptions
   - Escalation rationale documented
""",
    
    "frequently_asked_questions.txt": """
Frequently Asked Questions

Q: How do I set up payment arrangement?
A: Call our customer service at 1-800-RAG-HELP or use the website.
   You can arrange payments for 30, 60, or 90 days with 0% interest.
   Must be initiated before day 60 of delinquency.

Q: Can late fees be waived?
A: Late fees may be waived in hardship cases. Valid reasons include:
   - Medical emergency (up to 60-day extension)
   - Job loss (90-day forbearance program)
   - Natural disaster (fees waived for affected areas)
   Contact us to discuss your specific situation.

Q: What happens if I don't pay after 90 days?
A: After 90 days:
   - Account referred to collections
   - Credit report impact reported
   - Legal action may be pursued
   - Additional fees and interest accrued

Q: Can I appeal a late fee?
A: Yes, appeals can be submitted within 30 days of the fee.
   Valid reasons: Billing error, system error, documented hardship.
   Average resolution time is 5-7 business days.

Q: How long does the appeals process take?
A: Average resolution time is 5-7 business days.
   You'll receive a decision via email or mail.
   If denied, you can escalate to management level.

Q: Do I get a grace period?
A: New customers get a 5-day automatic grace period.
   Military personnel get 30-day grace period.
   Others: Contact support to discuss options.
"""
}


def create_demo_docs():
    """Create sample documents in uploads directory."""
    upload_dir = Path("app/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, content in SAMPLE_DOCS.items():
        filepath = upload_dir / filename
        with open(filepath, "w") as f:
            f.write(content)
        print(f"✓ Created {filename}")
    
    print(f"\n✓ Demo documents ready in app/uploads/")
    print(f"  Start server and upload these files to index them.")


def main():
    print("=" * 60)
    print("RAGify Demo Setup")
    print("=" * 60)
    print()
    
    print("Step 1: Creating sample documents...")
    create_demo_docs()
    
    print()
    print("Step 2: Start the server with optimal demo settings:")
    print()
    print("  For best demo experience:")
    print("  $env:RAGIFY_CHUNK_SIZE = '800'")
    print("  $env:RAGIFY_CHUNK_OVERLAP = '150'")
    print("  .\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000")
    print()
    
    print("Step 3: Upload the documents:")
    print("  1. Open http://localhost:8000")
    print("  2. Upload all .txt files from app/uploads/")
    print("  3. Wait for indexing to complete")
    print()
    
    print("Step 4: Try demo questions:")
    print("  - 'What is our late fee policy?'")
    print("  - 'If a payment is 45 days late, what fees apply?'")
    print("  - 'Can late fees be waived?'")
    print("  - 'How long is the appeals process?'")
    print()
    
    print("=" * 60)
    print("Demo setup complete! Ready to impress clients.")
    print("=" * 60)


if __name__ == "__main__":
    main()
