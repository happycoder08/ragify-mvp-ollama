# RAGify Demo Script - Employee Onboarding Use Case

## Demo Scenario
**Context**: New employee's first day at TechCorp Solutions. They have the onboarding guide but want quick answers without reading 50+ pages.

## Demo Flow

### 1. FIRST DAY BASICS

**Question**: "What time should I arrive on my first day?"
**Expected Answer**: 8:00 AM at main reception on 3rd floor

**Question**: "What documents do I need to bring on my first day?"
**Expected Answer**: Government-issued ID, signed offer letter, completed I-9 form

**Question**: "How do I set up my email signature?"
**Expected Answer**: Specific format with name, title, phone, email, website in Arial 10pt

---

### 2. REMOTE WORK QUESTIONS

**Question**: "How many days can I work remotely in a week?"
**Expected Answer**: Up to 3 days per week, with Tuesdays and Thursdays required in-office

**Question**: "What are the core hours I need to be available?"
**Expected Answer**: 10:00 AM - 3:00 PM

**Question**: "Do I need to have my camera on for meetings?"
**Expected Answer**: Video on for team meetings, optional for large all-hands

---

### 3. MANAGER MEETING PREP

**Question**: "What should I ask my manager in our first 1:1?"
**Expected Answer**: 
- "What does success look like in my first 30 days?"
- "Who are the key people I should connect with?"
- "What's the best way to reach you if I have questions?"
- "Are there any team norms I should know?"
- "What resources should I review?"

---

### 4. BENEFITS & TIME OFF

**Question**: "How many vacation days do I get?"
**Expected Answer**: 15 days per year (prorated first year)

**Question**: "When does my health insurance start?"
**Expected Answer**: Coverage starts on your first day

**Question**: "What if I need to take a sick day?"
**Expected Answer**: Notify manager ASAP via Slack/email. No doctor's note needed unless out 3+ days

---

### 5. OFFICE LOGISTICS

**Question**: "What's the dress code?"
**Expected Answer**: Business casual - nice jeans/slacks, collared shirt/blouse

**Question**: "Can I bring my dog to work?"
**Expected Answer**: Yes, dog-friendly office. Must be well-behaved and registered with facilities

**Question**: "How do I get reimbursed for an expense?"
**Expected Answer**: Use Expensify, upload receipt, categorize, submit for manager approval

---

### 6. COMMUNICATION GUIDELINES

**Question**: "When should I use email vs Slack?"
**Expected Answer**: 
- Email: Formal communications, external stakeholders, documentation
- Slack: Quick questions, team updates, informal communication

**Question**: "How quickly do I need to respond to messages?"
**Expected Answer**: Slack within 2 hours during core hours, email within 24 hours

---

### 7. PROFESSIONAL DEVELOPMENT

**Question**: "Does the company provide learning budget?"
**Expected Answer**: $1,500 annual learning budget per employee, plus 1 conference per year

---

### 8. MULTI-STEP GUIDANCE

**Question**: "Walk me through what happens in my first week"
**Expected Answer**: Should provide sequential breakdown of Days 1-5 activities

**Question**: "What are my 30-60-90 day goals?"
**Expected Answer**: Should outline milestones for each period

---

## Key Demo Talking Points

### Speed & Efficiency
- "Notice how you get instant answers without scrolling through a 50-page PDF"
- "Employees can ask in natural language - no need to know exact keywords"
- "Answers are contextual and specific, not generic boilerplate"

### Accuracy & Sources
- "Every answer includes sources - employees can verify and read more"
- "RAG retrieves the exact section relevant to the question"
- "No hallucinations - answers come directly from your documents"

### Use Cases Beyond Onboarding
- Employee handbook Q&A
- IT support documentation
- Sales playbooks and objection handling
- Compliance and policy questions
- Product documentation for customer support
- Training material for new features

### ROI Benefits
- **Reduced HR/IT tickets**: Self-service reduces support burden
- **Faster onboarding**: New hires get productive faster
- **Better compliance**: Consistent, accurate policy answers
- **24/7 availability**: No waiting for someone to answer
- **Easy updates**: Update documents, RAG automatically uses new info

---

## Demo Tips

1. **Start with simple questions** to show basic retrieval
2. **Progress to complex questions** that need multi-paragraph context
3. **Show "not in document" response** with unrelated question
4. **Upload second document** to show multi-document retrieval
5. **Ask same question different ways** to show natural language understanding

## Sample "Not Found" Questions
- "What's the company's stock price?" (not in onboarding doc)
- "Who is the CEO's favorite sports team?" (irrelevant)
- These should return: "I could not find anything relevant..."

---

## Next Steps After Demo

- Discuss client's specific use cases
- Identify which documents to index first
- Plan integration with existing systems (Slack, Teams, etc.)
- Define success metrics (ticket reduction, time saved, etc.)
