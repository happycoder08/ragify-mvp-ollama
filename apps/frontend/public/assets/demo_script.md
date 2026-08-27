# RAGify Demo Script

## Step 1: Document Indexing
**What to show:** Navigate to the `/docs` page and demonstrate document upload.

**Script:**
> "First, let's see how easy it is to get started. Simply upload your documents—PDFs, Word files, or text documents. RAGify automatically indexes them into a vector database. You can see the status here: documents go from 'pending' to 'indexed' within seconds."

**Key points:**
- No manual data entry required
- Automatic chunking and embedding
- Real-time status tracking

---

## Step 2: Grounded Question
**What to show:** Navigate to `/query` and ask a question that IS covered in your documents.

**Script:**
> "Now let's ask a question about information that's actually in the documents. Watch how RAGify retrieves the most relevant evidence and generates a grounded answer based solely on your data."

**Example questions:**
- "What is the vacation policy?"
- "What are the security requirements for contractors?"
- "How do I submit an expense report?"

**Key points:**
- Answers are grounded in your documents
- No hallucinations—only factual responses
- Fast retrieval (<2 seconds typical)

---

## Step 3: Evidence Transparency
**What to show:** Expand the evidence panels to show source snippets.

**Script:**
> "Here's what makes RAGify trustworthy: every answer comes with evidence. You can expand these cards to see the exact text snippets used, with your query terms highlighted. This is full transparency—you can verify every claim."

**Key points:**
- Collapsible evidence cards
- Highlighted matching terms
- Source filenames and chunk IDs
- Copy snippets to clipboard

**Pro tip:** Enable the debug drawer (Ctrl+D) to show retrieval scores and backend details.

---

## Step 4: Refusal Demo
**What to show:** Ask a question that is NOT in the documents.

**Script:**
> "But what happens if you ask about something that's not in your documents? RAGify refuses to answer rather than making something up. This is the 'grounding gate'—a safety mechanism that prevents hallucinations."

**Example questions:**
- "What's the weather today?"
- "Who won the Super Bowl last year?"
- "Tell me about quantum computing."

**Key points:**
- Clear refusal banner
- Explanation of why it couldn't answer
- No evidence shown (because there is none)
- Prevents misinformation

---

## Step 5: Privacy & Deployment
**What to show:** Return to slides or discuss architecture.

**Script:**
> "Let's talk about privacy and deployment. RAGify is designed for enterprise environments:
> 
> **Privacy:**
> - Your data never leaves your infrastructure
> - Self-hosted deployment options (Azure, AWS, on-prem)
> - No data sent to third-party APIs
> - Full GDPR/SOC2 compliance support
> 
> **Deployment Options:**
> - **Cloud**: Deploy to your Azure subscription or AWS account
> - **On-Premises**: Run on your own hardware
> - **Hybrid**: Index on-prem, query from cloud
> 
> **Multi-tenancy:**
> - Isolated collections per tenant
> - Role-based access control
> - Audit logs for compliance
> 
> This means you get the power of AI without compromising on security or control."

**Key points:**
- Emphasize data sovereignty
- Highlight deployment flexibility
- Mention compliance frameworks
- Discuss tenant isolation

---

## Demo Tips

### Preparation
- Upload 2-3 sample documents before the demo
- Test both grounded and refusal questions
- Have demo mode enabled (`VITE_DEMO_MODE=true`)
- Clear browser cache if needed

### Pacing
- **Step 1-2**: 2 minutes (upload + first query)
- **Step 3**: 1 minute (evidence review)
- **Step 4**: 1 minute (refusal demo)
- **Step 5**: 2 minutes (architecture discussion)
- **Total**: ~6 minutes + Q&A

### Common Questions
**Q: How accurate is the retrieval?**
A: Vector similarity typically achieves 90%+ recall on well-structured documents. You can tune `top_k` and chunking strategies.

**Q: What document formats are supported?**
A: PDF, DOCX, TXT, and more. We extract text and preserve structure (headings, tables).

**Q: Can it handle multiple languages?**
A: Yes! The embedding models support 100+ languages.

**Q: What's the cost?**
A: Depends on deployment model. Self-hosted costs are primarily compute (GPU optional). Cloud pricing varies by provider.

---

## Closing

**Script:**
> "So that's RAGify: secure, transparent, and grounded in your documents. No hallucinations, full evidence, and complete control over your data. Ready to get started?"

**Next steps:**
- Schedule technical deep-dive
- Provide trial access
- Share deployment guide
- Discuss custom integrations
