"""Debug chunking logic for EMAIL SIGNATURE section."""
from app.services.ingestion import chunk_text_sections

text = """4. EMAIL SIGNATURE SETUP (11:30 AM)
   - Required format:
     [Your Name]
     [Your Job Title]
     TechCorp Solutions
     Phone: [Your Extension]
     Email: firstname.lastname@techcorp.com
     Website: www.techcorp.com
   
   - Optional: LinkedIn profile link, pronouns
   - Font: Arial, 10pt, color #333333"""

print("Input text length:", len(text))
print("\nCalling chunk_text_sections with chunk_size=300, overlap=50...")

chunks = chunk_text_sections(text, chunk_size=300, overlap=50)

print(f"\nGot {len(chunks)} chunks:")
for i, chunk in enumerate(chunks):
    print(f"\n{'='*60}")
    print(f"Chunk {i} (length={len(chunk)}):")
    print(f"{'='*60}")
    print(chunk)
    print(f"Has 'signature': {'signature' in chunk.lower()}")
    print(f"Has 'arial': {'arial' in chunk.lower()}")
    print(f"Has '10pt': {'10pt' in chunk.lower()}")
