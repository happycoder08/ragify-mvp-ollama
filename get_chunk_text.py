from app.services import clients

clients.initialize_chroma_client()

# Adjust collection name if you use multi-tenancy
collection = clients.get_chroma_client().get_or_create_collection("documents_default")
result = collection.get(ids=["10_Employee_Onboarding_Guide_1.txt_9"])

if result and result.get("documents"):
    print("Chunk text for 10_Employee_Onboarding_Guide_1.txt_9:")
    print(result["documents"][0])
else:
    print("Chunk not found.")
