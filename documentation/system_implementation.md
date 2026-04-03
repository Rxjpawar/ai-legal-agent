# 🏛️ Legal Research & Drafting Multi-Agent System

An enterprise-grade **Multi-Agent Legal AI System** built using **LangGraph + RAG + Vector Search**, designed to assist with:

- Legal research  
- Case analysis  
- Structured document drafting  
- Citation verification  
- Legal reasoning with supporting evidence  

---

#  Overview

This system implements a **Retrieval-Augmented Generation (RAG)** enhanced multi-agent architecture to perform structured legal reasoning and drafting workflows.

It ensures:

- Evidence-backed outputs  
- Proper citation tracking  
- Reduced hallucinations  
- Jurisdiction-aware research  
- Review and verification loop  

---

# System Architecture
User Query
↓
Legal Research Agent
↓
Case Analysis Agent
↓
Document Drafting Agent
↓
Review Agent
↓
Final Legal Output (Verified + Cited)

Orchestrated using **LangGraph state machine workflows**.

---

#  Agents

## 1. Legal Research Agent

### Responsibilities

- Retrieve relevant case laws  
- Retrieve statutes and legal codes  
- Apply jurisdiction filters  
- Perform semantic search via embeddings  
- Return ranked legal authorities  

### Retrieval Pipeline
Query → Embedding → Qdrant Search → Top-K Authorities


---

## 2. Case Analysis Agent

Extracts:

- Legal principles  
- Ratio decidendi  
- Key facts  
- Holdings  
- Applicability to the current issue  

Structured reasoning format:
Issue → Rule → Application → Conclusion

---

## 3️⃣ Document Drafting Agent

Generates:

- Legal notices  
- Contracts  
- Case briefs  
- Petitions  
- Memorandums  

Features:

- Clause-level drafting  
- Context-aware citation injection  
- Jurisdiction-sensitive formatting  

---

## 4. Review Agent

Validates:

- Citation correctness  
- Logical consistency  
- Unsupported claims  
- Missing authorities  
- Hallucinated references 
