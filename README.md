# XAI Design Explainer
XAI Design Explainer — What this repo does
Goal. Build a small but solid demo of explainable LLM-assisted design reasoning for engineering. The app retrieves design evidence (FAISS if available, otherwise BM25), generates a short rationale tied to that evidence, highlights evidence tokens inside the explanation, and logs human study ratings with basic uncertainty calibration. It’s designed as a starter for PhD/industry case studies and user experiments.
Key features
•	RAG backends (plug-in):
o	Uses FAISS + embeddings when faiss-cpu (or GPU) is installed.
o	Falls back to a lightweight BM25 retriever if FAISS isn’t present.
•	Explainability:
o	Generates concise, readable rationales that reference retrieved items.
o	Token-level evidence highlighting inside the explanation.
o	Reports a simple calibrated confidence (demo; pair with ECE/Brier in studies).
•	Human-Centered Evaluation:
o	Built-in Human Study Logger writes one row per rating to outputs/human_study_log.csv.
o	Fields include anonymized annotator id and political leaning to help you balance panels.
•	Industrial examples (placeholders):
o	Two sample evidence items (Altair/FEA study; parametric CAD) live in sample_data/evidence.jsonl.
o	Swap this file with your real design notes, FEA summaries, or CAD guidelines at any time.
How it works (architecture in 5 steps)
1.	Load evidence. sample_data/evidence.jsonl is read into memory. Each line is a JSON object with id, title, text, tags, source, date.
2.	Indexing / retrieval.
o	If FAISS + embeddings are available: build a vector index and use similarity search.
o	Otherwise: BM25 ranks documents by term relevance (no external models needed).
3.	Rationale generation. A small, deterministic template turns retrieved snippets + user preferences into a grounded design rationale. (This is where you can plug an LLM later.)
4.	Evidence highlighting. Tokens present in the retrieved snippets are marked inside the rationale so readers see why each claim appears.
5.	Human study logging. The UI records trust/clarity/usefulness (1–5), annotator id, and political leaning to outputs/human_study_log.csv for further analysis.
Repository layout
XAI-Design-Explainer/
├─ app.py                      # Gradio UI + RAG + highlighting + logger
├─ requirements.txt            # Minimal deps
├─ sample_data/
│  └─ evidence.jsonl           # Industrial placeholder evidence (swap with your own)
├─ outputs/
│  └─ .gitkeep                 # Ratings CSV gets written here at runtime
├─ .gitignore
├─ LICENSE
└─ README.md
Run locally / in Colab
pip install -r requirements.txt
python app.py
•	A Gradio link appears. Open it in your browser.
•	Enter a design brief and preferences/constraints → click Generate rationale.
•	Inspect the evidence-highlighted explanation and citations.
•	In “Human-centered evaluation,” give scores and Save rating.
•	Find the log at: outputs/human_study_log.csv.
Human Study CSV schema
column	meaning
ts	timestamp (UTC)
query	design brief entered by the user
preference	preference/constraints text
answer	rationale (HTML tags stripped before logging)
confidence	demo confidence in [0..1] (use alongside ECE/Brier in analysis)
annotator_id	anonymized id (e.g., rater01)
pol_leaning	left / center / right / other (panel balance)
trust	1–5
clarity	1–5
usefulness	1–5
Extending the project (plug-in points)
•	LLM backends: wrap your model/API and call it in generate_rationale(...) to replace the template with true LLM reasoning.
•	Retrievers: keep FAISS and BM25, or add Elastic/Weaviate by mirroring the retrieve(...) function signature.
•	Evidence highlighting: current token-level marking can be upgraded to span-level using alignment or attention-based attribution.
•	Evaluation: add faithfulness tests (contrastive retrieval, counterfactual near-misses) and report calibration (ECE/Brier) in your notebooks.
•	Industrial datasets: replace sample_data/evidence.jsonl with curated FEA/CAD notes, link to PDFs, or store structured metadata.
Why this matters (for GenAIDE / XAI in design)
•	Transparency: shows where claims come from during preference-based optimization.
•	Trust & usability: combines readable rationales, visual highlighting, and human ratings.
•	Research-ready: small, auditable codebase you can run anywhere; ready for user studies and for swapping in stronger LLMs/RAG backends.
License: MIT (see LICENSE). Please ensure you only load evidence you are allowed to use and respect data licenses and platform terms.
