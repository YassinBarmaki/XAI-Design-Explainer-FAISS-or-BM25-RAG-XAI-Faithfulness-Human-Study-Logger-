import os, re, time, json
from pathlib import Path
from typing import List, Tuple
import numpy as np, pandas as pd
import gradio as gr

BASE = Path(__file__).parent
DATA = BASE / "sample_data"
OUT  = BASE / "outputs"
DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
EVID_PATH = DATA / "evidence.jsonl"

def _bm25_tok(t): return re.findall(r"[A-Za-z0-9\-_.]+", (t or "").lower())

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        from math import log
        self.docs=[_bm25_tok(d) for d in docs]; self.N=len(self.docs); self.k1=k1; self.b=b
        self.df={}; self.avgdl=np.mean([len(d) for d in self.docs]) if self.docs else 0
        for d in self.docs:
            for w in set(d): self.df[w]=self.df.get(w,0)+1
        self.idf={w: log((self.N-df+0.5)/(df+0.5)+1) for w,df in self.df.items()}
    def score(self, q, i):
        q_t=_bm25_tok(q); d=self.docs[i]; dl=len(d) or 1; s=0.0
        for w in q_t:
            f=d.count(w)
            if w in self.idf:
                s += self.idf[w] * (f*(self.k1+1)/(f + self.k1*(1-self.b + self.b*dl/(self.avgdl or 1))))
        return s
    def search(self, q, k=5):
        if not self.docs: return []
        sc=[(self.score(q,i), i) for i in range(self.N)]; sc.sort(reverse=True)
        return [i for s,i in sc[:k] if s>0]

def load_evidence(path=EVID_PATH):
    items=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): items.append(json.loads(line))
    return items

EVID = load_evidence()
DOCS = [f"{r['title']}\n{r['text']}" for r in EVID]

# سعی در استفاده از FAISS؛ در صورت نبود، BM25
USE_FAISS=False
try:
    import faiss; USE_FAISS=True
except Exception:
    USE_FAISS=False

if USE_FAISS and DOCS:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    embeddings=None
    try:
        if os.environ.get("OPENAI_API_KEY"):
            embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
    except Exception:
        embeddings=None
    if embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    VSTORE = FAISS.from_texts(DOCS, embedding=embeddings, metadatas=[{"id":x["id"],"source":x["source"]} for x in EVID])
else:
    VSTORE = BM25(DOCS)

def retrieve(query, k=3):
    if isinstance(VSTORE, BM25):
        idxs=VSTORE.search(query, k=k)
        return [(DOCS[i], {"id":EVID[i]["id"], "source":EVID[i]["source"]}) for i in idxs]
    else:
        hits=VSTORE.similarity_search_with_score(query, k=k)
        out=[]
        for doc, sc in hits:
            md=getattr(doc,"metadata",{}) or {}
            out.append((doc.page_content, {"id":md.get("id"),"source":md.get("source"),"score":float(sc)}))
        return out

def evidence_highlight(answer:str, evs:List[str])->str:
    txt=answer
    for i,ev in enumerate(evs, start=1):
        toks=set(_bm25_tok(ev))
        def repl(m):
            w=m.group(0)
            return f"<mark>{w}</mark>" if w.lower() in toks and len(w)>2 else w
        txt=re.sub(r"[A-Za-z0-9\-_.]+", repl, txt)
        txt+=f"\n\n<span style='opacity:.7'>[EVIDENCE {i}: {ev[:160]}...]</span>"
    return txt

def generate_rationale(query:str, snippets:List[str], pref:str)->Tuple[str,float]:
    parts=[]
    if pref.strip(): parts.append(f"Preference acknowledged: {pref.strip()}.")
    if snippets:
        parts.append("Design rationale (grounded):")
        for i,s in enumerate(snippets, start=1):
            pick=' '.join(_bm25_tok(s)[:18])
            parts.append(f"- Based on evidence {i}: \"{pick}\"")
    else:
        parts.append("No evidence retrieved; answer remains cautious.")
    parts.append("Trade-off: preserve stiffness/stress limits while reducing weight; obey tooling constraints on fillets.")
    conf=0.6+0.1*min(3, len(snippets))
    parts.append(f"Uncertainty: calibrated confidence ≈ {conf:.2f} (ECE/Brier to report).")
    return "\n".join(parts), conf

LOG = (OUT/"human_study_log.csv")
if not LOG.exists():
    pd.DataFrame(columns=["ts","query","preference","answer","confidence","annotator_id","pol_leaning","trust","clarity","usefulness"]).to_csv(LOG, index=False)

def log_row(q,pref,ans,cfg,aid,lean,tr,cl,us):
    row={"ts":time.strftime("%Y-%m-%d %H:%M:%S"),"query":q,"preference":pref,"answer":re.sub(r"<.*?>","",ans or ""),
         "confidence":float(cfg),"annotator_id":aid,"pol_leaning":lean,"trust":int(tr),"clarity":int(cl),"usefulness":int(us)}
    df=pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    df=pd.concat([df, pd.DataFrame([row])], ignore_index=True); df.to_csv(LOG, index=False)
    return str(LOG)

def ui_generate(q,pref,topk):
    hits=retrieve(q, k=int(topk)); snips=[h[0] for h in hits]
    ans, conf = generate_rationale(q, snips, pref)
    ans_html = evidence_highlight(ans, snips)
    cites = "\n".join([f"[{i+1}] {h[1].get('source','-')} (id={h[1].get('id','-')})" for i,h in enumerate(hits)])
    return ans_html, f"{conf:.2f}", cites, "\n\n".join(snips)

def ui_log(q,p,a_html,conf,aid,lean,tr,cl,us):
    path=log_row(q,p,a_html,conf,aid,lean,tr,cl,us); return f"Saved → {path}"

def launch():
    with gr.Blocks(title="XAI Design Explainer (RAG + XAI)") as demo:
        gr.Markdown("## XAI Design Explainer — FAISS-or-BM25 RAG, Evidence Highlighting, Human Study Logger")
        q=gr.Textbox(label="Design brief/query", value="Propose a lighter bracket while preserving stiffness near load path.")
        pref=gr.Textbox(label="Engineer preferences/constraints", value="Minimize weight; fillet radius ≥ 3mm; avoid hard-to-tool cavities.")
        k=gr.Slider(1,8,value=3,step=1,label="Top-k evidence")
        go=gr.Button("Generate rationale", variant="primary")
        ans=gr.HTML(label="Rationale (with token-level evidence highlighting)")
        conf=gr.Textbox(label="Calibrated confidence (demo)")
        cites=gr.Textbox(label="Citations"); ev=gr.Textbox(label="Retrieved evidence", lines=6)
        gr.Markdown("### Human-centered evaluation")
        annot=gr.Textbox(label="Annotator ID", value="rater01")
        lean=gr.Dropdown(["left","center","right","other"], value="center", label="Political leaning")
        trust=gr.Slider(1,5, value=4, step=1, label="Trust"); clar=gr.Slider(1,5, value=4, step=1, label="Clarity"); usef=gr.Slider(1,5, value=4, step=1, label="Usefulness")
        save=gr.Button("Save rating"); saved=gr.Textbox(label="Log status")
        go.click(ui_generate, [q,pref,k], [ans,conf,cites,ev])
        save.click(ui_log, [q,pref,ans,conf,annot,lean,trust,clar,usef], [saved])
    demo.launch(share=False)

if __name__=="__main__": launch()