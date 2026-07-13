# Demo Script — BMW X1 Specification Guide

> Demo document: `BMW_Guide.pdf` (BMW X1 spec guide, AU, Oct 2021).
> Why this corpus: the specs are **not** in any LLM's training data in this exact form,
> so correct answers *prove* retrieval works — and guardrails matter.
> (Legacy Transformer-paper questions at the bottom.)

## Quick Demo Questions (30-second demo)

### 1. Basic Retrieval
**Question:** "How much torque does the sDrive18d produce?"
**Expected:** "330 Nm" — retrieved from the model overview page

### 2. Specific Spec
**Question:** "What is the 0-100 km/h time of the sDrive20i?"
**Expected:** "7.6 seconds" — exact figure from the spec table

### 3. Equipment Question
**Question:** "What split do the foldable rear seats have?"
**Expected:** "40:20:40" — from standard equipment list

## Impressive Demo Questions

### 4. Package Contents
**Question:** "What features are included in the Enhancement Package?"
**Expected:** Metallic paint + panorama glass sunroof (+ 19" wheels)

### 5. Trim Detail
**Question:** "What colour are the callipers on the M Sport Brakes?"
**Expected:** "Blue" (red available on request) — buried in the M Sport Plus Package section

### 6. Model Codes
**Question:** "What is the model code of the sDrive18d?"
**Expected:** "52AC" — exact-identifier retrieval

### 7. Follow-up (Tests Memory)
**First:** "What engine does the sDrive18i have?"
**Then:** "And how fast does it get to 100?"
**Expected:** Second answer resolves "it" via conversation memory → 9.6 sec

## 🛡️ Guardrail Demo (the differentiator)

Ask these with the BMW guide loaded:

### 8. Off-topic Fact
**Question:** "What is the capital of France?"
**Expected:** *"The uploaded documents do not contain this information"* — NOT "Paris"!
**Why it matters:** the model obviously *knows* Paris. Refusing proves answers are
grounded in the document, not parametric memory.

### 9. Adversarially Close Off-topic
**Question:** "What is the price of a Tesla Model 3?"
**Expected:** Refusal. This one is hard — car-adjacent wording retrieves BMW chunks
with relevance 0.68, nearly indistinguishable from real questions. The retrieval
threshold alone can't block it; the document-only system prompt catches it.
**Talking point:** defense in depth — two layers because each one alone is bypassable.

### 10. Explain the Mechanism
Point at `agents.py`: if no chunk scores ≥ 0.65 relevance, the retriever tool refuses
to feed the LLM context; the agent may retry once with different wording, then must
tell the user. Threshold calibrated with `evaluate.py`: answerable questions scored
0.664–0.858, off-topic 0.597–0.683.

## 📏 Evaluation Demo

```bash
python evaluate.py                 # retrieval eval — seconds, no LLM cost
python evaluate.py --generation    # full scorecard — ~2 min
```

Show the scorecard and explain each metric:

- **Recall@5 = 100%** — every golden question's answer chunk is retrieved
- **Correctness = 100%** (12/12) — substring check against gold facts
- **Faithfulness = 100%** (12/12) — LLM-as-judge: every claim supported by retrieved context
- **Refusal rate = 100%** (5/5) — all off-topic questions correctly refused

**Talking point:** "Every change I made — chunk size, threshold, prompt — was
validated against this golden dataset. That's the difference between tweaking
vibes and engineering."

## 📚 Multi-Document Demo

Load **both** BMW_Guide.pdf and Attention_is_all_you_need.pdf, then:

1. **Scoped refusal:** tick ONLY BMW_Guide.pdf and ask *"How many attention heads
   does the model use?"* → refused (best BMW relevance 0.635, below the 0.65
   threshold). Tick the paper → answered ("8"). Same question, different scope,
   guardrail composes with document filtering.
2. **Cross-domain isolation:** with both ticked, ask a BMW question and show the
   retrieved chunks all come from BMW_Guide.pdf — metadata filters + relevance
   ranking keep domains from contaminating each other.
3. **Talking point:** "Retrieval is scoped with a `doc_name` metadata filter —
   the same mechanism production systems use for multi-tenant isolation, just
   with tenant_id from the JWT instead of a sidebar checkbox."

## Questions to Demonstrate Controls

### Temperature Demo
Same question at 0.0 vs 0.7: "Describe the M Sport Package" → precise list vs chattier prose

### Top-k Demo
- **k=1:** "What does the Comfort Package include?" → may miss items
- **k=5:** same question → complete list

### Retrieval Scores
Enable "Show Retrieval Scores": scores are relevance normalized to 0–1, higher = better.
"0.85 means high confidence; below our 0.65 threshold the system refuses to answer."

## Recruiter-Friendly Explanations

1. **Grounding**: "Ask it the capital of France — it refuses. Every answer is provably from the document."
2. **Agent Reasoning**: "The agent *decides* to retrieve; for 'thanks!' it skips retrieval entirely — cheaper and faster than fixed-pipeline RAG"
3. **Chunking**: "800 characters with 150 overlap so facts don't get split across chunk boundaries"
4. **Calibration**: "The refusal threshold isn't a guess — I measured relevance-score distributions for answerable vs off-topic questions and set it between them"
5. **Evaluation**: "12 golden questions + 5 off-topic traps, scored on recall, correctness, faithfulness, and refusal rate"

## Quick Demo Script (2 minutes)

```
1. Upload BMW_Guide.pdf → show ingestion success
2. Ask: "How much torque does the sDrive18d produce?" → 330 Nm
3. Enable "Show Retrieval Scores" → point at relevance ~0.8
4. Ask: "What is the capital of France?" → refusal (the wow moment)
5. Explain the two guardrail layers (threshold + document-only prompt)
6. Terminal: python evaluate.py → show 100% recall + calibration ranges
7. Close: "every design decision here was measured, not guessed"
```

## Technical Talking Points

- "OpenAI embeddings + ChromaDB for semantic search, LangGraph ReAct agent on top"
- "Normalized relevance scores (0–1) — I actually fixed a bug where raw distances were displayed as similarities"
- "Relevance-threshold guardrail, calibrated from measured score distributions"
- "Two-layer hallucination defense: tool-level threshold + document-only system prompt"
- "Golden-dataset eval harness: recall@5, correctness, LLM-judged faithfulness, refusal rate"
- "Found and fixed a real guardrail bypass: the agent answered 'Paris' without calling the retriever until the prompt forced retrieval for all factual questions"
- "Best bug of the project: the retriever tool truncated chunks to 300 chars, so the agent retrieved the right chunk but couldn't see the answer inside it — fixing one line took correctness from 75% to 100%"

---

## Legacy: "Attention is All You Need" Questions

If demoing with the Transformer paper instead:

1. "What is the main contribution of this paper?" — abstract retrieval
2. "What were the BLEU scores on WMT 2014 English-to-German?" — exact figures (28.4)
3. "What optimizer and learning rate schedule did they use?" — Adam, custom warmup schedule
4. "Explain the scaled dot-product attention mechanism" — formula section
5. Follow-up pair: "What is positional encoding?" → "Why is it necessary?" — memory demo

---

**Pro Tip:** Practice the demo 2-3 times. Lead with a correct BMW answer, then the
France refusal — the contrast between "knows the document" and "refuses what it
can't verify" is the strongest 20 seconds of the demo.
