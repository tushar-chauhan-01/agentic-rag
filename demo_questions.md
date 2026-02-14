# Demo Questions for "Attention is All You Need"

## Quick Demo Questions (30-second demo)

### 1. Basic Retrieval
**Question:** "What is the main contribution of this paper?"
**Expected:** Should retrieve abstract/introduction and summarize the Transformer architecture

### 2. Technical Detail
**Question:** "Explain the scaled dot-product attention mechanism"
**Expected:** Should retrieve the specific section with formula and explanation

### 3. Comparison
**Question:** "How does the Transformer architecture differ from RNN-based models?"
**Expected:** Should retrieve multiple sections comparing architectures

## Impressive Demo Questions (Show off capabilities)

### 4. Summarization
**Question:** "Summarize the key innovations of the Transformer in 3 bullet points"
**Expected:** Agent should use summarization tool, synthesize multiple sections

### 5. Specific Results
**Question:** "What were the BLEU scores on the WMT 2014 English-to-German translation task?"
**Expected:** Should retrieve exact numbers from results section

### 6. Architecture Details
**Question:** "Describe the multi-head attention mechanism"
**Expected:** Should retrieve technical details with formula

### 7. Training Details
**Question:** "What optimizer and learning rate schedule did they use?"
**Expected:** Should find specific training hyperparameters

## Questions to Show Agentic Behavior

### 8. Multi-Step Reasoning
**Question:** "Why is self-attention better than recurrence? Give me the computational advantages."
**Expected:** Agent retrieves → evaluates → retrieves more specific info → synthesizes

### 9. Comparison Question
**Question:** "Compare the model size and training time of the Transformer to previous models"
**Expected:** Agent should retrieve from multiple sections and compare

### 10. Follow-up Question (Tests Memory)
**First:** "What is positional encoding?"
**Then:** "Why is it necessary in the Transformer?"
**Expected:** Agent remembers context from previous question

## Questions to Demonstrate Controls

### Temperature Demo
Ask same question at different temperatures:
- **Temperature 0.0:** "Explain attention mechanism" → Factual, precise
- **Temperature 0.7:** "Explain attention mechanism" → More creative, verbose

### Top-k Demo
Ask same question with different top-k:
- **k=1:** "What is the Transformer?" → Minimal context
- **k=5:** "What is the Transformer?" → Rich context

## Recruiter-Friendly Explanations

When showing the demo, explain:

1. **Retrieval Scores**: "Notice the similarity scores - 0.85 is high confidence, showing the system found relevant content"

2. **Agent Reasoning**: "The agent decided to retrieve because this is a knowledge question, not something it can answer from memory"

3. **Chunking Strategy**: "I chunked the paper into 800-token segments with 150-token overlap to maintain context"

4. **Temperature Control**: "Temperature controls randomness - lower values give more deterministic, factual answers"

5. **Top-k Tradeoff**: "Top-k is precision vs recall - too low misses context, too high adds noise"

## Failure Cases to Show (Demonstrates Understanding)

### When RAG Fails
**Question:** "What do you think about the Transformer's impact on modern AI?"
**Expected:** Should acknowledge this requires opinion, not just retrieval

### Out of Scope
**Question:** "How does BERT use Transformers?"
**Expected:** Should say "not in the document" (BERT came later)

## Quick Demo Script (1 minute)

```
1. Upload "attention_is_all_you_need.pdf"
2. Show ingestion success
3. Ask: "What is the main contribution?"
4. Point out:
   - Agent reasoning
   - Retrieved chunks with scores
   - Synthesized answer
5. Adjust temperature slider
6. Ask same question - show difference
7. Show "Retrieved Context" section
8. Ask follow-up: "How does multi-head attention work?"
9. Show memory working
```

## Technical Talking Points

- "I used OpenAI embeddings for semantic search"
- "ChromaDB for vector storage with persistence"
- "LangChain ReAct agent for agentic reasoning"
- "Chunk size optimized for this domain (800 tokens)"
- "Temperature and top-k exposed for runtime control"
- "Retrieval scores provide transparency"

---

**Pro Tip:** Practice the demo 2-3 times so you can explain smoothly while showing the UI.
