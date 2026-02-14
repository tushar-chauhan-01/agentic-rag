#!/bin/bash
# Monitor ChromaDB and Streamlit logs in real-time

echo "=========================================="
echo "MONITORING STREAMLIT UPLOAD"
echo "=========================================="
echo ""
echo "Initial DB state:"
ls -lah chroma_db/ | tail -5
echo ""
echo "DB size: $(du -sh chroma_db/ | cut -f1)"
echo ""
echo "=========================================="
echo "Now upload a PDF in the UI and watch below:"
echo "=========================================="
echo ""

# Watch both DB changes and Streamlit logs
watch -n 1 'echo "=== ChromaDB Files ===" && ls -lah chroma_db/ 2>/dev/null | tail -10 && echo "" && echo "DB Size: $(du -sh chroma_db/ 2>/dev/null | cut -f1)" && echo "" && echo "=== Recent Streamlit Logs ===" && tail -15 /private/tmp/claude-501/-Users-tusharchauhan-Desktop-Projects-agentic-rag/tasks/b3f393e.output 2>/dev/null | grep -v "^$"'
