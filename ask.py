#!/usr/bin/env python
"""
ask.py - Simple script to query the RAG Enterprise Brain
Usage: python ask.py "Your question here"
"""

import requests
import sys
import json

API_URL = "http://localhost:8000/query"

def ask(question, top_k=5):
    """Send a question to the RAG system"""
    response = requests.post(
        API_URL,
        json={"query": question, "top_k": top_k}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n{'='*60}")
        print(f"📝 Question: {data['query']}")
        print(f"{'='*60}")
        print(f"⏱️  Processing time: {data['processing_ms']}ms")
        print(f"📊 Chunks searched: {data['total_chunks_searched']}")
        print(f"🔄 RAKE used: {'Yes' if data['rake_used'] else 'No'}")
        print(f"\n✅ Found {data['total_answers']} answer(s):\n")
        
        for i, answer in enumerate(data['answers'], 1):
            print(f"{'─'*50}")
            print(f"Answer {i}:")
            print(f"  📖 Text: {answer['text']}")
            print(f"  📄 Source: {answer['filename']} (Page {answer['page_number']})")
            print(f"  🎯 Score: {answer['span_score']:.4f}")
            
            # Show context if answer is short
            if len(answer['text']) < 100 and len(answer.get('chunk_text', '')) > 0:
                print(f"\n  📚 Full context:")
                context = answer['chunk_text'][:500]
                print(f"     {context}...")
            
            if answer.get('bbox'):
                bbox = answer['bbox']
                print(f"  📍 Position: ({bbox['x0']:.3f}, {bbox['y0']:.3f}) to ({bbox['x1']:.3f}, {bbox['y1']:.3f})")
            print()
            
    elif response.status_code == 404:
        print(f"\n❌ No answer found for: {question}")
        print("   Try rephrasing your question or checking your documents.")
    else:
        print(f"\n❌ Error: {response.status_code}")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text)

def interactive_mode():
    """Run in interactive mode"""
    print("="*60)
    print("RAG Enterprise Brain - Interactive Q&A")
    print("="*60)
    print("Commands: 'quit' to exit, 'help' for tips")
    print()
    
    while True:
        question = input("\n❓ You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        elif question.lower() == 'help':
            print("\n💡 Tips for better answers:")
            print("  • Be specific (e.g., 'What data is needed?' not 'Tell me about data')")
            print("  • Use keywords from your documents")
            print("  • Ask one question at a time")
            print("  • Try rephrasing if no answer found")
            continue
        elif not question:
            continue
        
        ask(question)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # No command line argument, run interactive mode
        interactive_mode()
    else:
        question = " ".join(sys.argv[1:])
        ask(question)