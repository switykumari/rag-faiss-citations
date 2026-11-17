"""Ask questions about the documents in docs/."""
import sys

from rag import TOP_K, RagIndex, answer


def ask(index, question):
    retrieved = index.search(question, TOP_K)
    print("\nRetrieved chunks:")
    for i, c in enumerate(retrieved, start=1):
        print(f"  [{i}] {c['source']} (page {c['page']})  score={c['score']:.2f}")
    print("\nANSWER:\n" + answer(question, retrieved))


def main():
    print("Building index (the first run downloads the embedding model)...")
    index = RagIndex()
    print(f"Indexed {len(index.chunks)} chunks.\n")

    if len(sys.argv) > 1:
        ask(index, " ".join(sys.argv[1:]))
        return

    while True:
        question = input("\nAsk a question (or press Enter to quit): ").strip()
        if not question:
            break
        ask(index, question)


if __name__ == "__main__":
    main()
