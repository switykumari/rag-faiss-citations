"""Compare keyword search vs embedding search: is the right chunk in the top-k?"""
from rag import RagIndex, keyword_search

# (question, file that holds the answer, phrase that must be in the retrieved chunk)
TESTS = [
    # Questions that reuse words from the documents
    ("How many days of annual leave do I get?", "leave_policy.md", "24"),
    ("What should I do when the VPN shows error 809?", "vpn_setup.md", "809"),
    ("What is the daily limit for food expenses on trips?", "expense_policy.md", "1500"),
    ("By when must I file a claim for my receipts?", "expense_policy.md", "30 days"),
    ("Where do I ask for help with remote access problems?", "vpn_setup.md", "#it-help"),
    ("When do I need a medical certificate for being unwell?", "leave_policy.md", "doctor"),
    # Paraphrased questions that share few words with the documents
    ("Am I allowed to bank my holiday balance for later?", "leave_policy.md", "carried forward"),
    ("What proof do I show after a short illness?", "leave_policy.md", "doctor"),
    ("How many weeks off after a baby arrives?", "leave_policy.md", "16 weeks"),
    ("The tunnel will not establish because my router blocks it", "vpn_setup.md", "809"),
    ("Which app approves the second step when I sign in remotely?", "vpn_setup.md", "authenticator"),
    ("What is the spending cap for lunch and dinner when travelling?", "expense_policy.md", "1500"),
    ("How soon will I get my money back after filing?", "expense_policy.md", "10 working days"),
    ("Is a senior manager sign-off needed for big purchases?", "expense_policy.md", "director"),
    ("What seat class applies on short flights?", "expense_policy.md", "economy"),
    ("Until which date do extra holiday days stay valid?", "leave_policy.md", "31 March"),
]


def is_hit(results, source, phrase):
    return any(
        r["source"] == source and phrase.lower() in r["text"].lower() for r in results
    )


def main():
    index = RagIndex()
    print(f"Chunks indexed: {len(index.chunks)}")
    for k in (1, 3):
        kw_hits = emb_hits = 0
        print(f"\n=== top-{k} ===")
        print(f"{'Question':<62} {'Keyword':<8} {'Embedding'}")
        print("-" * 82)
        for question, source, phrase in TESTS:
            kw = is_hit(keyword_search(index.chunks, question, k), source, phrase)
            emb = is_hit(index.search(question, k), source, phrase)
            kw_hits += kw
            emb_hits += emb
            print(f"{question[:60]:<62} {'HIT' if kw else 'miss':<8} {'HIT' if emb else 'miss'}")
        n = len(TESTS)
        print("-" * 82)
        print(f"Keyword hit rate:   {kw_hits}/{n} = {kw_hits / n:.0%}")
        print(f"Embedding hit rate: {emb_hits}/{n} = {emb_hits / n:.0%}")


if __name__ == "__main__":
    main()
