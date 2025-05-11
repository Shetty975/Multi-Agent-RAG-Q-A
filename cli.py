# cli.py
from tools import agent_router

def main():
    print("Multi-Agent RAG CLI Assistant (type 'exit' to quit)\n")
    while True:
        query = input(">> ")
        if query.lower() == 'exit':
            break
        response = agent_router(query)
        print(f"\n[Agent Branch]: {response['route']}")
        if 'context' in response:
            print("[Top Chunks]:")
            for c in response['context']:
                print(f"- {c.strip()[:100]}...")
        print(f"[Answer]: {response['result']}\n")

if __name__ == "__main__":
    main()
