from workflow.graph import build_graph
from rag.vector_store import build_vector_store

def main():
    build_vector_store()
    while True:
        query = input("Ask Question: ")
        if query.lower()=="exit":
            break

        else:
            app = build_graph()


            result = app.invoke({
                "query": query,
                "retrieved_context": None,
                "analysis": None,
                "draft_document": None,
                "reviewed_document": None,
            })

            print("Final Legal Document:")
            print(result["reviewed_document"])


if __name__ == "__main__":
    main()