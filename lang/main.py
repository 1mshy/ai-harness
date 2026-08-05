import sys

from graph import BASE_URL, build_graph, discover_model


def main() -> None:
    topic = " ".join(sys.argv[1:]) or "Why is the sky blue?"
    print(f"server: {BASE_URL}")
    print(f"model:  {discover_model()}")
    print(f"topic:  {topic}\n")

    graph = build_graph()
    state = {"topic": topic, "outline": "", "draft": "", "critique": "", "approved": False, "revision": 0}

    for update in graph.stream(state, stream_mode="updates"):
        for node, out in update.items():
            state.update(out)
            if node == "plan":
                print(f"--- plan ---\n{out['outline']}\n")
            elif node == "write":
                print(f"--- draft (revision {out['revision']}) ---\n{out['draft']}\n")
            elif node == "review":
                verdict = "approved" if out["approved"] else "needs revision"
                print(f"--- review: {verdict} ---\n{out['critique']}\n")

    print("=== final draft ===")
    print(state["draft"])


if __name__ == "__main__":
    main()

docker run -d \
  --name unitroni_lv_chip2unification \
  -e MYSQL_DATABASE=unitroni_lv_chip2unification \
  -e MYSQL_ALLOW_EMPTY_PASSWORD=true \
  -p 3308:3306 \
  -v unitroni_lv_chip2unification-data:/var/lib/mysql \
  mysql:8.0
