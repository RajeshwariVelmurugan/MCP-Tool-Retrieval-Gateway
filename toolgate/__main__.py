"""python -m toolgate <command>

  serve      start the dashboard + API            (python -m toolgate serve --port 8000)
  eval       run the offline evaluation in the terminal
  collect    collect real tools from servers.json into data/tools.json
  gateway    run the MCP gateway (stdio by default; --http for streamable HTTP)
  e2e        end-to-end LLM evaluation incl. native tool search (needs ANTHROPIC_API_KEY)
"""
import sys


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    rest = sys.argv[2:]
    if cmd == "serve":
        import argparse

        from .server import main as serve

        ap = argparse.ArgumentParser()
        ap.add_argument("--host")
        ap.add_argument("--port", type=int)
        a = ap.parse_args(rest)
        serve(a.host, a.port)
    elif cmd == "eval":
        from .evaluation.runner import main as run

        run(rest)
    elif cmd == "collect":
        from .collect import main as run

        raise SystemExit(run(rest))
    elif cmd == "gateway":
        from .gateway import main as run

        run(rest)
    elif cmd == "e2e":
        from .evaluation.e2e import main as run

        raise SystemExit(run(rest))
    else:
        print(__doc__)
        raise SystemExit(0 if cmd in ("-h", "--help", "help") else 2)


if __name__ == "__main__":
    main()
