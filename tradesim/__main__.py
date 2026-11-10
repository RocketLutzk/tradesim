import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "backtest":
        from .backtest import main as run
    elif cmd == "paper":
        from .paper import main as run
    else:
        print("Використання: python -m tradesim [backtest|paper]")
        sys.exit(1)
    run()


if __name__ == "__main__":
    main()
