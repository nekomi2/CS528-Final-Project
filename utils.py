import signal


def graceful_exit(signum, frame):
    print("\nShutting down gracefully...")
    raise KeyboardInterrupt
