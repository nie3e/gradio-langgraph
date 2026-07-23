import os
from gradio_langgraph.frontend import interface

SERVER_NAME = os.getenv("SERVER_NAME", None)
AUTH_USER = os.getenv("AUTH_USER", None)
AUTH_PASS = os.getenv("AUTH_PASS", None)


def main() -> int:
    """Entry point function for the web app."""
    demo = interface.create_app()
    auth = None
    if AUTH_USER:
        auth = [AUTH_USER, AUTH_PASS]
    demo.queue(max_size=10).launch(server_name=SERVER_NAME, auth=auth)

    return 0


if __name__ == "__main__":
    exit(main())
