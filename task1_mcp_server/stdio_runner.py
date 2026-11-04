"""
Task 1: Standalone STDIO Runner with Protocol Isolation
Reads JSON-RPC lines from sys.stdin, passes to DirectMCPProtocolHandler,
and writes replies strictly to sys.stdout while flushing.
All diagnostic messages go exclusively to sys.stderr.
"""

import sys
import asyncio
import logging

from task1_mcp_server.server import create_customer_mcp_server, DirectMCPProtocolHandler

logger = logging.getLogger("mcp_runner")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("[RUNNER] %(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


async def main():
    logger.info("Starting STDIO Protocol Loop. STDOUT is locked for JSON-RPC only.")
    server = create_customer_mcp_server()
    protocol_handler = DirectMCPProtocolHandler(server)

    loop = asyncio.get_running_loop()

    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            logger.info("STDIN EOF reached. Exiting runner loop.")
            break

        response = await protocol_handler.handle_message(line)
        if response:
            # Write strictly to stdout and flush immediately
            sys.stdout.write(response + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Exiting cleanly.")
