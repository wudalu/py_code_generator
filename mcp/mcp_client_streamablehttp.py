import asyncio
import json
import logging
import traceback
import random
from typing import Dict, Any, Callable, Optional, Awaitable, Union, List
import aiohttp # Keep for potential external session passing

# Import the transport layer
from .transports.streamable_http_client_transport import StreamableHttpClientTransport

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Specific logger name

# MCP Constants
MCP_SESSION_ID_HEADER = "Mcp-Session-Id"
JSONRPC_VERSION = "2.0"
ACCEPT_HEADER_VALUE = "application/json, text/event-stream"
SSE_ACCEPT_HEADER_VALUE = "text/event-stream"

# Default retry settings for SSE connection
SSE_RETRY_DELAY = 3 # seconds
SSE_MAX_RETRIES = 5 # Number of retries before giving up

class StreamableHttpClient:
    """
    Implements the application logic layer for an MCP client using Streamable HTTP.
    Uses StreamableHttpClientTransport for handling network communication.
    Provides methods for initialization, sending messages, and closing the connection.
    Manages pending requests to match responses if needed (optional, using Futures).
    Includes specific business methods mirroring CodeEmbeddingClient.
    """

    def __init__(self, base_url: str,
                 message_handler: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
                 client_session: Optional[aiohttp.ClientSession] = None):
        """
        Initializes the client application layer.

        Args:
            base_url: The base URL of the MCP server (e.g., "http://localhost:8080").
            message_handler: An optional async function to call for *all* messages received
                             from the server (responses, notifications, server requests).
                             If not provided, responses to requests sent via `send_request`
                             will complete the Future returned by that method.
            client_session: Optional existing aiohttp.ClientSession to pass to the transport.
                           If provided, the caller is responsible for its lifecycle.
        """
        self._message_handler = message_handler
        self._pending_requests: Dict[Union[str, int], asyncio.Future] = {} # id -> Future
        # Initialize the transport, passing our internal message dispatcher
        self.transport = StreamableHttpClientTransport(
            base_url=base_url,
            on_server_message=self._dispatch_server_message,
            client_session=client_session
        )

    async def _dispatch_server_message(self, message: Dict[str, Any]):
        """Internal dispatcher called by the transport layer for every server message."""
        logger.debug(f"Dispatching server message: {message}")
        msg_id = message.get("id")

        # Check if it's a response to a pending request we track
        if msg_id is not None and msg_id in self._pending_requests:
            future = self._pending_requests.pop(msg_id)
            if not future.done(): # Avoid setting result/exception on already done future
                if "error" in message:
                    # Create a more specific error if possible
                    error_obj = message['error']
                    exception = RuntimeError(f"Server Error Code {error_obj.get('code', 'N/A')}: {error_obj.get('message', 'Unknown error')}")
                    future.set_exception(exception)
                else:
                    future.set_result(message.get("result")) # Resolve future with the result part
            else:
                 logger.warning(f"Future for request ID {msg_id} was already done when response arrived.")

            # If a general message handler also exists, call it *after* resolving the future
            if self._message_handler:
                 try:
                     # Schedule the handler call instead of awaiting directly in dispatcher
                     asyncio.create_task(self._safe_call_message_handler(message))
                 except Exception as e:
                     logger.error(f"Error scheduling external message_handler for response {msg_id}: {e}", exc_info=True)

        # If it's not a response we track, or we have a general handler, call the handler
        elif self._message_handler:
            try:
                 # Schedule the handler call
                 asyncio.create_task(self._safe_call_message_handler(message))
            except Exception as e:
                 logger.error(f"Error scheduling external message_handler: {e}", exc_info=True)
        # If no handler and not a tracked response, log it
        elif msg_id is None:
             logger.debug(f"Received untracked server message (notification/server request) without handler: {message}")
        else: # Received a response for a request not sent via send_request or already handled
             logger.warning(f"Received unexpected/untracked response for id {msg_id}: {message}")

    async def _safe_call_message_handler(self, message: Dict[str, Any]):
         """Safely calls the external message handler in a new task."""
         try:
             await self._message_handler(message)
         except Exception as e:
             logger.error(f"Error executing external message_handler: {e}", exc_info=True)

    @property
    def is_connected(self) -> bool:
        """Checks if the underlying transport is connected."""
        return self.transport.is_connected

    @property
    def session_id(self) -> Optional[str]:
         """Gets the current session ID from the transport."""
         return self.transport.session_id

    async def initialize(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Connects and initializes the session using the transport layer.

        Args:
            params: Initialization parameters.

        Returns:
            The capabilities dictionary from the server's InitializeResult.
        """
        logger.info("Initializing client connection...")
        try:
            init_result = await self.transport.connect(init_params=params)
            logger.info("Client connection initialized successfully.")
            return init_result
        except Exception as e:
            logger.error(f"Client initialization failed: {e}")
            raise # Re-raise the exception

    async def send_notification(self, method: str, params: Optional[Union[Dict, List]] = None):
        """
        Sends a JSON-RPC notification (message without 'id') to the server.

        Args:
            method: The method name.
            params: The parameters (optional).
        """
        logger.debug(f"Sending notification: method={method}, params={params}")
        message = {
            "jsonrpc": JSONRPC_VERSION,
            "method": method,
        }
        if params is not None:
            message["params"] = params
        # Use transport's send method directly
        await self.transport.send(message)

    async def send_request(self, method: str, params: Optional[Union[Dict, List]] = None, request_id: Optional[Union[str, int]] = None) -> Any:
        """
        Sends a JSON-RPC request (message with 'id') to the server and returns the result.
        This method uses an internal future to wait for the response.

        Args:
            method: The method name.
            params: The parameters (optional).
            request_id: Optional request ID. If None, a random integer ID is generated.

        Returns:
            The result part of the JSON-RPC response.

        Raises:
            RuntimeError: If the server returns a JSON-RPC error response.
            ConnectionError: If sending fails at the transport level.
            TimeoutError: If the request times out waiting for a response.
        """
        if request_id is None:
            # Generate a unique enough ID (simple approach)
            request_id = f"req-{random.randint(10000, 99999)}-{int(asyncio.get_running_loop().time())}"

        logger.debug(f"Sending request: method={method}, id={request_id}, params={params}")
        message = {
            "jsonrpc": JSONRPC_VERSION,
            "method": method,
            "id": request_id
        }
        if params is not None:
            message["params"] = params

        # Create a future to wait for the response
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_requests[request_id] = future

        try:
            await self.transport.send(message)
            # Wait for the future to be resolved by the dispatcher
            # Add a timeout to prevent waiting indefinitely
            # TODO: Make timeout configurable (e.g., via __init__)
            return await asyncio.wait_for(future, timeout=60.0)
        except asyncio.TimeoutError as e:
             logger.error(f"Request {request_id} ({method}) timed out after 60s.")
             # Ensure future is cleaned up on timeout
             self._pending_requests.pop(request_id, None)
             # Set exception on the future? Or just raise? Raise is cleaner for caller.
             # future.set_exception(e) # Avoid this if raising
             raise e
        except Exception as e:
            # If sending failed or another error occurred, remove the pending future
            self._pending_requests.pop(request_id, None)
            logger.error(f"Failed request {request_id} ({method}): {e}")
            # Don't wrap connection errors, raise them directly
            raise e
        # No finally block needed to pop, as it's popped in dispatcher or exception handlers

    # --- Business Methods (Mirroring CodeEmbeddingClient) ---

    async def process_files(self, file_paths: List[str], collection_name: str = "code_collection") -> Dict[str, Any]:
        """
        Process code files: load, split, embed, store.

        Args:
            file_paths: List of code file paths.
            collection_name: Collection name, defaults to code_collection.

        Returns:
            Processing result dictionary.
        """
        logger.info(f"Calling tool: process_files with {len(file_paths)} files for collection '{collection_name}'.")
        params = {
            "file_paths": file_paths,
            "collection_name": collection_name
        }
        try:
            result = await self.send_request("process_files", params)
            # Assuming server returns a dictionary directly
            return result if isinstance(result, dict) else {"status": "error", "message": f"Unexpected result type: {type(result)}"}
        except Exception as e:
            logger.error(f"Error calling process_files tool: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    async def search_code(self, query: str, k: int = 5, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant code snippets.

        Args:
            query: The search query string.
            k: Number of results to return, defaults to 5.
            collection_name: Optional collection name.

        Returns:
            List of search result dictionaries.
        """
        logger.info(f"Calling tool: search_code with query '{query[:50]}...', k={k}, collection='{collection_name}'.")
        params: Dict[str, Any] = {
            "query": query,
            "k": k
        }
        if collection_name:
            params["collection_name"] = collection_name

        try:
            result = await self.send_request("search_code", params)
             # Assuming server returns a list of dictionaries directly
            if isinstance(result, list):
                 return result
            else:
                 logger.error(f"Unexpected result type for search_code: {type(result)}. Expected list.")
                 return []
        except Exception as e:
            logger.error(f"Error calling search_code tool: {e}", exc_info=True)
            return []

    async def load_collection(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load an existing vector store collection.

        Args:
            collection_name: Optional collection name.

        Returns:
            Loading result dictionary.
        """
        logger.info(f"Calling tool: load_collection for collection '{collection_name}'.")
        params = {}
        if collection_name:
            params["collection_name"] = collection_name

        try:
            result = await self.send_request("load_collection", params)
            return result if isinstance(result, dict) else {"status": "error", "message": f"Unexpected result type: {type(result)}"}
        except Exception as e:
            logger.error(f"Error calling load_collection tool: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    async def list_collections(self) -> Dict[str, List[str]]:
        """
        List all available collections (simulated via a tool call for now).
        Note: MCP resource reading is not directly mapped here yet.
              We assume a 'list_collections' tool exists on the server.
        """
        logger.info("Calling tool: list_collections")
        try:
            # This assumes a tool named 'list_collections' exists on the server.
            # A more accurate MCP implementation might use resource reading if the server supports it.
            result = await self.send_request("list_collections")
            if isinstance(result, dict) and "collections" in result and isinstance(result["collections"], list):
                 return result
            else:
                 logger.error(f"Unexpected result format for list_collections: {result}")
                 return {"collections": [], "error": "Invalid response format"}
        except Exception as e:
            logger.error(f"Error calling list_collections tool: {e}", exc_info=True)
            return {"collections": [], "error": str(e)}

    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a specific collection (simulated via a tool call).
        Note: MCP resource reading is not directly mapped here yet.
              We assume a 'get_collection_info' tool exists on the server.
        """
        logger.info(f"Calling tool: get_collection_info for '{collection_name}'.")
        if not collection_name:
             return {"error": "Collection name cannot be empty"}
        try:
             # Assumes a tool named 'get_collection_info' exists.
             result = await self.send_request("get_collection_info", {"collection_name": collection_name})
             return result if isinstance(result, dict) else {"name": collection_name, "error": f"Unexpected result type: {type(result)}"}
        except Exception as e:
             logger.error(f"Error calling get_collection_info tool: {e}", exc_info=True)
             return {"name": collection_name, "error": str(e)}

    async def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """Calls the delete_collection tool on the server."""
        if not self.is_connected:
            raise RuntimeError("Client is not connected.")
        params = {"collection_name": collection_name}
        logger.info(f"Calling tool: delete_collection for '{collection_name}'.")
        try:
            result = await self.send_request("delete_collection", params)
            return result
        except Exception as e:
            logger.error(f"Error calling delete_collection tool: {e}", exc_info=True)
            # Return a consistent error format
            return {"status": "error", "message": str(e)}

    # Note: get_code_search_prompt maps to MCP prompt concept, which isn't directly
    #       implemented via tool calls in this basic structure.
    #       If the server exposes prompt generation as a tool, it could be added here.
    # async def get_code_search_prompt(self, query: str) -> str: ...

    # --- Lifecycle Methods ---

    async def close(self, send_delete: bool = True):
        """
        Disconnects the client using the transport layer.

        Args:
            send_delete: Whether to ask the transport to send a DELETE request
                         to terminate the session on the server.
        """
        logger.info("Closing client connection...")
        # Cancel any pending request futures
        pending_ids = list(self._pending_requests.keys())
        if pending_ids:
             logger.info(f"Cancelling {len(pending_ids)} pending request(s)...")
             for req_id in pending_ids:
                 future = self._pending_requests.pop(req_id, None)
                 if future and not future.done():
                     future.cancel(f"Client closing connection while request {req_id} was pending.")
        self._pending_requests.clear()

        # Delegate to transport for actual disconnection
        await self.transport.disconnect(send_delete=send_delete)
        logger.info("Client connection closed.")

# --- Example Usage (Updated) --- #

# Define the application-level message handler (optional)
async def my_app_message_handler(message: Dict[str, Any]):
    # This handler receives *all* messages: responses, notifications, server requests
    logger.info(f"APP_HANDLER received: {json.dumps(message)}")
    if "method" in message and message["method"] == "$/serverNotification":
        logger.info(f"--> APP_HANDLER: Got a server notification: {message.get('params')}")
    # Note: Responses handled by send_request futures won't typically reach here unless
    #       the message handler is explicitly called by the dispatcher for them too.
    #       (Current dispatcher calls handler AFTER resolving future).

async def main():
    server_url = "http://localhost:8080" # Ensure the refactored server is running
    client = StreamableHttpClient(server_url, my_app_message_handler) # With general handler

    try:
        # Initialize
        init_result = await client.initialize(params={"client_info": "refactored_client_v1"})
        logger.info(f"APP: Initialization result: {init_result}")
        logger.info(f"APP: Connected with Session ID: {client.session_id}")

        if not client.is_connected:
             logger.error("APP: Failed to connect!")
             return

        # List initial collections
        collections_info = await client.list_collections()
        logger.info(f"APP: Initial collections: {collections_info}")

        # Process some files (assuming server has access or paths are relative/absolute)
        # Create dummy files for example if needed
        dummy_files = []
        # try:
        #      with open("dummy_code1.py", "w") as f: f.write("print('hello')\\ndef func1(): pass")
        #      with open("dummy_code2.py", "w") as f: f.write("import os\\n# Example")
        #      dummy_files = ["dummy_code1.py", "dummy_code2.py"]
        #      process_result = await client.process_files(dummy_files, collection_name="my_test_coll")
        #      logger.info(f"APP: Process files result: {process_result}")
        # except Exception as e:
        #      logger.error(f"APP: Error processing dummy files: {e}")
        # finally:
        #      import os
        #      for f in dummy_files:
        #          if os.path.exists(f): os.remove(f)

        # Search for code
        search_results = await client.search_code(query="print hello world", k=2, collection_name="my_test_coll")
        logger.info(f"APP: Search results for 'print hello world':")
        for res in search_results:
            logger.info(f"  - {res.get('content', '')[:80]}... (Metadata: {res.get('metadata')})")

        # Get info for a collection
        info = await client.get_collection_info("my_test_coll")
        logger.info(f"APP: Info for 'my_test_coll': {info}")

        # Send a notification
        await client.send_notification(
            method="$/clientNotification",
            params={"status": "Example finished!"}
        )

        # Keep client running briefly to potentially receive server notifications
        logger.info("APP: Example done. Waiting a few seconds for any server messages...")
        await asyncio.sleep(5)

    except ConnectionError as e:
        logger.error(f"APP: Connection failed: {e}")
    except RuntimeError as e:
         logger.error(f"APP: Runtime error (possibly server error): {e}")
    except asyncio.TimeoutError as e:
         logger.error(f"APP: Request timed out: {e}")
    except Exception as e:
        logger.error(f"APP: An unexpected error occurred: {e}", exc_info=True)
    finally:
        if client.is_connected:
            await client.close()
        else:
             logger.info("APP: Client was not connected or already closed.")

if __name__ == "__main__":
    # Configure logging for the example
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Silence noisy libraries in debug mode if needed
    logging.getLogger("aiohttp").setLevel(logging.INFO) # Less verbose aiohttp
    logging.getLogger("aiohttp_sse_client").setLevel(logging.INFO)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("APP: Client example stopped by user.") 