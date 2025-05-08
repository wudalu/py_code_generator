import asyncio
import json
import logging
from typing import Dict, Any, Callable, Optional, Set, Awaitable, List

# Import the transport layer
from .transports.streamable_http_server_transport import StreamableHttpServerTransport

# Import necessary components from embedding module
from embedding import EmbeddingPipeline, get_embedding_model
from embedding.storage.chroma import ChromaVectorStore # Assuming Chroma is used
from embedding.splitter.ast_python import AstPythonSplitter
from embedding.splitter.fallback import FallbackSplitter

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use specific logger name

# MCP Constants (could be moved to a shared constants file)
MCP_SESSION_ID_HEADER = "Mcp-Session-Id"
JSONRPC_VERSION = "2.0"

class StreamableHttpServer:
    """
    Implements the application logic layer for an MCP server using Streamable HTTP.
    Uses StreamableHttpServerTransport for handling network communication.
    Focuses on tool registration and processing JSON-RPC messages.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self._tools: Dict[str, Callable[..., Awaitable[Any]]] = {} # method_name -> async function
        # Initialize the transport layer, passing our message processor method
        self.transport = StreamableHttpServerTransport(
            message_processor=self._process_message, # Pass the bound method
            host=host,
            port=port
        )
        # Placeholder for the initialized pipeline
        self.embedding_pipeline: Optional[EmbeddingPipeline] = None
        # Keep a direct reference to the store if needed for methods not in pipeline
        self._vector_store: Optional[ChromaVectorStore] = None

    # Method to initialize embedding components and register tools
    async def initialize_embedding_pipeline(self, persist_directory="chroma_db_store"):
        """Initializes the embedding pipeline and registers its methods as tools."""
        logger.info("Initializing embedding components...")
        try:
            # 1. Load Embedding Model
            # Model configuration is now read from the imported 'config' module by get_embedding_model
            logger.info("Loading embedding model via get_embedding_model() using configuration...")
            embedding_model = get_embedding_model()

            # 2. Initialize Vector Store
            # TODO: Get persist directory from config
            logger.info(f"Initializing Chroma vector store at: {persist_directory}")
            self._vector_store = ChromaVectorStore(persist_directory=persist_directory, embedding_function=embedding_model)

            # 3. Initialize Splitters
            splitters = {
                "python": AstPythonSplitter(),
                "fallback": FallbackSplitter()
            }

            # 4. Create Embedding Pipeline instance
            self.embedding_pipeline = EmbeddingPipeline(
                embedding_model=embedding_model,
                vector_store=self._vector_store,
                splitters=splitters
            )
            logger.info("EmbeddingPipeline initialized successfully.")

            # 5. Register tools from the pipeline (and potentially the store)
            # self.register_tool(self.embedding_pipeline.process_files) # Registering sync method directly is problematic
            # Register the async wrapper instead
            self.register_tool(self.process_files_tool, name="process_files")
            # Add wrappers for methods likely on the vector store, explicitly naming them
            self.register_tool(self.search_code_tool, name="search_code")
            self.register_tool(self.load_collection_tool, name="load_collection")
            self.register_tool(self.list_collections_tool, name="list_collections")
            self.register_tool(self.get_collection_info_tool, name="get_collection_info")
            # Register the new delete tool
            self.register_tool(self.delete_collection_tool, name="delete_collection")

            logger.info("Embedding tools registered.")

        except Exception as e:
            logger.exception("Failed to initialize embedding pipeline.")
            raise RuntimeError("Embedding pipeline initialization failed") from e

    # --- Async Wrapper for process_files --- 
    async def process_files_tool(self, file_paths: List[str], collection_name: str = "code_collection") -> Dict[str, Any]:
        """MCP Tool async wrapper for synchronous EmbeddingPipeline.process_files."""
        if not self.embedding_pipeline:
            raise RuntimeError("Embedding pipeline not initialized.")
        logger.info(f"Tool: process_files - files={len(file_paths)}, collection={collection_name}")
        try:
            # Run the synchronous process_files method in a thread
            result = await asyncio.to_thread(
                self.embedding_pipeline.process_files,
                file_paths,
                collection_name
            )
            return result
        except Exception as e:
            logger.error(f"Error during process_files_tool: {e}", exc_info=True)
            # Mimic error structure expected by client
            return {"status": "error", "message": str(e)}

    # --- Wrapper methods for tools potentially living on VectorStore --- 
    # These assume the methods exist on the self._vector_store object
    # Adjust based on actual VectorStore interface in embedding/storage/interface.py

    async def search_code_tool(self, query: str, k: int = 5, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """MCP Tool wrapper for vector store search."""
        if not self._vector_store:
            raise RuntimeError("Vector store not initialized.")
        logger.info(f"Tool: search_code - query='{query[:30]}...', k={k}, collection={collection_name}")
        # Assuming vector_store has a search method compatible with these args
        # The actual search might not be async, wrap if necessary
        try:
            # Chroma search methods might be synchronous
            results = await asyncio.to_thread(self._vector_store.search, query, k, collection_name)
            # Format results if needed, Chroma might return Document objects
            return results # Adapt based on actual return type
        except Exception as e:
            logger.error(f"Error during search_code_tool: {e}", exc_info=True)
            return [] # Return empty list on error

    async def load_collection_tool(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """MCP Tool wrapper for loading a collection."""
        if not self._vector_store:
            raise RuntimeError("Vector store not initialized.")
        logger.info(f"Tool: load_collection - collection={collection_name}")
        # Chroma usually loads implicitly on init or first access.
        # This might just confirm the collection exists or set a default.
        try:
            # Replace with actual method if available, e.g., self._vector_store.set_collection(collection_name)
            # Simulate success for now if no explicit load needed
            exists = await asyncio.to_thread(self._vector_store.collection_exists, collection_name)
            if exists:
                return {"status": "success", "message": f"Collection '{collection_name}' confirmed/loaded."}
            else:
                return {"status": "error", "message": f"Collection '{collection_name}' does not exist."}
        except Exception as e:
            logger.error(f"Error during load_collection_tool: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    async def list_collections_tool(self) -> Dict[str, List[str]]:
        """MCP Tool wrapper for listing collections."""
        if not self._vector_store:
            raise RuntimeError("Vector store not initialized.")
        logger.info(f"Tool: list_collections")
        try:
            # Use the correct method name: get_collections()
            collection_names = await asyncio.to_thread(self._vector_store.get_collections)
            return {"collections": collection_names}
        except Exception as e:
            logger.error(f"Error during list_collections_tool: {e}", exc_info=True)
            # Return error in the expected format
            return {"collections": [], "error": str(e)}

    async def get_collection_info_tool(self, collection_name: str) -> Dict[str, Any]:
        """MCP Tool wrapper for getting collection info."""
        if not self._vector_store:
            raise RuntimeError("Vector store not initialized.")
        logger.info(f"Tool: get_collection_info - collection={collection_name}")
        try:
            # Assuming vector_store has a method like get_collection_info()
            info = await asyncio.to_thread(self._vector_store.get_collection_info, collection_name)
            return info
        except Exception as e:
            logger.error(f"Error during get_collection_info_tool: {e}", exc_info=True)
            return {"name": collection_name, "error": str(e)}

    async def delete_collection_tool(self, collection_name: str) -> Dict[str, Any]:
        """MCP Tool wrapper for deleting a collection."""
        if not self._vector_store:
            raise RuntimeError("Vector store not initialized.")
        logger.info(f"Tool: delete_collection - collection={collection_name}")
        try:
            await asyncio.to_thread(self._vector_store.delete_collection, collection_name)
            return {"status": "success", "message": f"Collection '{collection_name}' deleted successfully."}
        except Exception as e:
            logger.error(f"Error during delete_collection_tool: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    # --- End Wrapper Methods --- 

    def register_tool(self, func: Callable[..., Awaitable[Any]], name: Optional[str] = None):
        """Registers an async function as an available tool."""
        tool_name = name or func.__name__
        # Ensure we register the bound methods correctly if they come from an instance
        if hasattr(func, '__self__'):
            logger.debug(f"Registering bound method: {tool_name}")
        else:
            logger.debug(f"Registering function: {tool_name}")

        if not asyncio.iscoroutinefunction(func):
            # If it's a synchronous function (like potentially from ChromaDB client),
            # we might need to wrap it. However, our wrappers above use asyncio.to_thread.
            # Let's enforce async def for direct registration for now.
            # For simplicity, assume wrappers handle sync methods.
            # Remove process_files from this exclusion list as we now register an async wrapper
            # if tool_name not in ["search_code_tool", "load_collection_tool", "list_collections_tool", "get_collection_info_tool"]:
            # Allow wrappers (like search_code_tool) even if underlying func isn't async.
            # If a function isn't explicitly wrapped (like example_tool), it must be async.
            if not tool_name.endswith("_tool"): # Heuristic: assume wrappers end with _tool
                raise TypeError(f"Tool function '{tool_name}' must be an async function (defined with 'async def') or have an async wrapper registered.")

        if tool_name in self._tools:
            logger.warning(f"Tool '{tool_name}' is already registered. Overwriting.")
        # Use the actual function object as the value
        self._tools[tool_name] = func
        logger.info(f"Registered tool: {tool_name}")

    async def _process_message(self, session_id: Optional[str], msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Callback method passed to the transport layer to process incoming messages.
        Handles tool dispatching based on the 'method' field.
        Args:
            session_id: The session ID associated with the message (can be None for initialize).
            msg: The received JSON-RPC message dictionary.
        Returns:
            A JSON-RPC response dictionary if the message was a request, or None for notifications.
            Raises exceptions which the transport layer should catch and format as JSON-RPC errors.
        """
        logger.debug(f"Processing message for session {session_id}: {msg}")
        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params")

        if method == "initialize":
            # Application-specific initialization logic can go here if needed
            # Example: Return capabilities based on registered tools
            # Ensure pipeline is initialized before responding to initialize?
            if not self.embedding_pipeline:
                logger.warning("Embedding pipeline not initialized during initialize request. Attempting init now.")
                try:
                    # Run sync initialization in executor to avoid blocking event loop
                    await asyncio.to_thread(self.initialize_embedding_pipeline)
                except Exception as e:
                    logger.error(f"Failed to initialize embedding pipeline during request: {e}")
                    raise RuntimeError("Server setup error during initialization") from e

            capabilities = {
                "registered_tools": list(self._tools.keys()),
                # Add other server capabilities here
                "search_k_default": 5, # Example capability
                "default_collection": "code_collection" # Example
            }
            logger.info(f"Responding to initialize for new session {session_id}")
            return {"jsonrpc": JSONRPC_VERSION, "result": {"capabilities": capabilities}, "id": msg_id}

        # Check if pipeline/tools are ready for other methods
        if not self.embedding_pipeline:
            logger.error(f"Cannot process method '{method}': Embedding pipeline not initialized.")
            # Raise error only if it was a request
            if msg_id is not None:
                raise RuntimeError("Server error: Embedding system not ready.")
            else:
                return None # Ignore notification if not ready

        # --- Other standard MCP methods (shutdown, etc.) could be handled here --- 
        elif method == "$/cancelRequest":
            # Application logic for cancellation
            logger.info(f"Received cancel request notification: {params}")
            # Notifications do not return a response object
            return None

        elif method and method in self._tools:
            tool_func = self._tools[method]
            # Map tool names if needed (e.g., if registered wrapper has different name)
            actual_tool_name = tool_func.__name__ # Get name of the function/method itself
            logger.info(f"Dispatching method '{method}' to tool '{actual_tool_name}' for session {session_id}")
            try:
                # Execute the registered async tool function/wrapper
                if isinstance(params, dict):
                    result = await tool_func(**params)
                elif isinstance(params, list):
                    result = await tool_func(*params)
                elif params is None:
                    result = await tool_func() # Call without params
                else:
                    # Invalid params type for tool call according to JSON-RPC
                    raise ValueError("Parameters must be a structured value (object or array) or omitted.")

                if msg_id is not None: # It's a request, return result
                    return {"jsonrpc": JSONRPC_VERSION, "result": result, "id": msg_id}
                else: # It's a notification, successful execution means return None
                    return None
            except Exception as e:
                logger.error(f"Error executing tool '{method}' (mapped to '{actual_tool_name}') for session {session_id}: {e}", exc_info=True)
                # Re-raise the exception; the transport layer will format the JSON-RPC error
                raise e

        elif method:
            # Method not found
            logger.warning(f"Method '{method}' not found for session {session_id}")
            if msg_id is not None:
                # Raise an exception that corresponds to 'Method not found'
                raise ValueError(f"Method not found: {method}") # Transport will map this
            else:
                return None # Notification for unknown method, ignore
        else:
            # Invalid message structure (e.g., missing 'method' but not initialize)
            # This case should ideally be caught by the transport's basic validation
            # but we can add a fallback.
            logger.warning(f"Received invalid message structure (no method): {msg}")
            if msg_id is not None:
                raise ValueError("Invalid Request: Missing method.")
            else:
                return None

    # --- Methods for server to send messages to client --- 
    # These now delegate to the transport layer

    async def send_server_notification(self, session_id: str, method: str, params: Any):
        """Sends a notification from the server to the client via the transport."""
        if not session_id:
            logger.error("Cannot send notification: session_id is required.")
            return
        logger.debug(f"Sending notification {method} to session {session_id}")
        message = {"jsonrpc": JSONRPC_VERSION, "method": method, "params": params}
        await self.transport.send_message_to_client(session_id, message)

    async def send_server_request(self, session_id: str, method: str, params: Any, request_id: Any):
        """Sends a request from the server to the client via the transport."""
        if not session_id:
            logger.error("Cannot send request: session_id is required.")
            return
        logger.debug(f"Sending request {method} (id: {request_id}) to session {session_id}")
        message = {"jsonrpc": JSONRPC_VERSION, "method": method, "params": params, "id": request_id}
        await self.transport.send_message_to_client(session_id, message)

    # --- Server Lifecycle --- 

    async def start(self):
        """Initializes the embedding pipeline and starts the server transport."""
        logger.info("Starting MCP Streamable HTTP Server (Application Layer)...")
        # Initialize embedding pipeline before starting transport
        await self.initialize_embedding_pipeline()
        await self.transport.start()
        logger.info("Server application layer ready.")

    async def stop(self):
        """Stops the underlying server transport."""
        logger.info("Stopping MCP Streamable HTTP Server (Application Layer)...")
        await self.transport.stop()
        logger.info("Server application layer stopped.")

# --- Example Usage --- #

# Define an example async tool function
async def example_tool(param1: str, param2: int) -> str:
    """An example async tool function."""
    logger.info(f"Executing example_tool with param1='{param1}', param2={param2}")
    await asyncio.sleep(1) # Simulate async work
    processed_info = f"Processed '{param1}' with value {param2}"
    logger.info("example_tool finished.")
    return processed_info

# Define another example tool
async def get_server_time() -> Dict[str, str]:
    """Returns the current server time."""
    logger.info("Executing get_server_time tool.")
    now = asyncio.get_event_loop().time()
    return {"server_time_monotonic": str(now), "iso_time": asyncio.to_thread(datetime.datetime.now().isoformat)}
    # Note: Using asyncio.to_thread for potentially blocking datetime call is good practice
    # Requires importing datetime: import datetime

async def main():
    # Configure logging for the example
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Silence noisy libraries
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.INFO)

    logger.info("Setting up server...")

    # The server now initializes the pipeline internally on start
    server = StreamableHttpServer(host="0.0.0.0", port=8080) # Listen on all interfaces

    # Note: Tools are registered within server.start() -> initialize_embedding_pipeline()
    # server.register_tool(example_tool) # Can register additional non-embedding tools here if needed

    # Example: Start a background task to send a notification after a delay
    async def delayed_notification_task():
        await asyncio.sleep(25) # Longer delay
        # Need access to transport's sessions
        active_sessions = list(server.transport._sessions.keys())
        if active_sessions:
            target_session = active_sessions[0] # Just notify the first session
            logger.info(f"[BG Task] Sending delayed server notification to session {target_session}")
            import datetime # Import inside task if not global
            await server.send_server_notification(
                target_session,
                "$/serverNotification",
                {"message": "This is a delayed message from the server!", "timestamp": datetime.datetime.now().isoformat()}
            )
        else:
            logger.info("[BG Task] No active sessions to send notification to.")

    try:
        await server.start() # This now also initializes embedding pipeline
        logger.info("Server started successfully. Creating background notification task.")
        bg_task = asyncio.create_task(delayed_notification_task())

        # Keep the server running until interrupted
        stop_event = asyncio.Event()
        await stop_event.wait()

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown signal received.")
    except Exception as e:
        logger.exception("Server encountered an unhandled error.") # Log full traceback
    finally:
        logger.info("Shutting down server...")
        if 'bg_task' in locals() and bg_task and not bg_task.done():
            logger.info("Cancelling background task...")
            bg_task.cancel()
        await server.stop()
        logger.info("Server shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user.") 