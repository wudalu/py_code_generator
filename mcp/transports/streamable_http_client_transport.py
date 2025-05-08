import asyncio
import json
import logging
import traceback
from typing import Dict, Any, Callable, Optional, Awaitable

import aiohttp
from aiohttp_sse_client.client import EventSource # Using an external library for SSE client

logger = logging.getLogger(__name__)

# MCP Constants
MCP_SESSION_ID_HEADER = "Mcp-Session-Id"
JSONRPC_VERSION = "2.0"
ACCEPT_HEADER_VALUE = "application/json, text/event-stream"
SSE_ACCEPT_HEADER_VALUE = "text/event-stream"

# Default retry settings for SSE connection
SSE_RETRY_DELAY = 3 # seconds
SSE_MAX_RETRIES = 5 # Number of retries before giving up (set to None or 0 to disable retry)

class StreamableHttpClientTransport:
    """
    Handles the transport layer for an MCP client using Streamable HTTP.
    Manages aiohttp session, HTTP requests (POST/GET/DELETE), SSE connection,
    session ID, Last-Event-ID, and connection retries.
    Uses a callback to pass all received server messages to the application layer.
    """
    def __init__(self,
                 base_url: str,
                 on_server_message: Callable[[Dict[str, Any]], Awaitable[None]],
                 client_session: Optional[aiohttp.ClientSession] = None):
        """
        Initializes the client transport.

        Args:
            base_url: The base URL of the MCP server (e.g., "http://localhost:8080").
            on_server_message: An async function called whenever a complete JSON-RPC message
                               is received from the server (via POST response or SSE).
            client_session: Optional existing aiohttp.ClientSession to use.
        """
        if not base_url.endswith('/'):
            base_url += '/'
        self.mcp_endpoint = base_url + "mcp"
        self._on_server_message = on_server_message
        # If a session is passed in, the caller is responsible for its lifecycle
        self._external_session = client_session is not None
        # Don't create session here, create it in connect if not provided externally
        self._client_session: Optional[aiohttp.ClientSession] = client_session 
        self._session_id: Optional[str] = None
        self._last_event_id: Optional[str] = None
        self._sse_listener_task: Optional[asyncio.Task] = None
        self._is_connected = False # Represents if initialized and listener *should* be running
        self._connection_lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    async def connect(self, init_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Connects and initializes the MCP session via a POST request.
        Starts the background SSE listener upon successful initialization.

        Args:
            init_params: Initialization parameters for the server.

        Returns:
            The result part of the JSON-RPC initialize response.

        Raises:
            ConnectionError: If connection/initialization fails.
            ValueError: If the server response is malformed.
            RuntimeError: If already connected.
        """
        async with self._connection_lock:
            if self._is_connected:
                raise RuntimeError("Transport is already connected. Call disconnect() first.")

            # Create ClientSession here if not provided externally
            if not self._external_session and (self._client_session is None or self._client_session.closed):
                logger.debug("Transport: Creating internal aiohttp ClientSession.")
                self._client_session = aiohttp.ClientSession()
            elif self._client_session is None: # Should not happen if external_session is True, but check anyway
                raise RuntimeError("Transport Error: External session provided but is None.")

            init_request = {
                "jsonrpc": JSONRPC_VERSION,
                "method": "initialize",
                "params": init_params or {},
                "id": 0 # Standard ID for initialization
            }
            logger.info(f"Transport: Sending initialize request to {self.mcp_endpoint}")
            headers = {"Accept": ACCEPT_HEADER_VALUE}

            try:
                async with self._client_session.post(self.mcp_endpoint, json=init_request, headers=headers) as response:
                    response.raise_for_status()

                    self._session_id = response.headers.get(MCP_SESSION_ID_HEADER)
                    if not self._session_id:
                        raise ValueError("Transport Error: Server did not return Mcp-Session-Id header.")
                    logger.info(f"Transport: Received session ID: {self._session_id}")

                    content_type = response.headers.get("Content-Type", "")

                    # Handle immediate JSON response for initialize
                    if "application/json" in content_type:
                        response_data = await response.json()
                        if response_data.get("id") != init_request["id"]:
                            raise ValueError("Transport Error: Initialize response ID mismatch.")
                        if "error" in response_data:
                            raise ConnectionError(f"Transport Error: Server initialization failed: {response_data['error']}")
                        if "result" not in response_data:
                             raise ValueError("Transport Error: Initialize response missing 'result'.")

                        logger.info("Transport: Initialization successful (JSON response).")
                        initialize_result = response_data["result"]
                        # Also pass the full response to the handler in case it needs it
                        # Though typically only the result is needed by the app layer caller
                        await self._on_server_message(response_data)

                    # Handle SSE response for initialize
                    elif "text/event-stream" in content_type:
                        logger.info("Transport: Initialization response is SSE stream. Result expected via SSE.")
                        # The result will arrive via the SSE listener which we start next.
                        # We need to consume *this* response stream now though.
                        # This complicates things slightly. For now, assume result is on main SSE.
                        initialize_result = {"status": "Initialization result pending on SSE stream"}
                        # Start consuming this specific stream to potentially get the init result faster?
                        # Or rely on the main listener? Rely on main listener for simplicity first.
                        logger.warning("Handling initialize response via SSE stream is simplified, result might be delayed.")
                        # We still need to drain this response body even if we don't parse it here.
                        await response.read() # Consume the body

                    else:
                         raise ValueError(f"Transport Error: Unexpected Content-Type during initialization: {content_type}")

                    # Start background listener ONLY after successful session ID retrieval
                    self._is_connected = True
                    self._sse_listener_task = asyncio.create_task(self._listen_sse_with_retry())
                    return initialize_result # Return the result part to the application layer

            except aiohttp.ClientError as e:
                logger.error(f"Transport: Connection error during initialization: {e}", exc_info=True)
                self._is_connected = False # Ensure state is correct on failure
                self._session_id = None
                raise ConnectionError(f"Transport: Failed to connect or initialize: {e}") from e
            except Exception as e:
                logger.error(f"Transport: Error during initialization: {e}", exc_info=True)
                self._is_connected = False
                self._session_id = None
                # Don't close session here if it was passed externally
                if not self._external_session and not self._client_session.closed:
                    await self._client_session.close()
                raise

    async def send(self, message: Dict[str, Any]):
        """
        Sends a JSON-RPC message via POST.
        Handles different response types (202, JSON, SSE) from the server.

        Args:
            message: The JSON-RPC message dictionary.

        Raises:
            ConnectionError: If not connected or connection fails.
            RuntimeError: If client is not connected.
        """
        if not self._is_connected or not self._session_id:
            raise RuntimeError("Transport Error: Client is not connected. Call connect() first.")

        # Use the shared session for sending messages
        # await self._ensure_client_session() # Removed this erroneous call
        headers = {MCP_SESSION_ID_HEADER: self.session_id}
        logger.debug(f"Transport: Sending message: {message}")

        # Define an inner function to perform the POST request
        async def _do_post():
            try:
                # Removed explicit ClientTimeout to debug "Timeout context manager should be used inside a task" error
                async with self._client_session.post(self.mcp_endpoint, json=message, headers=headers) as response:
                    response_text = await response.text()
                    logger.debug(f"Transport: Received response status: {response.status}, body: {response_text[:100]}...")
                    # Handle status codes *before* raising for status to check 202 etc.
                    status_code = response.status
                    
                    # Check for non-error statuses first
                    if status_code == 202:
                        logger.debug("Transport: POST accepted (202).")
                        return # Success, no body expected
                    
                    # If not 202, raise for other errors (4xx/5xx)
                    response.raise_for_status()
                    
                    # Process successful responses (e.g., 200 OK) with content
                    content_type = response.headers.get("Content-Type", "")
                    logger.debug(f"Transport: POST response {status_code}, Content-Type: {content_type}")

                    if "application/json" in content_type:
                        response_data = await response.json() # Use already read text?
                        try:
                            response_data = json.loads(response_text)
                            logger.debug(f"Transport: Received JSON response: {response_data}")
                            await self._on_server_message(response_data)
                        except json.JSONDecodeError:
                             logger.error(f"Transport: Failed to decode JSON from response: {response_text[:200]}...")
                             raise ValueError("Invalid JSON received from server") # Raise error on bad JSON

                    elif "text/event-stream" in content_type:
                        logger.info("Transport: Receiving SSE stream in response to POST.")
                        # Handle this stream in the foreground
                        # NOTE: Passing the raw response object into another task might be tricky.
                        # This part might need rethinking if foreground SSE is common for POST.
                        # For now, we assume JSON or 202 based on previous logic. Reading text already happened.
                        logger.warning("Transport: Handling SSE stream in POST response is not fully implemented here.")
                        # await self._handle_foreground_sse_stream(response) # Cannot pass response easily

                    else:
                        # body_text = await response.text() # Already read
                        logger.warning(f"Transport: Unexpected Content-Type for POST response {status_code}: {content_type}. Body: {response_text[:200]}...")
                        # Attempt to parse as JSON just in case?
                        try:
                             response_data = json.loads(response_text)
                             await self._on_server_message(response_data)
                        except json.JSONDecodeError:
                             logger.error("Transport: Could not parse unexpected POST response body as JSON.")
                             # Raise or just log?
                             raise ValueError(f"Unexpected Content-Type: {content_type}")
                             
            except aiohttp.ClientError as e:
                logger.error(f"Transport: Connection error during POST: {e}")
                raise ConnectionError(f"Transport: Failed to send message: {e}") from e
            except Exception as e:
                logger.error(f"Transport: Unexpected error during POST: {e}", exc_info=True)
                raise # Re-raise other exceptions

        # Explicitly create and await a task for the POST operation
        try:
            # logger.debug(f\"Transport: Creating task for POST request {message.get('id', 'notify')}...\")
            # Use create_task to ensure the operation runs in a definite task context
            # post_task = asyncio.create_task(_do_post(), name=f\"mcp_post_{message.get('id', 'notify')}\")
            # await post_task
            # Reverting: Directly await the coroutine, assuming ClientSession creation in connect is sufficient.
            await _do_post()
            
        except ConnectionError: # Catch specific error raised by _do_post
            raise # Re-raise connection errors
        except Exception as e:
            # Catch any other exceptions from _do_post task
            logger.error(f"Transport: Unexpected error sending message (outer scope): {e}", exc_info=True)
            # Re-raise other exceptions
            raise RuntimeError(f"Unexpected error during send: {e}") from e

    async def _handle_foreground_sse_stream(self, response: aiohttp.ClientResponse):
        """Handles an SSE stream received in response to a POST, running in the foreground."""
        try:
            # TODO: Consider timeout for this foreground stream?
            async with EventSource(response, read_timeout=None) as event_source:
                 async for event in event_source:
                    logger.debug(f"Transport: Received event from POST-SSE: id={event.id}, data={event.data}")
                    # According to spec, event.id from POST response stream should *not* update Last-Event-ID
                    if event.data:
                        try:
                            message = json.loads(event.data)
                            await self._on_server_message(message)
                        except json.JSONDecodeError:
                            logger.error(f"Transport: Failed to decode JSON from POST-SSE data: {event.data}")
                        except Exception as e:
                            logger.error(f"Transport: Error processing message from POST-SSE: {e}", exc_info=True)
        except aiohttp.ClientError as e:
             logger.error(f"Transport: Error reading POST SSE stream: {e}")
             # Don't disconnect here, maybe the main SSE is still fine?
        except Exception as e:
             logger.error(f"Transport: Unexpected error handling POST SSE stream: {e}", exc_info=True)
        finally:
             logger.info("Transport: Finished handling SSE stream from POST response.")

    async def _listen_sse_with_retry(self):
        """Manages the background SSE listener task with retries."""
        retries = 0
        max_retries = SSE_MAX_RETRIES if SSE_MAX_RETRIES is not None else float('inf')
        run_indefinitely = SSE_MAX_RETRIES is None

        while self._is_connected and (run_indefinitely or retries < max_retries):
            try:
                await self._listen_sse_once()
                # If it exits cleanly (server closed), maybe reset retries or stop?
                logger.info("Transport: SSE listener connection closed by server.")
                # Let's assume clean close means we should retry unless shutdown is signaled
                if not self._is_connected: break
                retries = 0 # Reset retries after a successful connection that was closed by server
                await asyncio.sleep(1) # Short delay before reconnecting after clean close

            except aiohttp.ClientResponseError as e:
                logger.warning(f"Transport: SSE connection error (status {e.status}): {e.message}. Retrying...")
                # Specific handling for session expiry
                if e.status in [400, 404]: # Bad Request or Not Found might indicate invalid session
                     logger.error(f"Transport: Session {self._session_id} likely invalid/expired ({e.status}). Stopping listener.")
                     self._is_connected = False # Stop trying
                     await self._on_server_message({"internal_error": f"Session {self._session_id} invalid/expired ({e.status})"})
                     break
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                logger.warning(f"Transport: SSE connection failed ({type(e).__name__}). Retrying...")
            except asyncio.CancelledError:
                logger.info("Transport: SSE listener task cancelled.")
                break
            except Exception as e:
                logger.error(f"Transport: Unexpected error in SSE listener: {e}", exc_info=True)
                # Continue retrying for unexpected errors for now

            if not self._is_connected: # Check flag again before sleep/retry
                 break

            retries += 1
            if run_indefinitely or retries < max_retries:
                 delay = SSE_RETRY_DELAY * (2 ** min(retries - 1, 6)) # Exponential backoff capped at ~3 mins
                 logger.info(f"Transport: Waiting {delay}s before SSE retry attempt {retries}...")
                 await asyncio.sleep(delay)
            else:
                 logger.error(f"Transport: SSE connection failed after {max_retries} retries. Stopping listener.")
                 self._is_connected = False
                 await self._on_server_message({"internal_error": "SSE connection failed permanently"})
                 break

        logger.info("Transport: Background SSE listener task finished.")

    async def _listen_sse_once(self):
        """Attempts to connect and listen to the SSE stream once."""
        headers = {MCP_SESSION_ID_HEADER: self.session_id}
        logger.info(f"Transport: Attempting new SSE connection.")
        try:
            # Ensure the client session is created if it doesn't exist
            # await self._ensure_client_session() # Removed this erroneous call

            # Try removing method='GET' as ClientSession.request might handle it
            # or EventSource might pass it implicitly in a way that conflicts.
            # Removed explicit timeout from EventSource as well for now
            async with EventSource(self.mcp_endpoint, 
                                 session=self._client_session, 
                                 headers=headers) as event_source:
                                 # timeout=self.timeout) as event_source: # Temporarily removed timeout
                logger.info(f"Transport: SSE connection established for session {self.session_id}. Listening...")
                async for event in event_source:
                    if not self._is_connected: break # Check flag during event processing
                    logger.debug(f"Transport: Received SSE event: id={event.id}, data={event.data}")
                    if event.id:
                        self._last_event_id = event.id # Store last *successfully processed* ID?
                    if event.data:
                        try:
                            message = json.loads(event.data)
                            await self._on_server_message(message)
                        except json.JSONDecodeError:
                            logger.error(f"Transport: Failed to decode JSON from SSE data: {event.data}")
                        except Exception as e:
                             logger.error(f"Transport: Error in on_server_message callback: {e}", exc_info=True)
        except Exception as e:
            # Let the retry loop handle logging the specific error type
            raise e # Re-raise to be caught by _listen_sse_with_retry

    async def disconnect(self, send_delete: bool = True):
        """Disconnects the client, stops the listener, and optionally terminates the server session."""
        async with self._connection_lock:
            if not self._is_connected:
                logger.info("Transport: Already disconnected.")
                # Close session only if we own it and it's open
                if not self._external_session and not self._client_session.closed:
                     await self._client_session.close()
                return

            logger.info("Transport: Disconnecting...")
            self._is_connected = False # Signal loops to stop

            # Cancel listener task
            if self._sse_listener_task and not self._sse_listener_task.done():
                self._sse_listener_task.cancel()
                try:
                    await asyncio.wait_for(self._sse_listener_task, timeout=5.0)
                except asyncio.CancelledError:
                    logger.info("Transport: SSE listener task successfully cancelled.")
                except asyncio.TimeoutError:
                     logger.warning("Transport: Timed out waiting for SSE listener task to cancel.")
                except Exception as e:
                    logger.error(f"Transport: Error during SSE listener task cancellation: {e}", exc_info=True)
            self._sse_listener_task = None

            # Send DELETE request
            if send_delete and self._session_id:
                # Ensure session exists before trying to use it for DELETE
                if self._client_session and not self._client_session.closed:
                    logger.info(f"Transport: Sending DELETE to terminate session {self._session_id}")
                    headers = {MCP_SESSION_ID_HEADER: self._session_id}
                    try:
                        async with self._client_session.delete(self.mcp_endpoint, headers=headers, timeout=5.0) as response:
                            if response.status == 204:
                                logger.info(f"Transport: Session {self._session_id} terminated successfully on server.")
                            elif response.status in [404, 400]:
                                 logger.info(f"Transport: Session {self._session_id} already terminated or invalid on server.")
                            elif response.status == 405:
                                logger.info("Transport: Server does not support DELETE for session termination.")
                            else:
                                logger.warning(f"Transport: Unexpected status {response.status} terminating session.")
                    except asyncio.TimeoutError:
                        logger.warning("Transport: Timed out sending DELETE request.")
                    except aiohttp.ClientError as e:
                        logger.warning(f"Transport: Failed to send DELETE request: {e}")
                    except Exception as e:
                        logger.error(f"Transport: Unexpected error during session termination: {e}", exc_info=True)
                else:
                    logger.warning("Transport: Cannot send DELETE, client session is closed or None.")

            # Close aiohttp session ONLY if we created it and it's still open
            if not self._external_session and self._client_session and not self._client_session.closed:
                await self._client_session.close()
                logger.info("Transport: Owned aiohttp ClientSession closed.")
            # elif self._external_session:
            #     logger.info("Transport: Using external ClientSession, not closing it.")

            # Reset state
            self._client_session = None # Clear reference if internal
            self._session_id = None
            self._last_event_id = None
            logger.info("Transport: Disconnected.") 