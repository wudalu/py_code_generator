import asyncio
import json
import uuid
import logging
from typing import Dict, Any, Callable, Optional, Set, Awaitable

from aiohttp import web
from aiohttp.web_request import Request
from aiohttp.web_response import Response, StreamResponse

logger = logging.getLogger(__name__)

# MCP Constants
MCP_SESSION_ID_HEADER = "Mcp-Session-Id"
JSONRPC_VERSION = "2.0"

class StreamableHttpServerTransport:
    """
    Handles the transport layer for an MCP server using Streamable HTTP.
    Manages HTTP connections, SSE streams, and sessions via aiohttp.
    Uses callbacks to interact with the application logic layer.
    """
    def __init__(self,
                 message_processor: Callable[[Optional[str], Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]],
                 host: str = "127.0.0.1",
                 port: int = 8080):
        """
        Initializes the transport layer.

        Args:
            message_processor: An async callable that takes (session_id, request_message)
                               and returns an optional response_message. This is called
                               by the transport when a valid JSON-RPC message is received via POST.
                               The transport handles JSON-RPC error formatting for transport-level issues
                               or if message_processor raises an exception.
            host: The hostname to bind the server to.
            port: The port to bind the server to.
        """
        self.host = host
        self.port = port
        self._message_processor = message_processor
        self.app = web.Application()
        self.app.router.add_route('*', '/mcp', self._handle_mcp_request)
        self._sessions: Dict[str, Dict[str, Any]] = {} # session_id -> {'sse_writer': Optional[StreamResponse], 'active_tasks': Set[asyncio.Task], ...}
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    async def _handle_mcp_request(self, request: Request) -> Response:
        """Handles incoming requests to the /mcp endpoint (main router)."""
        session_id = request.headers.get(MCP_SESSION_ID_HEADER)

        # TODO: Add Origin validation based on configuration

        if request.method == 'POST':
            return await self._handle_post(request, session_id)
        elif request.method == 'GET':
            return await self._handle_get(request, session_id)
        elif request.method == 'DELETE':
            return await self._handle_delete(request, session_id)
        else:
            return web.Response(status=405, text="Method Not Allowed")

    async def _handle_post(self, request: Request, session_id: Optional[str]) -> Response:
        """Handles POST requests, parsing messages and calling the message_processor."""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return self._make_jsonrpc_error_response(None, -32700, "Parse error")

        # --- Session Validation ---
        is_initialize = False
        if isinstance(body, dict) and body.get("method") == "initialize":
             is_initialize = True
        elif isinstance(body, list) and any(isinstance(msg, dict) and msg.get("method") == "initialize" for msg in body):
             is_initialize = True

        if session_id and session_id not in self._sessions and not is_initialize:
            logger.warning(f"Received POST with invalid session ID: {session_id}")
            return web.json_response(
                {'jsonrpc': JSONRPC_VERSION, 'error': {'code': -32000, 'message': 'Invalid or expired session ID'}, 'id': None},
                status=400 # Or 404 if session expired
            )
        # --- End Session Validation ---


        if isinstance(body, list): # Batch request
            # Simple sequential processing for now
            responses_data = []
            for msg in body:
                resp_data = await self._process_and_handle_single_post_message(msg, session_id)
                if resp_data: # Don't add None (e.g., for notifications)
                    responses_data.append(resp_data)

            if not responses_data: # Only notifications/responses received
                 return web.Response(status=202, text="Accepted")
            else:
                 # Return JSON array for batch results
                 return web.json_response(responses_data)

        elif isinstance(body, dict): # Single message
            response_data = await self._process_and_handle_single_post_message(body, session_id)

            if response_data is None: # Notification successfully processed
                return web.Response(status=202, text="Accepted")
            # Special handling for initialize result from processor needed for header
            elif "_session_id_internal" in response_data:
                 new_session_id = response_data.pop("_session_id_internal")
                 headers = {MCP_SESSION_ID_HEADER: new_session_id}
                 return web.json_response(response_data, headers=headers)
            else: # Standard request response
                return web.json_response(response_data)
        else:
            return self._make_jsonrpc_error_response(None, -32600, "Invalid Request: Expected JSON object or array.")


    async def _process_and_handle_single_post_message(self, msg: Dict[str, Any], session_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Processes a single message from a POST request, validates structure,
        calls the message_processor, and handles session creation for initialize.
        Returns the data for the JSON-RPC response, or None for notifications.
        Includes internal keys like _session_id_internal for initialize.
        Formats JSON-RPC errors if processing fails.
        """
        if not isinstance(msg, dict) or msg.get("jsonrpc") != JSONRPC_VERSION:
            return self._make_jsonrpc_error_data(None, -32600, "Invalid Request: Invalid JSON-RPC structure.")

        msg_id = msg.get("id")
        method = msg.get("method")

        # Handle initialize specifically for session creation *before* calling processor
        if method == "initialize":
            if session_id and session_id in self._sessions:
                 return self._make_jsonrpc_error_data(msg_id, -32001, "Session already initialized")

            new_session_id = str(uuid.uuid4())
            self._sessions[new_session_id] = {'sse_writer': None, 'active_tasks': set()}
            logger.info(f"Transport: Initialized new session: {new_session_id}")
            # Pass the new session_id to the processor (it might not need it, but good practice)
            session_id = new_session_id # Use the new ID for the processor call

        elif "method" in msg and session_id not in self._sessions:
             # Non-initialize request requires a valid session
             return self._make_jsonrpc_error_data(msg_id, -32000, "Session not initialized or invalid")

        elif "result" in msg or "error" in msg:
            # Client sent a response/error via POST - spec says server MAY ignore
            logger.warning(f"Received client response/error via POST (ignoring): {msg}")
            return None # Indicate no response needed

        # Call the application layer processor
        try:
            app_response_data = await self._message_processor(session_id, msg)

            # If initialize, add the internal session ID key for the caller (_handle_post)
            if method == "initialize" and app_response_data and "error" not in app_response_data:
                app_response_data["_session_id_internal"] = session_id # The new_session_id

            return app_response_data # Can be None for notifications

        except Exception as e:
            logger.error(f"Message processor failed for method '{method}': {e}", exc_info=True)
            if msg_id is not None: # Only return error response if it was a request
                # Use a generic server error code
                return self._make_jsonrpc_error_data(msg_id, -32000, f"Server error processing request: {e}")
            else:
                return None # Error processing notification, just log


    async def _handle_get(self, request: Request, session_id: Optional[str]) -> Response:
        """Handles GET requests to establish an SSE connection."""
        if not session_id or session_id not in self._sessions:
            logger.warning(f"Received GET request for invalid/missing session ID: {session_id}")
            return web.Response(status=404, text="Session Not Found")

        accept_header = request.headers.get("Accept", "")
        if "text/event-stream" not in accept_header:
            return web.Response(status=406, text="Not Acceptable (Requires Accept: text/event-stream)")

        response = StreamResponse(
            status=200, reason='OK',
            headers={
                'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache',
                'Connection': 'keep-alive', 'Access-Control-Allow-Origin': '*' # Adjust in production
            }
        )
        await response.prepare(request)

        session = self._sessions[session_id]
        # Handle potential existing writer (e.g., client reconnects with GET)
        if session.get('sse_writer'):
            logger.warning(f"Session {session_id} already has an active SSE stream. Replacing.")
            try:
                # Attempt to cancel the previous stream's keepalive task
                if hasattr(session['sse_writer'], '_keepalive_task') and session['sse_writer']._keepalive_task:
                    session['sse_writer']._keepalive_task.cancel()
                await session['sse_writer'].write_eof()
            except Exception as e:
                 logger.debug(f"Ignoring error closing previous SSE writer for {session_id}: {e}")

        session['sse_writer'] = response
        logger.info(f"SSE stream opened for session: {session_id}")

        # Handle Last-Event-ID for resume
        last_event_id = request.headers.get("Last-Event-ID")
        if last_event_id:
             logger.info(f"Client {session_id} requested resume from event ID: {last_event_id}")
             # TODO: Implement message replay logic here if required
             # await self._replay_missed_messages(session_id, last_event_id, response)

        # Keep connection alive - associate the keepalive task with the response object
        keepalive_task = asyncio.create_task(self._sse_keepalive(response, session_id))
        response._keepalive_task = keepalive_task # Store reference for cancellation
        session.setdefault('active_tasks', set()).add(keepalive_task)

        try:
            await keepalive_task # Wait for the keepalive task to finish (on disconnect/error)
        except asyncio.CancelledError:
             logger.info(f"SSE keepalive task cancelled for session {session_id}.")
             # Ensure EOF is written if cancelled gracefully
             if not response.prepared: await response.prepare(request) # Ensure prepared before write_eof
             if not response.eof_sent: await response.write_eof()
        except Exception as e:
             logger.error(f"SSE keepalive task ended with error for {session_id}: {e}", exc_info=True)
        finally:
            logger.info(f"Cleaning up SSE stream resources for session: {session_id}")
            session.get('active_tasks', set()).discard(keepalive_task)
            # Check if this is still the active writer before nullifying
            if session_id in self._sessions and self._sessions[session_id].get('sse_writer') is response:
                 self._sessions[session_id]['sse_writer'] = None
            # Ensure EOF is written on any exit path
            try:
                if not response.prepared: await response.prepare(request)
                if not response.eof_sent: await response.write_eof()
            except Exception as e:
                logger.debug(f"Error sending final EOF for {session_id}: {e}")

        return response

    async def _sse_keepalive(self, writer: StreamResponse, session_id: str):
        """Periodically sends SSE comments to keep the connection alive and detects disconnects."""
        while True:
            try:
                await writer.write(b': keepalive\n\n')
                await writer.drain()
                await asyncio.sleep(15) # Send keepalive every 15 seconds
            except (ConnectionResetError, asyncio.CancelledError) as e:
                logger.info(f"SSE connection closed or cancelled for {session_id}: {type(e).__name__}")
                break # Exit loop on disconnect or cancellation
            except Exception as e:
                logger.error(f"Error during SSE keepalive for {session_id}: {e}", exc_info=True)
                break # Exit loop on other errors

    async def _handle_delete(self, request: Request, session_id: Optional[str]) -> Response:
         """Handles DELETE requests to terminate a session."""
         if not session_id or session_id not in self._sessions:
             return web.Response(status=404, text="Session Not Found")

         logger.info(f"Terminating session {session_id} by client request.")
         await self._terminate_session(session_id)
         return web.Response(status=204) # No Content

    async def _terminate_session(self, session_id: str):
         """Cleans up resources associated with a session internal helper."""
         if session_id in self._sessions:
             session = self._sessions.pop(session_id) # Remove session immediately
             logger.info(f"Terminating session {session_id}. Cleaning up resources...")
             sse_writer = session.get('sse_writer')
             if sse_writer:
                 keepalive_task = getattr(sse_writer, '_keepalive_task', None)
                 if keepalive_task and not keepalive_task.done():
                     keepalive_task.cancel()
                 try:
                     if not sse_writer.eof_sent:
                          if not sse_writer.prepared:
                              # Cannot prepare response outside handler normally. Skip write_eof if not prepared.
                              pass
                          else:
                              await sse_writer.write_eof()
                 except Exception as e:
                      logger.warning(f"Error closing SSE writer for session {session_id}: {e}")

             # Cancel any other tasks associated with the session
             active_tasks = session.get('active_tasks', set())
             for task in active_tasks:
                  if task and not task.done():
                      task.cancel()
             # Optional: Wait for tasks to finish cancellation
             # await asyncio.gather(*[t for t in active_tasks if t], return_exceptions=True)

             logger.info(f"Session {session_id} resources cleaned up.")
         else:
             logger.warning(f"Attempted to terminate non-existent session: {session_id}")

    async def send_message_to_client(self, session_id: str, message: Dict[str, Any]):
         """
         Sends a JSON-RPC message (notification or server request) to a specific client via SSE.
         The message should be a complete JSON-RPC object.
         Handles SSE formatting and error logging.
         """
         if session_id not in self._sessions:
             logger.error(f"Cannot send SSE message: Session {session_id} not found.")
             return

         session = self._sessions[session_id]
         writer = session.get('sse_writer')

         if writer and not writer.prepared:
              logger.warning(f"SSE writer for session {session_id} exists but is not prepared. Cannot send.")
              return
         if writer and writer.eof_sent:
              logger.warning(f"SSE writer for session {session_id} exists but EOF already sent. Cannot send.")
              return

         if writer:
             try:
                 # TODO: Use incrementing/timestamp-based event IDs for proper resume
                 event_id = str(uuid.uuid4())
                 json_data = json.dumps(message)
                 sse_formatted = f"id: {event_id}\ndata: {json_data}\n\n"
                 await writer.write(sse_formatted.encode('utf-8'))
                 await writer.drain()
                 logger.debug(f"Sent SSE to {session_id}: id={event_id} data={json_data}")
             except ConnectionResetError:
                  logger.warning(f"Cannot send SSE to {session_id}: Connection reset by peer.")
                  session['sse_writer'] = None # Mark writer as invalid
                  # Consider trying to terminate the session fully
                  # asyncio.create_task(self._terminate_session(session_id))
             except Exception as e:
                  logger.error(f"Failed to send SSE message to session {session_id}: {e}", exc_info=True)
                  # Consider marking writer as invalid or terminating session
         else:
             logger.warning(f"Cannot send SSE message: No active writer for session {session_id}.")


    def _make_jsonrpc_error_response(self, msg_id: Optional[Any], code: int, message: str, status_code: int = 400) -> web.Response:
        """Creates a JSON-RPC error aiohttp response object."""
        return web.json_response(
            {"jsonrpc": JSONRPC_VERSION, "error": {"code": code, "message": message}, "id": msg_id},
            status=status_code
        )

    def _make_jsonrpc_error_data(self, msg_id: Optional[Any], code: int, message: str) -> Dict[str, Any]:
         """Creates a JSON-RPC error data dictionary (for batch responses or internal use)."""
         return {"jsonrpc": JSONRPC_VERSION, "error": {"code": code, "message": message}, "id": msg_id}

    async def start(self):
        """Starts the HTTP server transport."""
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        logger.info(f"MCP Streamable HTTP Transport started on http://{self.host}:{self.port}/mcp")

    async def stop(self):
        """Stops the HTTP server transport gracefully."""
        logger.info("Stopping MCP Transport...")
        # Terminate all active sessions first
        active_session_ids = list(self._sessions.keys())
        if active_session_ids:
             logger.info(f"Terminating {len(active_session_ids)} active session(s)...")
             await asyncio.gather(*[self._terminate_session(sid) for sid in active_session_ids])

        if self._site:
            await self._site.stop()
            logger.info("Transport site stopped.")
        if self._runner:
            await self._runner.cleanup()
            logger.info("Transport runner cleaned up.")
        logger.info("MCP Transport stopped.") 