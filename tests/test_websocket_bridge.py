"""Tests for the WebSocket LLM bridge (WebSocketLLMService).

Three layers:

- ``TestProtocol``        — unit tests for the wire protocol (no I/O).
- ``TestWebSocketHardening`` — client robustness (timeouts, reconnection,
  malformed-message tolerance, remote errors) driven by a controllable
  in-process fake server.
- ``TestWebSocketBridgeE2E`` — full Pipecat pipeline against the *real* Bun
  bridge in ``js-bridge/``, verifying the protocol end to end.  Skipped when
  ``bun`` is not installed.
"""

import asyncio
import json
import os
import shutil
import unittest
from pathlib import Path
from typing import Any, Optional

from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_adk import WebSocketLLMService
from pipecat_adk.frames import (
    VqlContextFrame,
    VqlFunctionCallResultFrame,
    VqlFunctionCallsStartedFrame,
    VqlLLMFullResponseEndFrame,
    VqlLLMFullResponseStartFrame,
    VqlLLMTextFrame,
)
from pipecat_adk.websocket import protocol

from tests.mocks import TestRunner, Turn
from tests.ws_fake_server import FakeBridgeServer

REPO_ROOT = Path(__file__).resolve().parent.parent
BUN = shutil.which("bun")


# ===========================================================================
# Protocol unit tests
# ===========================================================================


class TestProtocol(unittest.TestCase):
    """The wire protocol is the contract; verify it precisely."""

    def test_turn_run_round_trips(self):
        raw = protocol.encode_turn_run("t1", {"text": "hello"}, {"k": "v"})
        env = protocol.decode_message(raw)
        self.assertEqual(env["type"], protocol.MSG_TURN_RUN)
        self.assertEqual(env["turn_id"], "t1")
        self.assertEqual(env["payload"], {"text": "hello"})
        self.assertEqual(env["state_delta"], {"k": "v"})

    def test_decode_rejects_bad_version(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.decode_message(json.dumps({"v": 99, "type": "x"}))

    def test_decode_rejects_non_json(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.decode_message("not json at all")

    def test_decode_rejects_missing_type(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.decode_message(json.dumps({"v": 1}))

    def test_decode_accepts_bytes(self):
        env = protocol.decode_message(protocol.encode_session_end().encode("utf-8"))
        self.assertEqual(env["type"], protocol.MSG_SESSION_END)

    def test_build_text_frame_downstream(self):
        frame, directions = protocol.build_output_frame(
            {
                "v": 1,
                "type": "frame",
                "frame": protocol.FRAME_TEXT,
                "turn_id": "t1",
                "data": {"text": "hi", "invocation_id": "inv1"},
            }
        )
        self.assertIsInstance(frame, VqlLLMTextFrame)
        self.assertEqual(frame.text, "hi")
        self.assertEqual(frame.turn_id, "t1")
        self.assertEqual(frame.invocation_id, "inv1")
        self.assertEqual(directions, (FrameDirection.DOWNSTREAM,))

    def test_build_function_call_frames_go_both_directions(self):
        for frame_name in (
            protocol.FRAME_FUNCTION_CALLS_STARTED,
            protocol.FRAME_FUNCTION_CALL_IN_PROGRESS,
            protocol.FRAME_FUNCTION_CALL_RESULT,
        ):
            data: dict[str, Any] = {"invocation_id": "inv1"}
            if frame_name == protocol.FRAME_FUNCTION_CALLS_STARTED:
                data["function_calls"] = [
                    {"tool_call_id": "c1", "function_name": "fn", "arguments": {}}
                ]
            else:
                data["tool_call_id"] = "c1"
                data["function_name"] = "fn"
            _, directions = protocol.build_output_frame(
                {"v": 1, "type": "frame", "frame": frame_name, "turn_id": "t1", "data": data}
            )
            self.assertEqual(
                set(directions),
                {FrameDirection.UPSTREAM, FrameDirection.DOWNSTREAM},
                f"{frame_name} must be pushed both ways",
            )

    def test_build_function_calls_started_reconstructs_calls(self):
        frame, _ = protocol.build_output_frame(
            {
                "v": 1,
                "type": "frame",
                "frame": protocol.FRAME_FUNCTION_CALLS_STARTED,
                "turn_id": "t1",
                "data": {
                    "invocation_id": "inv1",
                    "function_calls": [
                        {
                            "tool_call_id": "c1",
                            "function_name": "get_weather",
                            "arguments": {"city": "SF"},
                        }
                    ],
                },
            }
        )
        self.assertIsInstance(frame, VqlFunctionCallsStartedFrame)
        self.assertEqual(len(frame.function_calls), 1)
        self.assertEqual(frame.function_calls[0].function_name, "get_weather")
        self.assertEqual(frame.function_calls[0].arguments, {"city": "SF"})

    def test_build_output_frame_rejects_unknown_frame(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.build_output_frame(
                {"v": 1, "type": "frame", "frame": "NopeFrame", "turn_id": "t1"}
            )

    def test_build_output_frame_rejects_missing_turn_id(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.build_output_frame(
                {"v": 1, "type": "frame", "frame": protocol.FRAME_TEXT, "data": {"text": "x"}}
            )

    def test_build_error_frame(self):
        frame = protocol.build_error_frame(
            {"v": 1, "type": "error", "data": {"message": "boom", "fatal": True}}
        )
        self.assertIsInstance(frame, ErrorFrame)
        self.assertEqual(frame.error, "boom")
        self.assertTrue(frame.fatal)


# ===========================================================================
# Test harness for driving WebSocketLLMService in isolation
# ===========================================================================


class _FrameCapture(FrameProcessor):
    """Records every frame that passes through, then forwards it."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[tuple[Frame, FrameDirection]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self.records.append((frame, direction))
        await self.push_frame(frame, direction)


class _ServiceHarness:
    """Runs a WebSocketLLMService between two frame-capturing processors.

    ``downstream`` sees frames the service pushes toward the TTS; ``upstream``
    sees frames it pushes back toward the transport (errors, function calls).
    """

    def __init__(self, llm_service: WebSocketLLMService) -> None:
        self.llm = llm_service
        self.upstream = _FrameCapture()
        self.downstream = _FrameCapture()
        self.pipeline = Pipeline([self.upstream, self.llm, self.downstream])
        self.task: Optional[PipelineTask] = None
        self._runner_task: Optional[asyncio.Task] = None

    async def __aenter__(self) -> "_ServiceHarness":
        self.task = PipelineTask(
            self.pipeline, params=PipelineParams(allow_interruptions=True)
        )
        runner = PipelineRunner()
        self._runner_task = asyncio.create_task(runner.run(self.task))
        # Let StartFrame propagate and the bridge client connect.
        await asyncio.sleep(0.5)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._runner_task is not None and not self._runner_task.done():
            self._runner_task.cancel()
            try:
                await self._runner_task
            except asyncio.CancelledError:
                pass

    async def queue(self, frame: Frame) -> None:
        assert self.task is not None
        await self.task.queue_frame(frame)

    def downstream_of(self, frame_type: type) -> list[Frame]:
        return [
            f
            for f, d in self.downstream.records
            if isinstance(f, frame_type) and d == FrameDirection.DOWNSTREAM
        ]

    def upstream_of(self, frame_type: type) -> list[Frame]:
        return [
            f
            for f, d in self.upstream.records
            if isinstance(f, frame_type) and d == FrameDirection.UPSTREAM
        ]

    async def wait_until(self, predicate, timeout: float = 6.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            if predicate():
                return
            if loop.time() >= deadline:
                raise TimeoutError("harness condition not met")
            await asyncio.sleep(0.02)


# ===========================================================================
# Client hardening tests (controllable fake server)
# ===========================================================================


class TestWebSocketHardening(unittest.IsolatedAsyncioTestCase):
    """Verify the client survives a misbehaving bridge."""

    async def test_basic_turn_streams_response_frames(self):
        async with FakeBridgeServer(script=["Hello there friend"]) as server:
            service = WebSocketLLMService(uri=server.uri)
            async with _ServiceHarness(service) as harness:
                await harness.queue(VqlContextFrame(turn_id="t1", text="hi"))
                await harness.wait_until(
                    lambda: harness.downstream_of(VqlLLMFullResponseEndFrame)
                )

            text = "".join(f.text for f in harness.downstream_of(VqlLLMTextFrame))
            self.assertEqual(text, "Hello there friend")
            self.assertEqual(len(harness.downstream_of(VqlLLMFullResponseStartFrame)), 1)
            self.assertEqual(len(harness.downstream_of(VqlLLMFullResponseEndFrame)), 1)

    async def test_function_call_frames_pushed_both_directions(self):
        script = [[{"call": {"name": "get_weather", "args": {"city": "SF"}}}, "Sunny."]]
        async with FakeBridgeServer(script=script) as server:
            service = WebSocketLLMService(uri=server.uri)
            async with _ServiceHarness(service) as harness:
                await harness.queue(VqlContextFrame(turn_id="t1", text="weather?"))
                await harness.wait_until(
                    lambda: harness.downstream_of(VqlLLMFullResponseEndFrame)
                )

            # Function-call frames must reach both the transport and the TTS side.
            self.assertTrue(harness.upstream_of(VqlFunctionCallsStartedFrame))
            self.assertTrue(harness.downstream_of(VqlFunctionCallsStartedFrame))
            self.assertTrue(harness.upstream_of(VqlFunctionCallResultFrame))
            self.assertTrue(harness.downstream_of(VqlFunctionCallResultFrame))

    async def test_turn_idle_timeout_surfaces_error(self):
        """A bridge that accepts a turn but never answers triggers a timeout."""
        timed_out = asyncio.Event()

        async with FakeBridgeServer(respond_to_turns=False) as server:
            service = WebSocketLLMService(uri=server.uri, turn_timeout=0.6)

            @service.event_handler("on_completion_timeout")
            async def _on_timeout(_service):
                timed_out.set()

            async with _ServiceHarness(service) as harness:
                await harness.queue(VqlContextFrame(turn_id="t1", text="hi"))
                await harness.wait_until(
                    lambda: harness.upstream_of(ErrorFrame), timeout=4.0
                )

            self.assertTrue(timed_out.is_set(), "on_completion_timeout should fire")
            errors = harness.upstream_of(ErrorFrame)
            self.assertTrue(any("timed out" in e.error for e in errors))
            # The matched start/end pair is still emitted on failure.
            self.assertEqual(len(harness.downstream_of(VqlLLMFullResponseEndFrame)), 1)

    async def test_remote_turn_error_surfaces_error_frame(self):
        async with FakeBridgeServer(turn_error="model exploded") as server:
            service = WebSocketLLMService(uri=server.uri)
            async with _ServiceHarness(service) as harness:
                await harness.queue(VqlContextFrame(turn_id="t1", text="hi"))
                await harness.wait_until(
                    lambda: harness.upstream_of(ErrorFrame), timeout=4.0
                )

            errors = harness.upstream_of(ErrorFrame)
            self.assertTrue(any("model exploded" in e.error for e in errors))
            self.assertEqual(len(harness.downstream_of(VqlLLMFullResponseEndFrame)), 1)

    async def test_malformed_messages_are_ignored(self):
        """Garbage frames before a valid response must not break the turn."""
        async with FakeBridgeServer(
            script=["clean response"], send_malformed_before_response=True
        ) as server:
            service = WebSocketLLMService(uri=server.uri)
            async with _ServiceHarness(service) as harness:
                await harness.queue(VqlContextFrame(turn_id="t1", text="hi"))
                await harness.wait_until(
                    lambda: harness.downstream_of(VqlLLMFullResponseEndFrame)
                )

            text = "".join(f.text for f in harness.downstream_of(VqlLLMTextFrame))
            self.assertEqual(text, "clean response")

    async def test_reconnects_after_connection_drop(self):
        """The client recovers when the bridge drops the connection."""
        async with FakeBridgeServer(
            script=["recovered ok"], drop_first_connection=True
        ) as server:
            service = WebSocketLLMService(uri=server.uri)
            async with _ServiceHarness(service) as harness:
                # Give the receive loop time to notice the drop and reconnect.
                await harness.wait_until(lambda: server.connections >= 2, timeout=6.0)
                await harness.queue(VqlContextFrame(turn_id="t1", text="hi"))
                await harness.wait_until(
                    lambda: harness.downstream_of(VqlLLMFullResponseEndFrame),
                    timeout=6.0,
                )

            text = "".join(f.text for f in harness.downstream_of(VqlLLMTextFrame))
            self.assertEqual(text, "recovered ok")
            self.assertGreaterEqual(server.connections, 2)

    async def test_application_heartbeat_keeps_connection_alive(self):
        async with FakeBridgeServer(script=["ok"]) as server:
            service = WebSocketLLMService(
                uri=server.uri, heartbeat_interval=0.2, heartbeat_timeout=0.2
            )
            async with _ServiceHarness(service) as harness:
                await asyncio.sleep(1.0)  # several heartbeat cycles
                # The turn still works — the connection was never force-closed.
                await harness.queue(VqlContextFrame(turn_id="t1", text="hi"))
                await harness.wait_until(
                    lambda: harness.downstream_of(VqlLLMFullResponseEndFrame)
                )

            pings = [m for m in server.received if m.get("type") == "ping"]
            self.assertGreater(len(pings), 0, "client should send application pings")
            self.assertEqual(server.connections, 1, "connection should not have dropped")

    async def test_interruption_cancels_turn_and_notifies_bridge(self):
        """A pipeline interruption aborts the turn and sends turn.cancel."""
        # A slow, long response so the interrupt lands mid-stream.
        async with FakeBridgeServer(
            script=["word " * 60], chunk_delay=0.05, chunks_per_text=60
        ) as server:
            service = WebSocketLLMService(uri=server.uri)
            async with _ServiceHarness(service) as harness:
                await harness.queue(VqlContextFrame(turn_id="t1", text="tell me a story"))
                # Wait until streaming has started.
                await harness.wait_until(
                    lambda: harness.downstream_of(VqlLLMTextFrame), timeout=4.0
                )
                from pipecat_adk.frames import VqlInterruptionFrame

                await harness.queue(VqlInterruptionFrame(turn_id="t1"))
                await harness.wait_until(
                    lambda: server.cancelled_turns, timeout=4.0
                )

            self.assertIn("t1", server.cancelled_turns)


# ===========================================================================
# End-to-end tests against the real Bun bridge
# ===========================================================================


class BunBridge:
    """Launches the real ``js-bridge`` Bun process for an end-to-end test."""

    def __init__(self, script: list) -> None:
        self._script = script
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader: Optional[asyncio.Task] = None
        self._lines: list[str] = []
        self.port: Optional[int] = None

    async def __aenter__(self) -> "BunBridge":
        env = {
            **os.environ,
            "BRIDGE_SCRIPT": json.dumps(self._script),
            "PORT": "0",
            "HOST": "127.0.0.1",
        }
        self._proc = await asyncio.create_subprocess_exec(
            "bun",
            "run",
            str(REPO_ROOT / "js-bridge" / "src" / "index.ts"),
            env=env,
            cwd=str(REPO_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._reader = asyncio.create_task(self._read_stdout())

        loop = asyncio.get_running_loop()
        deadline = loop.time() + 15.0
        while self.port is None:
            if loop.time() >= deadline:
                raise TimeoutError(f"Bun bridge did not start: {self._lines}")
            if self._proc.returncode is not None:
                raise RuntimeError(f"Bun bridge exited early: {self._lines}")
            await asyncio.sleep(0.05)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._proc.kill()
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except asyncio.CancelledError:
                pass

    async def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        async for raw in self._proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            self._lines.append(line)
            if line.startswith("BRIDGE_LISTENING"):
                self.port = int(line.split()[1])

    @property
    def uri(self) -> str:
        assert self.port is not None
        return f"ws://127.0.0.1:{self.port}"

    @property
    def heard_lines(self) -> list[str]:
        return [line for line in self._lines if line.startswith("HEARD ")]

    async def wait_for_heard(self, timeout: float = 5.0) -> str:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if self.heard_lines:
                return self.heard_lines[-1]
            await asyncio.sleep(0.05)
        raise TimeoutError(f"no [HEARD] line from bridge; stdout={self._lines}")


@unittest.skipUnless(BUN, "bun is not installed")
class TestWebSocketBridgeE2E(unittest.IsolatedAsyncioTestCase):
    """Full pipeline against the real Bun LLM component."""

    async def test_basic_interaction(self):
        async with BunBridge(["Hi, I am a bot"]) as bridge:
            service = WebSocketLLMService(uri=bridge.uri)
            async with TestRunner(llm_service=service) as runner:
                await runner.join()
                await runner.speak_and_wait_for_response("Hi, I am John", timeout=10.0)

            self.assertEqual(
                runner.transcript,
                [Turn("user", "Hi, I am John"), Turn("bot", "Hi, I am a bot")],
            )

    async def test_multi_turn_conversation(self):
        script = [
            "Hi! How can I help you today?",
            "That sounds interesting! Tell me more.",
            "Great, I'd be happy to assist with that.",
        ]
        async with BunBridge(script) as bridge:
            service = WebSocketLLMService(uri=bridge.uri)
            async with TestRunner(llm_service=service) as runner:
                await runner.join()
                await runner.speak_and_wait_for_response("Hello", timeout=10.0)
                await runner.speak_and_wait_for_response(
                    "I need help with a project", timeout=10.0
                )
                await runner.speak_and_wait_for_response("Can you assist me?", timeout=10.0)

            self.assertEqual(
                runner.transcript,
                [
                    Turn("user", "Hello"),
                    Turn("bot", "Hi! How can I help you today?"),
                    Turn("user", "I need help with a project"),
                    Turn("bot", "That sounds interesting! Tell me more."),
                    Turn("user", "Can you assist me?"),
                    Turn("bot", "Great, I'd be happy to assist with that."),
                ],
            )

    async def test_function_call_in_turn(self):
        """The JS component owns the tool loop — one turn yields call + text."""
        script = [
            [
                {
                    "call": {
                        "name": "get_weather",
                        "args": {"location": "San Francisco"},
                        "result": {"weather": "sunny", "temperature": "72 degrees"},
                    }
                },
                "The weather in San Francisco is sunny and 72 degrees!",
            ]
        ]
        async with BunBridge(script) as bridge:
            service = WebSocketLLMService(uri=bridge.uri)
            async with TestRunner(llm_service=service) as runner:
                await runner.join()
                await runner.speak_and_wait_for_response(
                    "What's the weather in San Francisco?", timeout=10.0
                )

            self.assertEqual(
                runner.last_bot_message,
                "The weather in San Francisco is sunny and 72 degrees!",
            )

    async def test_interruption_round_trips_heard_provenance(self):
        """Interrupting the bot sends turn.cancel and turn.completed to the bridge."""
        long_response = (
            "Hello! I'm so glad you're interested in learning about our company. "
            "We have a very long history that spans over fifty years and we have "
            "been pioneers in many different areas of the industry over time."
        )
        # Two responses: the long one, then a reply to the interrupting turn.
        async with BunBridge([long_response, "Sure, go ahead with your question."]) as bridge:
            service = WebSocketLLMService(uri=bridge.uri)
            async with TestRunner(llm_service=service, tts_delay=0.05) as runner:
                await runner.join()
                await runner.speak("Tell me about your company")
                await runner.interrupt_bot("Wait, I have a question", timeout=10.0)

                def _has_interruption_transcription(_bot_output, delta):
                    return any(
                        m.get("type") == "user-transcription"
                        and m.get("data", {}).get("final")
                        and "question" in m.get("data", {}).get("text", "").lower()
                        for m in delta
                    )

                await runner.wait_for(_has_interruption_transcription, timeout=10.0)

            # The bridge recorded what the user actually heard (interrupted).
            heard = await bridge.wait_for_heard(timeout=5.0)
            self.assertIn("interrupted=true", heard)

    async def test_clean_turn_reports_uninterrupted(self):
        async with BunBridge(["A short and complete answer."]) as bridge:
            service = WebSocketLLMService(uri=bridge.uri)
            async with TestRunner(llm_service=service) as runner:
                await runner.join()
                await runner.speak_and_wait_for_response("Tell me something", timeout=10.0)

            heard = await bridge.wait_for_heard(timeout=5.0)
            self.assertIn("interrupted=false", heard)


if __name__ == "__main__":
    unittest.main()
