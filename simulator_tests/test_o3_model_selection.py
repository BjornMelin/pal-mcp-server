#!/usr/bin/env python3
"""
O3 Model Selection Test

Tests that O3 models are properly selected and used when explicitly specified,
regardless of the default model configuration (even when set to auto).
Validates model selection via server logs.
"""

import datetime
import json
import re

from .base_test import BaseSimulatorTest


class O3ModelSelectionTest(BaseSimulatorTest):
    """Test O3 model selection and usage"""

    @staticmethod
    def _filter_logs_since(logs: str, start_time: str) -> str:
        """Filter logs to entries emitted at or after start_time (YYYY-MM-DD HH:MM:SS)."""
        filtered_lines = []
        for line in logs.splitlines():
            line_timestamp = line[:19]
            if line_timestamp >= start_time:
                filtered_lines.append(line)
        return "\n".join(filtered_lines)

    @staticmethod
    def _contains_o3_model_reference(model_logs: list[str]) -> bool:
        """Check model usage logs for explicit o3/o3-mini selection."""
        o3_pattern = re.compile(r"\b(o3-mini|o3)\b", re.IGNORECASE)
        return any(o3_pattern.search(line) for line in model_logs)

    @staticmethod
    def _response_indicates_workflow_routing(response: str) -> bool:
        """Check codereview response payload for model/provider routing metadata."""
        try:
            response_data = json.loads(response)
            if not isinstance(response_data, dict):
                return False
            metadata = response_data.get("metadata", {})
            model_name = metadata.get("model_name", "")
            provider = metadata.get("provider", "")
            if model_name and provider:
                return True
            return False
        except json.JSONDecodeError:
            return False

    @property
    def test_name(self) -> str:
        return "o3_model_selection"

    @property
    def test_description(self) -> str:
        return "O3 model selection and usage validation"

    def run_test(self) -> bool:
        """Test O3 model selection and usage"""
        try:
            self.logger.info(" Test: O3 model selection and usage validation")

            # Check which API keys are configured
            import os

            has_openai = bool(os.environ.get("OPENAI_API_KEY"))
            has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))

            # If only OpenRouter is configured, adjust test expectations
            if has_openrouter and not has_openai:
                self.logger.info("  ℹ️  Only OpenRouter configured - O3 models will be routed through OpenRouter")
                return self._run_openrouter_o3_test()

            # If neither OpenAI nor OpenRouter is configured, skip the test
            if not has_openai and not has_openrouter:
                self.logger.info("  ⚠️  Neither OpenAI nor OpenRouter API keys configured - skipping test")
                self.logger.info(
                    "  ℹ️  This test requires either OPENAI_API_KEY or OPENROUTER_API_KEY to be set in .env"
                )
                self.logger.info("  ✅ Test skipped (no API keys configured)")
                return True  # Return True to indicate test passed/skipped

            self.logger.info("  ℹ️  OpenAI API configured - validating effective provider routing for O3 models")

            # Setup test files for later use
            self.setup_test_files()

            # Mark test start time to avoid matching stale historical entries.
            log_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Test 1: Explicit O3 model selection
            self.logger.info("  1: Testing explicit O3 model selection")

            response1, _ = self.call_mcp_tool(
                "chat",
                {
                    "prompt": "Simple test: What is 2 + 2? Just give a brief answer.",
                    "model": "o3",
                    "temperature": 1.0,  # O3 only supports default temperature of 1.0
                },
            )

            if not response1:
                self.logger.error("  ❌ O3 model test failed")
                return False

            self.logger.info("  ✅ O3 model call completed")

            # Test 2: Explicit O3-mini model selection
            self.logger.info("  2: Testing explicit O3-mini model selection")

            response2, _ = self.call_mcp_tool(
                "chat",
                {
                    "prompt": "Simple test: What is 3 + 3? Just give a brief answer.",
                    "model": "o3-mini",
                    "temperature": 1.0,  # O3-mini only supports default temperature of 1.0
                },
            )

            if not response2:
                self.logger.error("  ❌ O3-mini model test failed")
                return False

            self.logger.info("  ✅ O3-mini model call completed")

            # Test 3: Another tool with O3 to ensure it works across tools
            self.logger.info("  3: Testing O3 with different tool (codereview)")

            # Create a simple test file
            test_code = """def add(a, b):
    return a + b

def multiply(x, y):
    return x * y
"""
            test_file = self.create_additional_test_file("simple_math.py", test_code)

            response3, _ = self.call_mcp_tool(
                "codereview",
                {
                    "step": "Review this simple code for quality and potential issues",
                    "step_number": 1,
                    "total_steps": 1,
                    "next_step_required": False,
                    "findings": "Starting code review analysis",
                    "relevant_files": [test_file],
                    "model": "o3",
                    "temperature": 1.0,  # O3 only supports default temperature of 1.0
                },
            )

            if not response3:
                self.logger.error("  ❌ O3 with codereview tool failed")
                return False

            self.logger.info("  ✅ O3 with codereview tool completed")

            # Validate model usage from server logs
            self.logger.info("  4: Validating model usage in logs")
            logs = self._filter_logs_since(self.get_recent_server_logs(lines=10000), log_start_time)

            openai_api_logs = [line for line in logs.split("\n") if "Sending request to openai API for" in line]
            openrouter_api_logs = [line for line in logs.split("\n") if "Sending request to openrouter API for" in line]
            openai_model_logs = [
                line for line in logs.split("\n") if "Using model:" in line and "openai provider" in line
            ]
            openrouter_model_logs = [
                line for line in logs.split("\n") if "Using model:" in line and "openrouter provider" in line
            ]
            chat_openai_logs = [line for line in logs.split("\n") if "Sending request to openai API for chat" in line]
            chat_openrouter_logs = [
                line for line in logs.split("\n") if "Sending request to openrouter API for chat" in line
            ]
            codereview_openai_logs = [
                line for line in logs.split("\n") if "Sending request to openai API for codereview" in line
            ]
            codereview_openrouter_logs = [
                line for line in logs.split("\n") if "Sending request to openrouter API for codereview" in line
            ]
            codereview_workflow_metadata_logs = [
                line for line in logs.split("\n") if "[WORKFLOW_METADATA] codereview: Added metadata" in line
            ]

            all_calls_completed = response1 is not None and response2 is not None and response3 is not None
            openai_routing_detected = len(openai_api_logs) >= 1 or len(openai_model_logs) >= 1
            openrouter_routing_detected = len(openrouter_api_logs) >= 1 or len(openrouter_model_logs) >= 1
            any_routing_detected = openai_routing_detected or openrouter_routing_detected
            chat_routed = len(chat_openai_logs) >= 1 or len(chat_openrouter_logs) >= 1
            workflow_routed = (
                len(codereview_openai_logs) >= 1
                or len(codereview_openrouter_logs) >= 1
                or len(codereview_workflow_metadata_logs) >= 1
                or self._response_indicates_workflow_routing(response3)
            )
            resolved_o3_model_detected = self._contains_o3_model_reference(openai_model_logs + openrouter_model_logs)

            self.logger.info(f"   OpenAI API call logs: {len(openai_api_logs)}")
            self.logger.info(f"   OpenRouter API call logs: {len(openrouter_api_logs)}")
            self.logger.info(f"   OpenAI model usage logs: {len(openai_model_logs)}")
            self.logger.info(f"   OpenRouter model usage logs: {len(openrouter_model_logs)}")
            self.logger.info(
                f"   Chat calls routed: OpenAI={len(chat_openai_logs)}, OpenRouter={len(chat_openrouter_logs)}"
            )
            self.logger.info(
                f"   Codereview calls routed: OpenAI={len(codereview_openai_logs)}, OpenRouter={len(codereview_openrouter_logs)}"
            )
            self.logger.info(f"   Codereview workflow metadata logs: {len(codereview_workflow_metadata_logs)}")
            self.logger.info(f"   O3/O3-mini model usage detected: {resolved_o3_model_detected}")

            success_criteria = [
                ("All O3/O3-mini calls completed", all_calls_completed),
                ("O3 routing detected in logs", any_routing_detected),
                ("Chat tool routed to a provider", chat_routed),
                ("Workflow tool routed to a provider", workflow_routed),
                ("Resolved model is o3/o3-mini", resolved_o3_model_detected),
            ]

            passed_criteria = sum(1 for _, passed in success_criteria if passed)
            self.logger.info(f"   Success criteria met: {passed_criteria}/{len(success_criteria)}")

            for criterion, passed in success_criteria:
                status = "✅" if passed else "❌"
                self.logger.info(f"    {status} {criterion}")

            if passed_criteria == len(success_criteria):
                if openai_routing_detected:
                    self.logger.info("  ✅ O3 model selection validation passed (direct OpenAI routing)")
                else:
                    self.logger.info("  ✅ O3 model selection validation passed (OpenRouter fallback routing)")
                return True

            self.logger.error("  ❌ O3 model selection validation failed")
            return False

        except Exception as e:
            self.logger.error(f"O3 model selection test failed: {e}")
            return False
        finally:
            self.cleanup_test_files()

    def _run_openrouter_o3_test(self) -> bool:
        """Test O3 model selection when using OpenRouter"""
        try:
            # Setup test files
            self.setup_test_files()
            log_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Test 1: O3 model via OpenRouter
            self.logger.info("  1: Testing O3 model via OpenRouter")

            response1, _ = self.call_mcp_tool(
                "chat",
                {
                    "prompt": "Simple test: What is 2 + 2? Just give a brief answer.",
                    "model": "o3",
                    "temperature": 1.0,
                },
            )

            if not response1:
                self.logger.error("  ❌ O3 model test via OpenRouter failed")
                return False

            self.logger.info("  ✅ O3 model call via OpenRouter completed")

            # Test 2: O3-mini model via OpenRouter
            self.logger.info("  2: Testing O3-mini model via OpenRouter")

            response2, _ = self.call_mcp_tool(
                "chat",
                {
                    "prompt": "Simple test: What is 3 + 3? Just give a brief answer.",
                    "model": "o3-mini",
                    "temperature": 1.0,
                },
            )

            if not response2:
                self.logger.error("  ❌ O3-mini model test via OpenRouter failed")
                return False

            self.logger.info("  ✅ O3-mini model call via OpenRouter completed")

            # Test 3: Codereview with O3 via OpenRouter
            self.logger.info("  3: Testing O3 with codereview tool via OpenRouter")

            test_code = """def add(a, b):
    return a + b

def multiply(x, y):
    return x * y
"""
            test_file = self.create_additional_test_file("simple_math.py", test_code)

            response3, _ = self.call_mcp_tool(
                "codereview",
                {
                    "step": "Review this simple code for quality and potential issues",
                    "step_number": 1,
                    "total_steps": 1,
                    "next_step_required": False,
                    "findings": "Starting code review analysis",
                    "relevant_files": [test_file],
                    "model": "o3",
                    "temperature": 1.0,
                },
            )

            if not response3:
                self.logger.error("  ❌ O3 with codereview tool via OpenRouter failed")
                return False

            self.logger.info("  ✅ O3 with codereview tool via OpenRouter completed")

            # Validate OpenRouter usage in logs
            self.logger.info("  4: Validating OpenRouter usage in logs")
            logs = self._filter_logs_since(self.get_recent_server_logs(lines=10000), log_start_time)

            # Check for OpenRouter API calls
            openrouter_api_logs = [
                line
                for line in logs.split("\n")
                if "openrouter" in line.lower() and ("API" in line or "request" in line)
            ]

            # Check for model resolution through OpenRouter
            openrouter_model_logs = [
                line for line in logs.split("\n") if "openrouter" in line.lower() and ("o3" in line or "model" in line)
            ]

            # Check for successful responses
            openrouter_response_logs = [
                line for line in logs.split("\n") if "openrouter" in line.lower() and "response" in line
            ]

            self.logger.info(f"   OpenRouter API logs: {len(openrouter_api_logs)}")
            self.logger.info(f"   OpenRouter model logs: {len(openrouter_model_logs)}")
            self.logger.info(f"   OpenRouter response logs: {len(openrouter_response_logs)}")

            # Success criteria for OpenRouter
            openrouter_used = len(openrouter_api_logs) >= 3 or len(openrouter_model_logs) >= 3
            all_calls_succeeded = response1 and response2 and response3

            success_criteria = [
                ("All O3 model calls succeeded", all_calls_succeeded),
                ("OpenRouter provider was used", openrouter_used),
            ]

            passed_criteria = sum(1 for _, passed in success_criteria if passed)
            self.logger.info(f"   Success criteria met: {passed_criteria}/{len(success_criteria)}")

            for criterion, passed in success_criteria:
                status = "✅" if passed else "❌"
                self.logger.info(f"    {status} {criterion}")

            if passed_criteria == len(success_criteria):
                self.logger.info("  ✅ O3 model selection via OpenRouter passed")
                return True
            else:
                self.logger.error("  ❌ O3 model selection via OpenRouter failed")
                return False

        except Exception as e:
            self.logger.error(f"OpenRouter O3 test failed: {e}")
            return False
        finally:
            self.cleanup_test_files()


def main():
    """Run the O3 model selection tests"""
    import sys

    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    test = O3ModelSelectionTest(verbose=verbose)

    success = test.run_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
