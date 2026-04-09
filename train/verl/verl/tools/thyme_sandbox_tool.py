# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Thyme Sandbox Tool for VeRL.
"""

import logging
import os
import pickle
import re
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.sandbox import execute_code_in_sandbox
from verl.utils.rollout_trace import rollout_trace_op

from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ThymeSandboxTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Configuration
        self.cache_dir = os.environ.get("THYME_CACHE_DIR") or config.get("cache_dir", "/tmp/thyme_sandbox_cache")
        self.exec_time_limit = config.get("exec_time_limit", 10)
        self.max_images_per_round = config.get("max_images_per_round", 10)

        os.makedirs(self.cache_dir, exist_ok=True)
        log_msg = f"Init ThymeSandboxTool with cache_dir={self.cache_dir}"
        logger.info(log_msg)
        print(f"[ThymeSandboxTool] {log_msg}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, create_kwargs: Optional[dict] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        """
        Create a new sandbox session instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = create_kwargs or {}

        self._instance_dict[instance_id] = {
            "execution_context": None,
            "image_data": create_kwargs.get("image_data"),
            "image_path": create_kwargs.get("image_path"),
            "case_id": create_kwargs.get("case_id", instance_id),
            "turn_count": 0,
        }

        logger.info(f"[ThymeSandboxTool] Created session {instance_id}")
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], agent_data: Any = None, **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")

        if not isinstance(code, str):
            code = str(code)

        if not code.strip():
            error_text = "Sandbox for unknown: Empty code provided"
            return ToolResponse(text=f"<sandbox_output>{error_text}</sandbox_output>"), 0.0, {}

        # Get session data
        if instance_id not in self._instance_dict:
            error_text = "Sandbox for unknown: Invalid session ID"
            return ToolResponse(text=f"<sandbox_output>{error_text}</sandbox_output>"), 0.0, {}

        session_data = self._instance_dict[instance_id]
        case_id = session_data["case_id"]
        turn_count = session_data["turn_count"]
        previous_execution_context = session_data["execution_context"]

        image_path = session_data.get("image_path")
        if not image_path and agent_data and hasattr(agent_data, "image_data"):
            image_data = agent_data.image_data
            if image_data and len(image_data) > 0:
                first_image = image_data[0]
                if isinstance(first_image, dict) and "path" in first_image:
                    image_path = first_image["path"]
                elif hasattr(first_image, "save"):
                    from PIL import Image as PILImage
                    if isinstance(first_image, PILImage.Image):
                        saved_path = os.path.join(self.cache_dir, f"{case_id}_input.jpg")
                        first_image.save(saved_path)
                        image_path = saved_path
                        logger.info(f"[ThymeSandboxTool] Saved PIL image to {saved_path}")
            if image_path:
                session_data["image_path"] = image_path

        if not image_path or not os.path.exists(image_path):
            logger.warning(f"[ThymeSandboxTool] No valid image_path found for {case_id}")

        temp_output_dir = os.path.join(
            self.cache_dir,
            f"{case_id}_turn{turn_count}"
        )
        os.makedirs(temp_output_dir, exist_ok=True)

        try:
            processed_img_paths, captured_stdout, error_msg, current_execution_context = \
                execute_code_in_sandbox(
                    code_to_execute=code,
                    input_image_path=image_path,
                    temp_output_dir=temp_output_dir,
                    item_id=case_id,
                    previous_execution_context=previous_execution_context
                )

            if current_execution_context is not None:
                session_data["execution_context"] = current_execution_context
            session_data["turn_count"] += 1

        except Exception as e:
            logger.error(f"[ThymeSandboxTool] Execution failed: {e}")
            # Match original behavior: skip (return empty ToolResponse) on failure
            return ToolResponse(), 0.0, {}

        if not processed_img_paths:
            logger.warning(f"[ThymeSandboxTool] No output from sandbox: {error_msg}")
            return ToolResponse(), 0.0, {}

        first_path = processed_img_paths[0]

        if os.path.exists(first_path):
            image_list = []
            for img_path in processed_img_paths[:self.max_images_per_round]:
                if os.path.exists(img_path):
                    if self._check_aspect_ratio(img_path):
                        image_list.append({"bytes": None, "path": os.path.abspath(img_path)})
                    else:
                        logger.warning(f"[ThymeSandboxTool] Invalid aspect ratio: {img_path}")
                        response_text = f"<sandbox_output>Sandbox for {case_id}: The Generated img has invalid aspect_ratio</sandbox_output>"
                        return ToolResponse(text=response_text), 0.0, {}

            if image_list:
                session_data["image_path"] = os.path.abspath(processed_img_paths[0])
                content = [{"type": "text", "text": "<sandbox_output>"}]
                for _ in image_list:
                    content.append({"type": "image"})
                content.append({"type": "text", "text": "</sandbox_output>"})
                return ToolResponse(content=content, image=image_list), 0.0, {"images": image_list}

            return ToolResponse(), 0.0, {}
        else:
            response_text = f"<sandbox_output>{first_path}</sandbox_output>"
            return ToolResponse(text=response_text), 0.0, {}

    def _check_aspect_ratio(self, img_path: str) -> bool:
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                width, height = img.size
                aspect_ratio = max(width, height) / min(width, height)
                return aspect_ratio <= 200
        except Exception as e:
            logger.error(f"[ThymeSandboxTool] Error checking aspect ratio: {e}")
            return False

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward (not used for this tool)."""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the session instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
            logger.info(f"[ThymeSandboxTool] Released session {instance_id}")
