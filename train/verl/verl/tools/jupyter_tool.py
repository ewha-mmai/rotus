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
Jupyter Tool for DeepEyesV2
"""

import base64
import hashlib
import logging
import os
import random
import time
from io import BytesIO
from typing import Any, Optional
from uuid import uuid4

import requests
from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.utils.rollout_trace import rollout_trace_op

from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Image initialization code template
INITIALIZATION_CODE_TEMPLATE = """
from PIL import Image
import base64
from io import BytesIO

def _fix_base64_padding(b64_str):
    # Remove data URL prefix if present
    if b64_str.startswith('data:'):
        b64_str = b64_str.split(',', 1)[-1]
    # Remove whitespace
    b64_str = b64_str.strip().replace('\\n', '').replace('\\r', '').replace(' ', '')
    # Fix padding
    padding_needed = len(b64_str) % 4
    if padding_needed:
        b64_str += '=' * (4 - padding_needed)
    return b64_str

_img_base64 = "{base64_image}"
_img_base64 = _fix_base64_padding(_img_base64)
image_1 = Image.open(BytesIO(base64.b64decode(_img_base64)))
"""


def pil_image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    img_bytes = buffer.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64


def base64_to_pil_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    # Remove data URL prefix if present
    if base64_string.startswith('data:'):
        base64_string = base64_string.split(',', 1)[-1]

    # Remove whitespace and newlines
    base64_string = base64_string.strip().replace('\n', '').replace('\r', '').replace(' ', '')

    # Fix padding
    padding_needed = len(base64_string) % 4
    if padding_needed:
        base64_string += '=' * (4 - padding_needed)

    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


def generate_session_id() -> str:
    """Generate a unique session ID."""
    salted_str = str(int(time.time())) + str(random.randint(10000, 99999))
    salted_hash_str = str(hex(hash(salted_str.encode('utf-8')))).split('0x')[-1]
    return salted_hash_str


class JupyterTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Configuration
        self.jupyter_url = config.get("jupyter_url", "http://localhost:8000/run_jupyter")
        self.backend_urls = config.get("backend_urls", None)  # e.g. ["http://localhost:28901", ...]
        self.default_timeout = config.get("default_timeout", 200)
        self.request_timeout = config.get("request_timeout", 240)
        self.max_images_per_round = config.get("max_images_per_round", 10)
        self.enable_image_init = config.get("enable_image_init", True)

        log_msg = f"Init JupyterTool with config: jupyter_url={self.jupyter_url}, timeout={self.default_timeout}"
        logger.info(log_msg)
        print(f"[JupyterTool] {log_msg}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, create_kwargs: Optional[dict] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        """
        Create a new Jupyter session instance.

        If image data is provided in create_kwargs, initialize image_1 variable.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Reuse existing session_id if provided (for multi-turn kernel state persistence)
        create_kwargs = create_kwargs or {}
        provided_session_id = create_kwargs.get("session_id")
        already_initialized = create_kwargs.get("already_initialized", False)
        session_id = provided_session_id if provided_session_id else generate_session_id()

        self._instance_dict[instance_id] = {
            "session_id": session_id,
            "initialized": already_initialized,
            "image_data": None,
        }

        # Initialize image only for new sessions
        image_data = create_kwargs.get("image_data")

        if not already_initialized and image_data and self.enable_image_init:
            init_result = await self._initialize_image(instance_id, image_data)
            if init_result:
                self._instance_dict[instance_id]["initialized"] = True
                self._instance_dict[instance_id]["image_data"] = image_data
                logger.info(f"[JupyterTool] Session {session_id} initialized with image")
            else:
                logger.warning(f"[JupyterTool] Failed to initialize image for session {session_id}")

        return instance_id, ToolResponse()

    async def _initialize_image(self, instance_id: str, image_data: Any) -> bool:
        """Initialize image_1 variable in the Jupyter session."""
        try:
            # Convert image to base64
            if isinstance(image_data, Image.Image):
                base64_image = pil_image_to_base64(image_data)
            elif isinstance(image_data, str):
                # Assume it's already base64
                base64_image = image_data
            elif isinstance(image_data, list) and len(image_data) > 0:
                # Take the first image
                if isinstance(image_data[0], Image.Image):
                    base64_image = pil_image_to_base64(image_data[0])
                else:
                    base64_image = image_data[0]
            else:
                logger.warning(f"[JupyterTool] Unsupported image_data type: {type(image_data)}")
                return False

            # Generate initialization code
            init_code = INITIALIZATION_CODE_TEMPLATE.format(base64_image=base64_image)

            # Execute initialization code
            session_id = self._instance_dict[instance_id]["session_id"]
            result = self._request_jupyter_execution(session_id, init_code)

            if result and result.get("status") == "success":
                return True
            else:
                logger.warning(f"[JupyterTool] Image initialization failed: {result}")
                return False

        except Exception as e:
            logger.error(f"[JupyterTool] Error initializing image: {e}")
            return False

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], agent_data: Any = None, **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """
        Execute Python code in the Jupyter session.

        Args:
            instance_id: The session instance ID
            parameters: Dict with 'code' key containing Python code to execute
            agent_data: Optional agent data containing image information

        Returns:
            Tuple of (ToolResponse, reward, metrics)
        """
        code = parameters.get("code", "")
        timeout = parameters.get("timeout", self.default_timeout)

        if not isinstance(code, str):
            code = str(code)

        if not code.strip():
            return ToolResponse(text="Error: Empty code provided"), 0.0, {}

        # Get session ID
        if instance_id not in self._instance_dict:
            return ToolResponse(text="Error: Invalid session ID"), 0.0, {}

        session_id = self._instance_dict[instance_id]["session_id"]

        # Initialize image from agent_data on first execution
        if agent_data and not self._instance_dict[instance_id]["initialized"] and self.enable_image_init:
            image_data = getattr(agent_data, "image_data", None)
            if image_data and len(image_data) > 0:
                logger.info(f"[JupyterTool] Initializing image for session {session_id}")
                init_success = await self._initialize_image(instance_id, image_data)
                if init_success:
                    self._instance_dict[instance_id]["initialized"] = True
                    logger.info(f"[JupyterTool] Image initialized successfully for session {session_id}")
                else:
                    logger.warning(f"[JupyterTool] Failed to initialize image for session {session_id}")

        # Execute code
        result = self._request_jupyter_execution(session_id, code, timeout)

        if not result:
            return ToolResponse(text="Error: Code execution failed - no response"), 0.0, {}

        if result.get("status") != "success":
            error_msg = result.get("stderr", "") or result.get("result", "Unknown error")
            return ToolResponse(text=f"Error: {error_msg}"), 0.0, {}

        # Build response text
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        images = result.get("images", [])

        response_parts = []
        if stdout:
            response_parts.append(f"Output:\n{stdout}")
        if stderr:
            response_parts.append(f"Stderr:\n{stderr}")
        if images:
            response_parts.append(f"Generated {len(images)} image(s)")

        response_text = "\n".join(response_parts) if response_parts else "Code executed successfully (no output)"

        # Handle generated images
        if images:
            image_list = images[:self.max_images_per_round]
            return ToolResponse(text=response_text, image=image_list), 0.0, {"images": image_list}

        return ToolResponse(text=response_text), 0.0, {}

    def _get_backend_url(self, session_id: str) -> str:
        """Select backend URL consistently based on session_id hash (avoids nginx round-robin session mismatch)."""
        if self.backend_urls:
            idx = int(hashlib.md5(session_id.encode()).hexdigest(), 16) % len(self.backend_urls)
            return self.backend_urls[idx].rstrip("/") + "/run_jupyter"
        return self.jupyter_url

    def _request_jupyter_execution(
        self, session_id: str, code: str, timeout: int = None
    ) -> Optional[dict]:
        """
        Send code execution request to Jupyter server.

        Compatible with DeepEyesV2's multimodal-ipython-sandbox API.
        """
        timeout = timeout or self.default_timeout
        url = self._get_backend_url(session_id)

        try:
            response = requests.post(
                url,
                json={
                    "session_id": session_id,
                    "code": code,
                    "timeout": timeout
                },
                timeout=self.request_timeout
            )
            response.raise_for_status()
            resjson = response.json()
            result_dict = resjson.get('output', {})

        except requests.exceptions.RequestException as e:
            logger.error(f"[JupyterTool] Request to Jupyter server failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[JupyterTool] Unexpected error: {e}")
            return None

        # Process images
        image_pil_list = []
        image_base64_list = result_dict.get("images", [])
        for idx, img_b64 in enumerate(image_base64_list):
            try:
                img_pil = base64_to_pil_image(img_b64)
                img_pil = self._maybe_resize_image(img_pil)
                image_pil_list.append(img_pil)
            except Exception as e:
                logger.warning(f"[JupyterTool] Failed to decode image {idx}: {e}")
                continue

        return {
            "status": resjson.get("status", "error"),
            "execution_time": resjson.get("execution_time", -1.0),
            "result": result_dict.get("result", ""),
            "stdout": result_dict.get("stdout", ""),
            "stderr": result_dict.get("stderr", ""),
            "images": image_pil_list,
        }

    def _maybe_resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image if needed (Qwen-VL requires min 32px).
        """
        height, width = image.height, image.width

        # Handle extreme aspect ratios
        if max(height, width) / min(height, width) > 200:
            max_val = max(height, width)
            min_val = min(height, width)
            old_scale = max_val / min_val
            max_ratio = min(150, old_scale / 2)
            target_max = int(min_val * max_ratio)

            if height > width:
                new_height = target_max
                new_width = int(width * old_scale / max_ratio)
            else:
                new_width = target_max
                new_height = int(height * old_scale / max_ratio)

            image = image.resize((int(new_width), int(new_height)), Image.LANCZOS)
            height, width = image.height, image.width

        # Ensure minimum size
        if min(height, width) < 32:
            scale = 32 / min(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)

        return image

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward (not used for this tool)."""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the session instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
