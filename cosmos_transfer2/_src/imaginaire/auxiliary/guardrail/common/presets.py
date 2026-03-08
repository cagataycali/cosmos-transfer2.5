# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

# Patch: allow imports to fail gracefully (e.g. when guardrails disabled)
try:
    from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.blocklist.blocklist import Blocklist
except ImportError:
    Blocklist = None

from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common.core import GuardrailRunner

try:
    from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.face_blur_filter.face_blur_filter import RetinaFaceFilter
except ImportError:
    RetinaFaceFilter = None

try:
    from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.qwen3guard.qwen3guard import Qwen3Guard
except ImportError:
    Qwen3Guard = None

try:
    from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.video_content_safety_filter.video_content_safety_filter import (
        VideoContentSafetyFilter,
    )
except ImportError:
    VideoContentSafetyFilter = None

from cosmos_transfer2._src.imaginaire.utils import log


def create_text_guardrail_runner(offload_model_to_cpu: bool = False) -> GuardrailRunner:
    """Create the text guardrail runner."""
    safety_models = []
    if Blocklist is not None:
        safety_models.append(Blocklist())
    if Qwen3Guard is not None:
        safety_models.append(Qwen3Guard(offload_model_to_cpu=offload_model_to_cpu))
    return GuardrailRunner(safety_models=safety_models)


def create_video_guardrail_runner(offload_model_to_cpu: bool = False) -> GuardrailRunner:
    """Create the video guardrail runner."""
    safety_models = []
    postprocessors = []
    if VideoContentSafetyFilter is not None:
        safety_models.append(VideoContentSafetyFilter(offload_model_to_cpu=offload_model_to_cpu))
    if RetinaFaceFilter is not None:
        postprocessors.append(RetinaFaceFilter(offload_model_to_cpu=offload_model_to_cpu))
    return GuardrailRunner(safety_models=safety_models, postprocessors=postprocessors)


def run_text_guardrail(prompt: str, guardrail_runner: GuardrailRunner) -> bool:
    """Run the text guardrail on the prompt, checking for content safety."""
    is_safe, message = guardrail_runner.run_safety_check(prompt)
    if not is_safe:
        log.critical(f"GUARDRAIL BLOCKED: {message}")
    return is_safe


def run_video_guardrail(frames: np.ndarray, guardrail_runner: GuardrailRunner) -> np.ndarray | None:
    """Run the video guardrail on the frames, checking for content safety and applying face blur."""
    is_safe, message = guardrail_runner.run_safety_check(frames)
    if not is_safe:
        log.critical(f"GUARDRAIL BLOCKED: {message}")
        return None

    frames = guardrail_runner.postprocess(frames)
    return frames
