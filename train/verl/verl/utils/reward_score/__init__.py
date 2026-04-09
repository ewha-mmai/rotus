# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    def _compute_hm_v5(solution, gt, info, prompt_val):
        from . import reward_score_hm_v5

        if isinstance(solution, (list, tuple)):
            merged_solution = "\n".join(str(item) for item in solution)
        else:
            merged_solution = str(solution)

        bridge_extra_info = dict(info or {})
        bridge_extra_info["question"] = bridge_extra_info.get("question") or (
            prompt_val if isinstance(prompt_val, str) else str(prompt_val)
        )
        if "is_unanswerable" not in bridge_extra_info:
            bridge_extra_info["is_unanswerable"] = (
                gt is None or (isinstance(gt, str) and len(gt.strip()) == 0)
            )

        tool_tag = bridge_extra_info.get("tool_tag", "grounding")
        max_assistant_turns = bridge_extra_info.get("max_generation_round", 5)

        return reward_score_hm_v5.compute_score(
            solution_str=merged_solution,
            ground_truth=gt,
            extra_info=bridge_extra_info,
            tool_tag=tool_tag,
            max_assistant_turns=max_assistant_turns,
        )

    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math_reward

        res = math_reward.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    elif data_source in [
        "vlm_multiturn",
        "vlm_tool_agent",
        "vstar_bench",
        "visual_probe_easy",
        "visual_probe_medium",
        "visual_probe_hard",
        "visual_probe_train",
        "deepeyes_train",
    ]:
        reward_fn = (extra_info or {}).get("general_qa_reward_fn", "hm_v5_trainverl")
        if reward_fn in ["hm_v5", "hm_v5_trainverl"]:
            res = _compute_hm_v5(solution_str, ground_truth, extra_info, kwargs.get("prompt", ""))
        else:
            from . import rotus

            res = rotus.compute_score(solution_str, ground_truth, extra_info)

    else:
        reward_fn = (extra_info or {}).get("general_qa_reward_fn", "")
        if reward_fn in ["hm_v5", "hm_v5_trainverl"]:
            res = _compute_hm_v5(solution_str, ground_truth, extra_info, kwargs.get("prompt", ""))
        else:
            raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
