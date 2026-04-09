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

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive_multithreads_tool")
class NaiveMultiThreadsToolRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

        self.extra_info = dict(kwargs.get("extra_info", {}))
        self.gpt_threads = int(kwargs.get("gpt_threads", 64))

        self.overlong_buffer_len = int(self.extra_info.get("overlong_buffer_len", 0))
        self.max_total_response_length = int(self.extra_info.get("max_total_response_length", 0) or 0)

    def _extract_responses_list(
        self,
        input_ids: torch.Tensor,
        multi_turn_response_mask: torch.Tensor,
    ) -> list[str]:
        diff = torch.diff(
            multi_turn_response_mask,
            prepend=torch.tensor([0], device=multi_turn_response_mask.device),
        )
        starts = torch.where(diff == 1)[0]

        mask_appended = torch.cat([multi_turn_response_mask, torch.tensor([0], device=multi_turn_response_mask.device)], dim=0)
        diff_end = torch.diff(mask_appended)
        ends = torch.where(diff_end == -1)[0]

        segments = []
        for start_idx, end_idx in zip(starts, ends):
            segments.append(input_ids[start_idx : end_idx + 1].tolist())

        return self.tokenizer.batch_decode(segments, skip_special_tokens=True)

    def _process_single(self, args):
        index, data_item = args

        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        if "multi_turn_response_mask" in data_item.batch:
            response_list = self._extract_responses_list(
                data_item.batch["input_ids"],
                data_item.batch["multi_turn_response_mask"],
            )
            response_str = "\n".join(response_list)
        else:
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        non_tensor = data_item.non_tensor_batch
        ground_truth = non_tensor.get("ground_truth")
        if ground_truth is None:
            ground_truth = non_tensor.get("reward_model", {}).get("ground_truth", "")

        data_source = non_tensor.get(self.reward_fn_key, non_tensor.get("data_source", ""))

        sample_extra_info = dict(non_tensor.get("extra_info", {}))
        merged_extra_info = dict(self.extra_info)
        merged_extra_info.update(sample_extra_info)

        question = non_tensor.get("raw_prompt", "")

        result = self.compute_score(
            prompt=question,
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=merged_extra_info,
        )

        if isinstance(result, dict):
            score = float(result.get("score", 0.0))
            acc_score = float(result.get("acc", 0.0))
            format_score = float(result.get("format", 0.0))
        elif isinstance(result, (tuple, list)) and len(result) >= 3:
            score = float(result[0])
            acc_score = float(result[1])
            format_score = float(result[2])
            result = {"score": score, "acc": acc_score, "format": format_score}
        else:
            score = float(result)
            acc_score = 0.0
            format_score = 0.0
            result = {"score": score, "acc": acc_score, "format": format_score}

        if self.overlong_buffer_len > 0 and self.max_total_response_length > 0:
            expected_len = self.max_total_response_length - self.overlong_buffer_len
            exceed_len = int(valid_response_length) - int(expected_len)
            overlong_reward = min(-exceed_len / self.overlong_buffer_len, 0.0)
            score += overlong_reward
        else:
            overlong_reward = 0.0

        result["score"] = score
        result["acc"] = acc_score
        result["format"] = format_score
        result["overlong_reward"] = overlong_reward

        return (
            index,
            score,
            acc_score,
            format_score,
            overlong_reward,
            int(valid_response_length),
            data_source,
            prompt_str,
            response_str,
            ground_truth,
            result,
        )

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        n_threads = max(1, self.gpt_threads)
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(self._process_single, (i, data[i])) for i in range(len(data))]
            results = [future.result() for future in as_completed(futures)]

        results.sort(key=lambda item: item[0])

        for i, result in enumerate(results):
            (
                _,
                score,
                acc_score,
                format_score,
                overlong_score,
                valid_response_length,
                data_source,
                prompt_str,
                response_str,
                ground_truth,
                raw_result,
            ) = result

            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = score

            reward_extra_info["acc"].append(acc_score)
            reward_extra_info["format"].append(format_score)
            reward_extra_info["overlong_reward"].append(overlong_score)
            reward_extra_info["raw_result"].append(raw_result)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", (score, acc_score, format_score))

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
