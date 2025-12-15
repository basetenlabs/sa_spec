# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Baseten

import pytest

import torch

import sa_spec


def test_add_request():
    prompt = [1, 2, 3, 4, 1, 2, 3, 1, 2]
    sa_spec.add_request(1, prompt)
    sa_spec.prepare([1])

    depth_out = torch.zeros((1,), dtype=torch.int32, device="cuda")
    draft_out = torch.zeros((1, 2), dtype=torch.int32, device="cuda")

    extension = [3, 4, 5]
    extension_len = 2
    accepted_in = torch.tensor([extension], dtype=torch.int32, device="cuda")
    accepted_lens_in = torch.tensor([extension_len], dtype=torch.int32, device="cuda")

    sa_spec.extend(depth_out, draft_out, accepted_in, accepted_lens_in)

    assert depth_out.item() == 4
    assert draft_out.tolist() == [[1, 2]]

    tokens_out = torch.zeros((11,), dtype=torch.int32, device="cuda")
    sa_spec.get_active_tokens_for_test(1, tokens_out)
    assert tokens_out.tolist() == prompt + extension[:extension_len]


def test_cuda_graph():

    batch_size = 1
    draft_len = 2
    request_id = 152423

    new_tokens = torch.empty((batch_size, draft_len + 1), dtype=torch.int32, device="cuda")
    new_tokens_len = torch.empty((batch_size,), dtype=torch.int32, device="cuda")

    depth_out = torch.empty((batch_size,), dtype=torch.int32, device="cuda")
    draft_out = torch.empty((batch_size, draft_len), dtype=torch.int32, device="cuda")
    

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sa_spec.extend(depth_out, draft_out, new_tokens, new_tokens_len)
    
    prompt = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1]
    sa_spec.add_request(request_id, prompt)
    sa_spec.prepare([request_id])


    new_tokens.copy_(torch.tensor(
        [[2, 3, 0]],
        dtype=torch.int32,
        device="cuda")
    )

    new_tokens_len.copy_(torch.tensor(
        [2],
        dtype=torch.int32,
        device="cuda")
    )

    graph.replay()
    torch.cuda.synchronize()

    active_tokens = torch.zeros((len(prompt) + 2,), dtype=torch.int32, device="cuda")
    sa_spec.get_active_tokens_for_test(request_id, active_tokens)
    assert active_tokens.tolist() == prompt + [2, 3]
    assert draft_out.tolist() == [[4, 5]]



@pytest.mark.parametrize("max_batch_size,prompts,extensions,test_case_id", [
    (2, [[1, 2, 3], [4, 5, 6]], [[[7, 8], [9]], [[10], [11, 12]]], 0),
    (1, [[1, 2, 3]], [[[4, 5], [6], [7]]], 1),
    (5, [[1, 2], [3, 4], [5, 6]], [[[7], [8]], [[9], [10]], [[11], [12]]], 2),
    (3, [[1], [2], [3], [4]], [[[5, 6, 7]], [[8, 9]], [[10]], [[11, 12, 13, 14]]], 3),
    (2, [[1, 2], [3, 4], [5, 6], [7, 8]], [[[9], [10]], [[11], [12]], [[13], [14]], [[15], [16]]], 4),
    (4, [[i, i+1, i+2] for i in range(100)], [[[i+3, i+4], [i+5], [i+6]] for i in range(100)], 5),
])
def test_batching(max_batch_size, prompts, extensions, test_case_id):
    assert len(prompts) == len(extensions)

    # Use unique request IDs per test case to avoid conflicts with singleton APIImpl
    request_ids = [1000 * (test_case_id + 1) + i for i in range(len(prompts))]

    correct_text = {}


    while len(prompts) > 0:

        assert len(prompts) == len(extensions)

        finished_indices = [i for i in range(len(prompts)) if len(extensions[i]) == 0][::-1]

        for i in finished_indices:
            del correct_text[request_ids[i]]
            del prompts[i]
            del extensions[i]
            del request_ids[i]

        for i in range(len(prompts)):
            if len(correct_text) < max_batch_size and request_ids[i] not in correct_text:
                correct_text[request_ids[i]] = prompts[i]
                sa_spec.add_request(request_ids[i], prompts[i])

                assert extensions[i] != []

        sa_spec.prepare(list(correct_text.keys()))

        new_tokens = []
        new_tokens_lens = []

        for request_id in correct_text:
            i = request_ids.index(request_id)
            new_tokens.append(extensions[i][0].copy())
            new_tokens_lens.append(len(new_tokens[-1]))
            correct_text[request_ids[i]] += extensions[i][0]
            del extensions[i][0]
        
        if len(new_tokens) == 0:
            continue

        batch_size = len(new_tokens)
        draft_len = max(new_tokens_lens)

        # pad to draft_len+1
        for row in new_tokens:
            row += [0] * (draft_len + 1 - len(row))
        
        new_tokens = torch.tensor(new_tokens, dtype=torch.int32, device="cuda")
        new_tokens_lens = torch.tensor(new_tokens_lens, dtype=torch.int32, device="cuda")

        depth_out = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
        draft_out = torch.zeros((batch_size, draft_len), dtype=torch.int32, device="cuda")

        sa_spec.extend(depth_out, draft_out, new_tokens, new_tokens_lens)

        for request_id in correct_text:
            tokens_out = torch.zeros((len(correct_text[request_id]),), dtype=torch.int32, device="cuda")
            sa_spec.get_active_tokens_for_test(request_id, tokens_out)
            assert tokens_out.tolist() == correct_text[request_id]

