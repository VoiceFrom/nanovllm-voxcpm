import copy

import pytest
import torch
from torch import nn

from nanovllm_voxcpm.layers.streaming_vae import BatchedStreamingVAEDecoder


@pytest.mark.parametrize(
    "module_name",
    [
        "nanovllm_voxcpm.layers.audio_vae",
        "nanovllm_voxcpm.layers.audio_vae_v2",
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
    ],
)
def test_streaming_vae_matches_full_decode_with_dynamic_batch_order(module_name, device):
    module = __import__(module_name, fromlist=["CausalConv1d", "CausalTransposeConv1d"])
    causal_conv = module.CausalConv1d
    causal_transpose_conv = module.CausalTransposeConv1d

    class TinyVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Sequential(
                causal_conv(2, 4, kernel_size=3, padding=1),
                nn.SiLU(),
                causal_transpose_conv(4, 1, kernel_size=4, stride=2, padding=1),
            )

        def decode(self, z):
            return self.decoder(z)

    torch.manual_seed(0)
    streaming_vae = TinyVAE().to(device)
    reference_vae = copy.deepcopy(streaming_vae)
    decoder = BatchedStreamingVAEDecoder(streaming_vae, causal_conv, causal_transpose_conv)

    a_chunks = [torch.randn(1, 2, 2, device=device) for _ in range(3)]
    b_chunks = [torch.randn(1, 2, 2, device=device) for _ in range(3)]
    expected_a = reference_vae.decode(torch.cat(a_chunks, dim=-1))
    expected_b = reference_vae.decode(torch.cat(b_chunks, dim=-1))

    first = decoder.decode_chunks(torch.cat([a_chunks[0], b_chunks[0]], dim=0), ["a", "b"])
    second = decoder.decode_chunks(torch.cat([b_chunks[1], a_chunks[1]], dim=0), ["b", "a"])
    third = decoder.decode_chunks(torch.cat([a_chunks[2], b_chunks[2]], dim=0), ["a", "b"])
    actual_a = torch.cat([first[0:1], second[1:2], third[0:1]], dim=-1)
    actual_b = torch.cat([first[1:2], second[0:1], third[1:2]], dim=-1)

    torch.testing.assert_close(actual_a, expected_a, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_b, expected_b, rtol=1e-5, atol=1e-6)


def test_streaming_vae_pads_to_fixed_batch_bucket_and_cleans_dummy_state():
    from nanovllm_voxcpm.layers.audio_vae import CausalConv1d, CausalTransposeConv1d

    class TinyVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Sequential(CausalConv1d(1, 1, kernel_size=3, padding=1))
            self.seen_batch_sizes = []

        def decode(self, z):
            self.seen_batch_sizes.append(z.size(0))
            return self.decoder(z)

    torch.manual_seed(4)
    vae = TinyVAE()
    reference_vae = copy.deepcopy(vae)
    decoder = BatchedStreamingVAEDecoder(
        vae,
        CausalConv1d,
        CausalTransposeConv1d,
        max_batch_size=4,
    )
    chunks = torch.randn(3, 1, 2)
    stream_ids = ["a", "b", "c"]

    actual = decoder.decode_chunks(chunks, stream_ids)

    torch.testing.assert_close(actual, reference_vae.decode(chunks))
    assert vae.seen_batch_sizes == [4]
    for layer_states in decoder._states.values():
        assert set(layer_states) == set(stream_ids)


def test_streaming_vae_warmup_initializes_all_buckets_without_retaining_state():
    from nanovllm_voxcpm.layers.audio_vae import CausalConv1d, CausalTransposeConv1d

    class TinyVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Sequential(CausalConv1d(1, 1, kernel_size=3, padding=1))
            self.seen_batch_sizes = []

        def decode(self, z):
            self.seen_batch_sizes.append(z.size(0))
            return self.decoder(z)

    vae = TinyVAE()
    decoder = BatchedStreamingVAEDecoder(
        vae,
        CausalConv1d,
        CausalTransposeConv1d,
        max_batch_size=4,
    )

    decoder.warmup(latent_channels=1, chunk_size=2)

    assert vae.seen_batch_sizes == [1, 2, 4]
    assert not decoder._initialized_streams
    assert all(not layer_states for layer_states in decoder._states.values())


def test_streaming_vae_primes_context_once_and_releases_state():
    from nanovllm_voxcpm.layers.audio_vae import CausalConv1d, CausalTransposeConv1d

    class TinyVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Sequential(
                CausalConv1d(1, 2, kernel_size=3, padding=1),
                CausalTransposeConv1d(2, 1, kernel_size=4, stride=2, padding=1),
            )

        def decode(self, z):
            return self.decoder(z)

    torch.manual_seed(1)
    streaming_vae = TinyVAE()
    reference_vae = copy.deepcopy(streaming_vae)
    decoder = BatchedStreamingVAEDecoder(streaming_vae, CausalConv1d, CausalTransposeConv1d)
    context = torch.randn(1, 1, 3)
    chunk = torch.randn(1, 1, 2)

    expected = reference_vae.decode(torch.cat([context, chunk], dim=-1))[..., -4:]
    first = decoder.decode_chunks(chunk, ["request"], [context])
    decoder.release("request")
    second = decoder.decode_chunks(chunk, ["request"], [context])

    torch.testing.assert_close(first, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(second, expected, rtol=1e-5, atol=1e-6)


def test_streaming_vae_rejects_duplicate_stream_ids():
    from nanovllm_voxcpm.layers.audio_vae import CausalConv1d, CausalTransposeConv1d

    class TinyVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Sequential(CausalConv1d(1, 1, kernel_size=3, padding=1))

        def decode(self, z):
            return self.decoder(z)

    decoder = BatchedStreamingVAEDecoder(TinyVAE(), CausalConv1d, CausalTransposeConv1d)
    with pytest.raises(ValueError, match="unique"):
        decoder.decode_chunks(torch.zeros(2, 1, 1), ["same", "same"])


def test_streaming_vae_preserves_regular_decode_path():
    from nanovllm_voxcpm.layers.audio_vae import CausalConv1d, CausalTransposeConv1d

    class TinyVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Sequential(
                CausalConv1d(1, 2, kernel_size=3, padding=1),
                CausalTransposeConv1d(2, 1, kernel_size=4, stride=2, padding=1),
            )

        def decode(self, z):
            return self.decoder(z)

    torch.manual_seed(2)
    vae = TinyVAE()
    reference_vae = copy.deepcopy(vae)
    BatchedStreamingVAEDecoder(vae, CausalConv1d, CausalTransposeConv1d)
    inputs = torch.randn(1, 1, 4)

    torch.testing.assert_close(vae.decode(inputs), reference_vae.decode(inputs))


def test_streaming_vae_failed_decode_does_not_advance_state():
    from nanovllm_voxcpm.layers.audio_vae import CausalConv1d, CausalTransposeConv1d

    class FailOnce(nn.Module):
        def __init__(self):
            super().__init__()
            self.should_fail = True

        def forward(self, x):
            if self.should_fail:
                self.should_fail = False
                raise RuntimeError("decode failed")
            return x

    class TinyVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.fail_once = FailOnce()
            self.decoder = nn.Sequential(
                CausalConv1d(1, 1, kernel_size=3, padding=1),
                self.fail_once,
            )

        def decode(self, z):
            return self.decoder(z)

    torch.manual_seed(3)
    vae = TinyVAE()
    reference_vae = copy.deepcopy(vae)
    reference_vae.fail_once.should_fail = False
    decoder = BatchedStreamingVAEDecoder(vae, CausalConv1d, CausalTransposeConv1d)
    chunk = torch.randn(1, 1, 2)

    with pytest.raises(RuntimeError, match="decode failed"):
        decoder.decode_chunks(chunk, ["request"])
    actual = decoder.decode_chunks(chunk, ["request"])

    torch.testing.assert_close(actual, reference_vae.decode(chunk))
