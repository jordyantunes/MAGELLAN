"""PoC microbenchmarks for the planned 4090 training-pipeline speedups.

Measures, in isolation, on the real model (flan-t5-base + 4 LoRA adapters, fp32,
eval mode — mirroring PeftInitializer and the lamorel forward paths):

  1. TF32 vs fp32 on the SAC-update forward shapes:
     - "critic-style" no-grad forward: 640 distinct contexts through encoder +
       1-token decoder (mirrors critic_target / policy_current_q_fwd)
     - "score-style" no-grad forward: 64 contexts encoded once, encoder output
       repeat-interleaved over 10 candidates, 640 decoder rows (mirrors
       target_fwd's score pass with pre_encode_inputs)
     - "score-style" forward+backward through LoRA params (mirrors policy_fwd)
  2. Pad-mask build: models.py per-token Python loop vs vectorized equality
     (asserts exact equivalence)
  3. torch.cuda.empty_cache() per-call cost (called 16x per update cycle today)
  4. LP-recompute chunking: short goal prompts scored in chunks of 256 vs 1024

Run:  .venv/bin/python scripts/poc_speedups.py
"""

import statistics

import torch
from peft import LoraConfig, get_peft_model
from transformers import T5ForConditionalGeneration

DEVICE = "cuda"
MODEL = "google/flan-t5-base"

N_STATES = 64          # gradient_batch_size in the tuned config
N_ACTIONS = 10         # avg possible actions per LittleZoo state
N_PAIRS = N_STATES * N_ACTIONS
CTX_LEN = 96           # typical LittleZoo prompt length (tokens)
DEC_LEN = 8            # candidate action length + pad/start
VOCAB = 32128


def set_tf32(on: bool):
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def cuda_time_ms(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def build_model():
    model = T5ForConditionalGeneration.from_pretrained(MODEL).to(DEVICE)
    model.gradient_checkpointing_enable()
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q", "v"],
                     lora_dropout=0.0, bias="none", task_type="SEQ_2_SEQ_LM")
    peft_model = get_peft_model(model, cfg)
    for extra in ["delayed_adapters", "sr_adapters", "critic_target"]:
        peft_model.add_adapter(extra, cfg)
    peft_model.set_adapter("default")
    for name, module in peft_model.named_modules():
        if "lm_head" in name and hasattr(module, "weight"):
            module.requires_grad_(False)
    # lamorel/PeftInitializer leaves the model in eval() and never calls train(),
    # so HF gradient checkpointing is inactive in practice — reproduce that.
    peft_model.eval()
    peft_model.config.use_cache = True
    return peft_model


def make_batch(n, ctx_len, dec_len):
    g = torch.Generator(device="cpu").manual_seed(0)
    ctx_ids = torch.randint(3, VOCAB, (n, ctx_len), generator=g).to(DEVICE)
    ctx_mask = torch.ones(n, ctx_len, dtype=torch.long, device=DEVICE)
    dec_ids = torch.randint(3, VOCAB, (n, dec_len), generator=g).to(DEVICE)
    dec_ids[:, 0] = 0  # decoder start (pad) token, as in __build_encoder_decoder_minibatch
    dec_mask = torch.ones(n, dec_len, dtype=torch.long, device=DEVICE)
    return ctx_ids, ctx_mask, dec_ids, dec_mask


def bench_critic_style(model):
    """640 distinct contexts -> encoder + 1-token decoder (critic_target shape)."""
    ctx_ids, ctx_mask, _, _ = make_batch(N_PAIRS, CTX_LEN, DEC_LEN)
    dec_ids = torch.zeros(N_PAIRS, 2, dtype=torch.long, device=DEVICE)
    dec_mask = torch.ones(N_PAIRS, 2, dtype=torch.long, device=DEVICE)

    def run():
        with torch.no_grad():
            model(input_ids=ctx_ids, attention_mask=ctx_mask,
                  decoder_input_ids=dec_ids, decoder_attention_mask=dec_mask,
                  output_hidden_states=True)

    return cuda_time_ms(run)


def bench_score_style(model, grad: bool):
    """64 contexts encoded once, repeated x10, 640 decoder rows (score shape)."""
    ctx_ids, ctx_mask, _, _ = make_batch(N_STATES, CTX_LEN, DEC_LEN)
    _, _, dec_ids, dec_mask = make_batch(N_PAIRS, CTX_LEN, DEC_LEN)
    mask_rep = ctx_mask.repeat_interleave(N_ACTIONS, dim=0)

    def run():
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            enc = model.encoder(input_ids=ctx_ids, attention_mask=ctx_mask,
                                return_dict=False)[0]
            enc_rep = enc.repeat_interleave(N_ACTIONS, dim=0)
            out = model(encoder_outputs=(enc_rep,), attention_mask=mask_rep,
                        decoder_input_ids=dec_ids, decoder_attention_mask=dec_mask)
            if grad:
                logits = out["logits"][:, :-1, :]
                tokens = dec_ids[:, 1:]
                logprobs = torch.gather(logits, 2, tokens[:, :, None]).squeeze(-1)
                loss = logprobs.sum()
                loss.backward()
                model.zero_grad(set_to_none=True)

    return cuda_time_ms(run, warmup=2, iters=5 if grad else 10)


def bench_mask():
    """models.py:40-45 loop vs vectorized; assert exact equivalence."""
    g = torch.Generator(device="cpu").manual_seed(1)
    output_tokens = torch.randint(0, 5, (N_PAIRS, DEC_LEN - 1), generator=g).to(DEVICE)
    pad_token = 0

    def loop_mask():
        mask = torch.ones(output_tokens.shape, dtype=torch.bool, device=DEVICE)
        for i, _output in enumerate(output_tokens):
            for j, _token in enumerate(_output):
                if _token != pad_token:
                    mask[i, j] = False
        return mask

    def vec_mask():
        return output_tokens == pad_token

    assert torch.equal(loop_mask(), vec_mask()), "mask implementations differ!"
    loop_ms = cuda_time_ms(loop_mask, warmup=1, iters=3)
    vec_ms = cuda_time_ms(vec_mask, warmup=3, iters=10)
    return loop_ms, vec_ms


def bench_empty_cache(model):
    """Cost of empty_cache after an update-like alloc churn."""
    ctx_ids, ctx_mask, dec_ids, dec_mask = make_batch(N_PAIRS, CTX_LEN, DEC_LEN)

    def churn_and_empty():
        with torch.no_grad():
            model(input_ids=ctx_ids, attention_mask=ctx_mask,
                  decoder_input_ids=dec_ids, decoder_attention_mask=dec_mask)
        torch.cuda.empty_cache()

    def churn_only():
        with torch.no_grad():
            model(input_ids=ctx_ids, attention_mask=ctx_mask,
                  decoder_input_ids=dec_ids, decoder_attention_mask=dec_mask)

    with_ms = cuda_time_ms(churn_and_empty, warmup=2, iters=8)
    without_ms = cuda_time_ms(churn_only, warmup=2, iters=8)
    return max(with_ms - without_ms, 0.0)


def bench_lp_chunks(model, chunk):
    """2560 short goal prompts, encoder + 1-token decoder, chunked."""
    n = 2560
    ctx_ids, ctx_mask, _, _ = make_batch(n, 32, 2)
    dec_ids = torch.zeros(n, 2, dtype=torch.long, device=DEVICE)
    dec_mask = torch.ones(n, 2, dtype=torch.long, device=DEVICE)

    def run():
        with torch.no_grad():
            for i in range(0, n, chunk):
                model(input_ids=ctx_ids[i:i + chunk],
                      attention_mask=ctx_mask[i:i + chunk],
                      decoder_input_ids=dec_ids[i:i + chunk],
                      decoder_attention_mask=dec_mask[i:i + chunk],
                      output_hidden_states=True)

    return cuda_time_ms(run, warmup=2, iters=5)


def main():
    assert torch.cuda.is_available()
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"building {MODEL} + 4 LoRA adapters (fp32, eval mode)...")
    model = build_model()

    results = []

    for name, fn in [
        ("critic-style no-grad fwd (640 ctx enc + dec1)", lambda: bench_critic_style(model)),
        ("score-style  no-grad fwd (64 enc -> 640 dec)", lambda: bench_score_style(model, grad=False)),
        ("score-style  fwd+bwd     (LoRA grads)", lambda: bench_score_style(model, grad=True)),
    ]:
        set_tf32(False)
        fp32_ms = fn()
        set_tf32(True)
        tf32_ms = fn()
        results.append((name, fp32_ms, tf32_ms, fp32_ms / tf32_ms))

    print(f"\n{'benchmark':<48}{'fp32 ms':>10}{'tf32 ms':>10}{'speedup':>9}")
    for name, a, b, s in results:
        print(f"{name:<48}{a:>10.1f}{b:>10.1f}{s:>8.2f}x")

    set_tf32(False)
    loop_ms, vec_ms = bench_mask()
    print(f"\nmask build ({N_PAIRS}x{DEC_LEN - 1}): loop {loop_ms:.1f} ms vs "
          f"vectorized {vec_ms:.3f} ms ({loop_ms / max(vec_ms, 1e-6):.0f}x) — equivalence OK")

    ec_ms = bench_empty_cache(model)
    print(f"empty_cache marginal cost: {ec_ms:.1f} ms/call "
          f"-> ~{ec_ms * 16 / 1000:.2f} s/cycle at 16 calls/cycle")

    set_tf32(True)
    t256 = bench_lp_chunks(model, 256)
    t1024 = bench_lp_chunks(model, 1024)
    scale = 2 * 25000 / 2560  # full LP recompute is ~25k goals x 2 adapters
    print(f"LP recompute (2560 short prompts, tf32): chunk 256 {t256:.0f} ms vs "
          f"chunk 1024 {t1024:.0f} ms ({t256 / t1024:.2f}x) "
          f"-> projected full recompute {t256 * scale / 1000:.1f} s vs {t1024 * scale / 1000:.1f} s")

    print(f"\npeak VRAM during benches: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
