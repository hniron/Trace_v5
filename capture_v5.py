#!/usr/bin/env python
"""方案B 采集：带"相位码"的 trace。

相位码（编码进 layer_id 字段）：
  0..47  = 真实各层
  100    = embed（输入词嵌入，层0之前）
  200    = output（final norm + lm_head + 采样 + 尾部 KV 管理）

这样 lm_head 在其 bf16 workspace 上的读也会被标成 output，层0-47 彻底干净。
同一次运行内 dump allocator 地图 + 抓 trace；只抓 decode（arm + seq_len 门控）。
"""

import json, os, time

START_FILE = "/tmp/hbm_v5_start"
ARM_FILE = "/tmp/hbm_v5_arm"
MAXTOK = int(os.environ.get("V5_MAXTOK", "20"))
NPROMPT = int(os.environ.get("V5_NPROMPT", "3"))
EMB, OUT = 100, 200

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number and explain it:",
    "Calculate the definite integral of x squared plus three x from zero to two:",
    "请用中文简要介绍长城的历史背景和它的象征意义。",
]


def main():
    from vllm import LLM, SamplingParams

    for p in (START_FILE, ARM_FILE):
        if os.path.exists(p):
            os.remove(p)

    print("加载模型...", flush=True)
    llm = LLM(model="/root/model/", trust_remote_code=True, dtype="auto",
              enforce_eager=True, tensor_parallel_size=1,
              gpu_memory_utilization=0.85, max_model_len=2048)

    # allocator 地图
    def dump_weights(model):
        ws = [[n, p.data_ptr(), p.numel() * p.element_size(),
               list(p.shape), str(p.dtype)] for n, p in model.named_parameters()]
        try:
            lt = list(getattr(model.config, "layer_types", []))
        except Exception:
            lt = []
        return {"weights": ws, "layer_types": lt}

    wr = llm.apply_model(dump_weights)[0]

    def dump_kv(self):
        mr = self.model_runner
        ivs = sorted((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
                     for t in mr.kv_caches)
        m = []
        for s, e in ivs:
            if m and s <= m[-1][1]:
                m[-1][1] = max(m[-1][1], e)
            else:
                m.append([s, e])
        return {"kv_ranges": m}

    kvr = llm.collective_rpc(dump_kv)[0]
    json.dump({"weights": wr["weights"], "layer_types": wr["layer_types"],
               "kv": kvr}, open("allocator_map.json", "w"))
    print("allocator 地图已存；权重 %.2fGB, KV %.2fGB" % (
        sum(w[2] for w in wr["weights"]) / 1e9,
        sum(e - s for s, e in kvr["kv_ranges"]) / 1e9), flush=True)

    # 相位 hook
    def setup(model):
        import os as _os
        import torch
        from torch.utils.cpp_extension import load_inline
        marker = load_inline(
            name="layer_marker_v5", cpp_sources="void launch_layer_marker(int code, unsigned long long stream);",
            cuda_sources="""#include <cuda_runtime.h>
extern "C" __global__ void layer_marker_kernel(int code){}
void launch_layer_marker(int code, unsigned long long stream){ layer_marker_kernel<<<1,1,0,(cudaStream_t)stream>>>(code); }
""", functions=["launch_layer_marker"], extra_cuda_cflags=["-O2"], verbose=False)

        def find(paths):
            for p in paths:
                o = model
                ok = True
                for a in p:
                    if hasattr(o, a):
                        o = getattr(o, a)
                    else:
                        ok = False
                        break
                if ok:
                    return o
            return None

        embed = find([("model", "embed_tokens"), ("embed_tokens",)])
        layers = find([("model", "layers"), ("model", "model", "layers"), ("layers",)])
        norm = find([("model", "norm"), ("norm",)])
        lmhead = find([("lm_head",)])
        got = dict(embed=embed is not None, layers=(len(layers) if layers else 0),
                   norm=norm is not None, lmhead=lmhead is not None)

        def stream():
            return torch.cuda.current_stream().cuda_stream

        def gate(seq):
            armed = _os.path.exists(ARM_FILE)
            if armed and seq == 1:
                if not _os.path.exists(START_FILE):
                    open(START_FILE, "w").close()
            else:
                if _os.path.exists(START_FILE):
                    _os.remove(START_FILE)

        def mkhook(code, is_embed=False):
            def hook(module, args, kwargs, _code=code, _emb=is_embed):
                if _emb:
                    x = kwargs.get("input_ids")
                    if x is None and args:
                        x = args[0]
                    seq = int(x.shape[0]) if hasattr(x, "shape") else -1
                    gate(seq)
                marker.launch_layer_marker(_code, int(stream()))
            return hook

        if embed is not None:
            embed.register_forward_pre_hook(mkhook(EMB, True), with_kwargs=True)
        for i, layer in enumerate(layers or []):
            layer.register_forward_pre_hook(mkhook(i), with_kwargs=True)
        if norm is not None:
            norm.register_forward_pre_hook(mkhook(OUT), with_kwargs=True)
        if lmhead is not None:
            lmhead.register_forward_pre_hook(mkhook(OUT), with_kwargs=True)
        return got

    print("hooks:", llm.apply_model(setup)[0], flush=True)

    llm.generate(["Hello"], SamplingParams(max_tokens=2, temperature=0.0))
    print("预热完成", flush=True)

    open(ARM_FILE, "w").close()
    print("arm，开始采集（decode，相位码 0-47/100/200）...", flush=True)
    for i, p in enumerate(PROMPTS[:NPROMPT]):
        t = time.time()
        o = llm.generate([p], SamplingParams(temperature=0.7, top_p=0.95, max_tokens=MAXTOK))
        print(f"[{i+1}/{NPROMPT}] {time.time()-t:.1f}s -> {o[0].outputs[0].text[:30]!r}", flush=True)

    for f in (START_FILE, ARM_FILE):
        if os.path.exists(f):
            os.remove(f)
    del llm
    print("采集完成", flush=True)


if __name__ == "__main__":
    main()
