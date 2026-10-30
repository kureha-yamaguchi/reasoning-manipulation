'''
Patch vLLM 0.10.1 so AWQ/GPTQ checkpoints of DeepSeek-V3/R1 load.

vLLM fuses q_a_proj and kv_a_proj_with_mqa into one MergedReplicatedLinear (fused_qkv_a_proj),
whose weight_loader slices shards along dim 0. Packed int4 params (AWQ qweight/qzeros [in, out/8],
scales [groups, out]) keep the output dimension on dim 1 and pack 8 values per element, so loading
fails with "expanded size of the tensor (264) must match the existing size (72)". The patch slices
along the param's own output_dim and divides offsets by packed_factor when that dim is packed.
FP8/BF16 checkpoints never reach the new branch (output_dim is 0, not packed). Idempotent.

uv run python bash_scripts/runpod/patch_vllm_awq_fused_qkv.py
'''

import importlib.util
import sys

MARKER = '# patched: reasoning-manipulation awq fused_qkv_a'
OLD = '''        else:
            shard_offset = sum(self.output_sizes[:loaded_shard_id])
            shard_size = self.output_sizes[loaded_shard_id]

        param[shard_offset:shard_offset + shard_size] = loaded_weight
'''
NEW = f'''        else:
            shard_offset = sum(self.output_sizes[:loaded_shard_id])
            shard_size = self.output_sizes[loaded_shard_id]
            {MARKER}
            output_dim = getattr(param, "output_dim", 0)
            if getattr(param, "packed_dim", None) == output_dim:
                shard_offset //= param.packed_factor
                shard_size //= param.packed_factor
            if output_dim != 0:
                param.data.narrow(output_dim, shard_offset, shard_size).copy_(loaded_weight)
                return

        param[shard_offset:shard_offset + shard_size] = loaded_weight
'''


def main():
    import vllm
    path = importlib.util.find_spec('vllm.model_executor.layers.linear').origin
    with open(path) as f:
        src = f.read()
    if MARKER in src:
        print(f'already patched: {path}')
        return
    if vllm.__version__ != '0.10.1' or src.count(OLD) != 1:
        sys.exit(f'unexpected vLLM source (version {vllm.__version__}); not patching {path}')
    with open(path, 'w') as f:
        f.write(src.replace(OLD, NEW))
    print(f'patched MergedReplicatedLinear.weight_loader in {path}')


if __name__ == '__main__':
    main()
