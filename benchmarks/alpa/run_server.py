import argparse
import asyncio
from collections import namedtuple
import logging
import math
import time

from alpa import (PipeshardParallel, get_global_cluster,
                  get_global_virtual_physical_mesh,
                  set_global_virtual_physical_mesh,
                  AutoShardingOption, ManualStageOption,
                  parallelize)
from alpa_serve import run_controller
from alpa_serve.placement_policy import (
    SelectiveReplication, ModelData, ParallelConfig)
from alpa.util import compute_bytes, compute_param_number, GB
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_leaves, tree_map
import numpy as np
import ray

from benchmarks.alpa.util import build_logger

BertModelConfig = namedtuple(
    "BertModelConfig",
    ["seq_len", "hidden_size", "num_layers", "num_heads", "vocab_size"])


bert_specs = {
    #                      S，   H,   L,  head,   V,
    "125M": BertModelConfig(1024, 768, 12, 12, 51200),
    "350M": BertModelConfig(1024, 1024, 24, 16, 51200),
    "760M": BertModelConfig(1024, 1536, 24, 16, 51200),
    "1.3B": BertModelConfig(1024, 2048, 24, 32, 51200),
    "2.6B": BertModelConfig(1024, 2560, 32, 32, 51200),
    "6.7B": BertModelConfig(1024, 4096, 32, 32, 51200),
    "15B": BertModelConfig(1024, 5120, 48, 40, 51200),
    "39B": BertModelConfig(1024, 8192, 48, 64, 51200),
    "76B": BertModelConfig(1024, 10240, 60, 80, 51200),
}


def get_hf_pt_sentiment_model():
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification)

    name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(name)
    device = "cuda"
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.to(device)
    model.eval()

    def infer_func(src):
        inputs = tokenizer(src, return_tensors="pt").to(device)
        outputs = model(**inputs)
        return outputs.logits.detach().cpu().numpy()

    return infer_func


class BertModel:
    def __init__(self, model_config, parallel_config):
        self.logger = logging.getLogger("bert_model")
        self.logger.setLevel(logging.INFO)
        tic = time.time()

        self.infer_func = self.get_alpa_model(model_config, parallel_config)

        self.logger.info(f"Init done. elapsed: {time.time() - tic:.2f} s")

    def get_hf_jax_model(self, model_config, parallel_config):
        # Load tokenizer
        from transformers import AutoTokenizer
        name = "nlptown/bert-base-multilingual-uncased-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(name)

        if False:  # Wether to use the official reference model
            from transformers.models.bert.modeling_flax_bert import (
                FlaxBertForSequenceClassification)
            m = FlaxBertForSequenceClassification.from_pretrained(name)
            params, bert_config, module = {"params": m.params}, m.config, m.module
            pad_seq_len = bert_config.max_position_embeddings
        else:
            from alpa.model.bert_model import (BertConfig,
                FlaxBertForSequenceClassificationModule)
            seq_len, hidden_size, num_layers, num_heads, vocab_size = model_config
            bert_config = BertConfig(
                num_labels=5,
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                intermediate_size=hidden_size * 4,
                num_hidden_layers=num_layers)
            module = FlaxBertForSequenceClassificationModule(
                bert_config, dtype=jnp.float16)
            args = (jnp.ones((1, 1), jnp.int32),) * 4 + (None,)
            params = module.init(jax.random.PRNGKey(0), *args)
            pad_seq_len = seq_len

        @jax.jit
        def forward_func(params, batch):
            return module.apply(params, head_mask=None, **batch)

        def infer_func(src, request):
            inputs = tokenizer(src,
                               max_length=pad_seq_len,
                               padding="max_length",
                               return_tensors="np")
            input_ids = inputs.input_ids
            batch = {
                "input_ids": input_ids,
                "attention_mask": inputs.attention_mask,
                "token_type_ids": inputs.token_type_ids,
                "position_ids": np.broadcast_to(np.arange(
                    np.atleast_2d(input_ids).shape[-1]), input_ids.shape),
            }
            outputs = forward_func(params, batch)
            return np.asarray(outputs.logits)

        return infer_func

    def get_alpa_model(self, model_config, parallel_config):
        from alpa import global_config
        from alpa.model.bert_model import (BertConfig,
            FlaxBertForSequenceClassificationModule)

        # Load tokenizer
        from transformers import AutoTokenizer
        name = "nlptown/bert-base-multilingual-uncased-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(name)

        # Create model
        seq_len, hidden_size, num_layers, num_heads, vocab_size = model_config
        dp, op, pp = parallel_config
        batch_size = 1
        dtype = params_dtype = jnp.float16
        add_manual_layer_marker = True
        num_manual_pipeline_stages = pp

        batch = {
            "input_ids": np.ones((batch_size, seq_len), np.int32),
            "attention_mask": np.ones((batch_size, seq_len), np.int32),
            "token_type_ids": np.ones((batch_size, seq_len), np.int32),
            "position_ids": np.ones((batch_size, seq_len), np.int32),
        }

        bert_config = BertConfig(
            num_labels=5,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_layers,
            type_vocab_size=2,
            add_manual_pipeline_markers=add_manual_layer_marker,
            pipeline_mp_size=num_manual_pipeline_stages,
        )
        module = FlaxBertForSequenceClassificationModule(
            bert_config, dtype=dtype)

        # Choose parallel method
        virtual_mesh = get_global_virtual_physical_mesh()
        num_devices = virtual_mesh.num_devices
        num_devices_per_host = virtual_mesh.num_devices_per_host
        assert virtual_mesh.num_devices == dp * op * pp

        logical_mesh_shape = (dp, op)
        num_mesh_devices = np.prod(logical_mesh_shape)
        if num_mesh_devices <= num_devices_per_host:
            physical_mesh_shape = (1, num_mesh_devices)
        else:
            assert num_mesh_devices % num_devices_per_host == 0
            physical_mesh_shape = (num_mesh_devices // num_devices_per_host,
                                   num_devices_per_host)

        parallel_method = PipeshardParallel(
            num_micro_batches=1,
            default_auto_sharding_option=AutoShardingOption(
                prefer_reduce_scatter=False,
                force_batch_dim_to_mesh_dim=0,
            ),
            pipeline_schedule="inference",
            layer_option="manual",
            stage_option=ManualStageOption(
                forward_stage_layer_ids=[[i] for i in range(pp)],
                submesh_physical_shapes=[physical_mesh_shape] * pp,
                submesh_logical_shapes=[logical_mesh_shape] * pp,
                submesh_autosharding_option_dicts=[{}] * pp))

        # Compile executable
        @parallelize(method=parallel_method)
        def forward_func(params, batch):
            return module.apply(params, head_mask=None, **batch)

        use_dummy_weights = True
        if use_dummy_weights:
            params = jax.eval_shape(module.init, jax.random.PRNGKey(0), **batch)
            params = tree_map(
                lambda x: jax.core.ShapedArray(x.shape, dtype=params_dtype),
                params)
        else:
            params = module.init(jax.random.PRNGKey(0), **batch)
            params = tree_map(lambda x: jax.asarray(x, dtype=params_dtype),
                params)

        executable = forward_func.get_executable(params, batch)
        executable.dump_debug_info("tmp")

        # Preshard params
        if use_dummy_weights:
            global_config.use_dummy_value_for_benchmarking = True

        params_ps = executable.get_input_placement_specs()[0]
        flat_params, in_tree = tree_flatten(params)
        flat_ps = tree_leaves(params_ps)
        params = tree_unflatten(
            in_tree,
            executable.mesh_group.shard_args_to_arrays(flat_ps, flat_params))
        global_config.use_dummy_value_for_benchmarking = False

        # Final inference function
        async def infer_func(src, request):
            request.scope["ts"].append(("c", time.time()))
            inputs = tokenizer(src,
                               max_length=seq_len,
                               padding="max_length",
                               return_tensors="np")
            input_ids = inputs.input_ids
            batch = {
                "input_ids": input_ids,
                "attention_mask": inputs.attention_mask,
                "token_type_ids": inputs.token_type_ids,
                "position_ids": np.broadcast_to(np.arange(
                    np.atleast_2d(input_ids).shape[-1]), input_ids.shape),
            }
            outputs = executable(params, batch)
            request.scope["ts"].append(("d", time.time()))
            return await outputs.logits.to_np_async()

        return infer_func

    async def handle_request(self, request):
        obj = await request.json()

        tic = time.time()
        res = await self.infer_func(obj["input"], request)
        latency = time.time() - tic
        self.logger.info(f"handle request. latency = {latency*1e3:.2f} ms")

        return {
            "logits": res.tolist(),
            "ts": request.scope["ts"],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--port", type=int, default=20001)
    args = parser.parse_args()

    ray.init(address="auto")

    controller = run_controller("localhost", port=args.port, name=None)
    controller.register_model.remote("alpa/bert-1",
        BertModel, (bert_specs["1.3B"],))
    controller.register_model.remote("alpa/bert-2",
        BertModel, (bert_specs["1.3B"],))

    if args.policy == "manual_1":
        group_id = 0
        controller.launch_mesh_group_manager.remote(group_id, [1, 1])
        controller.create_replica.remote(
            "alpa/bert-1", group_id, (ParallelConfig(1, 1, 1),))
        controller.create_replica.remote(
            "alpa/bert-2", group_id, (ParallelConfig(1, 1, 1),))

        group_id = 1
        controller.launch_mesh_group_manager.remote(group_id, [1, 1])
        controller.create_replica.remote(
            "alpa/bert-1", group_id, (ParallelConfig(1, 1, 1),))
        controller.create_replica.remote(
            "alpa/bert-2", group_id, (ParallelConfig(1, 1, 1),))
    elif args.policy == "manual_2":
        group_id = 0
        controller.launch_mesh_group_manager.remote(group_id, [1, 2])
        controller.create_replica.remote(
            "alpa/bert-1", group_id, (ParallelConfig(1, 1, 2),))
        controller.create_replica.remote(
            "alpa/bert-2", group_id, (ParallelConfig(1, 1, 2),))
    elif args.policy == "sr":
        raise NotImplementedError()
    else:
        raise ValueError(f"Invalid policy: {args.policy}")

    while True:
        pass
