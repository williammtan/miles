from argparse import Namespace
from types import SimpleNamespace

from miles.backends.megatron_utils.model_provider import _apply_bridge_runtime_config


def _args(mtp_num_layers: int) -> Namespace:
    return Namespace(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
        context_parallel_size=1,
        calculate_per_token_loss=True,
        variable_seq_lengths=True,
        mtp_num_layers=mtp_num_layers,
        attention_softmax_in_fp32=True,
        gradient_accumulation_fusion=True,
        fp32_residual_connection=False,
        deterministic_mode=False,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        recompute_modules=None,
        cpu_offloading_num_layers=0,
        distribute_saved_activations=False,
        tp_comm_overlap=False,
        fp8=None,
        fp8_recipe=None,
        attention_backend="flash",
        moe_token_dispatcher_type="allgather",
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
        moe_router_bias_update_rate=None,
        moe_aux_loss_coeff=None,
    )


def test_bridge_runtime_config_disables_checkpoint_mtp():
    provider = SimpleNamespace(mtp_num_layers=1)

    _apply_bridge_runtime_config(provider, _args(mtp_num_layers=0))

    assert provider.mtp_num_layers is None


def test_bridge_runtime_config_preserves_explicit_mtp_request():
    provider = SimpleNamespace(mtp_num_layers=None)

    _apply_bridge_runtime_config(provider, _args(mtp_num_layers=2))

    assert provider.mtp_num_layers == 2
