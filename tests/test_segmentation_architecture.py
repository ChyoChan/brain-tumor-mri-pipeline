from src.segmentation import AttentionUNet


def test_attention_convolutions_keep_legacy_checkpoint_keys():
    model = AttentionUNet(base_filters=2)
    expected = {"block.0.weight", "block.3.weight"} | {
        f"block.{index}.{name}"
        for index in (1, 4)
        for name in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")
    }
    blocks = ("enc1", "enc2", "enc3", "enc4", "bottleneck") + tuple(
        f"dec{index}.conv" for index in range(1, 5)
    )
    for name in blocks:
        assert set(model.get_submodule(name).state_dict()) == expected, name
