import torch
from vista_vlm.config.schema import VISTAConfig
from vista_vlm.utils.registry import Registry

def test_config_defaults():
    cfg = VISTAConfig()
    assert cfg.model.backbone == "dinov2_vits14"
    assert cfg.model.embed_dim == 384
    assert cfg.model.out_indices == [4, 8, 12]
    assert cfg.data.image_size == 518
    assert cfg.calibration.method == "split_conformal"
    assert cfg.calibration.target_false_alarm_rate == 0.05

def test_config_load_yaml(tmp_path):
    yaml_content = """
    data:
      category: "cable"
      image_size: 518
    model:
      embed_dim: 384
    """
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(yaml_content)

    cfg = VISTAConfig.load_from_yaml(str(cfg_file))
    assert cfg.data.category == "cable"
    assert cfg.data.image_size == 518
    assert cfg.model.embed_dim == 384

def test_registry():
    reg = Registry("Test")
    @reg.register("item1")
    class Item1:
        pass

    assert reg.get("item1") == Item1
    assert "item1" in reg.list_keys()

def test_pytorch_tensor_operations():
    t = torch.randn(2, 384, 37, 37)
    assert t.shape == (2, 384, 37, 37)
    mean_val = t.mean()
    assert not torch.isnan(mean_val)
