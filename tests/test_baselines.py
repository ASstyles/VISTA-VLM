import torch
from torch.utils.data import DataLoader
from vista_vlm.models.baselines.patchcore import PatchCore
from vista_vlm.models.baselines.padim import PaDiM
from vista_vlm.models.baselines.efficientad import EfficientAD
from vista_vlm.models.baselines.winclip import WinCLIP

def test_patchcore_fit_and_predict():
    model = PatchCore(coreset_sampling_ratio=0.1, image_size=518)

    class DictDataset(torch.utils.data.Dataset):
        def __len__(self): return 2
        def __getitem__(self, idx): return {"image": torch.randn(3, 518, 518)}

    dict_loader = DataLoader(DictDataset(), batch_size=2)
    model.fit(dict_loader)
    assert model.memory_bank is not None

    # Predict
    test_x = torch.randn(1, 3, 518, 518)
    out = model.predict(test_x)
    assert "image_score" in out
    assert "anomaly_map" in out
    assert out["anomaly_map"].shape == (1, 1, 518, 518)
    assert out["image_score"].shape == (1,)

def test_padim_fit_and_predict():
    model = PaDiM(embed_dim=64, image_size=518)
    class DictDataset(torch.utils.data.Dataset):
        def __len__(self): return 2
        def __getitem__(self, idx): return {"image": torch.randn(3, 518, 518)}

    dict_loader = DataLoader(DictDataset(), batch_size=2)
    model.fit(dict_loader)
    assert model.mean is not None
    assert model.inv_cov is not None

    test_x = torch.randn(1, 3, 518, 518)
    out = model.predict(test_x)
    assert "image_score" in out
    assert "anomaly_map" in out
    assert out["anomaly_map"].shape == (1, 1, 518, 518)

def test_efficientad_forward():
    model = EfficientAD(image_size=518, feature_dim=64)
    x = torch.randn(1, 3, 518, 518)
    out = model.predict(x)
    assert "image_score" in out
    assert "anomaly_map" in out
    assert out["anomaly_map"].shape == (1, 1, 518, 518)

def test_winclip_zero_shot():
    model = WinCLIP(embed_dim=512, image_size=518)
    x = torch.randn(1, 3, 518, 518)
    out = model.predict(x, category="bottle")
    assert "image_score" in out
    assert "anomaly_map" in out
    assert out["anomaly_map"].shape == (1, 1, 518, 518)
