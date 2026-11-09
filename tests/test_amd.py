"""An AMD GPU is seen (TensorFold 1.2 serves natively on HIP), and the page says what that gives today."""

from studio import catalog, machine


def card(root, n, vendor, device="0x7551", vram=None, product=None):
    d = root / f"card{n}" / "device"
    d.mkdir(parents=True)
    (d / "vendor").write_text(vendor + "\n")
    (d / "device").write_text(device + "\n")
    if vram is not None:
        (d / "mem_info_vram_total").write_text(str(vram))
    if product:
        (d / "product_name").write_text(product + "\n")


def test_amd_cards_from_sysfs(tmp_path):
    card(tmp_path, 0, "0x10de")                                            # an NVIDIA card: not this list's
    card(tmp_path, 1, "0x1002", vram=32 * 1024**3, product="AMD Radeon AI PRO R9700")
    card(tmp_path, 2, "0x1002", device="0x1586", vram=512 * 1024**2)       # an APU's carve-out: memory unknown
    gpus = machine.amd_gpus(tmp_path)
    assert [(g.name, g.memory_bytes) for g in gpus] == [("AMD Radeon AI PRO R9700", 32 * 1024**3), ("AMD GPU 0x1586", None)]
    assert machine.amd_gpus(tmp_path / "none") == []


def test_an_amd_machine_has_the_hip_backend_and_no_catalog_model_yet(monkeypatch):
    monkeypatch.setattr(machine.platform, "system", lambda: "Linux")
    monkeypatch.setattr(machine, "_nvidia_gpus", lambda: ([], None))
    monkeypatch.setattr(machine, "amd_gpus", lambda: [machine.Gpu(name="AMD Radeon AI PRO R9700", memory_bytes=32 * 1024**3)])
    m, _ = machine._probe()
    assert m.backend == "hip" and "dots.mocr" in m.backend_note and m.chip == "AMD Radeon AI PRO R9700"
    cat = catalog.Catalog()
    cat.refresh_sizes_async = lambda: None
    view = cat.view(m)
    assert view and not any(e["runs_here"] for e in view)
    assert "this one has HIP" in view[0]["runs_note"]
