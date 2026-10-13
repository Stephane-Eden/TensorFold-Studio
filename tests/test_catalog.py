from studio import catalog, pull


def test_a_download_cut_halfway_is_not_installed(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(pull, "cache_root", lambda: tmp_path)
    repo = tmp_path / "models--org--m"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("abc")
    (repo / "snapshots" / "abc").mkdir(parents=True)
    (repo / "blobs").mkdir()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(repo / "snapshots" / "abc"))
    assert catalog.installed("org/m")
    (repo / "blobs" / "deadbeef.incomplete").write_bytes(b"x")              # a pull the Studio stopped halfway
    assert not catalog.installed("org/m")


def test_a_snapshot_lighter_than_the_catalog_says_is_not_installed(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(pull, "cache_root", lambda: tmp_path)
    repo = tmp_path / "models--org--m"
    snap = repo / "snapshots" / "abc"
    (repo / "blobs").mkdir(parents=True)
    snap.mkdir(parents=True)
    (repo / "blobs" / "b1").write_bytes(b"x" * 1000)
    (snap / "model.safetensors").symlink_to(repo / "blobs" / "b1")
    (repo / "blobs" / "stale.incomplete").write_bytes(b"y")                 # an orphan of another revision
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))
    assert catalog.installed("org/m", expected_bytes=1000)                   # whole, the orphan does not count
    assert not catalog.installed("org/m", expected_bytes=2000)               # half of it came down
    (snap / "tokenizer.json").symlink_to(repo / "blobs" / "never")           # a link to a blob that never finished
    assert not catalog.installed("org/m", expected_bytes=1000)


def test_a_checkpoint_the_engine_refuses_is_not_asked_again_on_every_view(monkeypatch):
    from studio import catalog, native

    asked = []
    monkeypatch.setattr(native, "find_binary", lambda: "/x/tensorfold")
    monkeypatch.setattr(native, "info", lambda binary, model: asked.append(model) or None)
    cat = catalog.Catalog()
    cat.refresh_sizes_async = lambda: None
    e = next(x for x in cat.entries if x.native)
    assert cat.engine_facts(e) is None and cat.engine_facts(e) is None
    assert asked == [e.id]                                      # once, not on every Models or Home
    cat._facts_missed[e.id] -= catalog.FACTS_RETRY_S           # the memo aged out: asked again
    assert cat.engine_facts(e) is None and asked == [e.id, e.id]
