from gpufleet import perf
from gpufleet.models import JobSpec, Need
from gpufleet.providers.vast import Vast


def test_gpu_names_map_to_the_table():
    assert perf.canonical("NVIDIA GeForce RTX 4090") == "RTX 4090"
    assert perf.canonical("RTX_3090_Ti") == "RTX 3090 Ti"
    assert perf.canonical("NVIDIA RTX PRO 6000 Blackwell Server Edition") == "RTX PRO 6000 S"
    assert perf.canonical("G4") == "RTX PRO 6000 S"
    assert perf.canonical("L40S") == "L40S" and perf.canonical("L4") == "L4"
    assert perf.canonical("A100 SXM4") == "A100 SXM4" and not perf.info("H100 SXM")["rt"]


def test_vast_query_encodes_need_and_blocklists(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "x")
    v = Vast({"gpu_names": ["RTX 4090"], "max_bandwidth_usd_per_tb": 10})
    q = v._query(Need(max_price_h=0.5, min_vram_gb=20, spot=True), {"123", "bad"}, {"9"})
    assert q["type"] == "bid" and q["dph_total"] == {"lte": 0.5}
    assert q["machine_id"] == {"notin": [123]} and q["host_id"] == {"notin": [9]}
    assert q["cuda_max_good"] == {"gte": 13.0} and q["inet_up_cost"] == {"lte": 0.01}
    assert q["gpu_ram"] == {"gte": 20000}


def test_jobspec_roundtrip_and_validation():
    s = JobSpec.from_dict({"project": "p", "name": "n", "command": "x", "range": [1, 10],
                           "checkpoint": [{"src": "a/", "dst": "b/"}]})
    assert JobSpec.from_dict(s.to_dict()) == s
    for bad in ({"units": ["a"], "range": [1, 2]}, {}):
        try:
            JobSpec.from_dict({"project": "p", "name": "n", "command": "x", **bad})
        except ValueError:
            continue
        raise AssertionError(bad)
