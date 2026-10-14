from structllm.datagen.export import collect, salvage_prefix
import json


def record(problems_by_turn):
    messages = [{"role": "system", "content": "s"}]
    for turn in range(len(problems_by_turn)):
        messages += [
            {"role": "user", "content": f"u{turn}"},
            {"role": "assistant", "content": "", "train": True, "tool_calls": [{"id": f"c{turn}", "name": f"t{turn}", "arguments": {}}]},
            {"role": "tool", "content": "{}"},
            {"role": "assistant", "content": f"a{turn}", "train": True},
        ]
    return {
        "record_id": "r", "request_id": "q", "messages": messages, "accepted": False,
        "turns": [{"behaviour": "execute", "problems": p} for p in problems_by_turn],
        "tools": [{"type": "function", "function": {"name": f"t{i}"}} for i in range(4)] + [{"type": "function", "function": {"name": "late"}}],
        "tools_by_turn": [["t0"], ["t0", "t1"], ["t0", "t1", "t2"], ["t0", "t1", "t2", "t3", "late"]][: len(problems_by_turn)],
    }


def test_the_turns_before_the_first_failure_are_kept():
    prefix = salvage_prefix(record([[], [], ["no expected tool succeeded"], []]))
    assert [m["content"] for m in prefix["messages"] if m["role"] == "user"] == ["u0", "u1"]
    assert prefix["messages"][-1]["content"] == "a1"
    assert prefix["accepted"] and prefix["record_id"] == "r.prefix2" and len(prefix["turns"]) == 2


def test_only_tools_offered_by_then_are_kept():
    names = {t["function"]["name"] for t in salvage_prefix(record([[], [], ["x"]]))["tools"]}
    assert names == {"t0", "t1"}


def test_nothing_is_kept_when_the_first_turn_failed_or_everything_passed():
    assert salvage_prefix(record([["x"], []])) is None
    assert salvage_prefix(record([[], []])) is None


def test_collect_prefers_the_accepted_record_and_only_salvages_on_request(tmp_path):
    shard = tmp_path / "shard"
    shard.mkdir()
    (shard / "rejected.jsonl").write_text(json.dumps(record([[], ["x"]])) + "\n")
    assert collect(tmp_path) == []
    assert [r["record_id"] for r in collect(tmp_path, salvage=True)] == ["r.prefix1"]
    accepted = record([[], []]) | {"accepted": True, "record_id": "ok"}
    (shard / "accepted.jsonl").write_text(json.dumps(accepted) + "\n")
    assert [r["record_id"] for r in collect(tmp_path, salvage=True)] == ["ok"]


def test_assemble_joins_exports_normalises_the_prompt_and_keeps_eval_clean(tmp_path):
    from structllm.datagen.export import assemble_sft

    def row(i, scenario, system="old prompt\n\nActive project_id: p"):
        return {"id": i, "messages": [{"role": "system", "content": system}], "tools": [], "meta": {"scenario_id": scenario}}

    for name, train, evals in (("a", [row("1", "s1")], [row("2", "s2")]), ("b", [row("3", "s2")], [row("4", "s9")])):
        d = tmp_path / name
        d.mkdir()
        (d / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
        (d / "eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in evals))
    out = assemble_sft([tmp_path / "a", tmp_path / "b"], tmp_path / "out", system_prompt="NEW")
    train = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    evals = [json.loads(line) for line in (out / "eval.jsonl").read_text().splitlines()]
    assert [r["meta"]["scenario_id"] for r in evals] == ["s9"]  # s2 is trained on, so it left eval
    assert train[0]["messages"][0]["content"] == "NEW\n\nActive project_id: p"
    assert json.loads((out / "manifest.json").read_text())["eval_rows_removed_for_scenario_overlap"] == 1
