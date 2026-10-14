from structllm.datagen.grounding import claimed_numbers, ungrounded


def conversation(answer: str, result: dict, user: str = "relax it") -> list[dict]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": "", "tool_calls": [{"arguments": {"parameters": {"body": {"steps": 500}}}}]},
        {"role": "tool", "content": __import__("json").dumps(result)},
        {"role": "assistant", "content": answer},
    ]


def test_identifiers_are_not_numbers():
    assert claimed_numbers("revision a8029f98-dc00 of Zr6O4 and c1ccccc1") == []


def test_value_from_tool_result_is_grounded():
    assert ungrounded(conversation("Final energy 101.37 eV.", {"energy": 101.372})) == []


def test_invented_value_is_caught():
    problems = ungrounded(conversation("The porosity is 41.3 %.", {"energy": 101.372}))
    assert problems and problems[0][1] == ["41.3"]


def test_percent_of_fraction_and_difference_are_allowed():
    result = {"porosity": 0.413, "e0": 471.5, "e1": 121.3}
    assert ungrounded(conversation("Porosity 41.3 % and the energy fell by 350.2 eV.", result)) == []


def test_scientific_notation_mantissa():
    assert ungrounded(conversation("K_H = 1.741 x 10-4 mol/kg/Pa", {"henry": 0.0001741})) == []


def test_number_from_user_and_arguments_is_grounded():
    assert ungrounded(conversation("Ran 500 steps at 298 K.", {"ok": True}, user="at 298 K")) == []


def test_turn_without_tool_is_not_checked():
    messages = [{"role": "user", "content": "uptake?"}, {"role": "assistant", "content": "Tell me e.g. 298 K and 15:85 CO2/N2."}]
    assert ungrounded(messages) == []


def test_extreme_tool_numbers_do_not_break_the_check():
    result = {"tiny": 1e-320, "huge": 1e300, "zero": 0.0, "energy": 101.372}
    assert ungrounded(conversation("Final energy 101.37 eV.", result)) == []
