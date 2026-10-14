from structllm.behaviour import aggregate, expected_from_message, schemas_by_name, score_turn
from structllm.chat_format import ParsedTurn, parse_generation

TOOLS = [
    {"type": "function", "function": {"name": "relax", "description": "", "parameters": {
        "type": "object", "required": ["structure_revision_id"], "additionalProperties": False,
        "properties": {"structure_revision_id": {"type": "string"}, "steps": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "analyse", "description": "", "parameters": {"type": "object"}}},
]
EXPECTED_MESSAGE = {"role": "assistant", "tool_calls": [
    {"id": "x", "type": "function", "function": {"name": "relax", "arguments": {"structure_revision_id": "rev-1", "idempotency_key": "a"}}}]}


def parsed(name, arguments):
    return parse_generation(f'<tool_call>\n{{"name": "{name}", "arguments": {arguments}}}\n</tool_call>', "hermes")


def test_expected_calls_are_read_from_the_exported_shape():
    assert expected_from_message(EXPECTED_MESSAGE) == [{"name": "relax", "arguments": {"structure_revision_id": "rev-1", "idempotency_key": "a"}}]
    assert set(schemas_by_name(TOOLS)) == {"relax", "analyse"}


def test_the_teachers_call_scores_fully_and_ignores_idempotency_keys():
    score = score_turn(parsed("relax", '{"structure_revision_id": "rev-1"}'), expected_from_message(EXPECTED_MESSAGE), TOOLS)
    assert score["decision"] and score["names"] and score["args_valid"] and score["args_exact"] and score["known_tools"]


def test_a_copied_wrong_revision_is_valid_but_not_exact():
    score = score_turn(parsed("relax", '{"structure_revision_id": "rev-2"}'), expected_from_message(EXPECTED_MESSAGE), TOOLS)
    assert score["args_valid"] and not score["args_exact"]


def test_a_missing_required_argument_fails_the_schema():
    score = score_turn(parsed("relax", '{"steps": 5}'), expected_from_message(EXPECTED_MESSAGE), TOOLS)
    assert not score["args_valid"]


def test_wrong_tool_and_invented_tool():
    wrong = score_turn(parsed("analyse", "{}"), expected_from_message(EXPECTED_MESSAGE), TOOLS)
    assert wrong["decision"] and not wrong["names"]
    invented = score_turn(parsed("relax_everything", "{}"), expected_from_message(EXPECTED_MESSAGE), TOOLS)
    assert invented["known_tools"] is False


def test_answering_when_a_tool_was_expected_is_a_wrong_decision():
    score = score_turn(ParsedTurn(reasoning="", content="Sure, done.", tool_calls=[], parse_errors=[]), expected_from_message(EXPECTED_MESSAGE), TOOLS)
    assert score["decision"] is False


def test_calling_a_tool_when_the_teacher_answered_is_a_wrong_decision():
    assert score_turn(parsed("relax", '{"structure_revision_id": "r"}'), [], TOOLS)["decision"] is False


def test_unparseable_call_is_reported():
    assert score_turn(parse_generation("<tool_call>{not json}</tool_call>", "hermes"), [], TOOLS)["parsed"] is False


def test_reply_numbers_are_checked_against_the_conversation():
    prefix = [{"role": "user", "content": "relax it"}, {"role": "assistant", "content": "", "tool_calls": []},
              {"role": "tool", "content": '{"energy": 101.372}'}]
    good = score_turn(ParsedTurn("", "Final energy 101.37 eV.", [], []), [], TOOLS, prefix)
    bad = score_turn(ParsedTurn("", "Porosity is 41.3 %.", [], []), [], TOOLS, prefix)
    assert good["grounded"] is True and bad["grounded"] is False


def test_aggregate_uses_only_the_turns_a_measure_applies_to():
    rates = aggregate([
        score_turn(parsed("relax", '{"structure_revision_id": "rev-1"}'), expected_from_message(EXPECTED_MESSAGE), TOOLS),
        score_turn(ParsedTurn("", "ok", [], []), [], TOOLS),
    ])
    assert rates["decision"] == 1.0 and rates["names"] == 1.0 and rates["score"] == 1.0 and rates["turns"] == 2
