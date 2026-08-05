import pytest

from online_interactive_environments import (
    ActionSequence,
    ActionSequenceCompiler,
    ActionTick,
    DecisionKind,
    GenerationStatus,
    GenerationTelemetry,
    UnderflowPolicy,
    format_action_sequence,
    parse_action_sequence,
)


def test_standard_action_sequence_format_round_trip() -> None:
    value = ActionSequence(
        "KeyboardMouse",
        4,
        (ActionTick(("W", "MouseMove", "4", "-2"), True), ActionTick()),
    )

    assert parse_action_sequence(format_action_sequence(value)) == value


def sequence(actions: str, offset: int = 0) -> str:
    return f"Device KeyboardMouse\nTick {offset}\n<action>{actions}</action>"


def consume(compiler: ActionSequenceCompiler) -> tuple[str, ...]:
    decision = compiler.pull()
    assert decision.action is not None
    compiler.commit(decision)
    return decision.action.inputs


def test_parser_expands_repeat_and_observes_only_once() -> None:
    parsed = parse_action_sequence(sequence("Observe ; W x3 ; NoOp"))
    assert [tick.inputs for tick in parsed.ticks] == [("W",), ("W",), ("W",), ()]
    assert [tick.observe for tick in parsed.ticks] == [True, False, False, False]


def test_parser_rejects_observe_sharing_a_segment_with_inputs() -> None:
    with pytest.warns(RuntimeWarning, match="Observe 必须独立成段"):
        parsed = parse_action_sequence(sequence("Observe W ; A"))

    assert [tick.inputs for tick in parsed.ticks] == [(), ("A",)]
    assert [tick.observe for tick in parsed.ticks] == [False, False]


def test_parser_binds_observe_to_the_following_tick_only() -> None:
    parsed = parse_action_sequence(sequence("W ; Observe ; A ; S"))

    assert [tick.inputs for tick in parsed.ticks] == [("W",), ("A",), ("S",)]
    assert [tick.observe for tick in parsed.ticks] == [False, True, False]


def test_parser_warns_on_trailing_observe_segment() -> None:
    with pytest.warns(RuntimeWarning, match="末尾的 Observe 段没有后续动作"):
        parsed = parse_action_sequence(sequence("W ; Observe"))

    assert [tick.inputs for tick in parsed.ticks] == [("W",), ()]


def test_parser_warns_on_consecutive_observe_segments() -> None:
    with pytest.warns(RuntimeWarning, match="连续的 Observe 段"):
        parsed = parse_action_sequence(sequence("Observe ; Observe ; W"))

    assert [tick.observe for tick in parsed.ticks] == [False, True]


@pytest.mark.parametrize(
    "movement",
    ["MouseMove 4 -2", "MouseMove 4 -20", "MouseMove 0 0", "MouseMove -4 -2"],
)
def test_parser_accepts_single_digit_mouse_move_coordinates(movement: str) -> None:
    parsed = parse_action_sequence(sequence(f"Observe ; {movement} x3"))

    assert [tick.inputs for tick in parsed.ticks] == [("MouseMove", *movement.split()[1:])] * 3
    assert [tick.observe for tick in parsed.ticks] == [True, False, False]


@pytest.mark.parametrize("actions", ["MouseMove 4", "MouseMove 4 W"])
def test_parser_rejects_incomplete_single_digit_mouse_move(actions: str) -> None:
    with pytest.warns(RuntimeWarning, match="MouseMove 需要两个整数参数"):
        parsed = parse_action_sequence(sequence(actions))

    assert parsed.ticks[0].inputs == ()


def test_stream_submits_when_closing_tag_arrives() -> None:
    compiler = ActionSequenceCompiler()
    text = sequence("W ; A")

    submissions = compiler.feed(text)

    assert len(submissions) == 2
    assert [submission.accepted_ticks for submission in submissions] == [1, 1]
    assert compiler.pending_input == ""
    assert consume(compiler) == ("W",)


def test_stream_automatically_appends_observe_when_sequence_ends_without_it() -> None:
    compiler = ActionSequenceCompiler(UnderflowPolicy.WAIT, auto_observe=True)

    compiler.feed(sequence("W ; A"))
    assert consume(compiler) == ("W",)
    assert consume(compiler) == ("A",)
    decision = compiler.pull()

    assert decision.kind is DecisionKind.OBSERVE
    assert decision.action is None


def test_stream_does_not_duplicate_terminal_observe() -> None:
    compiler = ActionSequenceCompiler(UnderflowPolicy.WAIT, auto_observe=True)

    submissions = compiler.feed(sequence("W ; Observe ; NoOp"))

    assert len(submissions) == 2
    assert sum(item.accepted_ticks for item in submissions) == 2


def test_failed_stream_can_discard_buffered_future_without_advancing_tick() -> None:
    compiler = ActionSequenceCompiler(UnderflowPolicy.WAIT, auto_observe=True)
    compiler.feed(sequence("W ; A"))

    discarded = compiler.discard_buffered_from_current_tick()

    assert discarded == 3
    assert compiler.current_tick == 0
    assert compiler.buffered_ticks == 0
    assert compiler.pull().kind is DecisionKind.WAIT


def test_stream_accepts_single_character_chunks_once() -> None:
    compiler = ActionSequenceCompiler()
    submissions = []

    for character in sequence("W x2"):
        submissions.extend(compiler.feed(character))

    assert len(submissions) == 1
    assert submissions[0].accepted_ticks == 2
    assert consume(compiler) == ("W",)
    assert consume(compiler) == ("W",)


def test_stream_submits_completed_ticks_before_closing_tag() -> None:
    compiler = ActionSequenceCompiler()
    header = "Device KeyboardMouse\nTick 0\n<action>"

    first = compiler.feed(header + "W ;")

    assert len(first) == 1
    assert first[0].accepted_ticks == 1
    assert consume(compiler) == ("W",)

    second = compiler.feed("A ; D</action>")

    assert len(second) == 2
    assert consume(compiler) == ("A",)
    assert consume(compiler) == ("D",)


def test_stream_keeps_original_anchor_while_environment_advances() -> None:
    compiler = ActionSequenceCompiler()
    compiler.feed("Device KeyboardMouse\nTick 1\n<action>W ;")
    assert consume(compiler) == ()
    assert consume(compiler) == ("W",)
    assert consume(compiler) == ()

    submissions = compiler.feed("A ; D</action>")

    assert [submission.expired_ticks for submission in submissions] == [1, 0]
    assert consume(compiler) == ("D",)


@pytest.mark.parametrize("suffix", ["\n", "\r\n", "\t", "向前走一格。"])
def test_stream_leaves_content_after_closing_tag_pending(suffix: str) -> None:
    compiler = ActionSequenceCompiler()

    submissions = compiler.feed(sequence("W") + suffix)

    assert len(submissions) == 1
    assert submissions[0].accepted_ticks == 1
    assert compiler.pending_input == suffix


def test_stream_extracts_multiple_fixed_sequences_from_model_text() -> None:
    compiler = ActionSequenceCompiler()
    model_text = "plan: " + sequence("W") + " continue " + sequence("A", offset=2) + " trailing"

    submissions = compiler.feed(model_text)

    assert len(submissions) == 2
    assert submissions[0].start_tick == 0
    assert submissions[1].start_tick == 2
    assert compiler.retained_text == "plan:  continue "
    assert compiler.pending_input == " trailing"
    assert consume(compiler) == ("W",)
    assert consume(compiler) == ()
    assert consume(compiler) == ("A",)


def test_stream_retains_text_between_and_after_sequences() -> None:
    compiler = ActionSequenceCompiler()
    model_text = sequence("W") + " first explanation " + sequence("A") + " final explanation"

    submissions = compiler.feed(model_text)

    assert len(submissions) == 2
    assert compiler.retained_text == " first explanation "
    assert compiler.pending_input == " final explanation"
    assert compiler.drain_retained_text() == " first explanation "
    assert compiler.retained_text == ""


def test_stream_recovers_when_new_header_replaces_unclosed_sequence() -> None:
    compiler = ActionSequenceCompiler()
    truncated = "Device KeyboardMouse\nTick 0\n<action>W"

    assert compiler.feed(truncated) == ()
    with pytest.warns(RuntimeWarning, match="未闭合"):
        submissions = compiler.feed("\n" + sequence("A"))

    assert len(submissions) == 1
    assert consume(compiler) == ("A",)


def test_parser_rejects_metadata_outside_fixed_sequence() -> None:
    text = "<action>W</action>\nDevice Gamepad\nTick 5"

    with pytest.raises(ValueError, match="严格匹配"):
        parse_action_sequence(text)


def test_cold_start_applies_offset_from_current_tick() -> None:
    compiler = ActionSequenceCompiler()
    submission = compiler.submit(sequence("W ; A", offset=8))
    assert submission.cold_start is True
    assert submission.start_tick == 8
    for _ in range(8):
        assert consume(compiler) == ()
    assert consume(compiler) == ("W",)


def test_new_sequence_overwrites_from_relative_start() -> None:
    compiler = ActionSequenceCompiler()
    compiler.submit(sequence("A ; B ; C ; D"))
    assert consume(compiler) == ("A",)
    result = compiler.submit(sequence("X ; Y", offset=2))
    assert result.start_tick == 3
    assert consume(compiler) == ("B",)
    assert consume(compiler) == ("C",)
    assert consume(compiler) == ("X",)
    assert consume(compiler) == ("Y",)


def test_offset_is_relative_to_current_tick_after_consuming_previous_sequence() -> None:
    compiler = ActionSequenceCompiler()
    compiler.submit(sequence("W x8"))
    for _ in range(8):
        assert consume(compiler) == ("W",)

    result = compiler.submit(sequence("A x4", offset=2))

    assert result.start_tick == 10
    assert result.accepted_ticks == 4
    assert result.expired_ticks == 0
    assert compiler.buffered_ticks == 4
    assert consume(compiler) == ()
    assert consume(compiler) == ()
    assert consume(compiler) == ("A",)


def test_observe_pauses_tick_until_acknowledged() -> None:
    compiler = ActionSequenceCompiler()
    compiler.submit(sequence("Observe ; W"))
    first = compiler.pull()
    assert first.kind is DecisionKind.OBSERVE
    assert compiler.current_tick == 0
    compiler.observed()
    action = compiler.pull()
    assert action.kind is DecisionKind.ACTION
    assert action.action is not None and action.action.inputs == ("W",)
    compiler.commit(action)
    assert compiler.current_tick == 1


@pytest.mark.parametrize(
    ("policy", "expected_kind", "expected_source"),
    [
        (UnderflowPolicy.NOOP, DecisionKind.ACTION, "noop"),
        (UnderflowPolicy.WAIT, DecisionKind.WAIT, None),
    ],
)
def test_empty_buffer_policy(
    policy: UnderflowPolicy,
    expected_kind: DecisionKind,
    expected_source: str | None,
) -> None:
    decision = ActionSequenceCompiler(policy).pull()
    assert decision.kind is expected_kind
    assert decision.source == expected_source


def test_repeat_last_reuses_action_without_observe() -> None:
    compiler = ActionSequenceCompiler(UnderflowPolicy.REPEAT_LAST)
    compiler.submit(sequence("W"))
    consume(compiler)
    repeated = compiler.pull()
    assert repeated.source == "repeat_last"
    assert repeated.action is not None
    assert repeated.action.inputs == ("W",)
    assert repeated.action.observe is False


def test_overrun_budget_turns_underflow_into_wait() -> None:
    compiler = ActionSequenceCompiler(UnderflowPolicy.REPEAT_LAST, max_overrun_ticks=2)
    compiler.submit(sequence("W"))
    assert consume(compiler) == ("W",)

    assert consume(compiler) == ("W",)
    assert consume(compiler) == ("W",)

    assert compiler.overrun_ticks == 2
    assert compiler.overrun_exhausted is True
    assert compiler.pull().kind is DecisionKind.WAIT


def test_overrun_budget_is_released_when_the_queue_continues() -> None:
    compiler = ActionSequenceCompiler(UnderflowPolicy.NOOP, max_overrun_ticks=1)
    compiler.submit(sequence("W"))
    consume(compiler)
    assert consume(compiler) == ()
    assert compiler.overrun_exhausted is True

    compiler.submit(sequence("A"))
    assert consume(compiler) == ("A",)

    assert compiler.overrun_ticks == 0
    assert compiler.overrun_exhausted is False


def test_unlimited_overrun_budget_never_exhausts() -> None:
    compiler = ActionSequenceCompiler(UnderflowPolicy.NOOP, max_overrun_ticks=None)
    for _ in range(5):
        assert consume(compiler) == ()

    assert compiler.overrun_exhausted is False
    assert compiler.pull().kind is DecisionKind.ACTION


def test_zero_overrun_budget_stops_immediately_after_the_queue() -> None:
    compiler = ActionSequenceCompiler(UnderflowPolicy.NOOP, max_overrun_ticks=0)
    compiler.submit(sequence("W"))
    consume(compiler)

    assert compiler.pull().kind is DecisionKind.WAIT


def test_overrun_budget_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="max_overrun_ticks"):
        ActionSequenceCompiler(max_overrun_ticks=-1)


def test_scheduled_action_exposes_queued_tick_before_execution() -> None:
    compiler = ActionSequenceCompiler()
    compiler.submit(sequence("W MouseLeft ; A"))

    assert compiler.scheduled_action(0).inputs == ("W", "MouseLeft")
    assert compiler.scheduled_action(1).inputs == ("A",)
    with pytest.raises(KeyError):
        compiler.scheduled_action(2)


def test_stale_decision_cannot_be_committed_after_overwrite() -> None:
    compiler = ActionSequenceCompiler()
    compiler.submit(sequence("W"))
    stale = compiler.pull()
    compiler.submit(sequence("A"))
    with pytest.raises(RuntimeError, match="覆盖"):
        compiler.commit(stale)


@pytest.mark.parametrize("policy", [UnderflowPolicy.NOOP, UnderflowPolicy.REPEAT_LAST])
def test_stale_underflow_decision_cannot_skip_new_sequence(policy: UnderflowPolicy) -> None:
    compiler = ActionSequenceCompiler(policy)
    if policy is UnderflowPolicy.REPEAT_LAST:
        compiler.submit(sequence("Q"))
        consume(compiler)
    stale = compiler.pull()
    compiler.submit(sequence("W ; A"))

    with pytest.raises(RuntimeError, match="覆盖"):
        compiler.commit(stale)

    assert consume(compiler) == ("W",)
    assert consume(compiler) == ("A",)


def test_generation_recording_is_fixed_at_construction_and_disabled_by_default() -> None:
    compiler = ActionSequenceCompiler()

    assert compiler.begin_generation(request_id="ignored") is None
    compiler.feed(sequence("W"))

    assert compiler.generation_records == ()


def test_generation_records_preserve_start_order_and_complete_input() -> None:
    compiler = ActionSequenceCompiler(record_generations=True)

    first_id = compiler.begin_generation(request_id="provider-request")
    compiler.feed("prefix ")
    compiler.feed(sequence("W"))
    first = compiler.end_generation()

    second_id = compiler.begin_generation(request_id="provider-request")
    compiler.submit(sequence("A"))
    second = compiler.end_generation()

    assert first_id == "generation-0"
    assert second_id == "generation-1"
    assert first is not None and first.complete_input == "prefix " + sequence("W")
    assert second is not None and second.complete_input == sequence("A")
    assert [record.sequence_number for record in compiler.generation_records] == [0, 1]
    assert [record.status for record in compiler.generation_records] == [
        GenerationStatus.COMPLETED,
        GenerationStatus.COMPLETED,
    ]
    assert first.accepted_ticks == 1
    assert second.accepted_ticks == 1


def test_generation_record_tracks_waiting_ticks_before_content_and_action() -> None:
    compiler = ActionSequenceCompiler(record_generations=True)
    compiler.begin_generation()

    consume(compiler)
    compiler.feed("explanation only")
    consume(compiler)
    compiler.feed(sequence("W"))
    consume(compiler)
    record = compiler.end_generation()

    assert record is not None
    assert record.waiting_ticks == 2
    assert record.waiting_before_first_content_ticks == 1
    assert record.waiting_before_first_action_ticks == 2
    assert record.noop_waiting_ticks == 2
    assert record.repeated_waiting_ticks == 0
    assert record.max_consecutive_waiting_ticks == 2
    assert compiler.current_waiting_ticks == 0
    assert compiler.max_waiting_ticks == 2
    assert compiler.total_waiting_ticks == 2


def test_wait_policy_records_scheduler_wait_without_advancing_tick() -> None:
    compiler = ActionSequenceCompiler(
        UnderflowPolicy.WAIT,
        record_generations=True,
    )
    compiler.begin_generation()

    first = compiler.pull()
    compiler.record_wait(first)
    second = compiler.pull()
    compiler.record_wait(second)

    assert compiler.current_tick == 0
    assert compiler.total_waiting_ticks == 2
    assert compiler.max_waiting_ticks == 2
    record = compiler.end_generation()
    assert record is not None
    assert record.waiting_ticks == 2
    assert record.waiting_before_first_content_ticks == 2
    assert record.waiting_before_first_action_ticks == 2
    assert record.max_consecutive_waiting_ticks == 2


def test_repeat_last_is_counted_as_waiting_for_active_generation() -> None:
    compiler = ActionSequenceCompiler(
        UnderflowPolicy.REPEAT_LAST,
        record_generations=True,
    )
    compiler.submit(sequence("Q"))
    consume(compiler)
    compiler.begin_generation()

    assert consume(compiler) == ("Q",)
    record = compiler.end_generation()

    assert record is not None
    assert record.waiting_ticks == 1
    assert record.repeated_waiting_ticks == 1


def test_reset_cancels_active_generation_without_deleting_history() -> None:
    compiler = ActionSequenceCompiler(record_generations=True)
    compiler.begin_generation()
    compiler.feed("partial")

    compiler.reset()

    assert len(compiler.generation_records) == 1
    assert compiler.generation_records[0].status is GenerationStatus.CANCELLED
    assert compiler.generation_records[0].complete_input == "partial"


def test_provider_telemetry_is_preserved_on_completed_record() -> None:
    compiler = ActionSequenceCompiler(record_generations=True)
    compiler.begin_generation(provider="openai", model="test-model")
    compiler.feed(sequence("W"))
    telemetry = GenerationTelemetry(
        request_id="req-1",
        input_tokens=10,
        output_tokens=4,
        total_tokens=14,
    )

    record = compiler.end_generation(telemetry=telemetry)

    assert record is not None
    assert record.telemetry.provider == "openai"
    assert record.telemetry.model == "test-model"
    assert record.telemetry.request_id == "req-1"
    assert record.telemetry.total_tokens == 14
    assert record.time_to_first_content_ms is not None
    assert record.time_to_first_action_ms is not None
    assert record.first_content_to_complete_ms is not None
    assert record.total_observed_generation_ms is not None
