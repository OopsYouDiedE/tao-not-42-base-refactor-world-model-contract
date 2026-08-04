import pytest

from shared_tools.model_clients import ModelTransportError, iter_sse_json


def test_sse_json_parser_handles_comments_and_done_marker() -> None:
    events = tuple(
        iter_sse_json(
            (
                b": keep-alive\n",
                b'data: {"type":"content","text":"ok"}\n',
                b"data: [DONE]\n",
                b'data: {"ignored":true}\n',
            )
        )
    )

    assert events == ({"type": "content", "text": "ok"},)


def test_sse_json_parser_rejects_non_object_events() -> None:
    with pytest.raises(ModelTransportError, match="JSON 对象"):
        tuple(iter_sse_json(("data: []",)))
