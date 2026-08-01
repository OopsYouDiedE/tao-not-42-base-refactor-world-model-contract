from train.command_line import build_argument_parser


def test_training_parser_supports_skip_backward() -> None:
    parser = build_argument_parser("test", "model")
    arguments = parser.parse_args(
        ["--dataset-dir", "dataset", "--skip-backward"]
    )

    assert arguments.skip_backward is True
