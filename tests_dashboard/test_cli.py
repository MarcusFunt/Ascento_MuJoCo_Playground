from ascento_mjlab.cli import build_parser


def test_unified_cli_exposes_operations_commands():
    parser = build_parser()

    args = parser.parse_args(
        [
            "run",
            "start",
            "--task",
            "Ascento-Velocity-Flat",
            "--envs",
            "128",
            "--iterations",
            "200",
            "--",
            "--agent.learning-rate",
            "0.001",
        ]
    )

    assert args.task == "Ascento-Velocity-Flat"
    assert args.envs == 128
    assert args.training_args == ["--", "--agent.learning-rate", "0.001"]


def test_unified_cli_supports_low_noise_monitoring():
    args = build_parser().parse_args(["run", "monitor", "run123", "--once", "--json"])

    assert args.run_id == "run123"
    assert args.once is True
    assert args.json is True


def test_unified_cli_exposes_logs_comparison_and_dashboard_commands():
    parser = build_parser()

    logs = parser.parse_args(["run", "logs", "run123", "--tail", "25"])
    compare = parser.parse_args(["run", "compare", "run123", "run456", "--json"])
    dashboard = parser.parse_args(["dashboard", "status", "--port", "9000"])

    assert logs.tail == 25
    assert compare.run_ids == ["run123", "run456"]
    assert dashboard.port == 9000
