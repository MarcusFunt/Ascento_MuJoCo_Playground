from ascento_mjlab.evaluation.report import render_html


def test_html_evaluation_report_displays_escaped_gate_reason(tmp_path):
    report = tmp_path / "report.html"

    render_html(
        report,
        manifest={"suite_id": "synthetic", "checkpoint": "model.pt"},
        summary={},
        gate_payload={
            "status": "INVALID",
            "reason": "cannot read <checkpoint>",
            "gates": [],
        },
        worst={},
    )

    rendered = report.read_text(encoding="utf-8")
    assert "cannot read &lt;checkpoint&gt;" in rendered
    assert "cannot read <checkpoint>" not in rendered
