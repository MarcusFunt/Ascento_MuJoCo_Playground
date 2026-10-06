from dashboard.task_catalog import task_catalog, task_ids


def test_waypoint_heading_sweep_variants_are_available_to_dashboard():
    expected = {
        "Ascento-Locomotion-Gate-Hold-Control-Flat",
        "Ascento-Locomotion-Gate-Hold-Turn-12-Flat",
        "Ascento-Locomotion-Gate-Hold-Turn-25-Flat",
    }
    catalog = {item["id"]: item for item in task_catalog()}

    assert expected <= task_ids()
    assert expected <= catalog.keys()
    assert all(catalog[task_id]["supports_horizon"] is False for task_id in expected)
