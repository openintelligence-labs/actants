from __future__ import annotations

from click.testing import CliRunner

from actants.cli import common_options, make_app


def test_make_app_creates_group():
    app = make_app("myapp", help="my test app")

    @app.command()
    def hello():
        from actants.cli import success

        success("ok")

    runner = CliRunner()
    result = runner.invoke(app, ["hello"])
    assert result.exit_code == 0
    assert "ok" in result.output


def test_common_options_strips_flags_before_func_call():
    app = make_app("test")

    captured = {}

    @app.command()
    @common_options
    def run(name: str = "default"):
        captured["name"] = name

    runner = CliRunner()
    result = runner.invoke(app, ["run", "--debug"])
    assert result.exit_code == 0
    # Function got called without the framework flags
    assert captured["name"] == "default"


def test_common_options_passes_json_output_when_requested():
    app = make_app("test")
    captured = {}

    @app.command()
    @common_options
    def run(json_output: bool = False):
        captured["json_output"] = json_output

    runner = CliRunner()
    result = runner.invoke(app, ["run", "--json"])
    assert result.exit_code == 0
    assert captured["json_output"] is True
