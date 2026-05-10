from pathlib import Path
from unittest.mock import Mock

from quantagent.trading_graph import TradingGraph

PNG_BYTES = b"fake-png-bytes"


def _graph_with_png_bytes() -> Mock:
    drawable_graph = Mock()
    drawable_graph.draw_mermaid_png.return_value = PNG_BYTES

    compiled_graph = Mock()
    compiled_graph.get_graph.return_value = drawable_graph
    return compiled_graph


def test_export_stategraph_image_writes_png_under_artifact_root(
    mock_llm, mock_vision_llm, mock_toolkit, tmp_path
):
    tg = TradingGraph(use_checkpointing=False)
    tg.graph = _graph_with_png_bytes()

    image_path = tg.export_stategraph_image(
        artifacts_policy="path-only",
        artifacts_root=tmp_path,
        environment="paper",
        thread_id="thread-001",
        symbol="AAPL",
    )

    assert image_path is not None
    output_path = Path(image_path)
    assert output_path.exists()
    assert output_path.read_bytes() == PNG_BYTES
    assert output_path.parent == tmp_path.resolve() / "paper" / "thread-001" / "AAPL"


def test_build_stategraph_artifact_metadata_returns_path_only_reference(
    mock_llm, mock_vision_llm, mock_toolkit, tmp_path
):
    tg = TradingGraph(use_checkpointing=False)
    tg.graph = _graph_with_png_bytes()

    metadata = tg.build_stategraph_artifact_metadata(
        artifacts_policy="path-only",
        artifacts_root=tmp_path,
        run_id="run-123",
    )

    assert list(metadata) == ["stategraph_image_path"]
    assert isinstance(metadata["stategraph_image_path"], str)
    assert metadata["stategraph_image_path"].startswith(str(tmp_path.resolve()))
    assert "base64" not in metadata["stategraph_image_path"]
    assert Path(metadata["stategraph_image_path"]).exists()


def test_build_stategraph_artifact_metadata_skips_when_artifacts_disabled(
    mock_llm, mock_vision_llm, mock_toolkit, tmp_path
):
    tg = TradingGraph(use_checkpointing=False)
    tg.graph = _graph_with_png_bytes()

    metadata = tg.build_stategraph_artifact_metadata(
        artifacts_policy="none",
        artifacts_root=tmp_path,
        environment="paper",
    )

    assert metadata == {}
    assert not any(tmp_path.iterdir())
