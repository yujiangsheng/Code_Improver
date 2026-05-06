"""Unit tests for ada/workspace.py."""

import json
import tempfile
from pathlib import Path

import pytest

from ada.workspace import Workspace


class TestWorkspaceInit:
    """Test Workspace initialization."""

    def test_init_valid_directory(self, tmp_path: Path):
        """Test initialization with a valid directory."""
        # Create a temporary directory
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()
        
        ws = Workspace(str(test_dir))
        
        # Verify .ada directory was created
        assert (test_dir / ".ada").is_dir()
        
        # Verify artifact files were created
        for artifact in Workspace.ARTIFACTS:
            assert (test_dir / ".ada" / artifact).exists()

    def test_init_nonexistent_directory(self, tmp_path: Path):
        """Test initialization with a nonexistent directory."""
        nonexistent_dir = tmp_path / "nonexistent"
        
        with pytest.raises(NotADirectoryError):
            Workspace(str(nonexistent_dir))

    def test_init_artifacts_have_defaults(self, tmp_path: Path):
        """Test that artifact files have default content."""
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()
        
        ws = Workspace(str(test_dir))
        
        # Check profile.md has default content
        profile_content = (test_dir / ".ada" / "profile.md").read_text()
        assert "Project Profile" in profile_content
        
        # Check baseline.json has empty dict
        baseline_content = (test_dir / ".ada" / "baseline.json").read_text()
        assert baseline_content.strip() == "{}"


class TestWorkspaceResolve:
    """Test path resolution."""

    def test_resolve_relative_path(self, tmp_path: Path):
        """Test resolving a relative path."""
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()
        
        ws = Workspace(str(test_dir))
        
        # Resolve a relative path
        resolved = ws.resolve("some/file.txt")
        
        # Should be relative to target_dir
        assert str(resolved).startswith(str(test_dir))

    def test_resolve_absolute_path(self, tmp_path: Path):
        """Test resolving an absolute path within target_dir."""
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()
        
        ws = Workspace(str(test_dir))
        
        # Resolve an absolute path within target_dir
        abs_path = test_dir / "some/file.txt"
        resolved = ws.resolve(str(abs_path))
        
        assert resolved == abs_path

    def test_resolve_absolute_path_outside_raises(self, tmp_path: Path):
        """Test that resolving an absolute path outside target_dir raises PermissionError."""
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()
        
        ws = Workspace(str(test_dir))
        
        # Try to resolve a path outside target_dir
        outside_path = tmp_path / "outside.txt"
        
        with pytest.raises(PermissionError):
            ws.resolve(str(outside_path))

    def test_resolve_path_escaping_target(self, tmp_path: Path):
        """Test that paths escaping target directory raise PermissionError."""
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()
        
        ws = Workspace(str(test_dir))
        
        # Try to resolve a path outside target_dir
        outside_path = tmp_path.parent / "outside.txt"
        
        with pytest.raises(PermissionError):
            ws.resolve(str(outside_path))


class TestWorkspaceJournal:
    """Test journal operations."""

    def test_append_journal(self, tmp_path: Path):
        """Test appending to journal."""
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()
        
        ws = Workspace(str(test_dir))
        
        ws.append_journal("Test entry 1")
        
        journal_content = (test_dir / ".ada" / "journal.md").read_text()
        assert "Test entry 1" in journal_content

    def test_append_journal_multiple_entries(self, tmp_path: Path):
        """Test appending multiple entries to journal."""
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()
        
        ws = Workspace(str(test_dir))
        
        ws.append_journal("Entry A")
        ws.append_journal("Entry B")
        
        journal_content = (test_dir / ".ada" / "journal.md").read_text()
        assert "Entry A" in journal_content
        assert "Entry B" in journal_content


class TestWorkspaceMetrics:
    """Test metric operations."""

    def test_append_metric(self, tmp_path: Path):
        """Test appending a metric."""
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()
        
        ws = Workspace(str(test_dir))
        
        ws.append_metric("phase", "metric", "value")
        
        metrics_content = (test_dir / ".ada" / "metrics.csv").read_text()
        assert "phase,metric,value" in metrics_content

    def test_append_metric_multiple_rows(self, tmp_path: Path):
        """Test appending multiple metrics."""
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()
        
        ws = Workspace(str(test_dir))
        
        ws.append_metric("phase1", "metric1", "value1")
        ws.append_metric("phase2", "metric2", "value2")
        
        metrics_content = (test_dir / ".ada" / "metrics.csv").read_text()
        assert "phase1,metric1,value1" in metrics_content
        assert "phase2,metric2,value2" in metrics_content


class TestWorkspaceBaseline:
    """Test baseline operations."""

    def test_write_baseline(self, tmp_path: Path):
        """Test writing baseline data."""
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()
        
        ws = Workspace(str(test_dir))
        
        baseline_data = {"test": "data", "count": 42}
        ws.write_baseline(baseline_data)
        
        baseline_content = (test_dir / ".ada" / "baseline.json").read_text()
        parsed = json.loads(baseline_content)
        assert parsed == baseline_data

    def test_write_baseline_overwrites(self, tmp_path: Path):
        """Test that write_baseline overwrites existing content."""
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()
        
        ws = Workspace(str(test_dir))
        
        # Write first baseline
        ws.write_baseline({"first": "data"})
        
        # Write second baseline
        ws.write_baseline({"second": "data"})
        
        baseline_content = (test_dir / ".ada" / "baseline.json").read_text()
        parsed = json.loads(baseline_content)
        assert parsed == {"second": "data"}


class TestWorkspaceReadArtifact:
    """Test reading artifacts."""

    def test_read_artifact(self, tmp_path: Path):
        """Test reading an artifact."""
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()
        
        ws = Workspace(str(test_dir))
        
        # Write some content
        ws.write_artifact("profile.md", "# Custom Profile\n")
        
        # Read it back
        content = ws.read_artifact("profile.md")
        
        assert content == "# Custom Profile\n"

    def test_read_artifact_existing(self, tmp_path: Path):
        """Test reading an existing artifact."""
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()
        
        ws = Workspace(str(test_dir))
        
        # Read the default profile.md
        content = ws.read_artifact("profile.md")
        
        assert "Project Profile" in content
