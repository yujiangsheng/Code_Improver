"""Unit tests for ada/git_ops.py."""

import tempfile
from pathlib import Path

import pytest

from ada.git_ops import Git, GitError


class TestGitInit:
    """Test Git class initialization."""

    def test_init_valid_repo(self, tmp_path: Path):
        """Test initialization with a valid git repo."""
        # Create a temporary git repo
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
        
        git = Git(repo_dir)
        
        # Verify it's recognized as a repo
        assert git.is_repo() is True

    def test_init_nonexistent_repo(self, tmp_path: Path):
        """Test initialization with a non-git directory."""
        non_repo_dir = tmp_path / "non_repo"
        non_repo_dir.mkdir()
        
        git = Git(non_repo_dir)
        
        # Should not be recognized as a repo
        assert git.is_repo() is False

    def test_init_absolute_path(self, tmp_path: Path):
        """Test initialization with absolute path."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
        
        git = Git(repo_dir)
        
        # Verify repo_dir is resolved to absolute path
        assert git.repo_dir.is_absolute()


class TestGitCurrentBranch:
    """Test current_branch method."""

    def test_current_branch_returns_string(self, tmp_path: Path):
        """Test that current_branch returns a string."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
        
        # Create initial commit to have a valid HEAD
        (repo_dir / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, check=True, capture_output=True)
        
        git = Git(repo_dir)
        
        branch = git.current_branch()
        
        assert isinstance(branch, str)
        assert len(branch) > 0

    def test_current_branch_returns_master_or_main(self, tmp_path: Path):
        """Test that current_branch returns master or main."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
        
        # Create initial commit to have a valid HEAD
        (repo_dir / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, check=True, capture_output=True)
        
        git = Git(repo_dir)
        
        branch = git.current_branch()
        
        assert branch in ["master", "main"]


class TestGitStatus:
    """Test status method."""

    def test_status_returns_string(self, tmp_path: Path):
        """Test that status returns a string."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
        
        git = Git(repo_dir)
        
        status = git.status()
        
        assert isinstance(status, str)
        assert len(status) > 0

    def test_status_with_clean_repo(self, tmp_path: Path):
        """Test status with a clean repo (no changes)."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
        
        # Create initial commit
        (repo_dir / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, check=True, capture_output=True)
        
        git = Git(repo_dir)
        
        status = git.status()
        
        # Clean repo should show no changes
        assert " M" not in status  # No modified files


class TestGitDiff:
    """Test diff method."""

    def test_diff_returns_string(self, tmp_path: Path):
        """Test that diff returns a string."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
        
        # Create initial commit
        (repo_dir / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, check=True, capture_output=True)
        
        # Make a change
        (repo_dir / "README.md").write_text("# Test Repo\n\nSome content")
        
        git = Git(repo_dir)
        
        diff = git.diff()
        
        assert isinstance(diff, str)
        assert len(diff) > 0

    def test_diff_with_paths(self, tmp_path: Path):
        """Test diff with specific paths."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
        
        # Create initial commit
        (repo_dir / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, check=True, capture_output=True)
        
        # Make a change
        (repo_dir / "README.md").write_text("# Test Repo\n\nSome content")
        
        git = Git(repo_dir)
        
        diff = git.diff(paths=["README.md"])
        
        assert isinstance(diff, str)
        assert "README.md" in diff

    def test_diff_truncated(self, tmp_path: Path):
        """Test that large diffs are truncated."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
        
        # Create initial commit
        (repo_dir / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, check=True, capture_output=True)
        
        # Make a large change (exceeds 12000 characters)
        large_content = "# Test Repo\n\n" + "Line\n" * 2000
        (repo_dir / "README.md").write_text(large_content)
        
        git = Git(repo_dir)
        
        diff = git.diff()
        
        # Should be truncated at 12000 characters (the code caps at 12000)
        # The actual limit is 12000, so we check it's close to that limit
        assert len(diff) <= 12500  # Allow some overhead for header
        assert "... (diff truncated)" in diff


class TestGitCreateBranch:
    """Test create_branch method."""

    def test_create_branch_returns_name(self, tmp_path: Path):
        """Test that create_branch returns the branch name."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
        
        git = Git(repo_dir)
        
        branch_name = git.create_branch("test-branch")
        
        assert branch_name == "test-branch"

    def test_create_branch_checkout(self, tmp_path: Path):
        """Test that create_branch with checkout=True switches to new branch."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
        
        # Create initial commit
        (repo_dir / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, check=True, capture_output=True)
        
        git = Git(repo_dir)
        
        branch_name = git.create_branch("test-branch", checkout=True)
        
        # Current branch should be the new branch
        current = git.current_branch()
        assert current == branch_name

    def test_create_branch_no_checkout(self, tmp_path: Path):
        """Test that create_branch with checkout=False just creates branch."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
        
        # Create initial commit
        (repo_dir / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, check=True, capture_output=True)
        
        git = Git(repo_dir)
        
        branch_name = git.create_branch("test-branch", checkout=False)
        
        # Current branch should still be master/main
        current = git.current_branch()
        assert current in ["master", "main"]


class TestGitCommitAll:
    """Test commit_all method."""

    def test_commit_all_with_changes(self, tmp_path: Path):
        """Test commit_all with staged changes."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
        
        # Create initial commit
        (repo_dir / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, check=True, capture_output=True)
        
        # Make changes
        (repo_dir / "README.md").write_text("# Test Repo\n\nSome content")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        
        git = Git(repo_dir)
        
        result = git.commit_all("Add content")
        
        assert result["committed"] is True
        assert "sha" in result
        assert "message" in result

    def test_commit_all_no_changes(self, tmp_path: Path):
        """Test commit_all with no staged changes."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
        
        # Create initial commit
        (repo_dir / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, check=True, capture_output=True)
        
        git = Git(repo_dir)
        
        result = git.commit_all("No changes")
        
        assert result["committed"] is False
        assert result["reason"] == "no staged changes"

    def test_commit_all_returns_sha(self, tmp_path: Path):
        """Test that commit_all returns the commit SHA."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
        
        # Create initial commit
        (repo_dir / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, check=True, capture_output=True)
        
        # Make changes
        (repo_dir / "README.md").write_text("# Test Repo\n\nSome content")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        
        git = Git(repo_dir)
        
        result = git.commit_all("Add content")
        
        # SHA should be 12 characters (full SHA from git rev-parse)
        assert len(result["sha"]) == 12


class TestGitRevert:
    """Test revert_to method."""

    def test_revert_to_head(self, tmp_path: Path):
        """Test revert_to with HEAD."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
        
        # Create initial commit
        (repo_dir / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, check=True, capture_output=True)
        
        # Make changes
        (repo_dir / "README.md").write_text("# Test Repo\n\nSome content")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        
        git = Git(repo_dir)
        
        result = git.revert_to("HEAD")
        
        assert isinstance(result, str)
        assert len(result) > 0

    def test_revert_to_specific_commit(self, tmp_path: Path):
        """Test revert_to with a specific commit."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
        
        # Create initial commit
        (repo_dir / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, check=True, capture_output=True)
        
        # Make changes
        (repo_dir / "README.md").write_text("# Test Repo\n\nSome content")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add content"], cwd=repo_dir, check=True, capture_output=True)
        
        # Get SHA of initial commit
        initial_sha = subprocess.run(["git", "rev-parse", "HEAD~1"], cwd=repo_dir, capture_output=True, text=True, check=True).stdout.strip()
        
        git = Git(repo_dir)
        
        result = git.revert_to(initial_sha)
        
        assert isinstance(result, str)
        assert len(result) > 0
