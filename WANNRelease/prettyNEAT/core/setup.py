"""Environment setup, logging, and warning suppression."""
import os
import sys
import contextlib
import logging
import warnings


class FilteredStderr:
    """Custom stderr filter to suppress gym deprecation warnings."""

    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        # Store the original attributes we need to preserve
        for attr in [
            "mode",
            "name",
            "encoding",
            "errors",
            "newlines",
            "line_buffering",
            "write_through",
        ]:
            if hasattr(original_stderr, attr):
                setattr(self, attr, getattr(original_stderr, attr))

    def write(self, text):
        # Filter out gym deprecation warnings
        if "Gym has been unmaintained" in text:
            return len(text)  # Pretend we wrote it
        if "does not support NumPy 2.0" in text:
            return len(text)
        if "upgrade to Gymnasium" in text:
            return len(text)
        if "migration guide" in text.lower():
            return len(text)
        # Write everything else to original stderr
        return self.original_stderr.write(text)

    def flush(self):
        return self.original_stderr.flush()

    def close(self):
        return self.original_stderr.close()

    def __getattr__(self, name):
        # Delegate all other attributes to original stderr
        return getattr(self.original_stderr, name)


def setup_environment():
    """Configure environment variables and suppress warnings."""
    # Fix SDL2 library conflict between OpenCV and Pygame
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "0")

    # Install global stderr filter to suppress gym warnings
    _original_stderr = sys.stderr
    sys.stderr = FilteredStderr(_original_stderr)

    # Suppress gym step API deprecation warning
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*Initializing environment in old step API.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="gym.wrappers.step_api_compatibility",
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
    # Suppress Gym/NumPy 2.0 compatibility warnings
    warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
    warnings.filterwarnings("ignore", message=".*does not support NumPy 2.0.*")
    warnings.filterwarnings("ignore", message=".*upgrade to Gymnasium.*")


def setup_logging():
    """Configure root logger with basicConfig."""
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(message)s",
            datefmt="%H:%M",
            stream=sys.stdout,
            force=True,  # Override any existing configuration
        )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger


@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output (for gym deprecation warnings)."""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr
