"""Minimal conftest for test_profile_cli.py - avoids heavy dependencies"""
import pytest

# This is intentionally minimal to avoid importing numpy, talib, etc.
# Only used when running test_profile_cli.py in isolation
