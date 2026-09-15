def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: spawns lmp/gmx subprocesses or runs ~100-step MD (deselect with -m 'not slow')"
    )
