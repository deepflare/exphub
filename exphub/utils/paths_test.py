import pytest
from exphub.utils.paths import shorten_paths


@pytest.mark.parametrize(
    "paths,expected",
    [
        (
            ["/a/b/c/d", "/a/b/c/e"],
            {
                "/a/b/c/d": "d",
                "/a/b/c/e": "e"
            },
        ),
        (
            ["/a/b/c/d", "/a/b/c/e", "/a/b/c/f"],
            {
                "/a/b/c/d": "d",
                "/a/b/c/e": "e",
                "/a/b/c/f": "f"
            },
        ),
        (
            ["/a/b/c/d/e", "/a/b/c/e"],
            {
                "/a/b/c/d/e": "d/e",
                "/a/b/c/e": "c/e"
            },
        ),
        (
            ["/a/b/c/d/e", "/a/b/c/d/f", "/a/b/c/d/g"],
            {
                "/a/b/c/d/e": "e",
                "/a/b/c/d/f": "f",
                "/a/b/c/d/g": "g"
            },
        ),
    ],
)
def test_shorten_paths(paths, expected):
    result = shorten_paths(paths)
    assert result == expected
