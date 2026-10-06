import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QIcon  # noqa: E402

from sciqlop_tohban.resources import ICON_PATH  # noqa: E402


def test_icon_ships_with_the_package_and_loads(qapp):
    assert ICON_PATH.is_file()
    assert not QIcon(str(ICON_PATH)).isNull()
