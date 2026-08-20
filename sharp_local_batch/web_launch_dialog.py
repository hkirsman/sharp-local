"""Small launch window for Sharp Local web: show URL and open the browser."""

from __future__ import annotations

import os
import threading
import webbrowser
from typing import Callable

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def browser_url(host: str, port: int) -> str:
    """URL for a local browser (map bind-all hosts to loopback)."""
    if host in ("0.0.0.0", "::", "[::]"):
        return f"http://127.0.0.1:{port}"
    return f"http://{host}:{port}"


class WebLaunchWindow(QWidget):
    def __init__(self, url: str, version: str, on_quit: Callable[[], None]) -> None:
        super().__init__()
        self._url = url
        self._on_quit = on_quit

        self.setWindowTitle(f"Sharp Local web {version}")
        self.setMinimumWidth(420)

        title = QLabel("Server is running")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setWeight(QFont.Weight.DemiBold)
        title.setFont(title_font)

        hint = QLabel("Open this address in your browser:")
        hint.setStyleSheet("color: #666;")

        link = QLabel(f'<a href="{url}">{url}</a>')
        link.setTextFormat(Qt.TextFormat.RichText)
        link.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
            | Qt.TextInteractionFlag.TextSelectableByMouse
        )
        link.setOpenExternalLinks(True)
        link_font = QFont()
        link_font.setPointSize(14)
        link.setFont(link_font)

        open_btn = QPushButton("Open in Browser")
        open_btn.setDefault(True)
        open_btn.clicked.connect(self._open_browser)

        copy_btn = QPushButton("Copy Link")
        copy_btn.clicked.connect(self._copy_link)

        quit_btn = QPushButton("Quit")
        quit_btn.clicked.connect(self._quit)

        row = QHBoxLayout()
        row.addWidget(open_btn)
        row.addWidget(copy_btn)
        row.addStretch(1)
        row.addWidget(quit_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(link)
        layout.addSpacing(8)
        layout.addLayout(row)

    def _open_browser(self) -> None:
        QDesktopServices.openUrl(QUrl(self._url))

    def _copy_link(self) -> None:
        QGuiApplication.clipboard().setText(self._url)

    def _quit(self) -> None:
        self._on_quit()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._on_quit()
        event.accept()


def run_server_with_launch_dialog(
    *,
    host: str,
    port: int,
    version: str,
    start_server: Callable[[], None],
    open_browser: bool = True,
) -> None:
    """Start Flask in a background thread and show the launch dialog on the main thread."""
    url = browser_url(host, port)

    server_thread = threading.Thread(target=start_server, name="sharp-web-flask", daemon=True)
    server_thread.start()

    qt_app = QApplication.instance() or QApplication([])
    window = WebLaunchWindow(
        url,
        version,
        on_quit=lambda: _force_exit(qt_app),
    )
    window.show()
    window.raise_()
    window.activateWindow()

    if open_browser:
        # Delay slightly so the first paint / listener are more likely ready.
        def _open() -> None:
            webbrowser.open(url)

        threading.Timer(0.6, _open).start()

    qt_app.exec()


def _force_exit(qt_app: QApplication) -> None:
    qt_app.quit()
    # Flask's threaded server does not expose a clean shutdown from here.
    os._exit(0)
