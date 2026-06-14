from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.config import PROJECT_ROOT, update_env_file
from app.hardware import (
    detect_hardware,
    format_report,
    plan_acceleration,
    recommended_install_commands,
)
from app.service import AppService
from app.ui import theme


class SettingsDialog(QDialog):
    """Runtime-editable settings. Changes apply for this session; edit .env to
    persist across restarts."""

    def __init__(self, service: AppService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.setWindowTitle("Settings")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(12)

        # --- LLM ---
        llm_box = QGroupBox("LLM")
        llm_form = QFormLayout(llm_box)

        self._temperature = QDoubleSpinBox()
        self._temperature.setRange(0.0, 2.0)
        self._temperature.setSingleStep(0.05)
        self._temperature.setDecimals(2)
        self._temperature.setValue(service.settings.llm_temperature)
        llm_form.addRow("Temperature:", self._temperature)

        self._max_tokens = QSpinBox()
        self._max_tokens.setRange(64, 4096)
        self._max_tokens.setSingleStep(64)
        self._max_tokens.setValue(service.settings.llm_max_tokens)
        llm_form.addRow("Max tokens:", self._max_tokens)

        layout.addWidget(llm_box)

        # --- Retrieval ---
        ret_box = QGroupBox("Retrieval")
        ret_form = QFormLayout(ret_box)

        self._top_k_final = QSpinBox()
        self._top_k_final.setRange(1, 20)
        self._top_k_final.setValue(service.settings.top_k_final)
        ret_form.addRow("Results (final k):", self._top_k_final)

        self._top_k_dense = QSpinBox()
        self._top_k_dense.setRange(1, 100)
        self._top_k_dense.setValue(service.settings.top_k_dense)
        ret_form.addRow("Dense candidates:", self._top_k_dense)

        self._top_k_bm25 = QSpinBox()
        self._top_k_bm25.setRange(1, 100)
        self._top_k_bm25.setValue(service.settings.top_k_bm25)
        ret_form.addRow("BM25 candidates:", self._top_k_bm25)

        layout.addWidget(ret_box)

        # --- Hardware & acceleration ---
        hw_box = QGroupBox("Hardware & Acceleration")
        hw_layout = QVBoxLayout(hw_box)

        try:
            profile = detect_hardware()
            plan = plan_acceleration(profile, service.settings)
            report = format_report(profile, plan)
        except Exception as exc:  # probing must not break the dialog
            report = f"(hardware detection failed: {exc})"

        report_label = QLabel(report)
        report_label.setWordWrap(True)
        report_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        report_label.setStyleSheet(
            f"font-family: Consolas, monospace; font-size: 11px;"
            f" background: {theme.SURFACE_ALT}; border: 1px solid {theme.BORDER};"
            f" border-radius: 6px; padding: 8px;"
        )
        hw_layout.addWidget(report_label)

        hw_form = QFormLayout()
        self._acceleration = QComboBox()
        self._acceleration.addItems(["auto", "cpu", "gpu"])
        self._acceleration.setCurrentText(service.settings.acceleration)
        hw_form.addRow("Acceleration:", self._acceleration)

        self._embedding_device = QComboBox()
        self._embedding_device.addItems(["auto", "cpu", "cuda", "xpu"])
        self._embedding_device.setCurrentText(service.settings.embedding_device)
        hw_form.addRow("Embedding device:", self._embedding_device)
        hw_layout.addLayout(hw_form)

        # Baseline values, so Apply can detect changes and Cancel can roll back.
        self._orig_acceleration = service.settings.acceleration
        self._orig_embedding_device = service.settings.embedding_device

        hw_note = QLabel("Acceleration & device are saved to .env and take effect after restart.")
        hw_note.setProperty("muted", True)
        hw_layout.addWidget(hw_note)

        layout.addWidget(hw_box)

        note = QLabel("Changes apply for this session only. Edit .env to persist.")
        note.setProperty("muted", True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Close
        )
        apply_btn = buttons.button(QDialogButtonBox.StandardButton.Apply)
        apply_btn.setProperty("accent", True)
        apply_btn.clicked.connect(self._apply)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def _apply(self) -> None:
        # These take effect immediately (read per request).
        s = self.service.settings
        s.llm_temperature = self._temperature.value()
        s.llm_max_tokens = self._max_tokens.value()
        s.top_k_final = self._top_k_final.value()
        s.top_k_dense = self._top_k_dense.value()
        s.top_k_bm25 = self._top_k_bm25.value()

        # Restart-required: embedder/LLM are already built. Persist + confirm.
        new_accel = self._acceleration.currentText()
        new_device = self._embedding_device.currentText()
        if new_accel != self._orig_acceleration or new_device != self._orig_embedding_device:
            self._prompt_acceleration_change(new_accel, new_device)

    def _prompt_acceleration_change(self, new_accel: str, new_device: str) -> None:
        """Confirm restart-required changes. Restart / Save (no restart) / Cancel."""
        try:
            profile = detect_hardware()
            cmds = recommended_install_commands(profile, new_accel, new_device)
        except Exception:  # must not block applying settings
            cmds = []

        text = (
            "Acceleration and embedding-device changes only take effect after "
            "restarting SEFM."
        )
        if cmds:
            joined = "\n".join(cmds)
            text += (
                "\n\nThese settings also need extra modules. Run from the project "
                f"root (S:\\SEFM):\n\n{joined}"
            )

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Apply acceleration changes")
        box.setText(text)
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        restart_btn = box.addButton("Restart", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Save (without restarting)", QMessageBox.ButtonRole.ApplyRole)
        cancel_btn = box.addButton("Cancel (reset)", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(restart_btn)
        box.exec()
        clicked = box.clickedButton()

        if clicked is cancel_btn:
            # Roll back; persist nothing.
            self._acceleration.setCurrentText(self._orig_acceleration)
            self._embedding_device.setCurrentText(self._orig_embedding_device)
            return

        self._persist_acceleration(new_accel, new_device)
        if clicked is restart_btn:
            self._restart_app()
        # "Save (without restarting)": persisted; takes effect on next launch.

    def _persist_acceleration(self, new_accel: str, new_device: str) -> None:
        update_env_file(
            {
                "SEFM_ACCELERATION": new_accel,
                "SEFM_EMBEDDING_DEVICE": new_device,
            }
        )
        # Mirror into live settings and update the baseline so re-Apply won't re-prompt.
        self.service.settings.acceleration = new_accel
        self.service.settings.embedding_device = new_device
        self._orig_acceleration = new_accel
        self._orig_embedding_device = new_device

    def _restart_app(self) -> None:
        # Relaunch via `-m app gui` from the project root.
        from PySide6.QtCore import QProcess

        QProcess.startDetached(sys.executable, ["-m", "app", "gui"], str(PROJECT_ROOT))
        app = QApplication.instance()
        if app is not None:
            app.quit()
