#!/usr/bin/env python3
"""Desktop GUI for EDL-to-Archive Converter using CustomTkinter."""

from __future__ import annotations

import io
import sys
import threading
import webbrowser
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from src.config import (
    ensure_template_files,
    get_app_data_dir,
    get_edl_path,
    get_source_path,
    load_config,
    open_file_in_default_app,
    reset_template_file,
    save_config,
)
from src.converter import convert, validate_edl_file, validate_source_file
from src.exclusion import ExclusionRuleSyntaxError, parse_exclusion_rules
from src.updater import check_for_update_async, get_current_version

# Appearance
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

FPS_OPTIONS = ["24", "25", "30", "50", "60"]
DELIMITER_OPTIONS = ["comma", "tab"]


class UpdateDialog(ctk.CTkToplevel):
    """Dialog shown when a new version is available."""

    def __init__(self, parent, update_info: dict, config: dict):
        super().__init__(parent)
        self.config = config
        self.title("Update Available")
        self.geometry("480x320")
        self.resizable(False, False)
        self.grab_set()

        current = get_current_version()
        new_ver = update_info["version"]

        ctk.CTkLabel(
            self, text="A new version is available!",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(20, 5))

        ctk.CTkLabel(
            self, text=f"Current: v{current}  →  New: v{new_ver}",
            font=ctk.CTkFont(size=13),
        ).pack(pady=5)

        if update_info.get("body"):
            notes_box = ctk.CTkTextbox(self, height=120, width=420)
            notes_box.pack(padx=20, pady=10)
            notes_box.insert("1.0", update_info["body"])
            notes_box.configure(state="disabled")

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=10)

        ctk.CTkButton(
            btn_frame, text="Download",
            command=lambda: self._download(update_info["url"]),
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame, text="Skip This Version",
            fg_color="gray", hover_color="darkgray",
            command=lambda: self._skip(new_ver),
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame, text="Remind Me Later",
            fg_color="gray", hover_color="darkgray",
            command=self.destroy,
        ).pack(side="left", padx=5)

    def _download(self, url: str) -> None:
        webbrowser.open(url)
        self.destroy()

    def _skip(self, version: str) -> None:
        self.config["skipped_version"] = version
        save_config(self.config)
        self.destroy()


class App(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.title(f"EDL to Archive Converter  v{get_current_version()}")
        self.geometry("680x820")
        self.minsize(580, 700)

        # Load config and ensure template files exist
        self.config = load_config()
        ensure_template_files()

        self._build_ui()
        self._load_config_into_ui()

        # Check for updates in background
        check_for_update_async(self._on_update_check_result)

    # ── UI Construction ──────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Scrollable frame for the entire content
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        container = ctk.CTkScrollableFrame(self)
        container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        container.grid_columnconfigure(0, weight=1)

        row = 0

        # ── Input Files Section ──
        input_label = ctk.CTkLabel(
            container, text="Input Files",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        input_label.grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1

        # EDL file
        edl_frame = ctk.CTkFrame(container)
        edl_frame.grid(row=row, column=0, sticky="ew", pady=2)
        edl_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(edl_frame, text="EDL File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.edl_path_label = ctk.CTkLabel(
            edl_frame, text=str(get_edl_path()), anchor="w",
            font=ctk.CTkFont(size=11),
        )
        self.edl_path_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        edl_btn_frame = ctk.CTkFrame(edl_frame, fg_color="transparent")
        edl_btn_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=(0, 5))
        ctk.CTkButton(
            edl_btn_frame, text="Open in Editor", width=130,
            command=lambda: open_file_in_default_app(get_edl_path()),
        ).pack(side="left", padx=(0, 5))
        ctk.CTkButton(
            edl_btn_frame, text="Reset to Template", width=130,
            fg_color="gray", hover_color="darkgray",
            command=self._reset_edl,
        ).pack(side="left")
        row += 1

        # Source file
        source_frame = ctk.CTkFrame(container)
        source_frame.grid(row=row, column=0, sticky="ew", pady=2)
        source_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(source_frame, text="Source File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.source_path_label = ctk.CTkLabel(
            source_frame, text=str(get_source_path()), anchor="w",
            font=ctk.CTkFont(size=11),
        )
        self.source_path_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        source_btn_frame = ctk.CTkFrame(source_frame, fg_color="transparent")
        source_btn_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=(0, 5))
        ctk.CTkButton(
            source_btn_frame, text="Open in Editor", width=130,
            command=lambda: open_file_in_default_app(get_source_path()),
        ).pack(side="left", padx=(0, 5))
        ctk.CTkButton(
            source_btn_frame, text="Reset to Template", width=130,
            fg_color="gray", hover_color="darkgray",
            command=self._reset_source,
        ).pack(side="left")
        row += 1

        # ── Options Section ──
        opts_label = ctk.CTkLabel(
            container, text="Options",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        opts_label.grid(row=row, column=0, sticky="w", pady=(15, 5))
        row += 1

        opts_frame = ctk.CTkFrame(container)
        opts_frame.grid(row=row, column=0, sticky="ew", pady=2)
        opts_frame.grid_columnconfigure(1, weight=1)

        # FPS
        ctk.CTkLabel(opts_frame, text="FPS:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.fps_var = ctk.StringVar(value="25")
        self.fps_menu = ctk.CTkOptionMenu(opts_frame, variable=self.fps_var, values=FPS_OPTIONS, width=100)
        self.fps_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Delimiter
        ctk.CTkLabel(opts_frame, text="Delimiter:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.delimiter_var = ctk.StringVar(value="comma")
        self.delimiter_menu = ctk.CTkOptionMenu(opts_frame, variable=self.delimiter_var, values=DELIMITER_OPTIONS, width=100)
        self.delimiter_menu.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Collapse checkbox
        self.collapse_var = ctk.BooleanVar(value=True)
        self.collapse_cb = ctk.CTkCheckBox(opts_frame, text="Collapse consecutive entries", variable=self.collapse_var)
        self.collapse_cb.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Frames checkbox
        self.frames_var = ctk.BooleanVar(value=False)
        self.frames_cb = ctk.CTkCheckBox(opts_frame, text="Include frames in timecodes", variable=self.frames_var)
        self.frames_cb.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Output path
        ctk.CTkLabel(opts_frame, text="Output:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        output_row = ctk.CTkFrame(opts_frame, fg_color="transparent")
        output_row.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
        output_row.grid_columnconfigure(0, weight=1)

        self.output_var = ctk.StringVar(value="")
        self.output_entry = ctk.CTkEntry(output_row, textvariable=self.output_var)
        self.output_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ctk.CTkButton(
            output_row, text="Browse...", width=80,
            command=self._browse_output,
        ).grid(row=0, column=1)
        row += 1

        # ── Exclusion Rules Section ──
        excl_label = ctk.CTkLabel(
            container, text="Exclusion Rules",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        excl_label.grid(row=row, column=0, sticky="w", pady=(15, 2))
        row += 1

        ctk.CTkLabel(
            container,
            text='One rule per line. Syntax: field_name IS/INCLUDES "value". Lines starting with # are comments.',
            font=ctk.CTkFont(size=11),
            text_color="gray",
        ).grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1

        self.exclusion_text = ctk.CTkTextbox(container, height=120)
        self.exclusion_text.grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        # ── Convert Button ──
        self.convert_btn = ctk.CTkButton(
            container, text="Convert", height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._on_convert,
        )
        self.convert_btn.grid(row=row, column=0, sticky="ew", pady=(15, 5))
        row += 1

        # ── Log Section ──
        log_label = ctk.CTkLabel(
            container, text="Log",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        log_label.grid(row=row, column=0, sticky="w", pady=(15, 5))
        row += 1

        self.log_text = ctk.CTkTextbox(container, height=200, state="disabled")
        self.log_text.grid(row=row, column=0, sticky="ew", pady=2)

    # ── Config ←→ UI ─────────────────────────────────────────────────

    def _load_config_into_ui(self) -> None:
        self.fps_var.set(str(self.config.get("fps", 25)))
        self.delimiter_var.set(self.config.get("delimiter", "comma"))
        self.collapse_var.set(self.config.get("collapse", True))
        self.frames_var.set(self.config.get("frames", False))
        self.output_var.set(self.config.get("output_path", ""))

        exclusion = self.config.get("exclusion_rules", "")
        if exclusion:
            self.exclusion_text.insert("1.0", exclusion)

    def _save_ui_to_config(self) -> None:
        self.config["fps"] = int(self.fps_var.get())
        self.config["delimiter"] = self.delimiter_var.get()
        self.config["collapse"] = self.collapse_var.get()
        self.config["frames"] = self.frames_var.get()
        self.config["output_path"] = self.output_var.get()
        self.config["exclusion_rules"] = self.exclusion_text.get("1.0", "end-1c")
        save_config(self.config)

    # ── Actions ──────────────────────────────────────────────────────

    def _reset_edl(self) -> None:
        reset_template_file(get_edl_path())
        self._log("EDL file reset to template.")
        open_file_in_default_app(get_edl_path())

    def _reset_source(self) -> None:
        reset_template_file(get_source_path())
        self._log("Source file reset to template.")
        open_file_in_default_app(get_source_path())

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save output as...",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.output_var.set(path)

    def _on_convert(self) -> None:
        """Validate inputs, save config, and run conversion in a background thread."""
        self._save_ui_to_config()
        self._clear_log()

        edl_path = get_edl_path()
        source_path = get_source_path()
        output_path = self.output_var.get().strip()

        if not output_path:
            # Default output next to EDL in app data dir
            output_path = str(get_app_data_dir() / "DEF.csv")
            self.output_var.set(output_path)

        # Validate inputs
        self._log("Validating input files...")

        edl_errors = validate_edl_file(edl_path, fps=int(self.fps_var.get()))
        if edl_errors:
            self._log(f"\nEDL file errors ({edl_path}):")
            for err in edl_errors:
                self._log(f"  - {err}")
            self._log("\nPlease fix the errors above and try again.")
            return

        self._log("  EDL file OK")

        source_errors = validate_source_file(source_path)
        if source_errors:
            self._log(f"\nSource file errors ({source_path}):")
            for err in source_errors:
                self._log(f"  - {err}")
            self._log("\nPlease fix the errors above and try again.")
            return

        self._log("  Source file OK")

        # Parse exclusion rules from the text area
        exclusion_text = self.exclusion_text.get("1.0", "end-1c").strip()
        exclusion_rules = None
        if exclusion_text:
            try:
                exclusion_rules = parse_exclusion_rules(exclusion_text)
                self._log(f"  Loaded {len(exclusion_rules)} exclusion rules")
            except ExclusionRuleSyntaxError as e:
                self._log(f"\nExclusion rule error: {e}")
                self._log("Please fix the rule above and try again.")
                return

        # Disable button and run in background
        self.convert_btn.configure(state="disabled", text="Converting...")

        thread = threading.Thread(
            target=self._run_conversion,
            args=(edl_path, source_path, output_path, exclusion_rules),
            daemon=True,
        )
        thread.start()

    def _run_conversion(
        self,
        edl_path: Path,
        source_path: Path,
        output_path: str,
        exclusion_rules,
    ) -> None:
        """Run conversion in a background thread, capturing stdout."""
        fps = int(self.fps_var.get())
        collapse = self.collapse_var.get()
        delimiter = "," if self.delimiter_var.get() == "comma" else "\t"
        include_frames = self.frames_var.get()

        # Capture stdout to display in the log
        old_stdout = sys.stdout
        capture = io.StringIO()
        sys.stdout = capture

        try:
            convert(
                edl_path=edl_path,
                source_path=source_path,
                output_path=output_path,
                fps=fps,
                collapse=collapse,
                delimiter=delimiter,
                exclusion_rules=exclusion_rules,
                include_frames=include_frames,
            )
            sys.stdout = old_stdout
            output = capture.getvalue()
            self.after(0, self._on_conversion_done, output, output_path, None)
        except Exception as e:
            sys.stdout = old_stdout
            output = capture.getvalue()
            self.after(0, self._on_conversion_done, output, output_path, e)

    def _on_conversion_done(self, log_output: str, output_path: str, error: Exception | None) -> None:
        """Called on the main thread when conversion finishes."""
        if log_output:
            self._log(log_output)

        if error:
            self._log(f"\nConversion failed: {error}")
        else:
            self._log("\nDone!")
            # Open the Excel output if it exists
            excel_path = Path(output_path).with_suffix(".xlsx")
            if excel_path.exists():
                self._log(f"Opening {excel_path.name}...")
                open_file_in_default_app(excel_path)

        self.convert_btn.configure(state="normal", text="Convert")

    # ── Update Check ─────────────────────────────────────────────────

    def _on_update_check_result(self, result: dict | None) -> None:
        """Called from the background thread with the update check result."""
        if result is None:
            return
        # Schedule dialog on the main thread
        self.after(500, self._show_update_dialog, result)

    def _show_update_dialog(self, update_info: dict) -> None:
        skipped = self.config.get("skipped_version", "")
        if update_info["version"] == skipped:
            return
        UpdateDialog(self, update_info, self.config)

    # ── Logging ──────────────────────────────────────────────────────

    def _log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    # ── Lifecycle ────────────────────────────────────────────────────

    def destroy(self) -> None:
        self._save_ui_to_config()
        super().destroy()


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
