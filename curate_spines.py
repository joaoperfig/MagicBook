#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

import generate_spines as gs


SUFFIXES = ["a", "b", "c"]
DISPLAY_WIDTH = 220
DISPLAY_HEIGHT = 420


class CurateSpinesApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Curate Spines")

        gs.load_env_file(gs.BASE_DIR / ".env")
        self.prompt_template = gs.load_prompt()
        self.json_prompt_template = gs.load_json_prompt()

        self.final_dir = gs.PIC_DIR / "final"
        self.final_dir.mkdir(parents=True, exist_ok=True)

        rows = gs.load_curated_rows()
        self.rows = []
        for row in rows:
            if row.get("score") != "3" or not row.get("time"):
                continue
            time_slug = str(row.get("time", "")).strip().replace(":", "")
            if not time_slug:
                continue
            final_path = self.final_dir / f"{time_slug}.png"
            if final_path.exists():
                continue
            self.rows.append(row)
        self.total_rows = len(self.rows)
        self.current_index = 0

        self.selected_value = tk.StringVar(value="")
        self.current_image_paths: list[Path] = []
        self.current_images: list[ImageTk.PhotoImage | None] = []
        self.copy_preview_image: ImageTk.PhotoImage | None = None
        self.copy_source_path: Path | None = None
        self.copy_source_label = ""
        self.suppress_copy_prompt_once = False

        self.is_busy = False

        self._build_ui()
        self._bind_shortcuts()
        self._load_current_row()

    def _build_ui(self) -> None:
        self.root.geometry("900x720")
        self.root.minsize(900, 720)

        self.header_label = ttk.Label(self.root, text="", font=("Segoe UI", 12))
        self.header_label.pack(pady=10)

        self.copy_frame = ttk.Frame(self.root)
        self.copy_prompt_label = ttk.Label(
            self.copy_frame, text="", font=("Segoe UI", 11)
        )
        self.copy_prompt_label.pack(pady=8)

        self.copy_image_label = ttk.Label(self.copy_frame, text="No image")
        self.copy_image_label.pack(pady=8)

        copy_actions = ttk.Frame(self.copy_frame)
        copy_actions.pack(pady=8)

        self.copy_yes_button = ttk.Button(
            copy_actions, text="Yes", command=self._on_copy_yes
        )
        self.copy_yes_button.grid(row=0, column=0, padx=6)

        self.copy_no_button = ttk.Button(
            copy_actions, text="No", command=self._on_copy_no
        )
        self.copy_no_button.grid(row=0, column=1, padx=6)

        self.selection_frame = ttk.Frame(self.root)
        self.selection_frame.pack(fill="both", expand=True)

        images_frame = ttk.Frame(self.selection_frame)
        images_frame.pack(pady=10)

        self.image_labels: list[ttk.Label] = []
        self.radio_buttons: list[ttk.Radiobutton] = []
        for idx in range(3):
            column = ttk.Frame(images_frame)
            column.grid(row=0, column=idx, padx=10)

            image_label = ttk.Label(column, text="No image", anchor="center")
            image_label.bind(
                "<Button-1>",
                lambda _event, index=idx: self._select_spine(index),
            )
            image_label.pack()
            self.image_labels.append(image_label)

            radio = ttk.Radiobutton(
                column,
                text=f"Select {SUFFIXES[idx].upper()}",
                value=str(idx),
                variable=self.selected_value,
            )
            radio.pack(pady=6)
            self.radio_buttons.append(radio)

        actions = ttk.Frame(self.selection_frame)
        actions.pack(pady=10)

        self.redo_button = ttk.Button(actions, text="Redo", command=self._on_redo)
        self.redo_button.grid(row=0, column=0, padx=6)

        self.continue_button = ttk.Button(
            actions, text="Continue", command=self._on_continue
        )
        self.continue_button.grid(row=0, column=1, padx=6)

        self.status_label = ttk.Label(self.selection_frame, text="")
        self.status_label.pack(pady=6)

        self.cached_prompts_text = tk.Text(
            self.selection_frame, height=6, width=90, wrap="word"
        )
        self.cached_prompts_text.pack(padx=16, pady=6)
        self.cached_prompts_text.configure(state="disabled")

        self.redo_frame = ttk.Frame(self.root)

        redo_header = ttk.Label(
            self.redo_frame, text="Edit prompts and generate", font=("Segoe UI", 11)
        )
        redo_header.pack(pady=6)

        self.prompt_entries: list[tk.Text] = []
        for idx in range(3):
            prompt_label = ttk.Label(self.redo_frame, text=f"Prompt {idx + 1}")
            prompt_label.pack(anchor="w", padx=16)

            text = tk.Text(self.redo_frame, height=5, width=90, wrap="word")
            text.pack(padx=16, pady=4)
            self.prompt_entries.append(text)

        redo_actions = ttk.Frame(self.redo_frame)
        redo_actions.pack(pady=10)

        self.generate_new_button = ttk.Button(
            redo_actions, text="Generate New", command=self._on_generate_new
        )
        self.generate_new_button.grid(row=0, column=0, padx=6)

        self.generate_button = ttk.Button(
            redo_actions, text="Generate", command=self._on_generate
        )
        self.generate_button.grid(row=0, column=1, padx=6)

        self.cancel_button = ttk.Button(
            redo_actions, text="Back", command=self._show_selection_frame
        )
        self.cancel_button.grid(row=0, column=2, padx=6)

        self.redo_status_label = ttk.Label(self.redo_frame, text="")
        self.redo_status_label.pack(pady=6)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Key>", self._handle_keypress)

    def _handle_keypress(self, event: tk.Event) -> None:
        if self.is_busy:
            return
        key = (event.keysym or "").lower()
        if self.selection_frame.winfo_ismapped():
            if key in {"1", "2", "3"}:
                self.selected_value.set(str(int(key) - 1))
                return
            if key == "return":
                self._on_continue()
                return
            if key == "r":
                self._on_redo()
                return

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.is_busy = busy
        state = "disabled" if busy else "normal"
        for widget in [
            self.continue_button,
            self.redo_button,
            self.generate_new_button,
            self.generate_button,
            self.cancel_button,
            self.copy_yes_button,
            self.copy_no_button,
        ]:
            widget.configure(state=state)
        self.status_label.configure(text=message)
        self.redo_status_label.configure(text=message)

    def _load_current_row(self) -> None:
        if self.current_index >= self.total_rows:
            self.header_label.configure(text="All done.")
            self.selection_frame.pack_forget()
            self.redo_frame.pack_forget()
            self.copy_frame.pack_forget()
            return

        self.copy_frame.pack_forget()
        row = self.rows[self.current_index]
        time_label = (row.get("time") or "").strip()
        title = (row.get("title") or "").strip()
        author = (row.get("author") or "").strip()

        progress = f"{self.current_index + 1}/{self.total_rows}"
        self.header_label.configure(
            text=f"{time_label} | {title} | {author} | {progress}"
        )

        if self.suppress_copy_prompt_once:
            self.suppress_copy_prompt_once = False
        elif self._maybe_show_copy_prompt(time_label):
            return
        self._show_selection_frame()

        time_slug = time_label.replace(":", "")
        self.current_image_paths = [
            gs.OUTPUT_DIR / f"{time_slug}_{suffix}.png" for suffix in SUFFIXES
        ]
        self.current_images = []
        self.selected_value.set("")

        any_missing = False
        for idx, image_path in enumerate(self.current_image_paths):
            if image_path.exists():
                image = Image.open(image_path).convert("RGBA")
                image = self._crop_spine(image)
                image.thumbnail((DISPLAY_WIDTH, DISPLAY_HEIGHT))
                photo = ImageTk.PhotoImage(image)
                self.image_labels[idx].configure(image=photo, text="")
                self.image_labels[idx].image = photo
                self.current_images.append(photo)
                self.radio_buttons[idx].configure(state="normal")
            else:
                self.image_labels[idx].configure(text="Missing image", image="")
                self.image_labels[idx].image = None
                self.current_images.append(None)
                self.radio_buttons[idx].configure(state="disabled")
                any_missing = True

        if any_missing:
            self.status_label.configure(
                text="Missing images. Use Redo to generate new spines."
            )
            self.continue_button.configure(state="disabled")
        else:
            self.status_label.configure(text="")
            self.continue_button.configure(state="normal")

        cached_prompts = self._load_cached_prompts(time_label)
        self._set_cached_prompts_text(cached_prompts)

    def _on_continue(self) -> None:
        if self.is_busy:
            return
        selection = self.selected_value.get()
        if selection == "":
            messagebox.showwarning("Select a spine", "Please select a spine first.")
            return
        idx = int(selection)
        if idx < 0 or idx >= len(self.current_image_paths):
            return
        source_path = self.current_image_paths[idx]
        if not source_path.exists():
            messagebox.showerror("Missing file", "Selected image is missing.")
            return
        row = self.rows[self.current_index]
        time_label = (row.get("time") or "").strip()
        time_slug = time_label.replace(":", "")
        dest_path = self.final_dir / f"{time_slug}.png"
        shutil.copy2(source_path, dest_path)
        self.current_index += 1
        self._load_current_row()

    def _select_spine(self, index: int) -> None:
        if self.is_busy:
            return
        if index < 0 or index >= len(self.current_image_paths):
            return
        if not self.current_image_paths[index].exists():
            return
        self.selected_value.set(str(index))

    def _on_redo(self) -> None:
        if self.is_busy:
            return
        self._show_redo_frame()
        self._load_prompts_into_editor()

    def _on_generate_new(self) -> None:
        if self.is_busy:
            return
        row = self.rows[self.current_index]
        time_label = (row.get("time") or "").strip()
        title = (row.get("title") or "").strip()
        author = (row.get("author") or "").strip()
        prompt = self.prompt_template.format(
            title=title,
            author=author,
            time_label=time_label,
        ).strip()

        def task() -> list[str]:
            return gs.openai_chat_json(prompt, gs.DEFAULT_TEXT_MODEL)

        def on_success(descriptions: list[str]) -> None:
            for widget in self.prompt_entries:
                widget.delete("1.0", "end")
            for widget, text in zip(self.prompt_entries, descriptions, strict=True):
                widget.insert("1.0", text)
            self._show_redo_frame()
            self._focus_redo_prompt()
            self._set_busy(False, "")

        self._run_background(task, on_success, "Requesting new prompts...")

    def _on_generate(self) -> None:
        if self.is_busy:
            return
        row = self.rows[self.current_index]
        time_label = (row.get("time") or "").strip()
        title = (row.get("title") or "").strip()
        author = (row.get("author") or "").strip()
        time_slug = time_label.replace(":", "")

        descriptions = []
        for entry in self.prompt_entries:
            text = entry.get("1.0", "end").strip()
            if text:
                descriptions.append(text)
        if len(descriptions) < 3:
            messagebox.showwarning(
                "Missing prompts", "Please provide three prompt descriptions."
            )
            return
        descriptions = descriptions[:3]
        self._append_debug_prompts(time_label, title, author, descriptions)

        json_prompt = gs.build_image_prompt(
            self.json_prompt_template, descriptions, title, author
        )

        def task() -> list[Path]:
            layouts = gs.openai_chat_json_layouts(json_prompt, gs.DEFAULT_LAYOUT_MODEL)
            output_paths: list[Path] = []
            for suffix, layout in zip(SUFFIXES, layouts, strict=True):
                output_path = gs.OUTPUT_DIR / f"{time_slug}_{suffix}.png"
                image = gs.render_layout(layout)
                gs.save_layout_image(image, output_path)
                output_paths.append(output_path)
            return output_paths

        def on_success(_: list[Path]) -> None:
            self._show_selection_frame()
            self._load_current_row()
            self._set_busy(False, "")

        self._run_background(task, on_success, "Generating spines...")

    def _run_background(self, task, on_success, message: str) -> None:
        self._set_busy(True, message)

        def runner() -> None:
            try:
                result = task()
            except Exception as exc:  # noqa: BLE001
                self.root.after(0, self._on_error, exc)
                return
            self.root.after(0, on_success, result)

        threading.Thread(target=runner, daemon=True).start()

    def _on_error(self, exc: Exception) -> None:
        self._set_busy(False, "")
        messagebox.showerror("Error", str(exc))

    def _show_redo_frame(self) -> None:
        self.selection_frame.pack_forget()
        self.redo_frame.pack(fill="both", expand=True)
        self._focus_redo_prompt()

    def _show_selection_frame(self) -> None:
        self.redo_frame.pack_forget()
        self.copy_frame.pack_forget()
        self.selection_frame.pack(fill="both", expand=True)

    def _focus_redo_prompt(self) -> None:
        if not self.prompt_entries:
            return
        first = self.prompt_entries[0]
        first.focus_set()
        first.tag_add("sel", "1.0", "end")

    def _load_prompts_into_editor(self) -> None:
        row = self.rows[self.current_index]
        time_label = (row.get("time") or "").strip()
        cached = self._load_cached_prompts(time_label)
        for widget in self.prompt_entries:
            widget.delete("1.0", "end")
        if cached:
            for widget, text in zip(self.prompt_entries, cached, strict=True):
                widget.insert("1.0", text)
        self._focus_redo_prompt()

    def _set_cached_prompts_text(self, prompts: list[str]) -> None:
        self.cached_prompts_text.configure(state="normal")
        self.cached_prompts_text.delete("1.0", "end")
        if prompts:
            lines = [f"{label.upper()}: {text}" for label, text in zip(SUFFIXES, prompts)]
            self.cached_prompts_text.insert("1.0", "\n\n".join(lines))
        else:
            self.cached_prompts_text.insert("1.0", "No cached prompts.")
        self.cached_prompts_text.configure(state="disabled")

    def _load_cached_prompts(self, time_label: str) -> list[str]:
        if not gs.DEBUG_LAYOUT_JSONL.exists():
            return []
        entries: list[dict] = []
        try:
            with gs.DEBUG_LAYOUT_JSONL.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("time") == time_label and data.get("description"):
                        entries.append(data)
        except OSError:
            return []

        if not entries:
            return []

        by_suffix: dict[str, str] = {}
        extras: list[str] = []
        for entry in entries:
            description = str(entry.get("description", "")).strip()
            if not description:
                continue
            output = str(entry.get("output", ""))
            suffix = None
            for letter in SUFFIXES:
                if f"_{letter}.png" in output:
                    suffix = letter
                    break
            if suffix:
                by_suffix[suffix] = description
            else:
                extras.append(description)

        result: list[str] = []
        for letter in SUFFIXES:
            if letter in by_suffix:
                result.append(by_suffix[letter])
            elif extras:
                result.append(extras.pop(0))
            else:
                result.append("")

        if all(not item for item in result):
            return []
        return result

    def _append_debug_prompts(
        self, time_label: str, title: str, author: str, descriptions: list[str]
    ) -> None:
        time_slug = time_label.replace(":", "")
        gs.DEBUG_LAYOUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with gs.DEBUG_LAYOUT_JSONL.open("a", encoding="utf-8") as handle:
            for suffix, description in zip(SUFFIXES, descriptions, strict=True):
                entry = {
                    "time": time_label,
                    "title": title,
                    "author": author,
                    "description": description,
                    "output": str(gs.OUTPUT_DIR / f"{time_slug}_{suffix}.png"),
                    "layout": None,
                }
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _maybe_show_copy_prompt(self, time_label: str) -> bool:
        try:
            total_minutes = gs.parse_time_label(time_label)
        except ValueError:
            return False
        if total_minutes < 12 * 60:
            return False
        source_minutes = total_minutes - 12 * 60
        source_label = f"{source_minutes // 60:02d}:{source_minutes % 60:02d}"
        source_slug = source_label.replace(":", "")
        source_path = self.final_dir / f"{source_slug}.png"
        if not source_path.exists():
            return False
        self.copy_source_path = source_path
        self.copy_source_label = source_label
        self._show_copy_frame()
        return True

    def _show_copy_frame(self) -> None:
        if not self.copy_source_path:
            return
        self.selection_frame.pack_forget()
        self.redo_frame.pack_forget()
        self.copy_prompt_label.configure(
            text=f"Copy spine for {self.copy_source_label}?"
        )
        image = Image.open(self.copy_source_path).convert("RGBA")
        image.thumbnail((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        photo = ImageTk.PhotoImage(image)
        self.copy_preview_image = photo
        self.copy_image_label.configure(image=photo, text="")
        self.copy_image_label.image = photo
        self.copy_frame.pack(fill="both", expand=True)

    def _on_copy_yes(self) -> None:
        if self.is_busy or not self.copy_source_path:
            return
        row = self.rows[self.current_index]
        time_label = (row.get("time") or "").strip()
        time_slug = time_label.replace(":", "")
        dest_path = self.final_dir / f"{time_slug}.png"
        shutil.copy2(self.copy_source_path, dest_path)
        self.current_index += 1
        self.copy_source_path = None
        self.copy_source_label = ""
        self._load_current_row()

    def _on_copy_no(self) -> None:
        if self.is_busy:
            return
        self.copy_source_path = None
        self.copy_source_label = ""
        self.suppress_copy_prompt_once = True
        self._load_current_row()

    @staticmethod
    def _crop_spine(image: Image.Image) -> Image.Image:
        width, height = image.size
        target_width = 240
        target_height = 800
        x0 = max(0, int(round((width - target_width) / 2)))
        y0 = 0
        x1 = min(width, x0 + target_width)
        y1 = min(height, target_height)
        return image.crop((x0, y0, x1, y1))


def main() -> None:
    root = tk.Tk()
    app = CurateSpinesApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
