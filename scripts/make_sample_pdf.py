"""Synthesize a small ICS PDF (fault table + register map + text) for local
testing, exercising both text and table chunking paths."""

from __future__ import annotations

from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_vfd.pdf"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()

    _page_text(
        doc,
        title="VFD Series 9000 — Overview",
        body=(
            "The Series 9000 Variable Frequency Drive is a three-phase inverter "
            "rated for 0.75 kW to 22 kW motors. It supports Modbus RTU over RS-485 "
            "and provides 12 configurable digital inputs and 4 analog outputs.\n\n"
            "This manual covers fault codes, the Modbus register map, and "
            "commissioning steps for the standard variant (part number "
            "VFD9000-22A-S)."
        ),
    )

    _page_table(
        doc,
        title="Fault Codes",
        headers=["Code", "Name", "Description"],
        rows=[
            ["0x10", "Undervoltage", "DC bus below 380 V threshold."],
            ["0x21", "Overcurrent", "Output current exceeded 150% of rated for 1 s."],
            ["0x22", "Overvoltage", "DC bus exceeded 820 V."],
            ["0x33", "Overtemperature", "Heatsink above 85 C."],
            ["0x40", "Ground Fault", "Detected earth leakage on phase output."],
        ],
    )

    _page_table(
        doc,
        title="Modbus Register Map (Holding Registers)",
        headers=["Address", "Name", "Type", "Access"],
        rows=[
            ["0x2000", "Run Command", "U16", "R/W"],
            ["0x2001", "Output Current", "U16", "R"],
            ["0x2002", "Output Frequency", "U16", "R"],
            ["0x2003", "DC Bus Voltage", "U16", "R"],
            ["0x2010", "Acceleration Time", "U16", "R/W"],
            ["0x2011", "Deceleration Time", "U16", "R/W"],
        ],
    )

    _page_text(
        doc,
        title="Commissioning Steps",
        body=(
            "1. Verify mains voltage is within 380-480 V AC before applying power.\n"
            "2. Connect motor leads to U, V, W terminals; observe rotation.\n"
            "3. Set parameter P0.01 to motor rated current in amps.\n"
            "4. Set parameter P0.02 to motor rated frequency (50 or 60 Hz).\n"
            "5. Run autotune via keypad: MENU > AUTOTUNE > START.\n"
            "6. Verify fault code register reads 0x00 (no fault) after autotune.\n\n"
            "For Schneider charging station integration, refer to the EV9000 manual."
        ),
    )

    doc.save(OUT)
    doc.close()
    print(f"wrote {OUT}")


def _page_text(doc: pymupdf.Document, *, title: str, body: str) -> None:
    page = doc.new_page()
    page.insert_text((72, 72), title, fontsize=16, fontname="helv")
    page.insert_textbox(
        pymupdf.Rect(72, 108, 540, 720),
        body,
        fontsize=11,
        fontname="helv",
    )


def _page_table(
    doc: pymupdf.Document,
    *,
    title: str,
    headers: list[str],
    rows: list[list[str]],
) -> None:
    page = doc.new_page()
    page.insert_text((72, 72), title, fontsize=16, fontname="helv")
    n_cols = len(headers)
    col_w = (540 - 72) / n_cols
    y = 108
    row_h = 22
    grid = [headers, *rows]
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            x0 = 72 + c * col_w
            x1 = x0 + col_w
            y0 = y + r * row_h
            y1 = y0 + row_h
            page.draw_rect(pymupdf.Rect(x0, y0, x1, y1), color=(0, 0, 0), width=0.5)
            page.insert_text(
                (x0 + 4, y0 + 15),
                cell,
                fontsize=10,
                fontname="hebo" if r == 0 else "helv",
            )


if __name__ == "__main__":
    main()
