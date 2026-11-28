import html
from typing import Optional, Sequence

import pandas as pd

_TABLE_STYLE = "border-collapse:collapse;width:100%;font-size:13px;margin:6px 0;"
_TH_STYLE = (
    "text-align:left;padding:8px 10px;"
    "border:1px solid rgba(128,128,128,0.35);"
    "background:rgba(128,128,128,0.12);font-weight:600;"
)
_TD_BASE_STYLE = (
    "text-align:left;vertical-align:top;padding:8px 10px;"
    "border:1px solid rgba(128,128,128,0.35);"
    "white-space:pre-wrap;word-break:break-word;line-height:1.5;"
)
_TD_MONO_STYLE = (
    _TD_BASE_STYLE + "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px;"
)


def configure_notebook_display(
    max_colwidth: Optional[int] = None,
    max_columns: Optional[int] = None,
    width: int = 1000,
) -> None:
    options = (
        ("display.max_colwidth", max_colwidth),
        ("display.max_columns", max_columns),
        ("display.width", width),
        ("display.expand_frame_repr", False),
    )
    for key, value in options:
        pd.set_option(key, value)


def build_multiline_html_table(
    df: pd.DataFrame,
    monospace_columns: Sequence[str] = (),
) -> str:
    mono_set = frozenset(monospace_columns)
    col_styles = [_TD_MONO_STYLE if str(col) in mono_set else _TD_BASE_STYLE for col in df.columns]
    headers = "".join(f'<th style="{_TH_STYLE}">{html.escape(str(col))}</th>' for col in df.columns)
    body_rows = [
        "<tr>"
        + "".join(
            f'<td style="{style}">{html.escape(str(val))}</td>'
            for style, val in zip(col_styles, row_values)
        )
        + "</tr>"
        for row_values in df.itertuples(index=False, name=None)
    ]
    return (
        f'<table style="{_TABLE_STYLE}">'
        f"<thead><tr>{headers}</tr></thead>"
        f'<tbody>{"".join(body_rows)}</tbody>'
        "</table>"
    )


def display_multiline_table(
    df: pd.DataFrame,
    monospace_columns: Sequence[str] = (),
) -> None:
    from IPython.display import HTML, display

    display(HTML(build_multiline_html_table(df, monospace_columns=monospace_columns)))
