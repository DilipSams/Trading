"""
Professional Table Formatter for Trading System
================================================

Provides beautifully formatted tables with box-drawing characters.
Compatible with both terminal and file output.
Handles ANSI color codes correctly for width calculations.
"""

import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

# ANSI escape code pattern — matches color/style sequences like \033[32m
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def _strip_ansi(s: str) -> str:
    """Remove all ANSI escape codes from a string."""
    return _ANSI_RE.sub('', str(s))


def _visible_len(s: str) -> int:
    """Return the visible length of a string (ignoring ANSI codes)."""
    return len(_strip_ansi(s))


@dataclass
class ColumnConfig:
    """Configuration for a single column."""
    name: str
    width: int = 0  # Auto-calculate if 0
    align: str = 'left'  # 'left', 'right', 'center'
    format_spec: str = ''  # e.g., '.2f' for floats


class TableFormatter:
    """
    Professional table formatter with box-drawing characters.
    Auto-calculates column widths from data. Handles ANSI color codes.

    Usage:
        formatter = TableFormatter()
        formatter.add_column('Symbol', align='left')
        formatter.add_column('Sharpe', align='right', format_spec='.3f')
        formatter.add_row(['MSFT', 1.234])
        print(formatter.render())
    """

    # Box-drawing characters
    TOP_LEFT = '┌'
    TOP_RIGHT = '┐'
    BOTTOM_LEFT = '└'
    BOTTOM_RIGHT = '┘'
    HORIZONTAL = '─'
    VERTICAL = '│'
    T_DOWN = '┬'
    T_UP = '┴'
    T_RIGHT = '├'
    T_LEFT = '┤'
    CROSS = '┼'

    # Alternative: Double-line characters for headers
    DOUBLE_HORIZONTAL = '═'

    def __init__(self, title: Optional[str] = None, width: Optional[int] = None):
        """
        Initialize table formatter.

        Args:
            title: Optional table title
            width: Fixed table width (None = auto-calculate)
        """
        self.title = title
        self.fixed_width = width
        self.columns: List[ColumnConfig] = []
        self.rows: List[List[Any]] = []
        self.header_groups: Optional[List[tuple]] = None

    def add_column(self, name: str, width: int = 0, align: str = 'left',
                   format_spec: str = ''):
        """Add a column definition. width=0 means auto-calculate from data."""
        self.columns.append(ColumnConfig(name, width, align, format_spec))

    def add_row(self, values: List[Any]):
        """Add a data row."""
        if len(values) != len(self.columns):
            raise ValueError(f"Row has {len(values)} values but table has {len(self.columns)} columns")
        self.rows.append(values)

    def set_header_groups(self, groups: List[tuple]):
        """
        Set column groups for a two-row header.

        Args:
            groups: List of (group_name, n_columns) tuples.
                    Sum of n_columns must equal len(self.columns).
                    Use '' for groups with no spanning header.
        """
        total = sum(gc for _, gc in groups)
        if self.columns and total != len(self.columns):
            raise ValueError(
                f"Header groups span {total} columns but table has "
                f"{len(self.columns)} columns"
            )
        self.header_groups = groups

    def _format_value(self, value: Any, col: ColumnConfig) -> str:
        """Format a single value according to column spec."""
        if value is None:
            return ''

        if col.format_spec:
            try:
                if isinstance(value, (int, float)):
                    return f"{value:{col.format_spec}}"
            except:
                pass

        return str(value)

    def _calculate_widths(self) -> List[int]:
        """Calculate optimal column widths using visible length (ANSI-aware)."""
        widths = []

        for i, col in enumerate(self.columns):
            if col.width > 0:
                widths.append(col.width)
            else:
                # Auto-calculate: max of header and all values (visible chars only)
                max_width = _visible_len(col.name)
                for row in self.rows:
                    formatted = self._format_value(row[i], col)
                    max_width = max(max_width, _visible_len(formatted))
                widths.append(max_width + 2)  # Add padding

        # Widen columns if any group header name exceeds its span
        if self.header_groups:
            col_idx = 0
            for gname, gcols in self.header_groups:
                span_width = sum(widths[col_idx:col_idx + gcols]) + (gcols - 1)
                needed = _visible_len(gname) + 2  # 1-char padding each side
                if needed > span_width:
                    deficit = needed - span_width
                    per_col = deficit // gcols
                    remainder = deficit % gcols
                    for k in range(gcols):
                        widths[col_idx + k] += per_col + (1 if k < remainder else 0)
                col_idx += gcols

        return widths

    def _align_text(self, text: str, width: int, align: str) -> str:
        """Align text within given width (ANSI-aware padding)."""
        text = str(text)
        vis_len = _visible_len(text)
        inner = width - 2  # space for 1-char padding on each side

        if vis_len >= inner:
            # Text fills or exceeds the inner width — just add minimal padding
            return ' ' + text + ' '

        pad = inner - vis_len
        if align == 'right':
            return ' ' * (pad + 1) + text + ' '
        elif align == 'center':
            lp = pad // 2
            rp = pad - lp
            return ' ' * (lp + 1) + text + ' ' * (rp + 1)
        else:  # left
            return ' ' + text + ' ' * (pad + 1)

    def _render_separator(self, widths: List[int], left: str, mid: str,
                         right: str, horiz: str) -> str:
        """Render a separator line."""
        parts = [left]
        for i, width in enumerate(widths):
            parts.append(horiz * width)
            if i < len(widths) - 1:
                parts.append(mid)
        parts.append(right)
        return ''.join(parts)

    def render(self) -> str:
        """Render the complete table."""
        if not self.columns:
            return ""

        # Validate group column count (deferred from set_header_groups
        # in case groups were set before columns were added)
        if self.header_groups:
            total_gc = sum(gc for _, gc in self.header_groups)
            if total_gc != len(self.columns):
                raise ValueError(
                    f"Header groups span {total_gc} columns but table has "
                    f"{len(self.columns)} columns"
                )

        widths = self._calculate_widths()
        total_width = sum(widths) + len(widths) + 1

        lines = []

        # Title (if provided)
        if self.title:
            title_vis_len = _visible_len(self.title)
            lines.append(self.DOUBLE_HORIZONTAL * total_width)
            # Center title using visible length
            pad = total_width - title_vis_len
            lp = pad // 2
            rp = pad - lp
            title_line = ' ' * lp + self.title + ' ' * rp
            lines.append(title_line)
            lines.append(self.DOUBLE_HORIZONTAL * total_width)
            lines.append('')

        if self.header_groups:
            # === Two-row header with column groups ===

            # Row 1 top border: group boundaries get ┬, internal sub-cols get ─
            top_parts = [self.TOP_LEFT]
            col_idx = 0
            for gi, (gname, gcols) in enumerate(self.header_groups):
                span_width = sum(widths[col_idx:col_idx + gcols]) + (gcols - 1)
                top_parts.append(self.HORIZONTAL * span_width)
                col_idx += gcols
                if col_idx < len(widths):
                    top_parts.append(self.T_DOWN)
            top_parts.append(self.TOP_RIGHT)
            lines.append(''.join(top_parts))

            # Row 1 content: group names centered in their span
            grp_parts = [self.VERTICAL]
            col_idx = 0
            for gname, gcols in self.header_groups:
                span_width = sum(widths[col_idx:col_idx + gcols]) + (gcols - 1)
                grp_parts.append(self._align_text(gname, span_width, 'center'))
                col_idx += gcols
                grp_parts.append(self.VERTICAL)
            lines.append(''.join(grp_parts))

            # Separator between group row and sub-header row
            sep_parts = [self.T_RIGHT]
            col_idx = 0
            for gi, (gname, gcols) in enumerate(self.header_groups):
                for j in range(gcols):
                    sep_parts.append(self.HORIZONTAL * widths[col_idx])
                    col_idx += 1
                    if col_idx < len(widths):
                        # Check if this is a group boundary or internal sub-col boundary
                        next_group_start = sum(gc for _, gc in self.header_groups[:gi + 1])
                        if col_idx == next_group_start:
                            sep_parts.append(self.CROSS)
                        else:
                            sep_parts.append(self.T_DOWN)
            sep_parts.append(self.T_LEFT)
            lines.append(''.join(sep_parts))

            # Row 2: sub-column names
            header_parts = [self.VERTICAL]
            for col, width in zip(self.columns, widths):
                header_parts.append(self._align_text(col.name, width, 'center'))
                header_parts.append(self.VERTICAL)
            lines.append(''.join(header_parts))

            # Separator after sub-header
            lines.append(self._render_separator(widths, self.T_RIGHT, self.CROSS,
                                                self.T_LEFT, self.HORIZONTAL))
        else:
            # === Standard single-row header ===

            # Top border
            lines.append(self._render_separator(widths, self.TOP_LEFT, self.T_DOWN,
                                                self.TOP_RIGHT, self.HORIZONTAL))

            # Header row
            header_parts = [self.VERTICAL]
            for col, width in zip(self.columns, widths):
                header_parts.append(self._align_text(col.name, width, 'center'))
                header_parts.append(self.VERTICAL)
            lines.append(''.join(header_parts))

            # Header separator
            lines.append(self._render_separator(widths, self.T_RIGHT, self.CROSS,
                                                self.T_LEFT, self.HORIZONTAL))

        # Data rows
        for row in self.rows:
            row_parts = [self.VERTICAL]
            for value, col, width in zip(row, self.columns, widths):
                formatted = self._format_value(value, col)
                row_parts.append(self._align_text(formatted, width, col.align))
                row_parts.append(self.VERTICAL)
            lines.append(''.join(row_parts))

        # Bottom border
        lines.append(self._render_separator(widths, self.BOTTOM_LEFT, self.T_UP,
                                            self.BOTTOM_RIGHT, self.HORIZONTAL))

        return '\n'.join(lines)

    def print(self):
        """Print the table to console."""
        print(self.render())

    @classmethod
    def from_data(cls, columns: List[Dict[str, Any]], rows: List[List[Any]],
                  title: Optional[str] = None) -> 'TableFormatter':
        """
        One-call table creation with auto-width.

        Args:
            columns: List of dicts with keys: name, align (optional), format_spec (optional)
            rows: List of row value lists
            title: Optional table title

        Returns:
            TableFormatter instance ready to render

        Example:
            table = TableFormatter.from_data(
                columns=[
                    {'name': 'Symbol', 'align': 'left'},
                    {'name': 'Sharpe', 'align': 'right', 'format_spec': '.3f'},
                ],
                rows=[['MSFT', 1.234], ['AAPL', 2.567]],
                title="RESULTS"
            )
            print(table.render())
        """
        t = cls(title=title)
        for col in columns:
            t.add_column(
                name=col['name'],
                width=col.get('width', 0),
                align=col.get('align', 'left'),
                format_spec=col.get('format_spec', ''),
            )
        for row in rows:
            t.add_row(row)
        return t


def format_alpha_results(results: List[Dict[str, Any]]) -> str:
    """
    Format alpha validation results as a professional table.

    Args:
        results: List of dicts with keys: name, verdict, t_stat, sharpe_is,
                 sharpe_oos, decay, dsr, pbo, n_is, n_oos

    Returns:
        Formatted table string
    """
    table = TableFormatter(title="ALPHA VALIDATION RESULTS")

    table.add_column('Alpha', align='left')
    table.add_column('Verdict', align='center')
    table.add_column('t-stat', align='right', format_spec='+.2f')
    table.add_column('Sh(IS)', align='right', format_spec='+.2f')
    table.add_column('Sh(OOS)', align='right', format_spec='+.3f')
    table.add_column('Decay', align='right', format_spec='.2f')
    table.add_column('DSR', align='right', format_spec='.2f')
    table.add_column('PBO', align='right', format_spec='.2f')
    table.add_column('n_IS', align='right')
    table.add_column('n_OOS', align='right')

    for r in results:
        table.add_row([
            r.get('name', ''),
            r.get('verdict', ''),
            r.get('t_stat', 0),
            r.get('sharpe_is', 0),
            r.get('sharpe_oos', 0),
            r.get('decay', 0),
            r.get('dsr', 0),
            r.get('pbo', 0),
            r.get('n_is', 0),
            r.get('n_oos', 0),
        ])

    return table.render()


def format_backtest_results(symbol: str, results: Dict[str, Any]) -> str:
    """
    Format backtest results as a professional table.

    Args:
        symbol: Symbol name
        results: Dict with metrics (sharpe, dd, return, trades, etc.)

    Returns:
        Formatted table string
    """
    table = TableFormatter(title=f"BACKTEST RESULTS: {symbol}")

    table.add_column('Metric', align='left')
    table.add_column('Value', align='right')

    # Add rows
    metrics = [
        ('Total Return', f"{results.get('total_return', 0):.2f}%"),
        ('Sharpe Ratio', f"{results.get('sharpe', 0):.3f}"),
        ('Max Drawdown', f"{results.get('max_dd', 0):.2f}%"),
        ('Win Rate', f"{results.get('win_rate', 0):.1f}%"),
        ('Avg Win', f"{results.get('avg_win', 0):.2f}%"),
        ('Avg Loss', f"{results.get('avg_loss', 0):.2f}%"),
        ('Win/Loss Ratio', f"{results.get('win_loss_ratio', 0):.2f}x"),
        ('Total Trades', f"{results.get('total_trades', 0)}"),
    ]

    for metric, value in metrics:
        table.add_row([metric, value])

    return table.render()


def format_comparison_table(symbols: List[str], baseline: Dict[str, Dict],
                            enhanced: Dict[str, Dict], metric_name: str) -> str:
    """
    Format comparison table (baseline vs enhanced).

    Args:
        symbols: List of symbol names
        baseline: Dict[symbol -> metrics]
        enhanced: Dict[symbol -> metrics]
        metric_name: Metric to compare ('sharpe', 'dd', 'return')

    Returns:
        Formatted table string
    """
    table = TableFormatter(title=f"{metric_name.upper()} COMPARISON: Baseline vs Enhanced")

    table.add_column('Symbol', align='left')
    table.add_column('Baseline', align='right', format_spec='.3f')
    table.add_column('Enhanced', align='right', format_spec='.3f')
    table.add_column('Change', align='right', format_spec='+.3f')
    table.add_column('Change %', align='right', format_spec='+.1f')

    for sym in symbols:
        base_val = baseline[sym].get(metric_name, 0)
        enh_val = enhanced[sym].get(metric_name, 0)
        change = enh_val - base_val

        if abs(base_val) > 1e-6:
            change_pct = (change / abs(base_val)) * 100
        else:
            change_pct = 0

        table.add_row([sym, base_val, enh_val, change, change_pct])

    return table.render()


# Example usage and tests
if __name__ == "__main__":
    print("=" * 100)
    print("TABLE FORMATTER EXAMPLES")
    print("=" * 100)

    # Example 1: Alpha validation results (auto-width)
    print("\n[Example 1] Alpha Validation Results (auto-width)")
    print("-" * 100)

    alpha_results = [
        {'name': 'trend_follow', 'verdict': 'REJECT', 't_stat': 8.28, 'sharpe_is': 0.24,
         'sharpe_oos': 0.188, 'decay': 0.77, 'dsr': 1.00, 'pbo': 0.00, 'n_is': 552599, 'n_oos': 491627},
        {'name': 'mean_reversion', 'verdict': 'PASS', 't_stat': 4.31, 'sharpe_is': -0.17,
         'sharpe_oos': 0.092, 'decay': -0.54, 'dsr': 1.00, 'pbo': 0.20, 'n_is': 628514, 'n_oos': 557371},
        {'name': 'volume_price_divergence', 'verdict': 'PASS', 't_stat': 11.68, 'sharpe_is': 0.08,
         'sharpe_oos': 0.250, 'decay': 2.96, 'dsr': 1.00, 'pbo': 0.00, 'n_is': 618319, 'n_oos': 552007},
    ]

    print(format_alpha_results(alpha_results))

    # Example 2: Backtest results (auto-width)
    print("\n\n[Example 2] Backtest Results (auto-width)")
    print("-" * 100)

    backtest_data = {
        'total_return': 142.91,
        'sharpe': 0.859,
        'max_dd': 37.15,
        'win_rate': 58.5,
        'avg_win': 5.2,
        'avg_loss': -3.1,
        'win_loss_ratio': 1.68,
        'total_trades': 245,
    }

    print(format_backtest_results('MSFT', backtest_data))

    # Example 3: Custom table with hardcoded widths (still works)
    print("\n\n[Example 3] Custom Table (hardcoded widths)")
    print("-" * 100)

    table = TableFormatter(title="ASYMMETRIC STOPS: 5-SYMBOL TEST")
    table.add_column('Symbol', width=10, align='left')
    table.add_column('Sharpe', width=10, align='right', format_spec='.3f')
    table.add_column('Max DD', width=10, align='right', format_spec='.2f')
    table.add_column('Return', width=12, align='right', format_spec='+.2f')
    table.add_column('Stops', width=10, align='right')

    table.add_row(['MSFT', 0.542, 19.66, 49.49, '24L 18T'])
    table.add_row(['AAPL', 1.313, 24.35, 210.78, '25L 17T'])
    table.add_row(['SPY', 1.199, 11.50, 105.23, '9L 13T'])
    table.add_row(['GOOGL', 0.528, 28.19, 52.01, '27L 14T'])
    table.add_row(['META', 0.457, 33.24, 47.64, '33L 16T'])

    table.print()

    # Example 4: ANSI color codes handled correctly
    print("\n\n[Example 4] ANSI Color Codes (auto-width)")
    print("-" * 100)

    table = TableFormatter(title="COLORED VALUES TEST")
    table.add_column('Alpha', align='left')
    table.add_column('Change', align='right')
    table.add_column('Status', align='center')

    table.add_row(['volume_price_divergence', '\033[32m+0.042\033[0m', '\033[32mPASS\033[0m'])
    table.add_row(['trend_follow', '\033[31m-0.018\033[0m', '\033[31mFAIL\033[0m'])
    table.add_row(['mean_reversion', '\033[32m+0.125\033[0m', '\033[32mPASS\033[0m'])

    table.print()

    # Example 5: from_data() classmethod
    print("\n\n[Example 5] from_data() One-Call Creation")
    print("-" * 100)

    t = TableFormatter.from_data(
        columns=[
            {'name': 'Symbol', 'align': 'left'},
            {'name': 'Sharpe', 'align': 'right', 'format_spec': '.3f'},
            {'name': 'Return %', 'align': 'right', 'format_spec': '+.2f'},
        ],
        rows=[
            ['MSFT', 0.542, 49.49],
            ['AAPL', 1.313, 210.78],
            ['SPY', 1.199, 105.23],
        ],
        title="QUICK TABLE"
    )
    t.print()

    print("\n" + "=" * 100)
    print("[EXAMPLES COMPLETE]")
    print("=" * 100)
