"""SUMO network XML generation for NxN intersection grids.

Generates ``grid_{n}x{n}.net.xml`` and ``grid_{n}x{n}.sumocfg`` by
templating the single-intersection ``environment.net.xml`` structure and
substituting junction positions and edge IDs programmatically.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Lane geometry constants (taken from environment.net.xml)
# ---------------------------------------------------------------------------

# Distance from junction centre to the junction boundary (edge entry/exit point)
JB = 16.80  # junction boundary offset in metres

# Lane lateral offsets from junction centre for each approach direction.
# Index = lane number within edge (0-3).
_W_Y = [-11.20, -8.00, -4.80, -1.60]  # W2TL approach: y offsets
_N_X = [-11.20, -8.00, -4.80, -1.60]  # N2TL approach: x offsets
_E_Y = [11.20, 8.00, 4.80, 1.60]      # E2TL approach: y offsets
_S_X = [11.20, 8.00, 4.80, 1.60]      # S2TL approach: x offsets

# Outgoing lane offsets
_E_OUT_Y = [-11.20, -8.00, -4.80, -1.60]   # TL2E exit (matches W2TL approach)
_W_OUT_Y = [11.20, 8.00, 4.80, 1.60]       # TL2W exit (matches E2TL approach)
_N_OUT_X = [11.20, 8.00, 4.80, 1.60]       # TL2N exit (matches S2TL approach)
_S_OUT_X = [-11.20, -8.00, -4.80, -1.60]   # TL2S exit (matches N2TL approach)

LANE_SPEED = 13.89  # m/s

# ---------------------------------------------------------------------------
# Internal junction-box edge templates
# Coordinates are RELATIVE to junction centre (will be offset per junction).
# Format: (edge_suffix, [(lane_index, speed, length, [(x0,y0),(x1,y1),...]),... ])
# ---------------------------------------------------------------------------

_INTERNAL_TEMPLATES: list[tuple[str, list[tuple[int, float, float, list[tuple[float, float]]]]]] = [
    # N right-turn (N→W)
    ("0", [(0, 6.51, 9.03, [(-11.20, 16.80), (-11.55, 14.35), (-12.60, 12.60), (-14.35, 11.55), (-16.80, 11.20)])]),
    # N straight (N→S), 3 lanes
    ("1", [
        (0, 13.89, 33.60, [(-11.20, 16.80), (-11.20, -16.80)]),
        (1, 13.89, 33.60, [(-8.00, 16.80), (-8.00, -16.80)]),
        (2, 13.89, 33.60, [(-4.80, 16.80), (-4.80, -16.80)]),
    ]),
    # N left-turn (N→E)
    ("4", [(0, 11.36, 29.67, [(-1.60, 16.80), (-0.45, 8.75), (3.00, 3.00), (8.75, -0.45), (16.80, -1.60)])]),
    # E right-turn (E→N)
    ("5", [(0, 6.51, 9.03, [(16.80, 11.20), (14.35, 11.55), (12.60, 12.60), (11.55, 14.35), (11.20, 16.80)])]),
    # E straight (E→W), 3 lanes
    ("6", [
        (0, 13.89, 33.60, [(16.80, 11.20), (-16.80, 11.20)]),
        (1, 13.89, 33.60, [(16.80, 8.00), (-16.80, 8.00)]),
        (2, 13.89, 33.60, [(16.80, 4.80), (-16.80, 4.80)]),
    ]),
    # E left-turn (E→S)
    ("9", [(0, 11.36, 29.67, [(16.80, 1.60), (8.75, 0.45), (3.00, -3.00), (-0.45, -8.75), (-1.60, -16.80)])]),
    # S right-turn (S→E)
    ("10", [(0, 6.51, 9.03, [(11.20, -16.80), (11.55, -14.35), (12.60, -12.60), (14.35, -11.55), (16.80, -11.20)])]),
    # S straight (S→N), 3 lanes
    ("11", [
        (0, 13.89, 33.60, [(11.20, -16.80), (11.20, 16.80)]),
        (1, 13.89, 33.60, [(8.00, -16.80), (8.00, 16.80)]),
        (2, 13.89, 33.60, [(4.80, -16.80), (4.80, 16.80)]),
    ]),
    # S left-turn (S→W)
    ("14", [(0, 11.36, 29.67, [(1.60, -16.80), (0.45, -8.75), (-3.00, -3.00), (-8.75, 0.45), (-16.80, 1.60)])]),
    # W right-turn (W→S)
    ("15", [(0, 6.51, 9.03, [(-16.80, -11.20), (-14.35, -11.55), (-12.60, -12.60), (-11.55, -14.35), (-11.20, -16.80)])]),
    # W straight (W→E), 3 lanes
    ("16", [
        (0, 13.89, 33.60, [(-16.80, -11.20), (16.80, -11.20)]),
        (1, 13.89, 33.60, [(-16.80, -8.00), (16.80, -8.00)]),
        (2, 13.89, 33.60, [(-16.80, -4.80), (16.80, -4.80)]),
    ]),
    # W left-turn (W→N)
    ("19", [(0, 11.36, 29.67, [(-16.80, -1.60), (-8.75, -0.45), (-3.00, 3.00), (0.45, 8.75), (1.60, 16.80)])]),
]

# Request matrix (same for all junctions — topology is identical)
_REQUESTS = [
    (0,  "00000000000000000000", "00000100000111000000"),
    (1,  "00000000000000000000", "11111100001111000000"),
    (2,  "00000000000000000000", "11111100001111000000"),
    (3,  "00000000000000000000", "11111100001111000000"),
    (4,  "00000011110000000000", "11110011111111000000"),
    (5,  "00000011100000000000", "10000011100000000000"),
    (6,  "00000111100000011111", "10000111100000011111"),
    (7,  "00000111100000011111", "10000111100000011111"),
    (8,  "00000111100000011111", "10000111100000011111"),
    (9,  "01111111100000011110", "01111111100000011110"),
    (10, "00000000000000000000", "01110000000000010000"),
    (11, "00000000000000000000", "11110000001111110000"),
    (12, "00000000000000000000", "11110000001111110000"),
    (13, "00000000000000000000", "11110000001111110000"),
    (14, "00000000000000001111", "11110000001111001111"),
    (15, "00000000000000001110", "00000000001000001110"),
    (16, "00000111110000011110", "00000111111000011110"),
    (17, "00000111110000011110", "00000111111000011110"),
    (18, "00000111110000011110", "00000111111000011110"),
    (19, "00000111100111111110", "00000111100111111110"),
]

# Junction shape polygon (relative to centre)
_TL_SHAPE_REL = [
    (-12.80, 16.80), (12.80, 16.80), (13.24, 14.58), (13.80, 13.80),
    (14.58, 13.24), (15.58, 12.91), (16.80, 12.80), (16.80, -12.80),
    (14.58, -13.24), (13.80, -13.80), (13.24, -14.58), (12.91, -15.58),
    (12.80, -16.80), (-12.80, -16.80), (-13.24, -14.58), (-13.80, -13.80),
    (-14.58, -13.24), (-15.58, -12.91), (-16.80, -12.80), (-16.80, 12.80),
    (-14.58, 13.24), (-13.80, 13.80), (-13.24, 14.58), (-12.91, 15.58),
]

# TL phase logic (same 8 phases for all junctions)
_TL_PHASES = [
    (100, "GGGGrrrrrrGGGGrrrrrr"),
    (100, "yyyyrrrrrryyyyrrrrrr"),
    (100, "rrrrGrrrrrrrrrGrrrrr"),
    (100, "rrrryrrrrrrrrryrrrrr"),
    (100, "rrrrrGGGGrrrrrrGGGGr"),
    (100, "rrrrryyyyrrrrrryyyyr"),
    (100, "rrrrrrrrrGrrrrrrrrrG"),
    (100, "rrrrrrrrryrrrrrrrrry"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    """Format a float to 2 decimal places."""
    return f"{v:.2f}"


def _shape(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in points)


def _lanes_4(edge_id: str, offsets: list[float], sx: float, sy: float,
              ex: float, ey: float, is_x_offset: bool, length: float) -> str:
    """Emit 4 lane elements for a straight edge.

    Args:
        edge_id: Parent edge ID.
        offsets: Per-lane lateral offset values (4 values).
        sx, sy: Start x,y of the first lane *before* applying the offset.
                Offsets are added in the non-primary direction.
        ex, ey: End x,y (same logic).
        is_x_offset: True if offsets are applied to x coordinates (N/S edges);
                     False if applied to y coordinates (W/E edges).
        length: Declared edge length.
    """
    lines = []
    for i, off in enumerate(offsets):
        if is_x_offset:
            p1 = (sx + off, sy)
            p2 = (ex + off, ey)
        else:
            p1 = (sx, sy + off)
            p2 = (ex, ey + off)
        lines.append(
            f'        <lane id="{edge_id}_{i}" index="{i}" speed="{LANE_SPEED:.2f}"'
            f' length="{length:.2f}" shape="{_shape([p1, p2])}" />'
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal edge generation
# ---------------------------------------------------------------------------

def _internal_edges_xml(r: int, c: int, cx: float, cy: float) -> str:
    """Generate the internal (junction-box) edge elements for TL_{r}_{c}."""
    lines: list[str] = []
    prefix = f"TL_{r}_{c}"
    for suffix, lane_defs in _INTERNAL_TEMPLATES:
        edge_id = f":{prefix}_{suffix}"
        lines.append(f'    <edge id="{edge_id}" function="internal">')
        for lane_idx, speed, length, pts in lane_defs:
            offset_pts = [(cx + px, cy + py) for px, py in pts]
            lines.append(
                f'        <lane id="{edge_id}_0" index="{lane_idx}"'
                if len(lane_defs) == 1 else
                f'        <lane id="{edge_id}_{lane_idx}" index="{lane_idx}"'
            )
            # Replace the partial line with a full one
            lines[-1] = (
                f'        <lane id="{edge_id}_{lane_idx}" index="{lane_idx}"'
                f' speed="{speed:.2f}" length="{length:.2f}"'
                f' shape="{_shape(offset_pts)}" />'
            )
        lines.append("    </edge>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# External edge generation
# ---------------------------------------------------------------------------

def _approach_edge_xml(
    edge_id: str, from_id: str, to_id: str, length: float,
    offsets: list[float], is_x_offset: bool,
    src: tuple[float, float], dst: tuple[float, float],
) -> str:
    """Emit a 4-lane approach/departure edge."""
    sx, sy = src
    ex, ey = dst
    lines = [
        f'    <edge id="{edge_id}" from="{from_id}" to="{to_id}"'
        f' priority="-1" length="{length:.2f}">'
    ]
    lines.append(_lanes_4(edge_id, offsets, sx, sy, ex, ey, is_x_offset, length))
    lines.append("    </edge>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exit edge name helpers
# ---------------------------------------------------------------------------

def _exit_name(r: int, c: int, direction: str, n: int) -> str:
    """Return the SUMO edge ID for traffic exiting TL_{r}_{c} in *direction*."""
    if direction == "W":
        return f"E2TL_{r}_{c - 1}" if c > 0 else f"TL_{r}_{c}_2W"
    if direction == "E":
        return f"W2TL_{r}_{c + 1}" if c < n - 1 else f"TL_{r}_{c}_2E"
    if direction == "N":
        return f"S2TL_{r - 1}_{c}" if r > 0 else f"TL_{0}_{c}_2N"
    if direction == "S":
        return f"N2TL_{r + 1}_{c}" if r < n - 1 else f"TL_{n - 1}_{c}_2S"
    msg = f"Unknown direction: {direction}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Connection generation
# ---------------------------------------------------------------------------

# Template: (from_dir, from_lane, to_dir, to_lane, via_suffix, link_index, dir_char, state_char)
_CONN_TEMPLATE = [
    ("N", 0, "W", 0, "0", 0, "r", "O"),
    ("N", 0, "S", 0, "1", 1, "s", "O"),
    ("N", 1, "S", 1, "1", 2, "s", "O"),
    ("N", 2, "S", 2, "1", 3, "s", "O"),
    ("N", 3, "E", 3, "4", 4, "l", "o"),
    ("E", 0, "N", 0, "5", 5, "r", "o"),
    ("E", 0, "W", 0, "6", 6, "s", "o"),
    ("E", 1, "W", 1, "6", 7, "s", "o"),
    ("E", 2, "W", 2, "6", 8, "s", "o"),
    ("E", 3, "S", 3, "9", 9, "l", "o"),
    ("S", 0, "E", 0, "10", 10, "r", "O"),
    ("S", 0, "N", 0, "11", 11, "s", "O"),
    ("S", 1, "N", 1, "11", 12, "s", "O"),
    ("S", 2, "N", 2, "11", 13, "s", "O"),
    ("S", 3, "W", 3, "14", 14, "l", "o"),
    ("W", 0, "S", 0, "15", 15, "r", "o"),
    ("W", 0, "E", 0, "16", 16, "s", "o"),
    ("W", 1, "E", 1, "16", 17, "s", "o"),
    ("W", 2, "E", 2, "16", 18, "s", "o"),
    ("W", 3, "N", 3, "19", 19, "l", "o"),
]

# Internal (junction-box) to outgoing edge connections
# (from_suffix, from_lane, to_dir, to_lane, dir_char)
_INT_CONN_TEMPLATE = [
    ("0",  0, "W", 0, "r"),
    ("1",  0, "S", 0, "s"),
    ("1",  1, "S", 1, "s"),
    ("1",  2, "S", 2, "s"),
    ("4",  0, "E", 3, "l"),
    ("5",  0, "N", 0, "r"),
    ("6",  0, "W", 0, "s"),
    ("6",  1, "W", 1, "s"),
    ("6",  2, "W", 2, "s"),
    ("9",  0, "S", 3, "l"),
    ("10", 0, "E", 0, "r"),
    ("11", 0, "N", 0, "s"),
    ("11", 1, "N", 1, "s"),
    ("11", 2, "N", 2, "s"),
    ("14", 0, "W", 3, "l"),
    ("15", 0, "S", 0, "r"),
    ("16", 0, "E", 0, "s"),
    ("16", 1, "E", 1, "s"),
    ("16", 2, "E", 2, "s"),
    ("19", 0, "N", 3, "l"),
]


def _connections_xml(r: int, c: int, n: int) -> str:
    """Generate all connection elements for TL_{r}_{c}."""
    prefix = f"TL_{r}_{c}"
    lines: list[str] = []

    # External connections (incoming lane → junction box → outgoing edge)
    for from_dir, from_lane, to_dir, to_lane, via_sfx, link_idx, dir_c, state_c in _CONN_TEMPLATE:
        from_edge = f"{from_dir}2TL_{r}_{c}"
        to_edge = _exit_name(r, c, to_dir, n)
        via = f":{prefix}_{via_sfx}_0"
        lines.append(
            f'    <connection from="{from_edge}" to="{to_edge}"'
            f' fromLane="{from_lane}" toLane="{to_lane}"'
            f' via="{via}" tl="{prefix}" linkIndex="{link_idx}"'
            f' dir="{dir_c}" state="{state_c}"/>'
        )

    # Internal connections (junction box → outgoing edge)
    for sfx, from_lane, to_dir, to_lane, dir_c in _INT_CONN_TEMPLATE:
        from_int = f":{prefix}_{sfx}"
        to_edge = _exit_name(r, c, to_dir, n)
        lines.append(
            f'    <connection from="{from_int}" to="{to_edge}"'
            f' fromLane="{from_lane}" toLane="{to_lane}"'
            f' dir="{dir_c}" state="M"/>'
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_grid_network(n: int, out_dir: Path, spacing: int = 1500) -> None:
    """Write ``grid_{n}x{n}.net.xml`` to *out_dir*.

    Args:
        n: Grid dimension (number of intersections per side).
        out_dir: Directory to write the file into (created if absent).
        spacing: Distance between adjacent junction centres in metres.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"grid_{n}x{n}.net.xml"
    half = spacing // 2  # 750 m for spacing=1500

    # ---------- Compute junction positions ----------
    def tl_pos(r: int, c: int) -> tuple[float, float]:
        return float(c * spacing), float((n - 1 - r) * spacing)

    # ---------- Build XML sections ----------
    sections: list[str] = []

    # Header
    max_coord = (n - 1) * spacing + half
    sections.append('<?xml version="1.0" encoding="UTF-8"?>')
    sections.append(
        f'<net version="1.0" junctionCornerDetail="5" limitTurnSpeed="5.50"'
        f' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        f' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">'
    )
    sections.append("")
    sections.append(
        f'    <location netOffset="0.00,0.00"'
        f' convBoundary="{-half:.2f},{-half:.2f},{max_coord:.2f},{max_coord:.2f}"'
        f' origBoundary="-10000000000.00,-10000000000.00,10000000000.00,10000000000.00"'
        f' projParameter="!"/>'
    )
    sections.append("")

    # Internal (junction-box) edges for each TL
    for r in range(n):
        for c in range(n):
            cx, cy = tl_pos(r, c)
            sections.append(_internal_edges_xml(r, c, cx, cy))
            sections.append("")

    # External edges
    for r in range(n):
        cx, cy = tl_pos(r, 0)  # leftmost column

        # West border: W2TL_{r}_{0}  (DW_{r} → TL_{r}_{0})
        dw_id = f"DW_{r}"
        tl_id = f"TL_{r}_{0}"
        edge_id = f"W2TL_{r}_0"
        sections.append(_approach_edge_xml(
            edge_id, dw_id, tl_id, float(half),
            _W_Y, False,
            (-float(half), cy), (cx - JB, cy),
        ))

        # East exit: TL_{r}_{N-1}_2E  (TL_{r}_{N-1} → DE_{r})
        cx_last, cy_last = tl_pos(r, n - 1)
        de_id = f"DE_{r}"
        tl_last_id = f"TL_{r}_{n - 1}"
        exit_e_id = f"TL_{r}_{n - 1}_2E"
        sections.append(_approach_edge_xml(
            exit_e_id, tl_last_id, de_id, float(half),
            _E_OUT_Y, False,
            (cx_last + JB, cy_last), (cx_last + float(half), cy_last),
        ))

        # West exit: TL_{r}_{0}_2W  (TL_{r}_{0} → DW_{r})
        exit_w_id = f"TL_{r}_{0}_2W"
        sections.append(_approach_edge_xml(
            exit_w_id, tl_id, dw_id, float(half),
            _W_OUT_Y, False,
            (cx - JB, cy), (-float(half), cy),
        ))

        # East border: E2TL_{r}_{N-1}  (DE_{r} → TL_{r}_{N-1})
        edge_id = f"E2TL_{r}_{n - 1}"
        sections.append(_approach_edge_xml(
            edge_id, de_id, tl_last_id, float(half),
            _E_Y, False,
            (cx_last + float(half), cy_last), (cx_last + JB, cy_last),
        ))

    for c in range(n):
        _, cy_top = tl_pos(0, c)
        _, cy_bot = tl_pos(n - 1, c)
        cx_c, _ = tl_pos(0, c)

        # North border: N2TL_{0}_{c}  (DN_{c} → TL_{0}_{c})
        dn_id = f"DN_{c}"
        tl_top_id = f"TL_0_{c}"
        edge_id = f"N2TL_0_{c}"
        sections.append(_approach_edge_xml(
            edge_id, dn_id, tl_top_id, float(half),
            _N_X, True,
            (cx_c, cy_top + float(half)), (cx_c, cy_top + JB),
        ))

        # North exit: TL_{0}_{c}_2N  (TL_{0}_{c} → DN_{c})
        exit_n_id = f"TL_0_{c}_2N"
        sections.append(_approach_edge_xml(
            exit_n_id, tl_top_id, dn_id, float(half),
            _N_OUT_X, True,
            (cx_c, cy_top + JB), (cx_c, cy_top + float(half)),
        ))

        # South border: S2TL_{N-1}_{c}  (DS_{c} → TL_{N-1}_{c})
        ds_id = f"DS_{c}"
        tl_bot_id = f"TL_{n - 1}_{c}"
        edge_id = f"S2TL_{n - 1}_{c}"
        sections.append(_approach_edge_xml(
            edge_id, ds_id, tl_bot_id, float(half),
            _S_X, True,
            (cx_c, cy_bot - float(half)), (cx_c, cy_bot - JB),
        ))

        # South exit: TL_{N-1}_{c}_2S  (TL_{N-1}_{c} → DS_{c})
        exit_s_id = f"TL_{n - 1}_{c}_2S"
        sections.append(_approach_edge_xml(
            exit_s_id, tl_bot_id, ds_id, float(half),
            _S_OUT_X, True,
            (cx_c, cy_bot - JB), (cx_c, cy_bot - float(half)),
        ))

    # Inter-junction edges
    for r in range(n):
        for c in range(n):
            cx, cy = tl_pos(r, c)

            # Eastward inter-junction: W2TL_{r}_{c}  (TL_{r}_{c-1} → TL_{r}_{c})
            if c > 0:
                cx_prev = float((c - 1) * spacing)
                edge_id = f"W2TL_{r}_{c}"
                from_id = f"TL_{r}_{c - 1}"
                to_id = f"TL_{r}_{c}"
                sections.append(_approach_edge_xml(
                    edge_id, from_id, to_id, float(spacing),
                    _W_Y, False,
                    (cx_prev + JB, cy), (cx - JB, cy),
                ))

            # Westward inter-junction: E2TL_{r}_{c}  (TL_{r}_{c+1} → TL_{r}_{c})
            if c < n - 1:
                cx_next = float((c + 1) * spacing)
                edge_id = f"E2TL_{r}_{c}"
                from_id = f"TL_{r}_{c + 1}"
                to_id = f"TL_{r}_{c}"
                sections.append(_approach_edge_xml(
                    edge_id, from_id, to_id, float(spacing),
                    _E_Y, False,
                    (cx_next - JB, cy), (cx + JB, cy),
                ))

            # Southward inter-junction: N2TL_{r}_{c}  (TL_{r-1}_{c} → TL_{r}_{c})
            if r > 0:
                cy_prev = float((n - 1 - (r - 1)) * spacing)
                edge_id = f"N2TL_{r}_{c}"
                from_id = f"TL_{r - 1}_{c}"
                to_id = f"TL_{r}_{c}"
                sections.append(_approach_edge_xml(
                    edge_id, from_id, to_id, float(spacing),
                    _N_X, True,
                    (cx, cy_prev - JB), (cx, cy + JB),
                ))

            # Northward inter-junction: S2TL_{r}_{c}  (TL_{r+1}_{c} → TL_{r}_{c})
            if r < n - 1:
                cy_next = float((n - 1 - (r + 1)) * spacing)
                edge_id = f"S2TL_{r}_{c}"
                from_id = f"TL_{r + 1}_{c}"
                to_id = f"TL_{r}_{c}"
                sections.append(_approach_edge_xml(
                    edge_id, from_id, to_id, float(spacing),
                    _S_X, True,
                    (cx, cy_next + JB), (cx, cy - JB),
                ))

    # TL logic
    for r in range(n):
        for c in range(n):
            tl_id = f"TL_{r}_{c}"
            sections.append(f'    <tlLogic id="{tl_id}" type="static" programID="0" offset="0">')
            for dur, state in _TL_PHASES:
                sections.append(f'        <phase duration="{dur}" state="{state}"/>')
            sections.append("    </tlLogic>")
            sections.append("")

    # Dead-end junctions
    for r in range(n):
        _, cy = tl_pos(r, 0)
        inc_w = " ".join(f"TL_{r}_0_2W_{i}" for i in range(4))
        sections.append(
            f'    <junction id="DW_{r}" type="dead_end"'
            f' x="{-float(half):.2f}" y="{cy:.2f}"'
            f' incLanes="{inc_w}" intLanes="" shape="'
            f'{_fmt(-float(half))},{_fmt(cy)} {_fmt(-float(half))},{_fmt(cy + 12.80)}'
            f' {_fmt(-float(half))},{_fmt(cy)}"/>'
        )
        cx_last, _ = tl_pos(r, n - 1)
        inc_e = " ".join(f"TL_{r}_{n - 1}_2E_{i}" for i in range(4))
        sections.append(
            f'    <junction id="DE_{r}" type="dead_end"'
            f' x="{cx_last + float(half):.2f}" y="{cy:.2f}"'
            f' incLanes="{inc_e}" intLanes="" shape="'
            f'{_fmt(cx_last + float(half))},{_fmt(cy)}'
            f' {_fmt(cx_last + float(half))},{_fmt(cy - 12.80)}'
            f' {_fmt(cx_last + float(half))},{_fmt(cy)}"/>'
        )

    for c in range(n):
        cx, cy_top = tl_pos(0, c)
        inc_n = " ".join(f"TL_0_{c}_2N_{i}" for i in range(4))
        sections.append(
            f'    <junction id="DN_{c}" type="dead_end"'
            f' x="{cx:.2f}" y="{cy_top + float(half):.2f}"'
            f' incLanes="{inc_n}" intLanes="" shape="'
            f'{_fmt(cx)},{_fmt(cy_top + float(half))}'
            f' {_fmt(cx + 12.80)},{_fmt(cy_top + float(half))}'
            f' {_fmt(cx)},{_fmt(cy_top + float(half))}"/>'
        )
        _, cy_bot = tl_pos(n - 1, c)
        inc_s = " ".join(f"TL_{n - 1}_{c}_2S_{i}" for i in range(4))
        sections.append(
            f'    <junction id="DS_{c}" type="dead_end"'
            f' x="{cx:.2f}" y="{cy_bot - float(half):.2f}"'
            f' incLanes="{inc_s}" intLanes="" shape="'
            f'{_fmt(cx)},{_fmt(cy_bot - float(half))}'
            f' {_fmt(cx - 12.80)},{_fmt(cy_bot - float(half))}'
            f' {_fmt(cx)},{_fmt(cy_bot - float(half))}"/>'
        )

    sections.append("")

    # TL junctions
    for r in range(n):
        for c in range(n):
            cx, cy = tl_pos(r, c)
            tl_id = f"TL_{r}_{c}"
            prefix = tl_id

            inc_lanes = (
                " ".join(f"N2TL_{r}_{c}_{i}" for i in range(4)) + " " +
                " ".join(f"E2TL_{r}_{c}_{i}" for i in range(4)) + " " +
                " ".join(f"S2TL_{r}_{c}_{i}" for i in range(4)) + " " +
                " ".join(f"W2TL_{r}_{c}_{i}" for i in range(4))
            )

            int_lane_sfxs = [
                "0_0", "1_0", "1_1", "1_2", "4_0",
                "5_0", "6_0", "6_1", "6_2", "9_0",
                "10_0", "11_0", "11_1", "11_2", "14_0",
                "15_0", "16_0", "16_1", "16_2", "19_0",
            ]
            int_lanes = " ".join(f":{prefix}_{s}" for s in int_lane_sfxs)

            shape_pts = [(cx + px, cy + py) for px, py in _TL_SHAPE_REL]
            shape_str = _shape(shape_pts)

            req_lines = "\n".join(
                f'        <request index="{idx}" response="{resp}" foes="{foes}" cont="0"/>'
                for idx, resp, foes in _REQUESTS
            )

            sections.append(
                f'    <junction id="{tl_id}" type="traffic_light"'
                f' x="{cx:.2f}" y="{cy:.2f}"'
                f' incLanes="{inc_lanes}" intLanes="{int_lanes}" shape="{shape_str}">'
            )
            sections.append(req_lines)
            sections.append("    </junction>")
            sections.append("")

    # Connections
    for r in range(n):
        for c in range(n):
            sections.append(_connections_xml(r, c, n))
            sections.append("")

    sections.append("</net>")

    out_file.write_text("\n".join(sections), encoding="utf-8")


def generate_grid_sumocfg(n: int, out_dir: Path) -> None:
    """Write ``grid_{n}x{n}.sumocfg`` to *out_dir*.

    Args:
        n: Grid dimension.
        out_dir: Directory to write the file into (created if absent).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    net_file = f"grid_{n}x{n}.net.xml"
    routes_file = f"grid_{n}x{n}_routes.rou.xml"
    cfg = f"""<?xml version="1.0" encoding="UTF-8"?>

<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="{net_file}" />
        <route-files value="{routes_file}" />
    </input>

    <time>
        <begin value="0" />
    </time>

    <processing>
        <time-to-teleport value="-1" />
    </processing>

</configuration>
"""
    (out_dir / f"grid_{n}x{n}.sumocfg").write_text(cfg, encoding="utf-8")
