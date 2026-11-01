"""Derive walking-route gradients from a configured digital elevation model.

Adds ``grade_max_abs`` and ``grade_coverage`` columns to the network ``edges``
table.  The gradient is evaluated over a moving window along each edge (so a
short steep ramp is not averaged away by a long gentle approach), and edges
with insufficient valid DEM coverage are left with a null gradient, to be
treated as unknown.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from script_running_log import script_running_log
from shapely import segmentize
from shapely.geometry import MultiLineString
from sqlalchemy import text

import ghsci

SAMPLE_CHUNK = 500_000


def _geometry_coords(geometry, interval):
    """Return the densified (x, y) coordinates along a line geometry."""
    if geometry is None or geometry.is_empty:
        return np.empty((0, 2))
    try:
        geometry = segmentize(geometry, interval)
    except Exception:
        pass
    if isinstance(geometry, MultiLineString):
        parts = [np.asarray(part.coords) for part in geometry.geoms]
    else:
        parts = [np.asarray(geometry.coords)]
    parts = [part for part in parts if len(part)]
    if not parts:
        return np.empty((0, 2))
    return np.vstack(parts)


def _max_window_grade(distance, elevation, window):
    """Maximum absolute gradient over any window-length run of the profile.

    Only contiguous runs of valid elevations are considered.  Edges shorter
    than the window are evaluated between their endpoints when at least two
    valid samples exist.
    """
    distance = np.asarray(distance, dtype='float64')
    elevation = np.asarray(elevation, dtype='float64')
    valid = np.isfinite(elevation)
    if valid.sum() < 2 or distance[-1] <= 0:
        return np.nan

    # Split the profile into contiguous runs of valid samples
    run_ids = np.cumsum(np.diff(np.concatenate(([False], valid))) != 0)
    best = np.nan
    for run_id in np.unique(run_ids[valid]):
        positions = np.flatnonzero((run_ids == run_id) & valid)
        if len(positions) < 2:
            continue
        run_distance = distance[positions] - distance[positions[0]]
        run_elevation = elevation[positions]
        run_length = run_distance[-1]
        if run_length < 2:
            continue
        if run_length < window:
            # A genuinely short edge: use its endpoints
            grade = abs(run_elevation[-1] - run_elevation[0]) / run_length
            if np.isnan(best) or grade > best:
                best = grade
            continue
        ends = np.searchsorted(
            run_distance,
            run_distance + window,
            side='left',
        )
        for start in range(len(positions) - 1):
            end = ends[start]
            if end >= len(positions):
                continue
            run = run_distance[end] - run_distance[start]
            if run <= 0:
                continue
            grade = abs(run_elevation[end] - run_elevation[start]) / run
            if np.isnan(best) or grade > best:
                best = grade
    return best


def _edge_grades(r):
    """Compute maximum gradient and DEM coverage for every edge."""
    elevation_config = r.config['network'].get('elevation')
    if not elevation_config:
        print(
            '  No elevation data configured; edge gradients will not be '
            'derived.',
        )
        return None
    dem_path = elevation_config['data']
    if not os.path.isfile(dem_path):
        print(
            f'  Elevation data not found at {dem_path}; edge gradients will '
            'not be derived.',
        )
        return None
    interval = float(elevation_config['sample_interval'])
    window = float(elevation_config['grade_window'])
    min_coverage = float(elevation_config['min_coverage'])

    edges = r.get_gdf('SELECT ogc_fid, geom FROM edges')
    if edges.empty:
        print('  No network edges found; edge gradients will not be derived.')
        return None
    print(
        f'  Sampling {dem_path} at {interval:g} m intervals along '
        f'{len(edges):,} edges...',
    )

    profiles = [None] * len(edges)
    for i, geometry in enumerate(edges.geometry):
        coords = _geometry_coords(geometry, interval)
        if len(coords) >= 2:
            profiles[i] = coords
    available = [i for i, profile in enumerate(profiles) if profile is not None]
    if not available:
        print('  No usable edge geometries found.')
        return None

    points = np.vstack([profiles[i] for i in available])
    elevations = np.full(len(points), np.nan, dtype='float64')
    with rasterio.open(dem_path) as src:
        print(
            f'  DEM: {src.width} x {src.height} pixels, '
            f'{src.crs}, no-data {src.nodata}',
        )
        dem_crs = src.crs
        project_crs = CRS.from_user_input(f'EPSG:{r.config["crs"]["srid"]}')
        try:
            dem_crs_equivalent = CRS.from_user_input(
                dem_crs.to_wkt(),
            ).equals(project_crs, ignore_axis_order=True)
        except Exception:
            dem_crs_equivalent = False
        if not dem_crs_equivalent:
            print(
                '  Warning: the DEM coordinate reference system does not '
                'appear to match the network CRS.  Check that the DEM is in '
                'the same projected coordinate system as the study region '
                '(coordinates matching is assumed).',
            )
        bounds = src.bounds
        nodata = src.nodata
        for start in range(0, len(points), SAMPLE_CHUNK):
            stop = min(start + SAMPLE_CHUNK, len(points))
            chunk = points[start:stop]
            values = np.array(
                list(
                    src.sample(
                        chunk,
                        indexes=1,
                    ),
                ),
                dtype='float64',
            ).reshape(-1)
            in_bounds = (
                (chunk[:, 0] >= bounds.left)
                & (chunk[:, 0] <= bounds.right)
                & (chunk[:, 1] >= bounds.bottom)
                & (chunk[:, 1] <= bounds.top)
            )
            valid = in_bounds & np.isfinite(values)
            if nodata is not None:
                valid &= values != nodata
            elevations[start:stop] = np.where(valid, values, np.nan)
            print(
                f'    sampled {stop:,}/{len(points):,} profile points',
                end='\r',
                flush=True,
            )
    print()

    grade_max = np.full(len(edges), np.nan, dtype='float64')
    coverage = np.zeros(len(edges), dtype='float64')
    position = 0
    for i, profile in enumerate(profiles):
        if profile is None:
            continue
        n = len(profile)
        profile_elevation = elevations[position : position + n]
        position += n
        distances = np.concatenate(
            (
                [0.0],
                np.cumsum(
                    np.hypot(
                        np.diff(profile[:, 0]),
                        np.diff(profile[:, 1]),
                    ),
                ),
            ),
        )
        coverage[i] = np.isfinite(profile_elevation).sum() / n
        if coverage[i] >= min_coverage:
            grade_max[i] = _max_window_grade(
                distances,
                profile_elevation,
                window,
            )

    result = pd.DataFrame(
        {
            'ogc_fid': edges['ogc_fid'].astype('int64').to_numpy(),
            'grade_max_abs': grade_max,
            'grade_coverage': coverage,
        },
    )
    return result


def add_edge_gradients(r):
    """Add and populate gradient attributes on the network edges table."""
    elevation_config = r.config['network'].get('elevation')
    if not elevation_config:
        print(
            '\nNo elevation data is configured for this study region, so '
            'edge gradients are not derived.',
        )
        return
    edge_columns = set(
        r.get_df(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'edges'",
        )['column_name'],
    )
    if {'grade_max_abs', 'grade_coverage'}.issubset(edge_columns):
        known = r.get_df(
            'SELECT count(*) AS n FROM edges WHERE grade_max_abs IS NOT NULL',
        )['n'][0]
        if known:
            print(
                f'\nEdge gradients are already present for {int(known):,} '
                'edges; skipping.  Drop the grade_max_abs and grade_coverage '
                'columns to recalculate them.',
            )
            return

    try:
        result = _edge_grades(r)
    except Exception as error:
        print(
            '  Edge gradients could not be derived from the elevation data: '
            f'{error}',
        )
        return
    if result is None:
        return
    known = result['grade_max_abs'].notna()
    print(
        f'\nGradient derived for {int(known.sum()):,} of '
        f'{len(result):,} edges '
        f'({100 * known.mean():.1f}%).',
    )
    if known.any():
        print(
            '  Maximum edge gradient: '
            f'{result.loc[known, "grade_max_abs"].max():.3f} '
            '(rise / run)',
        )
    result['grade_max_abs'] = result['grade_max_abs'].astype(object)
    result.loc[~known, 'grade_max_abs'] = None
    result.to_sql(
        '_edge_gradients',
        r.engine,
        if_exists='replace',
        index=False,
    )
    with r.engine.begin() as connection:
        connection.execute(
            text(
                'ALTER TABLE edges ADD COLUMN IF NOT EXISTS '
                'grade_max_abs double precision;',
            ),
        )
        connection.execute(
            text(
                'ALTER TABLE edges ADD COLUMN IF NOT EXISTS '
                'grade_coverage double precision;',
            ),
        )
        connection.execute(
            text(
                'UPDATE edges e '
                'SET grade_max_abs = CAST(g.grade_max_abs AS double precision), '
                '    grade_coverage = CAST(g.grade_coverage AS double precision) '
                'FROM _edge_gradients g '
                'WHERE e.ogc_fid = g.ogc_fid;',
            ),
        )
        connection.execute(text('DROP TABLE IF EXISTS _edge_gradients;'))


def derive_edge_gradients(codename):
    start = time.time()
    script = '_03a_derive_edge_gradients'
    task = 'Derive walking-route gradients from elevation data'
    r = ghsci.Region(codename)
    add_edge_gradients(r)
    script_running_log(r.config, script, task, start)
    r.engine.dispose()


def main():
    try:
        codename = sys.argv[1]
    except IndexError:
        codename = None
    derive_edge_gradients(codename)


if __name__ == '__main__':
    main()
