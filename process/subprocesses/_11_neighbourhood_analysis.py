"""
Neighbourhood analysis.

This script creates neighbourhood indicators for sample points.  To run it, supply a study region code name.

It assumes network projected network nodes and edges have been generated and stored in a PostGIS database, which can be read from as a GeoDataFrame to generate a graph.

Once run, the sample points may be aggregated to a neighbourhood small area grid and for overall
city summaries by running 03_aggregation.py.

As a result of running the script, a layer is added to the study region's geopackage file
containing sample point indicators ("sample_points").  These indicators include:
1. average population and intersection density per sample sample point
2. accessibility, dailyliving and walkability score per sample point
"""

import os
import sys
import time

import geopandas as gpd

# Set up project and region parameters for GHSCIC analyses
import ghsci
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from geoalchemy2 import Geometry
from script_running_log import script_running_log
from setup_sp import (
    binary_access_score,
    build_dest_node_lookup,
    cal_dist_node_to_nearest_pois,
    cal_dist_nodes_to_nearest_pois_inmemory,
    create_full_nodes,
    drop_dest_node_lookup,
    filter_ids,
    graph_from_edge_arrays,
    nearest_poi_query_columns,
    neighbourhood_reachable_nodes,
    spatial_join_index_to_gdf,
)
from sqlalchemy import text
from tqdm import tqdm

# Hard coded density variable names
density_statistics = {
    'pop_per_sqkm': 'sp_local_nh_avg_pop_density',
    'intersections_per_sqkm': 'sp_local_nh_avg_intersection_density',
}


def _grid_mean_summariser(grid, gdf_nodes, nh_grid_fields):
    """Vectorised per-source grid-cell density mean, bit-matching the pandas form.

    Returns a function mapping a nearest-first array of reached node osmids to
    the per-field mean over the distinct grid cells of those nodes -- the exact
    quantity the networkx branch computes as
    ``grid.loc[gdf_nodes.loc[reached, 'grid_id'].dropna().unique(), fields].mean()``.
    That label-based pandas chain costs ~milliseconds per call against
    million-row frames (measured ~90 ms/source on Minneapolis' 1.19M-cell grid,
    turning the density stage into a 13.7 h run); this precomputes positional
    arrays once and reduces each source with numpy in microseconds.

    Bit-equality with the pandas expression: cells are visited in the same
    first-appearance (nearest-first) order, and each field is reduced as a
    fresh contiguous 1-D array with ``np.nansum`` -- the same zero-filled
    pairwise summation pandas' skipna mean (without bottleneck) applies along
    its blocks' contiguous axis.  (Reducing a (k, n_fields) array along axis 0
    instead accumulates sequentially and differs in the last ulp.)
    """
    node_osmids = gdf_nodes.index.to_numpy('int64')
    order = np.argsort(node_osmids, kind='stable')
    osmid_sorted = node_osmids[order]
    # per node: position of its grid cell in the grid frame, -1 where no cell
    gids = gdf_nodes['grid_id'].to_numpy('float64')[order]
    grid_pos_sorted = np.full(len(gids), -1, dtype='int64')
    notna = ~np.isnan(gids)
    grid_pos_sorted[notna] = grid.index.get_indexer(
        gids[notna].astype('int64'),
    )
    grid_columns = [
        np.ascontiguousarray(grid[field].to_numpy('float64'))
        for field in nh_grid_fields
    ]
    n_fields = len(nh_grid_fields)
    nan_row = np.full(n_fields, np.nan)

    def summarise(reached):
        idx = np.searchsorted(osmid_sorted, reached)
        if not (
            osmid_sorted[np.clip(idx, 0, len(osmid_sorted) - 1)] == reached
        ).all():
            missing = set(reached) - set(osmid_sorted)
            raise KeyError(
                f'reached nodes absent from the nodes table: {sorted(missing)[:5]}...',
            )
        pos = grid_pos_sorted[idx]
        pos = pos[pos >= 0]
        if not pos.size:
            return nan_row
        # first-appearance unique preserves the nearest-first cell order that
        # pandas' .unique() yields, keeping the mean's summation order identical
        rows = pos[np.sort(np.unique(pos, return_index=True)[1])]
        out = np.empty(n_fields)
        for j in range(n_fields):
            column = grid_columns[j][rows]  # fresh contiguous 1-D gather
            count = column.size - np.count_nonzero(np.isnan(column))
            out[j] = np.nansum(column) / count if count else np.nan
        return out

    return summarise


def compute_nodes_pop_intersect_density(
    r,
    edges,
    nodes,
    neighbourhood_distance,
    engine=None,
):
    """Calculate average population and intersection density for each intersection node.

    Takes mean values from the distinct grid cells reached within the
    neighbourhood buffer distance along the network.  The neighbourhood search
    runs either as a networkx all-pairs Dijkstra ('pgrouting'/default engine
    setting; the historical method) or as an in-memory scipy Dijkstra over the
    identical graph ('inmemory'), which reaches the identical node sets.  The
    in-memory engine visits the reached grid cells in nearest-first order
    (networkx's discovery order) and reduces them with a vectorised
    equivalent of the networkx branch's pandas mean (_grid_mean_summariser),
    so results agree to the last bit except in the astronomically rare case
    of exact float distance ties reordering the mean's summation.

    Returns the nodes_simple GeoDataFrame (grid nodes joined with density
    columns); no caching or database writes.
    """
    if engine is None:
        engine = pedestrian_routing_engine(r)
    grid = r.get_gdf(r.config['population_grid'], index_col='grid_id')
    print('  - Set up simple nodes')
    gdf_nodes = spatial_join_index_to_gdf(nodes, grid, dropna=False)
    # keep only the unique node id column
    gdf_nodes = gdf_nodes[['grid_id', 'geometry']]
    # drop any nodes which are na
    # (they are outside the buffered study region and not of interest)
    nodes_simple = gdf_nodes[~gdf_nodes.grid_id.isna()].copy()
    sampling = r.config.get('sampling', {})
    if sampling.get('sample_unpopulated_areas') or sampling.get(
        'custom_sample_points',
    ):
        # Sampling of areas lacking population data coverage has been
        # configured; also retain nodes associated with sample points
        # even if they do not intersect the population grid, so that
        # estimates can be derived for these points.  Local densities for
        # such nodes are estimated using any populated grid cells located
        # within the neighbourhood buffer distance.
        required_nodes = r.get_df(
            """
            SELECT n1 AS osmid FROM urban_sample_points
            UNION
            SELECT n2 AS osmid FROM urban_sample_points
            """,
        )['osmid']
        nodes_simple = gdf_nodes[
            (~gdf_nodes.grid_id.isna()) | gdf_nodes.index.isin(required_nodes)
        ].copy()
    gdf_nodes = gdf_nodes[['grid_id']]
    nh_grid_fields = list(density_statistics.keys())
    total_nodes = len(nodes_simple)
    if engine == 'inmemory':
        print(
            f'  - Generate {neighbourhood_distance}m neighbourhoods for nodes '
            '(in-memory Dijkstra) and summarise attributes (average value from '
            'unique associated grid cells within nh buffer distance)...',
        )
        graph, node_ids = graph_from_edge_arrays(
            edges.index.get_level_values('u').to_numpy('int64'),
            edges.index.get_level_values('v').to_numpy('int64'),
            edges['length'].to_numpy('float64'),
        )
        reachables = neighbourhood_reachable_nodes(
            graph,
            node_ids,
            nodes_simple.index.to_numpy('int64'),
            neighbourhood_distance,
        )
        summarise = _grid_mean_summariser(grid, gdf_nodes, nh_grid_fields)
        result = pd.DataFrame(
            [summarise(reached) for reached in reachables],
            columns=list(density_statistics.values()),
            index=nodes_simple.index.values,
        )
    else:
        G_proj = ox.graph_from_gdfs(
            nodes,
            edges,
            graph_attrs=None,
        ).to_undirected()
        # run all pairs analysis
        print(
            f'  - Generate {neighbourhood_distance}m neighbourhoods '
            'for nodes (All pairs Dijkstra shortest path analysis)',
        )
        all_pairs_d = pd.DataFrame(
            [
                (k, v.keys())
                for k, v in tqdm(
                    nx.all_pairs_dijkstra_path_length(
                        G_proj,
                        neighbourhood_distance,
                        'length',
                    ),
                    total=total_nodes,
                    unit='nodes',
                    desc=' ' * 18,
                    miniters=int(total_nodes / 100),
                )
            ],
            columns=['osmid', 'nodes'],
        ).set_index('osmid')
        # extract results
        print(
            '  - Summarise attributes (average value from unique associated grid cells within nh buffer distance)...',
        )
        result = pd.DataFrame(
            [
                tuple(
                    grid.loc[
                        gdf_nodes.loc[all_pairs_d.loc[n].nodes, 'grid_id']
                        .dropna()
                        .unique(),
                        nh_grid_fields,
                    ]
                    .mean()
                    .values,
                )
                for index, n in tqdm(
                    np.ndenumerate(nodes_simple.index.values),
                    total=total_nodes,
                    desc=' ' * 18,
                    miniters=int(total_nodes / 100),
                )
            ],
            columns=list(density_statistics.values()),
            index=nodes_simple.index.values,
        )
    return nodes_simple.join(result)


def node_level_neighbourhood_analysis(
    r,
    edges,
    nodes,
    neighbourhood_distance,
    engine=None,
):
    """First pass node-level neighbourhood analysis (Calculate average population and intersection density for each intersection node in study regions, taking mean values from distinct grid cells within neighbourhood buffer distance."""
    nh_startTime = time.time()
    # read from disk if exist
    if 'nodes_pop_intersect_density' in r.tables:
        print('  - Read population and intersection density from database.')
        nodes_simple = r.get_gdf(
            'nodes_pop_intersect_density',
            index_col='osmid',
            geom_col='geometry',
        )
    else:
        nodes_simple = compute_nodes_pop_intersect_density(
            r,
            edges,
            nodes,
            neighbourhood_distance,
            engine,
        )
        # save in geopackage (so output files are all kept together)
        with r.engine.connect() as connection:
            nodes_simple.to_postgis(
                'nodes_pop_intersect_density',
                connection,
                index='osmid',
            )
    print(
        'Time taken to calculate or load city local neighbourhood statistics: '
        f'{(time.time() - nh_startTime) / 60:.02f} mins',
    )
    return nodes_simple


def pedestrian_routing_engine(r):
    """Resolve the pedestrian accessibility routing engine for a region.

    Configured via the region's top-level ``routing_engine`` key: 'pgrouting'
    (default; banded pgr_drivingDistance lookup in PostGIS) or 'inmemory'
    (in-process scipy Dijkstra; identical results).
    """
    engine = str(r.config.get('routing_engine') or 'pgrouting').lower()
    if engine not in ('pgrouting', 'inmemory'):
        sys.exit(
            f"Unknown routing_engine '{engine}' "
            "(expected 'pgrouting' or 'inmemory').",
        )
    return engine


def _resolve_output_names(analysis, layer):
    """Resolve the output names a nearest-node analysis uses for one layer."""
    output_names = analysis['output_names'].copy()
    if len(analysis['layers']) > 1 and len(analysis['layers']) == len(
        analysis['output_names'],
    ):
        # assume that output names correspond to layers, and refresh per analysis
        output_names = [output_names[analysis['layers'].index(layer)]]
    return output_names


def _poi_column_plan(r):
    """Flat (layer, col_name, where_clause) plan over all active nearest-node analyses.

    Mirrors the per-analysis iteration of calculate_poi_accessibility so the
    in-memory engine computes exactly the columns the pgRouting engine would,
    using the shared nearest_poi_query_columns clause construction.
    """
    plan = []
    for analysis_key in r.indicators['nearest_node_analyses']:
        analysis = r.indicators['nearest_node_analyses'][analysis_key]
        for layer in analysis['layers']:
            if layer in r.tables and layer is not None:
                plan.extend(
                    (layer, col_name, where_clause)
                    for col_name, where_clause in nearest_poi_query_columns(
                        category_field=analysis['category_field'],
                        categories=analysis['categories'],
                        filter_field=analysis['filter_field'],
                        filter_iterations=analysis['filter_iterations'],
                        output_names=_resolve_output_names(analysis, layer),
                        output_prefix='sp_nearest_node_',
                    )
                )
    return plan


def calculate_poi_accessibility(r, engine=None):
    # Calculate accessibility to points of interest and walkability for sample points:
    # 1. using pgr_drivingDistance (or the equivalent in-memory Dijkstra engine,
    #    per the region's routing_engine setting) to calculate distance from nodes
    #    to nearest destinations (daily living destinations, public open space)
    # 2. calculate accessibiity score per sample point: transform accessibility
    #    distance to binary measure: 1 if access <= 400m, 0 otherwise
    # 3. calculate daily living score by summing the accessibiity scores to all
    #    POIs (excluding pos)
    # 4. calculate walkability score per sample point: get zscores for daily
    #    living accessibility, populaiton density and intersections population_density;
    #    sum these three zscores at sample point level
    print('\nCalculate accessibility to points of interest.')
    accessibility_distance = ghsci.settings['network_analysis'][
        'accessibility_distance'
    ]
    if engine is None:
        engine = pedestrian_routing_engine(r)
    node_index = pd.Index(
        r.get_df('SELECT osmid FROM nodes ORDER BY osmid')['osmid'].to_numpy(
            dtype='int64',
        ),
        name='osmid',
    )
    # Identify active destination layers and build the network distance lookup table.
    active_layers = {
        layer
        for analysis_key in r.indicators['nearest_node_analyses']
        for layer in r.indicators['nearest_node_analyses'][analysis_key][
            'layers'
        ]
        if layer is not None and layer in r.tables
    }
    if engine == 'inmemory':
        print(
            '  Routing engine: inmemory (exact in-process Dijkstra; '
            'pgRouting-equivalent results).',
        )
        nodes_dist_inmemory = cal_dist_nodes_to_nearest_pois_inmemory(
            r,
            _poi_column_plan(r),
            accessibility_distance,
            node_index,
        )
    else:
        print('  Building destination-node travel cost lookup table...')
        build_dest_node_lookup(r, active_layers, accessibility_distance)
    distance_results = {}
    print('\nCalculating nearest node analyses ...')
    for analysis_key in r.indicators['nearest_node_analyses']:
        print(f'\n\t- {analysis_key}')
        analysis = r.indicators['nearest_node_analyses'][analysis_key]
        for layer in analysis['layers']:
            if layer in r.tables and layer is not None:
                output_names = _resolve_output_names(analysis, layer)
                print(f'\t\t{output_names}')
                if engine == 'inmemory':
                    cols = [
                        col_name
                        for col_name, _ in nearest_poi_query_columns(
                            category_field=analysis['category_field'],
                            categories=analysis['categories'],
                            filter_field=analysis['filter_field'],
                            filter_iterations=analysis['filter_iterations'],
                            output_names=output_names,
                            output_prefix='sp_nearest_node_',
                        )
                    ]
                    distance_results[f'{analysis}_{layer}'] = (
                        nodes_dist_inmemory[cols]
                    )
                else:
                    distance_results[f'{analysis}_{layer}'] = (
                        cal_dist_node_to_nearest_pois(
                            r,
                            layer,
                            node_index=node_index,
                            category_field=analysis['category_field'],
                            categories=analysis['categories'],
                            filter_field=analysis['filter_field'],
                            filter_iterations=analysis['filter_iterations'],
                            output_names=output_names,
                            output_prefix='sp_nearest_node_',
                        )
                    )
            else:
                # create null results --- e.g. for GTFS analyses where no layer exists
                distance_results[f'{analysis_key}_{layer}'] = pd.DataFrame(
                    index=node_index,
                    columns=[
                        f'sp_nearest_node_{x}'
                        for x in analysis['output_names']
                    ],
                )
    if engine != 'inmemory':
        drop_dest_node_lookup(r)
    # concatenate analysis dataframes into one
    nodes_poi_dist = pd.concat(
        [distance_results[x] for x in distance_results],
        axis=1,
    )
    nodes_poi_dist = nodes_poi_dist[
        [x for x in nodes_poi_dist.columns if x.startswith('sp_nearest_node_')]
    ]
    # replace -999 values (meaning no destination reached in less than 400 metres) as nan
    nodes_poi_dist = (
        round(nodes_poi_dist, 0).replace(-999, np.nan).astype('Int64')
    )
    return nodes_poi_dist


# Average daytime weekday service frequency (minutes) at a public transport
# stop for it to be considered usable in the age-friendly transport indicators.
FREQUENT_PT_HEADWAY = 20

# Default condition identifying low-stress ('safe') walking links for the
# 'safe routes to public transport' indicator, evaluated against the network
# edges.  Only links whose recorded width is at least 1.5 m are traversed;
# links without a recorded width are excluded.  Overridable per study region
# via the 'safe_walking_edges' setting under 'network' in its configuration.
DEFAULT_SAFE_WALKING_EDGES = (
    "highway IN ('footway','path','pedestrian','living_street','residential')"
    " AND COALESCE(NULLIF(substring(width::text from '[0-9]+[.]?[0-9]*'), '')"
    '::numeric, 0) >= 1.5'
)


def _node_index(r):
    """Ordered index of network node osmids, matching nodes_poi_dist."""
    return pd.Index(
        r.get_df('SELECT osmid FROM nodes ORDER BY osmid')['osmid'].to_numpy(
            dtype='int64',
        ),
        name='osmid',
    )


def _empty_node_measure(node_index, column):
    """Null result for a node-level measure that could not be calculated."""
    return pd.Series(np.nan, index=node_index, name=column, dtype='Int64')


def _normalised_ids(values):
    """Stringify stop/route identifiers, dropping any float-style '.0' suffix,
    so identifiers match between the headway and route-stop tables however
    they were inferred by pandas when written to the database."""
    return pd.Series(
        [
            value[:-2]
            if str(value).strip().endswith('.0')
            else str(value).strip()
            for value in values
        ],
        index=getattr(values, 'index', None),
    )


def _nearest_node_distance_to_layer(
    r,
    layer,
    output_name,
    distance,
    node_index,
    engine,
    edge_where=None,
    dest_filter_field=None,
    dest_filter_iterations=None,
):
    """Distance from every network node to the nearest destination in a layer.

    A thin wrapper over the two routing engines used elsewhere in this module,
    returning a single ``sp_nearest_node_<output_name>`` Series (null where no
    destination is reachable within ``distance``).  ``edge_where`` optionally
    restricts routing to a subgraph of edges (for example the low-stress
    walking network).
    """
    column = f'sp_nearest_node_{output_name}'
    if engine == 'inmemory':
        plan = nearest_poi_query_columns(
            filter_field=dest_filter_field,
            filter_iterations=dest_filter_iterations,
            output_names=[output_name],
            output_prefix='sp_nearest_node_',
        )
        distances = cal_dist_nodes_to_nearest_pois_inmemory(
            r,
            [(layer, col_name, where_clause) for col_name, where_clause in plan],
            distance,
            node_index,
            where=edge_where,
        )
    else:
        build_dest_node_lookup(r, {layer}, distance, where=edge_where)
        try:
            distances = cal_dist_node_to_nearest_pois(
                r,
                layer,
                node_index=node_index,
                filter_field=dest_filter_field,
                filter_iterations=dest_filter_iterations,
                output_names=[output_name],
                output_prefix='sp_nearest_node_',
            )
        finally:
            drop_dest_node_lookup(r)
    return (
        round(distances[column], 0).replace(-999, np.nan).astype('Int64')
    )


def calculate_transit_destination_access(
    r,
    nodes_poi_dist,
    engine,
    distance,
    destination_column,
    output_name,
    label,
    singular_label,
    destination_label,
    stops_table,
):
    """Access to a destination category by public transport.

    Identifies frequent public transport stops from which a stop within
    walking distance of a destination is served by the same route (a
    direct service, with no interchange).  The network distance from every
    node to the nearest such stop is then measured along the pedestrian
    network, so sample points within the walking distance threshold of such a
    stop are attributed public transport access to that destination category.

    ``destination_column`` is the node-level walking distance to the
    destination category, ``output_name`` names the resulting
    ``sp_nearest_node_`` column, and ``label``/``singular_label``/
    ``destination_label`` are used in progress messages.
    """
    column = f'sp_nearest_node_{output_name}'
    node_index = nodes_poi_dist.index
    if (
        'pt_stops_headway' not in r.tables
        or 'pt_route_stops' not in r.tables
    ):
        print(
            f'  Public transport access to {label} skipped: public '
            'transport route-stop membership is not available for this study '
            'region.',
        )
        return _empty_node_measure(node_index, column)
    if destination_column not in nodes_poi_dist.columns:
        print(
            f'  Public transport access to {label} skipped: '
            f'{destination_label} destinations are not configured for this '
            'study region.',
        )
        return _empty_node_measure(node_index, column)

    stops = r.get_gdf('pt_stops_headway')
    stops['stop_id'] = _normalised_ids(stops['stop_id'])
    destination_distance = nodes_poi_dist[destination_column]

    def _stop_destination_distance(stop):
        # network walking distance from the stop to its nearest destination
        measures = []
        for node_col, offset_col in (
            ('n1', 'n1_distance'),
            ('n2', 'n2_distance'),
        ):
            node = stop.get(node_col)
            if node is None or pd.isna(node):
                continue
            node_distance = destination_distance.get(int(node))
            if node_distance is None or pd.isna(node_distance):
                continue
            offset = stop.get(offset_col)
            measures.append(
                float(node_distance)
                + (0.0 if offset is None or pd.isna(offset) else float(offset)),
            )
        return min(measures) if measures else np.nan

    stops['_destination_distance'] = stops.apply(
        _stop_destination_distance,
        axis=1,
    )
    destination_adjacent_stops = set(
        stops.loc[stops['_destination_distance'] <= distance, 'stop_id'],
    )

    route_stops = r.get_df('SELECT route_id, stop_id FROM pt_route_stops')
    if route_stops is None or route_stops.empty:
        print(
            f'  Public transport access to {label} skipped: no '
            'route-stop membership records found.',
        )
        return _empty_node_measure(node_index, column)
    route_stops['stop_id'] = _normalised_ids(route_stops['stop_id'])
    route_stops['route_id'] = _normalised_ids(route_stops['route_id'])

    # Routes serving at least one destination-adjacent stop ...
    destination_routes = set(
        route_stops.loc[
            route_stops['stop_id'].isin(destination_adjacent_stops),
            'route_id',
        ],
    )
    # ... and the frequent stops that those routes also serve.  A passenger can
    # travel between any two stops on the same route without interchanging.
    frequent_stops = stops.loc[
        stops['headway'].notna() & (stops['headway'] <= FREQUENT_PT_HEADWAY),
        'stop_id',
    ]
    access_stops = frequent_stops[
        frequent_stops.isin(
            set(
                route_stops.loc[
                    route_stops['route_id'].isin(destination_routes),
                    'stop_id',
                ],
            ),
        )
    ]
    print(
        f'  Transport access to {label}: {len(access_stops)} of '
        f'{len(frequent_stops)} frequent public transport stops provide a '
        f'direct service to within walking distance of {singular_label}.',
    )
    if access_stops.empty:
        return _empty_node_measure(node_index, column)

    access_gdf = stops[stops['stop_id'].isin(set(access_stops))].copy()
    with r.engine.begin() as connection:
        connection.execute(
            text(f'DROP TABLE IF EXISTS {stops_table}'),
        )
        access_gdf.to_postgis(
            stops_table,
            connection,
            index=False,
        )
    r.add_nearest_node_associations(stops_table)
    return _nearest_node_distance_to_layer(
        r,
        stops_table,
        output_name,
        distance,
        node_index,
        engine,
    )


def calculate_transit_health_services_access(
    r,
    nodes_poi_dist,
    engine,
    distance,
):
    """Access to health services by public transport."""
    return calculate_transit_destination_access(
        r,
        nodes_poi_dist,
        engine,
        distance,
        destination_column='sp_nearest_node_health_services',
        output_name='pt_transit_health_services',
        label='health services',
        singular_label='a health service',
        destination_label='health service',
        stops_table='pt_stops_health_transit',
    )


def _safe_walking_predicate(r):
    """Configured condition identifying low-stress walking links, or None when
    it cannot be evaluated for this study region (no network width attribute)."""
    network_config = r.config.get('network') or {}
    predicate = (
        network_config.get('safe_walking_edges') or DEFAULT_SAFE_WALKING_EDGES
    )
    if 'width' in predicate.lower():
        edge_columns = set(
            r.get_df(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = 'edges'",
            )['column_name'],
        )
        if 'width' not in edge_columns:
            print(
                '  Safe walking indicators skipped: the network edges have no '
                'footpath width attribute.  Re-run the network creation step '
                'to retain OSM width tags.',
            )
            return None
    try:
        safe_links = r.get_df(
            f'SELECT COUNT(*) AS n FROM edges WHERE {predicate}',
        )['n'][0]
        print(
            f'  Safe walking network: {safe_links} low-stress links satisfy '
            'the configured safe walking condition.',
        )
    except Exception as error:
        print(
            f'  Safe walking indicators: could not evaluate the safe walking '
            f'condition ({error}).',
        )
    return predicate


def calculate_safe_walking_access(r, engine, distance):
    """Walking access along low-stress ('safe') routes.

    An iteration of the frequent public transport and daily living access
    indicators, routed only over links considered safe for an older person to
    walk along, as defined by the configured condition on footpath width (see
    ``DEFAULT_SAFE_WALKING_EDGES`` and the region's ``network`` settings).

    Returns the network distance from every node to the nearest frequent public
    transport stop, and to the nearest fresh food market, convenience store and
    public transport stop (the daily living destinations).
    """
    columns = [
        'sp_nearest_node_pt_safe_freq_20',
        'sp_nearest_node_safe_fresh_food_market',
        'sp_nearest_node_safe_convenience',
        'sp_nearest_node_safe_pt_any',
    ]
    node_index = _node_index(r)
    predicate = _safe_walking_predicate(r)
    if predicate is None:
        return pd.DataFrame(
            np.nan,
            index=node_index,
            columns=columns,
            dtype='Int64',
        )
    layer_plans = {}
    if 'pt_stops_headway' in r.tables:
        layer_plans['pt_stops_headway'] = nearest_poi_query_columns(
            filter_field='headway',
            filter_iterations=[f'<={FREQUENT_PT_HEADWAY}'],
            output_names=['pt_safe_freq_20'],
            output_prefix='sp_nearest_node_',
        )
    if 'destinations' in r.tables:
        layer_plans['destinations'] = nearest_poi_query_columns(
            category_field='dest_name',
            categories=['fresh_food_market', 'convenience', 'pt_any'],
            output_names=[
                'safe_fresh_food_market',
                'safe_convenience',
                'safe_pt_any',
            ],
            output_prefix='sp_nearest_node_',
        )
    if not layer_plans:
        print(
            '  Safe walking indicators skipped: no destination layers are '
            'available for this study region.',
        )
        return pd.DataFrame(
            np.nan,
            index=node_index,
            columns=columns,
            dtype='Int64',
        )
    if engine == 'inmemory':
        plan = [
            (layer, col_name, where_clause)
            for layer, entries in layer_plans.items()
            for col_name, where_clause in entries
        ]
        distances = cal_dist_nodes_to_nearest_pois_inmemory(
            r,
            plan,
            distance,
            node_index,
            where=predicate,
        )
    else:
        build_dest_node_lookup(
            r,
            set(layer_plans),
            distance,
            where=predicate,
        )
        try:
            parts = []
            for layer, entries in layer_plans.items():
                options = {}
                if layer == 'pt_stops_headway':
                    options = {
                        'filter_field': 'headway',
                        'filter_iterations': [f'<={FREQUENT_PT_HEADWAY}'],
                        'output_names': ['pt_safe_freq_20'],
                    }
                else:
                    options = {
                        'category_field': 'dest_name',
                        'categories': [
                            'fresh_food_market',
                            'convenience',
                            'pt_any',
                        ],
                        'output_names': [
                            'safe_fresh_food_market',
                            'safe_convenience',
                            'safe_pt_any',
                        ],
                    }
                parts.append(
                    cal_dist_node_to_nearest_pois(
                        r,
                        layer,
                        node_index=node_index,
                        output_prefix='sp_nearest_node_',
                        **options,
                    ),
                )
            distances = pd.concat(parts, axis=1)
        finally:
            drop_dest_node_lookup(r)
    result = pd.DataFrame(index=node_index)
    for column in columns:
        if column in distances.columns:
            result[column] = (
                round(distances[column], 0)
                .replace(-999, np.nan)
                .astype('Int64')
            )
        else:
            result[column] = pd.Series(np.nan, index=node_index, dtype='Int64')
    return result


def calculate_sample_point_access_scores(
    r,
    nodes_simple,
    nodes_poi_dist,
    density_statistics,
    accessibility_distance,
):
    # read sample points from disk (in city-specific geopackage)
    sample_points = r.get_gdf('urban_sample_points')
    sample_points.columns = [
        'geometry' if x == 'geom' else x for x in sample_points.columns
    ]
    sample_points.set_geometry('geometry', inplace=True)
    sample_points = filter_ids(
        df=sample_points,
        query=f"""n1 in {nodes_simple.index.tolist()} and n2 in {nodes_simple.index.tolist()}""",
        message='Restrict sample points to those with two associated sample nodes...',
    )
    sample_points.set_index('point_id', inplace=True)
    # Estimate full distance to destinations for sample points
    full_nodes = create_full_nodes(
        sample_points,
        nodes_simple,
        nodes_poi_dist,
        list(density_statistics.values()),
    )
    sample_points = sample_points[
        ['grid_id', 'edge_ogc_fid', 'geometry']
    ].join(full_nodes, how='left')
    # create binary access scores evaluated against accessibility distance
    # Options for distance decay accessibility scores are available in setup_sp.py module
    distance_names = list(nodes_poi_dist.columns)
    access_score_names = [
        f"{x.replace('nearest_node','access')}_score" for x in distance_names
    ]
    # Join the access-score columns in one operation rather than a block insert
    # into the GeoDataFrame, mirroring the cycling path and avoiding frame
    # fragmentation as the number of destination columns grows.
    scores = binary_access_score(
        sample_points,
        distance_names,
        accessibility_distance,
    )
    # binary_access_score returns the distance_names columns in order; rename
    # positionally to the access-score names (as the block assignment did)
    scores.columns = access_score_names
    sample_points = sample_points.join(scores)
    return sample_points


def calculate_sample_point_indicators(
    r,
    sample_points,
):
    print('Calculating sample point specific analyses ...')
    # Accumulate the new indicator columns and join them in a single operation at
    # the end rather than inserting each into the GeoDataFrame as it is computed:
    # per-column insertion fragments the frame (pandas PerformanceWarning and
    # O(ncols^2) recopying) as the number of configured analyses grows.  The
    # read() accessor makes freshly-computed indicators visible to later analyses,
    # preserving the original var-by-var dependency chain exactly (e.g. daily
    # living reads the PT access score; walkability reads daily living).
    computed = {}

    def read(cols):
        # mirror sample_points[cols] (cols a str or list) but with freshly
        # computed indicators taking precedence over the base frame
        if isinstance(cols, str):
            return computed[cols] if cols in computed else sample_points[cols]
        return pd.concat(
            [
                (computed[c] if c in computed else sample_points[c]).rename(c)
                for c in cols
            ],
            axis=1,
        )

    # Defined in generated config file, e.g. daily living score, walkability index, etc
    for analysis in r.indicators['sample_point_analyses']:
        print(f'\t - {analysis}')
        for var in r.indicators['sample_point_analyses'][analysis]:
            variable = r.indicators['sample_point_analyses'][analysis][var]
            if 'layer' in variable and 'field' in variable:
                layer = variable['layer']
                field = variable['field']
                formula = variable.get('formula', 'intersection')
                if formula == 'intersection':
                    # retrieve polygon layer from database and assign value of new sample point variable based on intersection of sample points with the polygon layer, using the specified field from the polygon layer
                    gdf_polys = r.get_gdf(layer)
                    if gdf_polys is None:
                        # A missing layer (e.g. an Earth Engine indicator table
                        # that was not generated) would otherwise crash the
                        # whole analysis at `gdf_polys[[field, 'geom']]`.
                        # Report it and skip this variable instead.
                        print(
                            f"  Warning: could not retrieve layer '{layer}' "
                            f"for sample point variable '{var}'; skipping. "
                            'This usually indicates the corresponding '
                            'indicator table was not generated for this '
                            'study region.',
                        )
                        continue
                    joined = gpd.sjoin(
                        sample_points,
                        gdf_polys[[field, 'geom']],  # only keep needed columns
                        how='left',
                        predicate='within',  # or "intersects" depending on your use case
                    )
                    computed[var] = joined[field]
            elif 'columns' in variable and 'axis' in variable:
                columns = variable['columns']
                formula = variable['formula']
                axis = variable['axis']
                if formula == 'sum':
                    computed[var] = read(columns).sum(axis=axis)
                if formula == 'max':
                    computed[var] = read(columns).max(axis=axis)
                if formula == 'sum_of_z_scores':
                    block = read(columns)
                    computed[var] = ((block - block.mean()) / block.std()).sum(
                        axis=1,
                    )
                if formula.startswith('greater_than_or_equal_to'):
                    threshold = float(formula.split('(')[1].split(')')[0])
                    block = read(columns)
                    if isinstance(block, pd.DataFrame):
                        # elementwise formula; a single-column selection must
                        # be reduced to a series to form one output column
                        block = block.iloc[:, 0]
                    computed[var] = (block >= threshold).astype(int)
    if computed:
        new_columns = pd.DataFrame(computed, index=sample_points.index)
        # if an analysis reuses an existing column name, drop the old column first
        # so join replaces it (matching the original per-column overwrite); this is
        # a no-op for the standard analyses, which only ever add new columns
        overlap = [
            c for c in new_columns.columns if c in sample_points.columns
        ]
        if overlap:
            sample_points = sample_points.drop(columns=overlap)
        sample_points = sample_points.join(new_columns)
    # grid_id and edge_ogc_fid are integers; grid_id uses a nullable integer
    # type, as it may be null for sample points located in areas lacking
    # population data coverage (if such sampling has been configured)
    sample_points['grid_id'] = sample_points['grid_id'].astype('Int64')
    sample_points['edge_ogc_fid'] = sample_points['edge_ogc_fid'].astype(int)
    # remaining non-geometry fields are float
    sample_points[sample_points.columns[3:]] = sample_points[
        sample_points.columns[3:]
    ].astype(float)
    return sample_points


def neighbourhood_analysis(codename):
    start = time.time()
    script = '_11_neighbourhood_analysis'
    task = 'Analyse neighbourhood indicators for sample points'
    r = ghsci.Region(codename)
    destination_tables = [
        'destinations',
        'aos_public_any_nodes_30m_line',
        'aos_public_large_nodes_30m_line',
        'pt_stops_headway',
    ]
    # Conditional check to generate Earth Engine indicators
    if r.config['gee']:
        try:
            from _earth_engine_indicators import earth_engine_analysis

            earth_engine_analysis(r)
            destination_tables.append('pugs_nodes_30m_line')
            # Refresh cached table list so tables created by the Earth Engine
            # analysis (e.g. pugs_nodes_30m_line) are recognised below on a
            # first analysis pass
            r.tables = r.get_tables()
        except Exception as e:
            # Fail rather than continue with incomplete results that would
            # only surface as errors at report generation time
            raise Exception(
                f"Error occurred while running Earth Engine analysis: {e}",
            )
    print(
        'Pre-associating destinations with nearest nodes for accessibility analysis...',
    )
    for table in destination_tables:
        if table in r.tables:
            print(f'\t- {table}... ')
            r.add_nearest_node_associations(table)
    nodes = r.get_gdf('nodes', index_col='osmid')
    nodes.columns = ['geometry' if x == 'geom' else x for x in nodes.columns]
    nodes = nodes.set_geometry('geometry')
    edges = r.get_gdf('edges_simplified', index_col=['u', 'v', 'key'])
    edges.columns = ['geometry' if x == 'geom' else x for x in edges.columns]
    edges = edges.set_geometry('geometry')
    nodes_simple = node_level_neighbourhood_analysis(
        r,
        edges,
        nodes,
        ghsci.settings['network_analysis']['neighbourhood_distance'],
    )
    accessibility_distance = ghsci.settings['network_analysis'][
        'accessibility_distance'
    ]
    routing_engine = pedestrian_routing_engine(r)
    nodes_poi_dist = calculate_poi_accessibility(r, engine=routing_engine)
    # Age-friendly public transport indicators, which re-use the walking
    # network measures above:
    #   - public transport access to health services (walk to a frequent stop,
    #     direct service to a stop near a health service, walk to the health
    #     service);
    #   - public transport access to community and social facilities (the same
    #     composite, applied to the community facilities destination category);
    #   - safe routes to public transport (as the frequent public transport
    #     indicator, but routed only over low-stress walking links).
    nodes_poi_dist = nodes_poi_dist.join(
        calculate_transit_health_services_access(
            r,
            nodes_poi_dist,
            routing_engine,
            accessibility_distance,
        ),
    )
    nodes_poi_dist = nodes_poi_dist.join(
        calculate_transit_destination_access(
            r,
            nodes_poi_dist,
            routing_engine,
            accessibility_distance,
            destination_column='sp_nearest_node_community_facilities',
            output_name='pt_transit_community_facilities',
            label='community and social facilities',
            singular_label='a community and social facility',
            destination_label='community and social facility',
            stops_table='pt_stops_community_facilities_transit',
        ),
    )
    nodes_poi_dist = nodes_poi_dist.join(
        calculate_safe_walking_access(
            r,
            routing_engine,
            accessibility_distance,
        ),
    )

    sample_points = calculate_sample_point_access_scores(
        r,
        nodes_simple,
        nodes_poi_dist,
        density_statistics,
        accessibility_distance,
    )

    sample_points = calculate_sample_point_indicators(r, sample_points)

    print('Save to database...')
    # save the sample points with all the desired results to a new layer in the database
    sample_points.columns = [
        'geom' if x == 'geometry' else x for x in sample_points.columns
    ]
    sample_points = sample_points.set_geometry('geom')
    with r.engine.connect() as connection:
        sample_points.to_postgis(
            r.config['point_summary'],
            connection,
            index=True,
            if_exists='replace',
        )
    # output to completion log
    script_running_log(r.config, script, task, start)
    r.engine.dispose()


def main():
    try:
        codename = sys.argv[1]
    except IndexError:
        codename = None
    neighbourhood_analysis(codename)


if __name__ == '__main__':
    main()
