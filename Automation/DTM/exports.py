"""GeoTIFF/vector export helpers, perimeter clipping, and building lift utilities."""

from __future__ import annotations

import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.transform import Affine
from rasterio.features import geometry_mask, rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Polygon, LineString, Point, MultiLineString
from shapely.ops import linemerge, nearest_points, split, unary_union

# Bound by channel_modifier.py after the final DTMChannelModifier facade class is created.
# The original implementation references DTMChannelModifier inside many static methods;
# keeping that symbol here preserves the existing method bodies while allowing this file
# to stay focused on one part of the workflow.
DTMChannelModifier = None


class ExportMixin:
    """GeoTIFF/vector export helpers, perimeter clipping, and building lift utilities."""

    @staticmethod
    def _write_modifier_geotiff(modifier, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        meta = modifier.dtm_meta.copy()
        meta.update(
            {
                "driver": "GTiff",
                "height": modifier.dtm_data.shape[0],
                "width": modifier.dtm_data.shape[1],
                "count": 1,
                "dtype": "float32",
                "crs": modifier.dtm_crs,
                "transform": modifier.dtm_transform,
            }
        )
        data = ExportMixin._sanitize_geotiff_data(modifier)
        with rasterio.open(output_path, "w", **meta) as dest:
            dest.write(data, 1)

    @staticmethod
    def _sanitize_geotiff_data(modifier):
        data = np.asarray(modifier.dtm_data, dtype="float32").copy()
        nodata = modifier.dtm_meta.get("nodata") if modifier.dtm_meta else None
        if nodata is None:
            data[~np.isfinite(data)] = np.nan
            return data

        invalid = ~np.isfinite(data) | np.isclose(data, nodata)
        original = getattr(modifier, "original_dtm_data", None)
        if original is not None:
            original_arr = np.asarray(original)
            invalid |= ~np.isfinite(original_arr) | np.isclose(original_arr, nodata)

        if np.any(invalid):
            data[invalid] = float(nodata)
        return data

    @staticmethod
    def _apply_building_lift_to_modifier(modifier, buildings_shp_path=None, lift_m=0.0):
        lift = float(lift_m or 0.0)
        summary = {
            "enabled": bool(buildings_shp_path) and abs(lift) > 1e-9,
            "buildings_shp": str(buildings_shp_path) if buildings_shp_path else None,
            "lift_m": lift,
            "cells_lifted": 0,
        }
        if not summary["enabled"]:
            return summary

        buildings_path = Path(buildings_shp_path)
        if not buildings_path.exists():
            summary["warning"] = f"Building shapefile not found: {buildings_path}"
            print(f"Warning: {summary['warning']}")
            return summary

        buildings_gdf = gpd.read_file(buildings_path)
        if buildings_gdf.empty:
            summary["warning"] = f"Building shapefile has no features: {buildings_path}"
            print(f"Warning: {summary['warning']}")
            return summary

        if modifier.dtm_crs is not None:
            if buildings_gdf.crs is None:
                buildings_gdf = buildings_gdf.set_crs(modifier.dtm_crs, allow_override=True)
            elif buildings_gdf.crs != modifier.dtm_crs:
                buildings_gdf = buildings_gdf.to_crs(modifier.dtm_crs)

        geometries = [
            geometry
            for geometry in buildings_gdf.geometry
            if geometry is not None and not geometry.is_empty
        ]
        if not geometries:
            summary["warning"] = f"Building shapefile has no valid polygon geometry: {buildings_path}"
            print(f"Warning: {summary['warning']}")
            return summary

        mask = rasterize(
            geometries,
            out_shape=modifier.dtm_data.shape,
            transform=modifier.dtm_transform,
            fill=0,
            default_value=1,
            dtype="uint8",
            all_touched=True,
        ).astype(bool)

        nodata = modifier.dtm_meta.get("nodata") if modifier.dtm_meta else None
        if nodata is not None:
            mask &= ~np.isclose(modifier.dtm_data, nodata)

        cell_count = int(np.count_nonzero(mask))
        if cell_count:
            modifier.dtm_data = np.array(modifier.dtm_data, copy=True)
            modifier.dtm_data[mask] = modifier.dtm_data[mask].astype(float) + lift

        summary["cells_lifted"] = cell_count
        print(
            f"Applied building lift of {lift:g} m to {cell_count} raster cells "
            f"using {buildings_path}."
        )
        return summary

    @staticmethod
    def _delete_vector_sidecars(path):
        path = Path(path)
        if not path.parent.exists():
            return True
        deleted_all = True
        for sidecar in path.parent.glob(f"{path.stem}.*"):
            try:
                sidecar.unlink()
            except OSError:
                deleted_all = False
        return deleted_all

    @staticmethod
    def _write_gdf_with_locked_file_fallback(gdf, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not DTMChannelModifier._delete_vector_sidecars(output_path):
                raise PermissionError(f"Could not replace locked vector sidecars for {output_path}")
            gdf.to_file(output_path)
            return output_path
        except PermissionError:
            for index in range(1, 100):
                fallback = output_path.with_name(f"{output_path.stem}_new{index}{output_path.suffix}")
                try:
                    DTMChannelModifier._delete_vector_sidecars(fallback)
                    gdf.to_file(fallback)
                    print(
                        f"Warning: {output_path} is locked by another process; "
                        f"wrote {fallback} instead."
                    )
                    return fallback
                except PermissionError:
                    continue
            raise

    @staticmethod
    def _export_network_centerlines(channels, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "Channel": channel["name"],
                "CLSource": channel.get("centerline_source", "bank_lines"),
                "ProcCLSrc": channel.get("processing_centerline_source", "bank_lines"),
                "geometry": channel["processing_centerline"],
            }
            for channel in channels
        ]
        crs = channels[0]["processing_banks_gdf"].crs if channels else None
        gpd.GeoDataFrame(rows, crs=crs).to_file(output_path)

    @staticmethod
    def _export_network_bank_polygons(channels, output_path, offset_m=0.2):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        crs = channels[0]["processing_banks_gdf"].crs if channels else None
        for channel in channels:
            try:
                polygon_gdf = DTMChannelModifier.create_polygon_mask_from_banks(
                    channel["processing_banks_gdf"],
                    offset_m=offset_m,
                )
            except Exception as exc:
                print(f"Warning: bank polygon export failed for {channel['name']}: {exc}")
                continue

            for geometry in polygon_gdf.geometry:
                if geometry is None or geometry.is_empty:
                    continue
                rows.append(
                    {
                        "Channel": str(channel["name"])[:80],
                        "OffsetM": float(offset_m),
                        "geometry": geometry,
                    }
                )

        if not rows:
            print(f"Warning: no bank polygons were created for {output_path}.")
            return None

        bank_polygons_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
        written_path = DTMChannelModifier._write_gdf_with_locked_file_fallback(
            bank_polygons_gdf,
            output_path,
        )
        print(f"Bank polygon shapefile written to: {written_path}")
        return written_path

    @staticmethod
    def _export_network_perimeter(channels, output_path, offset_m=500.0, network=None):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        perimeter = DTMChannelModifier._build_clipped_network_perimeter(
            channels=channels,
            offset_m=offset_m,
            network=network,
        )
        crs = channels[0]["processing_banks_gdf"].crs if channels else None
        gpd.GeoDataFrame(
            [{"Name": f"Network Study Perimeter {offset_m}m", "geometry": perimeter}],
            crs=crs,
        ).to_file(output_path)

    @staticmethod
    def _build_clipped_network_perimeter(channels, offset_m=500.0, network=None):
        if not channels:
            raise ValueError("At least one channel is required to export a study perimeter.")

        network = network or {"channels": channels, "junctions": []}
        junctions = network.get("junctions", [])
        main_indices = {int(junction["main_index"]) for junction in junctions}
        tributary_by_index = {
            int(junction["tributary_index"]): junction
            for junction in junctions
        }

        clipped_polygons = []
        for channel in channels:
            centerline = channel.get("processing_centerline") or channel["centerline"]
            if centerline is None or centerline.is_empty:
                continue

            channel_perimeter = centerline.buffer(offset_m)
            if channel_perimeter.is_empty:
                continue

            bank_lines = DTMChannelModifier._line_strings(channel["banks_gdf"])
            sections = DTMChannelModifier._cross_sections_in_file_order(
                cross_section_csv=channel["cross_section_csv"],
                centerline=channel["centerline"],
                bank_lines=bank_lines,
            )
            if not sections:
                clipped_polygons.append(channel_perimeter)
                continue

            channel_index = int(channel.get("index", len(clipped_polygons)))
            is_main = channel_index in main_indices
            tributary_junction = tributary_by_index.get(channel_index)

            if not junctions or is_main or tributary_junction is None:
                channel_perimeter = DTMChannelModifier._clip_perimeter_between_cross_sections(
                    polygon=channel_perimeter,
                    channel_centerline=channel["centerline"],
                    start_section=sections[0],
                    end_section=sections[-1],
                )
            else:
                junction_point = Point(
                    float(tributary_junction["x"]),
                    float(tributary_junction["y"]),
                )
                channel_perimeter = DTMChannelModifier._clip_perimeter_at_cross_section(
                    polygon=channel_perimeter,
                    cut_section=sections[-1],
                    keep_point=junction_point,
                )

            if not channel_perimeter.is_empty:
                clipped_polygons.append(channel_perimeter)

        if not clipped_polygons:
            centerlines = [channel["processing_centerline"] for channel in channels]
            return unary_union(centerlines).buffer(offset_m)

        return unary_union(clipped_polygons)

    @staticmethod
    def _cross_sections_in_file_order(cross_section_csv, centerline, bank_lines=None):
        df = DTMChannelModifier._read_csv_auto(
            cross_section_csv,
            required_columns=("X", "Y"),
        )
        group_cols = [column for column in ["River", "Reach", "Station"] if column in df.columns]
        if not group_cols:
            group_cols = ["Station"] if "Station" in df.columns else []

        grouped = df.groupby(group_cols, sort=False) if group_cols else [(None, df)]
        sections = []
        for name, group in grouped:
            if len(group) < 2:
                continue
            coord_columns = ["X", "Y", "Z"] if "Z" in group.columns else ["X", "Y"]
            line = LineString(group[coord_columns].to_numpy(dtype=float))
            if line.length <= 0:
                continue
            station_name = str(name if not isinstance(name, tuple) else name[-1])
            center_point = DTMChannelModifier._cross_section_center_point_from_centerline(
                line,
                centerline,
                label=station_name,
                bank_lines=bank_lines,
            )
            sections.append(
                {
                    "station": station_name,
                    "line": line,
                    "center_point": center_point,
                    "centerline_measure": float(centerline.project(center_point)),
                }
            )
        return sections

    @staticmethod
    def _clip_perimeter_between_cross_sections(
        polygon,
        channel_centerline,
        start_section,
        end_section,
    ):
        if polygon.is_empty:
            return polygon

        start_measure = float(start_section["centerline_measure"])
        end_measure = float(end_section["centerline_measure"])
        if abs(end_measure - start_measure) <= 1e-6:
            return polygon

        start_keep = DTMChannelModifier._centerline_point_toward_measure(
            channel_centerline,
            from_measure=start_measure,
            toward_measure=end_measure,
        )
        end_keep = DTMChannelModifier._centerline_point_toward_measure(
            channel_centerline,
            from_measure=end_measure,
            toward_measure=start_measure,
        )

        clipped = DTMChannelModifier._clip_perimeter_at_cross_section(
            polygon=polygon,
            cut_section=start_section,
            keep_point=start_keep,
        )
        clipped = DTMChannelModifier._clip_perimeter_at_cross_section(
            polygon=clipped,
            cut_section=end_section,
            keep_point=end_keep,
        )
        return clipped

    @staticmethod
    def _centerline_point_toward_measure(centerline, from_measure, toward_measure):
        length = max(float(centerline.length), 0.0)
        from_measure = float(np.clip(from_measure, 0.0, length))
        toward_measure = float(np.clip(toward_measure, 0.0, length))
        direction = 1.0 if toward_measure >= from_measure else -1.0
        step = min(max(length * 0.01, 0.5), 5.0)
        target_measure = from_measure + direction * step
        if direction > 0:
            target_measure = min(target_measure, toward_measure, length)
        else:
            target_measure = max(target_measure, toward_measure, 0.0)
        if abs(target_measure - from_measure) <= 1e-9:
            target_measure = toward_measure
        return centerline.interpolate(target_measure)

    @staticmethod
    def _clip_perimeter_at_cross_section(polygon, cut_section, keep_point):
        if polygon.is_empty:
            return polygon

        extended_line = DTMChannelModifier._extended_cross_section_line(
            cut_section["line"],
            polygon,
        )
        if extended_line is None or extended_line.is_empty:
            return polygon

        try:
            pieces = list(split(polygon, extended_line).geoms)
        except Exception:
            return polygon

        polygonal_pieces = [
            piece
            for piece in pieces
            if piece.geom_type in {"Polygon", "MultiPolygon"} and not piece.is_empty
        ]
        if len(polygonal_pieces) <= 1:
            return polygon

        keep_point = Point(float(keep_point.x), float(keep_point.y))
        tolerance = max(polygon.length * 1e-9, 1e-6)
        selected = [
            piece
            for piece in polygonal_pieces
            if piece.buffer(tolerance).contains(keep_point)
            or piece.buffer(tolerance).touches(keep_point)
        ]
        if not selected:
            nearest_piece = min(polygonal_pieces, key=lambda piece: piece.distance(keep_point))
            selected = [nearest_piece]

        return unary_union(selected)

    @staticmethod
    def _extended_cross_section_line(cross_section_line, polygon):
        coords = list(cross_section_line.coords)
        if len(coords) < 2:
            return None

        start = np.asarray(coords[0][:2], dtype=float)
        end = np.asarray(coords[-1][:2], dtype=float)
        vector = end - start
        norm = np.linalg.norm(vector)
        if norm <= 0:
            return None
        unit = vector / norm

        minx, miny, maxx, maxy = polygon.bounds
        diagonal = float(np.hypot(maxx - minx, maxy - miny))
        extension = max(diagonal * 3.0, float(cross_section_line.length) * 3.0, 100.0)
        extended_start = start - unit * extension
        extended_end = end + unit * extension
        return LineString(
            [
                (float(extended_start[0]), float(extended_start[1])),
                (float(extended_end[0]), float(extended_end[1])),
            ]
        )

    @staticmethod
    def _export_connected_bank_products(
        network,
        output_dir,
        clip_buffer_m=5.0,
        nearest_cross_section_count=2,
        combined_prefix=None,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        products = []
        all_clipped_bank_gdfs = []
        all_junction_polygon_rows = []
        all_inner_polygon_rows = []
        combined_crs = None
        safe_combined_prefix = DTMChannelModifier._safe_name(combined_prefix or "network")
        pair_counts = {}
        for junction in network.get("junctions", []):
            main = network["channels"][junction["main_index"]]
            tributary = network["channels"][junction["tributary_index"]]
            pair_key = (str(tributary["name"]), str(main["name"]))
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

        for junction_index, junction in enumerate(network.get("junctions", []), start=1):
            main = network["channels"][junction["main_index"]]
            tributary = network["channels"][junction["tributary_index"]]
            junction_id = str(junction.get("id") or f"J{junction_index:03d}")
            safe_pair_name = (
                f"{DTMChannelModifier._safe_name(tributary['name'])}"
                f"__{DTMChannelModifier._safe_name(main['name'])}"
            )
            pair_key = (str(tributary["name"]), str(main["name"]))
            safe_junction_name = safe_pair_name
            if pair_counts.get(pair_key, 0) > 1:
                safe_junction_name = f"{safe_pair_name}_{DTMChannelModifier._safe_name(junction_id)}"
            junction_point = Point(float(junction["x"]), float(junction["y"]))
            merged_banks = DTMChannelModifier.build_connected_junction_banklines(
                channels=[tributary, main],
            )

            merged_path = output_dir / f"{safe_junction_name}_SEV_USTU_combined.shp"
            merged_path = DTMChannelModifier._write_gdf_with_locked_file_fallback(
                merged_banks,
                merged_path,
            )

            raw_clipped_banks = DTMChannelModifier._junction_bank_lines_between_cross_sections(
                tributary=tributary,
                main=main,
                junction=junction,
                junction_point=junction_point,
            )
            clipped_banks = DTMChannelModifier._join_gdf_line_features_by_proximity(
                raw_clipped_banks,
                tolerance=1.0,
            )
            if combined_crs is None and clipped_banks is not None:
                combined_crs = clipped_banks.crs
            clipped_banks = DTMChannelModifier._with_junction_export_fields(
                clipped_banks,
                junction=junction,
                junction_id=junction_id,
                main=main,
                tributary=tributary,
            )
            if clipped_banks is not None and not clipped_banks.empty:
                all_clipped_bank_gdfs.append(clipped_banks.copy())

            clipped_path = output_dir / f"{safe_junction_name}_SEV_USTU_junction_clipped.shp"
            clipped_path = DTMChannelModifier._write_gdf_with_locked_file_fallback(
                clipped_banks,
                clipped_path,
            )

            junction_polygon_path = None
            junction_inner_polygon_path = None
            try:
                junction_polygon = DTMChannelModifier._junction_bank_polygon_from_clipped_banks(
                    raw_clipped_banks,
                    bank_lines=DTMChannelModifier._line_strings(clipped_banks),
                    junction_point=junction_point,
                )
                if junction_polygon is not None and not junction_polygon.is_empty:
                    polygon_gdf = gpd.GeoDataFrame(
                        [
                            {
                                "JunctionId": junction_id[:80],
                                "JuncRow": junction.get("network_row_index"),
                                "main": str(main["name"])[:80],
                                "tributary": str(tributary["name"])[:80],
                                "geometry": junction_polygon,
                            }
                        ],
                        geometry="geometry",
                        crs=clipped_banks.crs,
                    )
                    all_junction_polygon_rows.extend(polygon_gdf.to_dict("records"))
                    junction_polygon_path = output_dir / f"{safe_junction_name}_SEV_USTU_junction_bank_polygon.shp"
                    junction_polygon_path = DTMChannelModifier._write_gdf_with_locked_file_fallback(
                        polygon_gdf,
                        junction_polygon_path,
                    )
                    inner_polygon = DTMChannelModifier._fresh_junction_inner_bed_polygon(
                        junction_bank_polygon=junction_polygon,
                        bank_lines=DTMChannelModifier._line_strings(clipped_banks),
                        offset_m=0.3,
                    )
                    if inner_polygon is not None and not inner_polygon.is_empty:
                        inner_polygon_gdf = gpd.GeoDataFrame(
                            [
                                {
                                    "JunctionId": junction_id[:80],
                                    "JuncRow": junction.get("network_row_index"),
                                    "main": str(main["name"])[:80],
                                    "tributary": str(tributary["name"])[:80],
                                    "offset_m": 0.3,
                                    "geometry": inner_polygon,
                                }
                            ],
                            geometry="geometry",
                            crs=clipped_banks.crs,
                        )
                        all_inner_polygon_rows.extend(inner_polygon_gdf.to_dict("records"))
                        junction_inner_polygon_path = output_dir / f"{safe_junction_name}_SEV_USTU_junction_inner_0p3m_polygon.shp"
                        junction_inner_polygon_path = DTMChannelModifier._write_gdf_with_locked_file_fallback(
                            inner_polygon_gdf,
                            junction_inner_polygon_path,
                        )
            except Exception as exc:
                print(f"Warning: junction bank polygon export failed for {safe_pair_name}: {exc}")

            products.append(
                {
                    "type": "junction",
                    "junction_id": junction_id,
                    "network_row_index": junction.get("network_row_index"),
                    "main": main["name"],
                    "tributary": tributary["name"],
                    "merged_banks_shp": str(merged_path),
                    "junction_clipped_banks_shp": str(clipped_path),
                    "junction_bank_polygon_shp": str(junction_polygon_path) if junction_polygon_path else None,
                    "junction_inner_bed_polygon_shp": str(junction_inner_polygon_path) if junction_inner_polygon_path else None,
                }
            )

        combined_product = DTMChannelModifier._write_combined_junction_products(
            output_dir=output_dir,
            safe_prefix=safe_combined_prefix,
            clipped_bank_gdfs=all_clipped_bank_gdfs,
            junction_polygon_rows=all_junction_polygon_rows,
            inner_polygon_rows=all_inner_polygon_rows,
            crs=combined_crs,
        )
        if combined_product:
            products.append(combined_product)

        return products

    @staticmethod
    def _with_junction_export_fields(gdf, junction, junction_id, main, tributary):
        if gdf is None or gdf.empty:
            return gdf

        export_gdf = gdf.copy()
        export_gdf["JunctionId"] = str(junction_id)[:80]
        export_gdf["JuncRow"] = junction.get("network_row_index")
        export_gdf["Main"] = str(main["name"])[:80]
        export_gdf["Tributary"] = str(tributary["name"])[:80]
        export_gdf["From"] = str(junction.get("from") or tributary["name"])[:80]
        export_gdf["To"] = str(junction.get("to") or main["name"])[:80]
        return export_gdf

    @staticmethod
    def _write_combined_junction_products(
        output_dir,
        safe_prefix,
        clipped_bank_gdfs,
        junction_polygon_rows,
        inner_polygon_rows,
        crs=None,
    ):
        combined_product = {
            "type": "all_junctions",
            "junction_count": len(junction_polygon_rows),
            "junction_clipped_banks_shp": None,
            "junction_bank_polygon_shp": None,
            "junction_inner_bed_polygon_shp": None,
        }

        if clipped_bank_gdfs:
            combined_clipped = gpd.GeoDataFrame(
                pd.concat(clipped_bank_gdfs, ignore_index=True),
                geometry="geometry",
                crs=clipped_bank_gdfs[0].crs,
            )
            combined_clipped_path = output_dir / f"{safe_prefix}_SEV_USTU_all_junction_clipped.shp"
            combined_clipped_path = DTMChannelModifier._write_gdf_with_locked_file_fallback(
                combined_clipped,
                combined_clipped_path,
            )
            combined_product["junction_clipped_banks_shp"] = str(combined_clipped_path)

        if junction_polygon_rows:
            polygon_gdf = gpd.GeoDataFrame(
                junction_polygon_rows,
                geometry="geometry",
                crs=crs,
            )
            polygon_path = output_dir / f"{safe_prefix}_SEV_USTU_all_junction_bank_polygons.shp"
            polygon_path = DTMChannelModifier._write_gdf_with_locked_file_fallback(
                polygon_gdf,
                polygon_path,
            )
            combined_product["junction_bank_polygon_shp"] = str(polygon_path)

        if inner_polygon_rows:
            inner_gdf = gpd.GeoDataFrame(
                inner_polygon_rows,
                geometry="geometry",
                crs=crs,
            )
            inner_path = output_dir / f"{safe_prefix}_SEV_USTU_all_junction_inner_0p3m_polygons.shp"
            inner_path = DTMChannelModifier._write_gdf_with_locked_file_fallback(
                inner_gdf,
                inner_path,
            )
            combined_product["junction_inner_bed_polygon_shp"] = str(inner_path)

        if not any(
            combined_product[key]
            for key in (
                "junction_clipped_banks_shp",
                "junction_bank_polygon_shp",
                "junction_inner_bed_polygon_shp",
            )
        ):
            return None
        return combined_product

    @staticmethod
    def build_connected_junction_banklines(channels, proximity_tolerance=1.0):
        rows = []
        crs = None
        all_lines = []

        for channel in channels:
            bank_gdf = gpd.read_file(channel["bank_shp_path"])
            if crs is None:
                crs = bank_gdf.crs
            for line in DTMChannelModifier._line_strings(bank_gdf):
                all_lines.append(line)

        merged_lines = DTMChannelModifier._join_lines_by_endpoint_proximity(
            all_lines,
            tolerance=proximity_tolerance,
        )
        for index, line in enumerate(merged_lines, start=1):
            rows.append(
                {
                    "LineId": index,
                    "Source": "raw_SEV_USTU",
                    "geometry": line,
                }
            )

        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)

    @staticmethod
    def _join_gdf_line_features_by_proximity(gdf, tolerance=1.0):
        if gdf is None or gdf.empty:
            return gdf

        joined_lines = DTMChannelModifier._join_lines_by_endpoint_proximity(
            DTMChannelModifier._line_strings(gdf),
            tolerance=tolerance,
        )
        rows = [
            {
                "LineId": index,
                "Source": "junction_clip",
                "geometry": line,
            }
            for index, line in enumerate(joined_lines, start=1)
        ]
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)

    @staticmethod
    def _join_lines_by_endpoint_proximity(lines, tolerance=1.0):
        if not lines:
            return []

        snapped_lines = DTMChannelModifier._snap_line_endpoints(lines, tolerance=tolerance)
        try:
            merged = linemerge(unary_union(snapped_lines))
            merged_lines = DTMChannelModifier._line_strings(merged)
            return merged_lines if merged_lines else snapped_lines
        except Exception:
            return snapped_lines

    @staticmethod
    def _snap_line_endpoints(lines, tolerance=1.0):
        endpoints = []
        for line_index, line in enumerate(lines):
            coords = list(line.coords)
            if not coords:
                continue
            endpoints.append((line_index, 0, Point(coords[0][:2])))
            endpoints.append((line_index, -1, Point(coords[-1][:2])))

        replacements = {}
        visited = set()
        for index, endpoint in enumerate(endpoints):
            if index in visited:
                continue
            cluster = [endpoint]
            visited.add(index)
            for other_index in range(index + 1, len(endpoints)):
                if other_index in visited:
                    continue
                if endpoint[2].distance(endpoints[other_index][2]) <= float(tolerance):
                    cluster.append(endpoints[other_index])
                    visited.add(other_index)

            if len(cluster) == 1:
                continue
            mean_x = sum(item[2].x for item in cluster) / len(cluster)
            mean_y = sum(item[2].y for item in cluster) / len(cluster)
            for line_index, endpoint_index, _ in cluster:
                replacements[(line_index, endpoint_index)] = Point(mean_x, mean_y)

        snapped = []
        for line_index, line in enumerate(lines):
            coords = list(line.coords)
            if not coords:
                continue
            if (line_index, 0) in replacements:
                coords[0] = DTMChannelModifier._coord_like_point(replacements[(line_index, 0)], coords[0])
            if (line_index, -1) in replacements:
                coords[-1] = DTMChannelModifier._coord_like_point(replacements[(line_index, -1)], coords[-1])
            snapped.append(LineString(coords))
        return snapped

    @staticmethod
    def _junction_bank_lines_between_cross_sections(tributary, main, junction, junction_point):
        crs = main["banks_gdf"].crs or tributary["banks_gdf"].crs
        rows = []

        main_bounds = DTMChannelModifier._main_junction_cross_section_bounds(
            channel=main,
            junction_point=junction_point,
        )
        tributary_bounds = DTMChannelModifier._tributary_junction_cross_section_bounds(
            channel=tributary,
            endpoint_name=junction["tributary_endpoint"],
        )

        for role, channel, bounds in (
            ("main", main, main_bounds),
            ("tributary", tributary, tributary_bounds),
        ):
            if bounds is None:
                continue
            measure_range = bounds["measure_range"]
            cut_sections = bounds["cut_sections"]
            bank_gdf = channel.get("processing_banks_gdf")
            if bank_gdf is None or bank_gdf.empty:
                bank_gdf = channel.get("banks_gdf")
            if bank_gdf is None or bank_gdf.empty:
                bank_gdf = gpd.read_file(channel["bank_shp_path"])
            for bank_index, bank_line in enumerate(DTMChannelModifier._line_strings(bank_gdf), start=1):
                selected_lines = DTMChannelModifier._line_parts_by_cross_section_measure_range(
                    bank_line=bank_line,
                    centerline=channel["centerline"],
                    measure_range=measure_range,
                    cut_sections=cut_sections,
                )
                for part_index, line in enumerate(selected_lines, start=1):
                    coords = list(line.coords)
                    start_cut = DTMChannelModifier._endpoint_cross_section_cut_label(
                        Point(coords[0][:2]),
                        role=role,
                        cut_sections=cut_sections,
                    )
                    end_cut = DTMChannelModifier._endpoint_cross_section_cut_label(
                        Point(coords[-1][:2]),
                        role=role,
                        cut_sections=cut_sections,
                    )
                    rows.append(
                        {
                            "Channel": channel["name"][:80],
                            "Role": role,
                            "BankId": bank_index,
                            "PartId": part_index,
                            "FromM": float(measure_range[0]),
                            "ToM": float(measure_range[1]),
                            "StartCut": start_cut,
                            "EndCut": end_cut,
                            "geometry": line,
                        }
                    )

        columns = [
            "Channel",
            "Role",
            "BankId",
            "PartId",
            "FromM",
            "ToM",
            "StartCut",
            "EndCut",
            "geometry",
        ]
        return gpd.GeoDataFrame(rows, columns=columns, geometry="geometry", crs=crs)

    @staticmethod
    def _main_junction_cross_section_bounds(channel, junction_point):
        bank_lines = DTMChannelModifier._line_strings(channel["banks_gdf"])
        sections = DTMChannelModifier._cross_sections_in_file_order(
            cross_section_csv=channel["cross_section_csv"],
            centerline=channel["centerline"],
            bank_lines=bank_lines,
        )
        sections.sort(key=lambda section: section["centerline_measure"])
        if len(sections) < 2:
            return None

        junction_measure = channel["centerline"].project(junction_point)
        lower = [
            section for section in sections
            if section["centerline_measure"] < junction_measure
        ]
        upper = [
            section for section in sections
            if section["centerline_measure"] > junction_measure
        ]

        if lower and upper:
            start_section = max(lower, key=lambda section: section["centerline_measure"])
            end_section = min(upper, key=lambda section: section["centerline_measure"])
        else:
            nearest = sorted(
                sections,
                key=lambda section: abs(section["centerline_measure"] - junction_measure),
            )[:2]
            if len(nearest) < 2:
                return None
            nearest.sort(key=lambda section: section["centerline_measure"])
            start_section, end_section = nearest

        measure_range = (
            float(start_section["centerline_measure"]),
            float(end_section["centerline_measure"]),
        )
        return {
            "measure_range": measure_range,
            "cut_sections": [start_section, end_section],
        }

    @staticmethod
    def _main_junction_cross_section_range(channel, junction_point):
        measures = DTMChannelModifier._cross_section_centerline_measures(
            cross_section_csv=channel["cross_section_csv"],
            centerline=channel["centerline"],
            bank_lines=DTMChannelModifier._line_strings(channel["banks_gdf"]),
        )
        if len(measures) < 2:
            return None
        junction_measure = channel["centerline"].project(junction_point)
        lower = [measure for measure in measures if measure < junction_measure]
        upper = [measure for measure in measures if measure > junction_measure]

        if lower and upper:
            return (max(lower), min(upper))
        nearest = sorted(measures, key=lambda measure: abs(measure - junction_measure))[:2]
        if len(nearest) < 2:
            return None
        return (min(nearest), max(nearest))

    @staticmethod
    def _tributary_junction_cross_section_bounds(channel, endpoint_name):
        bank_lines = DTMChannelModifier._line_strings(channel["banks_gdf"])
        sections = DTMChannelModifier._cross_sections_in_file_order(
            cross_section_csv=channel["cross_section_csv"],
            centerline=channel["centerline"],
            bank_lines=bank_lines,
        )
        sections.sort(key=lambda section: section["centerline_measure"])
        if not sections:
            return None

        endpoint_measure = 0.0 if endpoint_name == "start" else channel["centerline"].length
        tolerance = max(float(channel["centerline"].length) * 1e-9, 1e-6)
        if endpoint_name == "start":
            candidates = [
                section for section in sections
                if section["centerline_measure"] > endpoint_measure + tolerance
            ]
            section = min(
                candidates,
                key=lambda item: item["centerline_measure"],
                default=None,
            )
        else:
            candidates = [
                section for section in sections
                if section["centerline_measure"] < endpoint_measure - tolerance
            ]
            section = max(
                candidates,
                key=lambda item: item["centerline_measure"],
                default=None,
            )
        if section is None:
            section = min(
                sections,
                key=lambda item: abs(item["centerline_measure"] - endpoint_measure),
            )

        section_measure = float(section["centerline_measure"])
        return {
            "measure_range": (
                min(float(endpoint_measure), section_measure),
                max(float(endpoint_measure), section_measure),
            ),
            "cut_sections": [section],
        }

    @staticmethod
    def _tributary_junction_cross_section_range(channel, endpoint_name):
        measures = DTMChannelModifier._cross_section_centerline_measures(
            cross_section_csv=channel["cross_section_csv"],
            centerline=channel["centerline"],
            bank_lines=DTMChannelModifier._line_strings(channel["banks_gdf"]),
        )
        if not measures:
            return None

        endpoint_measure = 0.0 if endpoint_name == "start" else channel["centerline"].length
        tolerance = max(float(channel["centerline"].length) * 1e-9, 1e-6)
        if endpoint_name == "start":
            candidates = [measure for measure in measures if measure > endpoint_measure + tolerance]
            section_measure = (
                min(candidates)
                if candidates
                else min(measures, key=lambda measure: abs(measure - endpoint_measure))
            )
        else:
            candidates = [measure for measure in measures if measure < endpoint_measure - tolerance]
            section_measure = (
                max(candidates)
                if candidates
                else min(measures, key=lambda measure: abs(measure - endpoint_measure))
            )

        return (min(endpoint_measure, section_measure), max(endpoint_measure, section_measure))

    @staticmethod
    def _cross_section_centerline_measures(cross_section_csv, centerline, bank_lines=None):
        df = DTMChannelModifier._read_csv_auto(
            cross_section_csv,
            required_columns=("X", "Y", "Z"),
        )
        group_cols = [col for col in ["River", "Reach", "Station"] if col in df.columns]
        if not group_cols:
            group_cols = ["Station"] if "Station" in df.columns else []

        grouped = df.groupby(group_cols) if group_cols else [(None, df)]
        measures = []
        for _, group in grouped:
            if len(group) < 2 or not {"X", "Y"}.issubset(group.columns):
                continue
            coords = group[["X", "Y"]].to_numpy(dtype=float)
            line = LineString(coords)
            center_point = DTMChannelModifier._cross_section_center_point_from_centerline(
                line,
                centerline,
                bank_lines=bank_lines,
            )
            measures.append(float(centerline.project(center_point)))

        return sorted(set(round(measure, 6) for measure in measures))

    @staticmethod
    def _line_parts_by_cross_section_measure_range(
        bank_line,
        centerline,
        measure_range,
        cut_sections=None,
    ):
        start_measure, end_measure = sorted([float(measure_range[0]), float(measure_range[1])])
        if abs(end_measure - start_measure) <= 1e-6:
            return []

        if cut_sections:
            pieces = DTMChannelModifier._split_line_at_cross_sections(
                line=bank_line,
                cut_sections=cut_sections,
            )
        else:
            coords = list(bank_line.coords)
            pieces = [
                LineString([coords[index], coords[index + 1]])
                for index in range(len(coords) - 1)
            ]
        selected_segments = []
        measure_tolerance = max((end_measure - start_measure) * 1e-9, 1e-6)
        for segment in pieces:
            if segment is None or segment.is_empty or segment.length <= 1e-9:
                continue
            midpoint = segment.interpolate(0.5, normalized=True)
            measure = centerline.project(midpoint)
            if start_measure - measure_tolerance <= measure <= end_measure + measure_tolerance:
                selected_segments.append(segment)

        if not selected_segments:
            return []

        try:
            merged = linemerge(unary_union(selected_segments))
            return DTMChannelModifier._line_strings(merged)
        except Exception:
            return selected_segments

    @staticmethod
    def _line_parts_by_centerline_measure_range(bank_line, centerline, measure_range):
        return DTMChannelModifier._line_parts_by_cross_section_measure_range(
            bank_line=bank_line,
            centerline=centerline,
            measure_range=measure_range,
            cut_sections=None,
        )

    @staticmethod
    def _endpoint_cross_section_cut_label(point, role, cut_sections, tolerance=1.0):
        if point is None or point.is_empty:
            return ""
        candidates = []
        for section in cut_sections or []:
            section_line = section.get("line")
            if section_line is None or section_line.is_empty:
                continue
            distance = float(point.distance(section_line))
            if distance <= float(tolerance):
                candidates.append((distance, section))
        if not candidates:
            return ""
        _, section = min(candidates, key=lambda item: item[0])
        station = section.get("station", "")
        return f"{role}:{station}"[:80]

    @staticmethod
    def _split_line_at_cross_sections(line, cut_sections, snap_tolerance=1.0):
        if line is None or line.is_empty or line.length <= 0:
            return []

        cut_distances = []
        for section in cut_sections or []:
            cut_line = DTMChannelModifier._extended_cross_section_line(
                section["line"],
                line,
            )
            if cut_line is None or cut_line.is_empty:
                continue

            intersection = line.intersection(cut_line)
            points = DTMChannelModifier._points_from_geometry(intersection)
            if not points and line.distance(cut_line) <= float(snap_tolerance):
                try:
                    point_on_line, _ = nearest_points(line, cut_line)
                    points = [point_on_line]
                except Exception:
                    points = []

            for point in points:
                distance = float(line.project(point))
                if 1e-6 < distance < float(line.length) - 1e-6:
                    cut_distances.append(distance)

        if not cut_distances:
            return [line]

        unique_distances = []
        for distance in sorted(cut_distances):
            if not unique_distances or abs(distance - unique_distances[-1]) > 1e-6:
                unique_distances.append(distance)

        distances = [0.0] + unique_distances + [float(line.length)]
        pieces = []
        for start_distance, end_distance in zip(distances[:-1], distances[1:]):
            piece = DTMChannelModifier._line_substring_by_distance(
                line,
                start_distance,
                end_distance,
            )
            if piece is not None and not piece.is_empty and piece.length > 1e-9:
                pieces.append(piece)
        return pieces

    @staticmethod
    def _line_substring_by_distance(line, start_distance, end_distance):
        length = float(line.length)
        if length <= 0:
            return None

        start_distance = float(np.clip(start_distance, 0.0, length))
        end_distance = float(np.clip(end_distance, 0.0, length))
        if end_distance < start_distance:
            start_distance, end_distance = end_distance, start_distance
        if end_distance - start_distance <= 1e-9:
            return None

        coords = [(float(coord[0]), float(coord[1])) for coord in line.coords]
        if len(coords) < 2:
            return None

        def interpolate_segment(p0, p1, fraction):
            return (
                float(p0[0] + (p1[0] - p0[0]) * fraction),
                float(p0[1] + (p1[1] - p0[1]) * fraction),
            )

        output_coords = []
        traversed = 0.0
        tolerance = 1e-9
        for index in range(len(coords) - 1):
            p0 = coords[index]
            p1 = coords[index + 1]
            segment_length = float(Point(p0).distance(Point(p1)))
            if segment_length <= 0:
                continue
            next_traversed = traversed + segment_length
            if next_traversed < start_distance - tolerance:
                traversed = next_traversed
                continue
            if traversed > end_distance + tolerance:
                break

            local_start = max(start_distance, traversed)
            local_end = min(end_distance, next_traversed)
            if local_end < local_start - tolerance:
                traversed = next_traversed
                continue

            start_fraction = (local_start - traversed) / segment_length
            end_fraction = (local_end - traversed) / segment_length
            start_point = interpolate_segment(p0, p1, start_fraction)
            end_point = interpolate_segment(p0, p1, end_fraction)

            if not output_coords or Point(output_coords[-1]).distance(Point(start_point)) > 1e-8:
                output_coords.append(start_point)
            if Point(output_coords[-1]).distance(Point(end_point)) > 1e-8:
                output_coords.append(end_point)
            traversed = next_traversed

        if len(output_coords) < 2:
            return None
        return LineString(output_coords)
