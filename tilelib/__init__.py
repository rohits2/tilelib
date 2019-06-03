from itertools import product
from pathlib import Path
from typing import Set, List, Dict

import json
import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from PIL import Image
from PIL.ImageDraw import Draw
from matplotlib import pyplot as plt
from mercantile import Tile, bounds, LngLatBbox, parent, simplify, xy, xy_bounds, bounding_tile
from mercantile import children
from loguru import logger

from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.ops import transform

EPS = 1e-9


def get_strided_tiles(root: Tile, zoom: int = 15):
    child_bounds = children(root, zoom=zoom)
    min_x = min((child.x for child in child_bounds))
    min_y = min((child.y for child in child_bounds))
    max_x = max((child.x for child in child_bounds))
    max_y = max((child.y for child in child_bounds))
    n = max_x - min_x + 1
    strided_tiles = []
    for x, y in product(np.arange(min_x, max_x + 0.1, 0.5), np.arange(min_y, max_y + 0.1, 0.5)):
        strided_tiles += [Tile(float(x), float(y), zoom)]
    return strided_tiles


def get_stack_path(tile: Tile):
    x, y, z = tile
    return Path(f"jpg/{int(z)}/{int(x)}/{int(y)}.jpg")


def get_res(tile: Tile, source_zoom: int = 18):
    x, y, z = tile
    return (1 << (source_zoom - z)) * 256


class TileEngine:
    def __init__(self, use_max_res=True):
        self.stacks: Dict[Tile, Path] = {}
        self.max_res = use_max_res

    def add_stack(self, stack_path: Path, stack_tile: Tile):
        if type(stack_path) == str:
            stack_path = Path(stack_path)
        assert stack_path.exists()
        self.stacks[stack_tile] = stack_path

    def add_directory(self, stack_dir: Path):
        if type(stack_dir) == str:
            stack_dir = Path(stack_dir)
        assert stack_dir.exists()
        assert stack_dir.is_dir()
        count = 0
        for subdir in stack_dir.iterdir():
            if subdir.is_dir():
                _, z, x, y, *_ = subdir.name.split("_")
                x, y, z = map(int, (x, y, z))
                self[x, y, z] = subdir 
                count+=1
        logger.info(f"Found {count} tilestacks")

    def __getitem__(self, tile: Tile):
        return self.get_image(tile)

    def __setitem__(self, stack_tile: Tile, stack_path: Path):
        self.add_stack(stack_path, stack_tile)

    def get_image(self, *tile: Tile, source_zoom: int = 18, black_fail: bool = False):
        if len(tile) == 1:
            tile = tile[0]
        x, y, z = tile
        if not self.max_res:
            source_zoom = z
        parents = [
            parent(Tile(int(fn(x)), int(fn(y)), z), zoom=i) for i, fn in product(range(z + 1), [np.ceil, np.floor])
        ]
        for supertile in parents:
            if supertile in self.stacks:
                return self.__make_image(self.stacks[supertile], tile, source_zoom)
        raise FileNotFoundError(f"Could not find a parent tilestack for requested subtile {tile}!")

    def __make_image(self, stack_path: Path, tile: Tile, source_zoom: int = 18, black_fail: bool = False):
        x, y, z = tile
        if z != int(z):
            raise ValueError("Cannot make fractionally-zoomed images!")

        # If this is a primitive tile request, simply return the raw image
        if x == int(x) and y == int(y) and z >= source_zoom:
            path = stack_path / get_stack_path(tile)
            with path.open('rb') as f:
                img = plt.imread(f) / 255
                if len(img.shape) < 3 or img.shape[2] != 3:
                    print(f"WARNING: {tile} image not RGB! Will attempt to upconvert grayscale to RGB.")
                    h, w, *_ = img.shape
                    tmp = np.zeros((h, w, 3))
                    tmp[:, :, 0] = img
                    tmp[:, :, 1] = img
                    tmp[:, :, 2] = img
                    img = tmp
                return img
        # If this is a super-resolution tile request, recurse into the mosaic and return the whole.
        elif x == int(x) and y == int(y) and z < source_zoom:
            res = get_res(tile, source_zoom=source_zoom)
            mosaic = np.zeros((res, res, 3))
            source_tiles = children(tile)
            try:
                mosaic[:res // 2, :res // 2, :] = self.get_image(source_tiles[0], source_zoom=source_zoom)
                mosaic[:res // 2, res // 2:, :] = self.get_image(source_tiles[1], source_zoom=source_zoom)
                mosaic[res // 2:, :res // 2, :] = self.get_image(source_tiles[3], source_zoom=source_zoom)
                mosaic[res // 2:, res // 2:, :] = self.get_image(source_tiles[2], source_zoom=source_zoom)
            except FileNotFoundError as e:
                raise FileNotFoundError(str(e) + f"\n This tile was requested as part of a mosaic for {tile}.")
            return mosaic

        # If it is a fractile request, recurse into the mosaic and return the requested segment.
        source_tiles = [
            Tile(np.trunc(x), np.trunc(y), z),
            Tile(np.trunc(x), np.ceil(y), z),
            Tile(np.ceil(x), np.trunc(y), z),
            Tile(np.ceil(x), np.ceil(y), z),
        ]
        res = get_res(tile, source_zoom=source_zoom)
        mosaic = np.zeros((res * 2, res * 2, 3))

        try:
            mosaic[:res, :res, :] = self.get_image(source_tiles[0], source_zoom=source_zoom)
            mosaic[:res, res:, :] = self.get_image(source_tiles[2], source_zoom=source_zoom)
            mosaic[res:, :res, :] = self.get_image(source_tiles[1], source_zoom=source_zoom)
            mosaic[res:, res:, :] = self.get_image(source_tiles[3], source_zoom=source_zoom)
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e) + f"\nThis tile was requested as part of a mosaic for {tile}.")

        x_cut = int((x - int(x)) * res)
        y_cut = int((y - int(y)) * res)

        return mosaic[y_cut:y_cut + res, x_cut:x_cut + res, :].copy()


LAT_LNG_BUFFER = 0.01

WIDTH_FACTOR = 90 / 1e7  # Bad history lesson here
raster_table = {
    "motorway": 2,
    "trunk": 10,
    "primary": 8,
    "residential": 3.,
    "secondary": 5.,
    "tertiary": 3.,
    "unclassified": 2.,
    "service": 2,
    "pedestrian": 1,
    "track": 2,
    "escape": 3,
    "footway": 1,
    "motorway_link": 8,
    "trunk_link": 4,
    "primary_link": 4,
    "secondary_link": 3,
    "tertiary_link": 3,
}


class OSMRoadEngine:
    def __init__(self, use_overpass=True):
        self.cached_tiles: Set[Tile] = set()
        self.cache: gpd.GeoDataFrame = gpd.GeoDataFrame([], geometry=[], crs={'init': 'epsg:4326'})
        self.use_overpass = use_overpass

    def __get_from_cache(self, tile: Tile) -> gpd.GeoDataFrame:
        if self.cache is None:
            raise ValueError("Tile cache not found!")
        bbox: LngLatBbox = bounds(tile)
        west, south, east, north = bbox
        west -= LAT_LNG_BUFFER
        east += LAT_LNG_BUFFER
        north += LAT_LNG_BUFFER
        south -= LAT_LNG_BUFFER
        try:
            index = self.cache.sindex
            possible_matches_index = list(index.intersection(bbox))
            return self.cache.iloc[possible_matches_index]
        except:
            print("[WARNING] Spatial index failure! Falling back to direct indexing...")
            return self.cache.cx[west:east, south:north]

    def __get_from_osmnx(self, tile: Tile) -> gpd.GeoDataFrame:
        bbox: LngLatBbox = bounds(tile)
        west, south, east, north = bbox
        west -= LAT_LNG_BUFFER
        east += LAT_LNG_BUFFER
        north += LAT_LNG_BUFFER
        south -= LAT_LNG_BUFFER
        try:
            graph = ox.graph_from_bbox(north, south, east, west, simplify=False, retain_all=True)
            gdfs = ox.graph_to_gdfs(graph)
            for gdf in gdfs:
                self.cache = self.cache.append(gdf, ignore_index=True, sort=False)
        except ox.core.EmptyOverpassResponse:
            pass
        except ValueError as e:
            if "not enough values" in str(e):
                print(f"[WARNING] Could not load tile {tile}! Assuming it is empty...")
            else:
                raise e
        self.cached_tiles |= {tile}
        self.cached_tiles = set(simplify(*self.cached_tiles))
        return self.__get_from_cache(tile)

    def load_directory(self, stack_dir: Path, verbose: bool = False):
        files = self.__recursive_dir_search(stack_dir)
        logger.info(f"Found {len(files)} road files!")
        self.load_files(*files, verbose=verbose)

    def __recursive_dir_search(self, stack_dir):
        files = []
        if type(stack_dir) == str:
            stack_dir = Path(stack_dir)
        assert stack_dir.exists()
        assert stack_dir.is_dir()
        for sub in stack_dir.iterdir():
            if sub.is_dir():
                files += self.__recursive_dir_search(sub)
            elif "osm_roads" in sub.name and sub.name[-4:] == ".shp":
                files += [sub]
        return files

    def load_files(self, *files: List[Path], verbose: bool = False):
        files = [Path(file) for file in files]
        if verbose:
            from tqdm import tqdm
            files = tqdm(files, desc="RoadEngine Load")
        for file in files:
            if not file.exists():
                raise FileNotFoundError(f"Could not load {file.absolute()}!")
            gdf = gpd.read_file(file.absolute())
            self.cache = self.cache.append(gdf, ignore_index=True, sort=False)

        tileset = set()
        geometry = gdf['geometry']
        if verbose:
            geometry = tqdm(geometry, desc="RoadEngine Quadtree")
        for geom in geometry:
            if type(geom) != LineString:
                continue
            root = bounding_tile(*geom.bounds)
            tileset |= {root}
        self.cached_tiles |= set(simplify(*tileset))
        if verbose:
            logger.info("RoadEngine is building an R-Tree...")
        self.cache.sindex
        if verbose:
            logger.info("RoadEngine R-Tree done!")

    def save(self, host_dir="/tmp"):
        Path(host_dir).mkdir(exist_ok=True, parents=True)
        geometry_path = Path(host_dir) / "osm_cache.shp"
        tiles_path = Path(host_dir) / "osm_cache.json"
        self.cache = self.cache[[not isinstance(x, Point) for x in self.cache.geometry]]
        self.cache.to_file(geometry_path.absolute(), driver="ESRI Shapefile")
        with tiles_path.open("w+") as f:
            json.dump(list(self.cached_tiles), f)

    def prefetch(self, tile: Tile, host_dir: str = None, verbose: bool = False):
        x, y, z = tile
        if z != int(z):
            raise ValueError("Fractional zooms not allowed!")
        if x == int(x) and y == int(y):
            x, y = int(x), int(y)
            tile_parents = [parent(tile, zoom=i) for i in range(z + 1)]
            for tile_parent in tile_parents:
                if tile_parent in self.cached_tiles:
                    return
            if z < 14:
                blob = children(tile, zoom=14)
                if verbose:
                    from tqdm import tqdm
                    blob = tqdm(blob, desc="OpenStreetMap Prefetch")
                for child in blob:
                    self.prefetch(child)
            else:
                self.__get_from_osmnx(tile)
            if host_dir is not None:
                self.save(host_dir)
            return
        source_tiles = [
            Tile(np.trunc(x), np.trunc(y), z),
            Tile(np.trunc(x), np.ceil(y), z),
            Tile(np.ceil(x), np.trunc(y), z),
            Tile(np.ceil(x), np.ceil(y), z),
        ]
        for tile in source_tiles:
            self.prefetch(tile)
        return

    def gdf_from_tile(self, tile: Tile) -> gpd.GeoDataFrame:
        x, y, z = tile
        if z != int(z):
            raise ValueError("Fractional zooms not allowed!")
        if x == int(x) and y == int(y):
            x, y, z = int(x), int(y), int(z)
            tile_parents = [parent((x, y, z), zoom=i) for i in range(z + 1)]
            for tile_parent in tile_parents:
                if tile_parent in self.cached_tiles:
                    return self.__get_from_cache(tile)
            if self.use_overpass:
                print(f"[WARNING] Fetching {tile} from OpenStreetMap! This is slower than using regional shapefiles.")
                return self.__get_from_osmnx(tile)
            else:
                return self.__get_from_cache(tile)
        source_tiles = [
            Tile(np.trunc(x), np.trunc(y), z),
            Tile(np.trunc(x), np.ceil(y), z),
            Tile(np.ceil(x), np.trunc(y), z),
            Tile(np.ceil(x), np.ceil(y), z),
        ]
        returned_gdfs = [self.gdf_from_tile(tile) for tile in source_tiles]
        return pd.concat(returned_gdfs, ignore_index=True)

    def get_raster(self, tile: Tile, raster_table: Dict[str, float] = raster_table):
        try:
            gdf = self.gdf_from_tile(tile)
            buf = buffer_geometry(gdf, raster_table)
            return rasterize_geometry(buf, tile)
        except ValueError as e:
            if "Empty data" in str(e) or "not enough" in str(e):
                print(f"[WARNING] No data found for {tile}! Returning empty list...")
                return rasterize_geometry([], tile)
            else:
                raise e

    def __getitem__(self, tile: Tile):
        return self.get_raster(tile)


def rasterize_geometry(geometry: List[Polygon], tile: Tile, source_zoom: int = 18) -> np.ndarray:
    wm_geometry = [transform(xy, shape) for shape in geometry]

    res = get_res(tile, source_zoom=source_zoom)
    wm_bounds = xy_bounds(tile)

    def imgspace_transform(xs, ys):
        xs = np.array(xs)
        ys = np.array(ys)
        xs -= wm_bounds.left
        ys -= wm_bounds.bottom
        xs /= (wm_bounds.right - wm_bounds.left)
        ys /= (wm_bounds.top - wm_bounds.bottom)
        xs *= res
        ys *= res
        ys = res - ys
        return xs, ys

    img_geometry = [transform(imgspace_transform, shape) for shape in wm_geometry]
    img_geometry = [list(poly) if type(poly) == MultiPolygon else poly for poly in img_geometry]

    img = Image.new('L', (res, res))
    draw = Draw(img)
    for polygon in img_geometry:
        if type(polygon) != Polygon:
            print(f"Skipping non-polygon {type(polygon)}!")
            continue
        draw.polygon(list(polygon.exterior.coords), fill=1)
        for interior_hole in polygon.interiors:
            draw.polygon(list(interior_hole.coords), fill=0)

    ar = np.array(img, dtype=np.float32)
    return ar


def buffer_geometry(gdf: gpd.GeoDataFrame, width_table: Dict[str, float] = raster_table,
                    default_width: float = 1.5) -> List[Polygon]:
    roads: List[Polygon] = []
    for row in gdf.itertuples():
        if hasattr(row, 'highway'):
            road_class = row.highway
        else:
            road_class = row.fclass
        if type(road_class) == list:
            road_class = road_class[0]
        width = width_table.get(road_class, default_width) * WIDTH_FACTOR
        roads += [row.geometry.buffer(width)]
    return roads
