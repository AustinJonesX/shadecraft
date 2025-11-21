import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import box
import cv2
from tqdm import tqdm

def reproject_footprints(geojson_path, dst_crs):
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.to_crs(dst_crs)
    return gdf

def clip_footprints_to_raster(gdf, raster_bounds):
    minx, miny, maxx, maxy = raster_bounds
    raster_bbox = box(minx, miny, maxx, maxy)
    return gdf[gdf.intersects(raster_bbox)]

def rasterize_buildings(gdf, out_shape, transform):
    shapes = ((geom, 1) for geom in gdf.geometry)
    mask = features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    return mask

def compute_edges(rgb_image):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges.astype(np.uint8)

def extract_shadows(rgbnir):
    rgb = rgbnir[..., :3].astype(np.float32)
    nir = rgbnir[..., 3].astype(np.float32)

    # Simple shadow detection: low RGB brightness + low NIR
    brightness = rgb.mean(axis=-1)
    shadow_mask = (brightness < 70) & (nir < 60)

    return shadow_mask.astype(np.uint8)

def preprocess(
    raster_path,
    footprints_path,
    save_dir
):
    os.makedirs(save_dir, exist_ok=True)

    print("Loading NAIP raster...")
    with rasterio.open(raster_path) as src:
        img = src.read().transpose(1, 2, 0)
        transform = src.transform
        bounds = src.bounds
        crs = src.crs
        h, w = img.shape[:2]

    print("Reprojecting footprints...")
    gdf = reproject_footprints(footprints_path, crs)

    print("Clipping footprints to raster...")
    gdf = clip_footprints_to_raster(gdf, bounds)

    print("Rasterizing building masks...")
    mask = rasterize_buildings(gdf, (h, w), transform)

    print("Computing edges...")
    edges = compute_edges(img[..., :3])

    print("Extracting shadow mask...")
    shadows = extract_shadows(img)

    print("Saving outputs...")
    np.save(os.path.join(save_dir, "image.npy"), img)
    np.save(os.path.join(save_dir, "mask.npy"), mask)
    np.save(os.path.join(save_dir, "edges.npy"), edges)
    np.save(os.path.join(save_dir, "shadows.npy"), shadows)

    print("Done.")


if __name__ == "__main__":
    preprocess(
        raster_path="/content/shadecraft/data/raw/tempe_2013_naip.tif",
        footprints_path="/content/shadecraft/data/osm/tempe_buildings.geojson",
        save_dir="/content/shadecraft/data/processed/2013"
    )
