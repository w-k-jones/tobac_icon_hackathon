import pathlib
import sys
from datetime import datetime, timedelta
import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from iris.analysis.cartography import area_weights
from scipy.ndimage import labeled_comprehension

import intake
import healpy
import tobac

cmd_folder = pathlib.Path.cwd().parent.absolute()
if cmd_folder not in sys.path:
    sys.path.insert(0, str(cmd_folder))

# from utils import data_formats, get_tb, calc_area_and_precip, is_track_mcs

parser = argparse.ArgumentParser(description="""ICON tracking using tobac""")
parser.add_argument("date", help="Date on which to start process", type=str)
parser.add_argument("hours", help="Number of hours to process", type=float)
parser.add_argument("-offset", help="Number of days to offset from start date", default=0, type=int)
parser.add_argument(
    "-s", help="path to save output data", default="../data_out", type=str
)

args = parser.parse_args()

start_date = datetime.strptime(args.date, "%Y-%m-%d-%H:%M:%S")
start_date = start_date + timedelta(days=args.offset)
end_date = start_date + timedelta(hours=args.hours)
save_path = pathlib.Path(args.s)

def get_tb(olr):
    """
    This function converts outgoing longwave radiation to brightness temperatures.

    Args:
        olr(xr.DataArray or numpy array): 2D field of model output with OLR

    Returns:
        tb(xr.DataArray or numpy array): 2D field with estimated brightness temperatures
    """
    # constants
    aa = 1.228
    bb = -1.106e-3  # K−1
    # Planck constant
    sigma = 5.670374419e-8  # W⋅m−2⋅K−4

    # flux equivalent brightness temperature
    Tf = (abs(olr) / sigma) ** (1.0 / 4)
    tb = (((aa**2 + 4 * bb * Tf) ** (1.0 / 2)) - aa) / (2 * bb)
    return tb


def main() -> None:
    print("Loading data", flush=True)
    cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
    dataset = cat.ICON.ngc4008(time="PT15M", zoom=9).to_dask()
    
    lon = xr.DataArray(np.arange(0.05, 360, 0.1), dims=("lon",), name="lon", attrs=dict(units="degrees", standard_name="longitude"))
    lat = xr.DataArray(np.arange(59.95, -60, -0.1), dims=("lat",), name="lat", attrs=dict(units="degrees", standard_name="latitude"))
    
    pix = xr.DataArray(
        healpy.ang2pix(dataset.crs.healpix_nside, *np.meshgrid(lon, lat), nest=True, lonlat=True),
        coords=(lat, lon),
    )

    bt = get_tb(dataset.rlut.sel(time=slice(start_date, end_date-timedelta(minutes=1))).isel(cell=pix))

    dt = 900  # in seconds
    dxy = 11100  # in meter (for Latitude)

    parameters_features = dict(
        dxy=dxy,
        threshold=[241, 233, 225],
        n_min_threshold=10,
        min_distance=2.5*dxy,
        target="minimum",
        position_threshold="center",
        PBC_flag="hdim_2",
        statistic={"feature_min_BT": np.nanmin},
    )

    parameters_segments = dict(
        threshold=241, target="minimum", PBC_flag="hdim_2", seed_3D_flag="box", seed_3D_size=11,
    )

    print(datetime.now(), f"Commencing feature detection", flush=True)
    features = tobac.feature_detection_multithreshold(
        bt.to_iris(),
        **parameters_features,
    )

    # Convert feature_min_BT to float dtype as the default of 'None' means that it will be an object array
    features["feature_min_BT"] = features["feature_min_BT"].to_numpy().astype(float)

    print(datetime.now(), f"Commencing segmentation", flush=True)
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Warning: converting a masked element to nan.*",
    )
    warnings.filterwarnings(
        "ignore",
        # category=FutureWarning,
        message="FutureWarning: Calling float on a sing*",
    )
    segments, features = tobac.segmentation.segmentation(
        features, bt.to_iris(), dxy, **parameters_segments,
    )

    del bt
    precip = dataset.pr.sel(time=slice(start_date, end_date-timedelta(minutes=1))).isel(cell=pix)
    
    # Get area array and calculate area of each segment
    segment_slice = segments[0]
    segment_slice.coord("latitude").guess_bounds()
    segment_slice.coord("longitude").guess_bounds()
    area = area_weights(segment_slice, normalize=False)

    features["area"] = np.nan
    features["max_precip"] = np.nan
    features["total_precip"] = np.nan

    features_t = features["time"].to_numpy()
    for time, mask in zip(precip["time"].data, segments.slices_over("time")):
        wh = features_t == time
        if np.any(wh):
            feature_areas = labeled_comprehension(
                area, mask.data, features[wh]["feature"], np.sum, area.dtype, np.nan
            )
            features.loc[wh, "area"] = feature_areas

            step_precip = precip.sel({"time": time}).values
            max_precip = labeled_comprehension(
                step_precip,
                mask.data,
                features[wh]["feature"],
                np.max,
                area.dtype,
                np.nan,
            )

            features.loc[wh, "max_precip"] = max_precip

            feature_precip = labeled_comprehension(
                area * step_precip,
                mask.data,
                features[wh]["feature"],
                np.sum,
                area.dtype,
                np.nan,
            )

            features.loc[wh, "total_precip"] = feature_precip
    
    out_ds = features.set_index(features.feature).to_xarray()

    out_ds = out_ds.rename_vars(
        {
            # "latitude": "feature_latitude",
            # "longitude": "feature_longitude",
            "time": "time_feature",
            "cell": "feature_cell_id",
            "track": "feature_track_id",
            "hdim_1": "y",
            "hdim_2": "x",
            "num": "detection_pixel_count",
            "feature_min_BT": "min_BT",
            "ncells": "segmentation_pixel_count",
        }
    )

    all_feature_labels = xr.DataArray.from_iris(segments)
    all_feature_labels =  all_feature_labels * np.isin(
        all_feature_labels, out_ds.feature.values
    )
    all_feature_labels.name = "all_feature_labels"

    out_ds = out_ds.assign_coords(all_feature_labels.coords)

    out_ds = xr.merge(
        [
            out_ds,
            all_feature_labels,
        ]
    )

    # Add compression encoding
    comp = dict(zlib=True, complevel=5, shuffle=True)
    for var in out_ds.data_vars:
        var_type = out_ds[var].dtype
        if np.issubdtype(var_type, np.integer) or np.issubdtype(var_type, np.floating):
            out_ds[var].encoding.update(comp)

    out_ds.to_netcdf(save_path / f"tobac_{start_date.strftime('%Y%m%d-%H%M%S')}_{end_date.strftime('%Y%m%d-%H%M%S')}_ICON_feature_mask_file.nc")

    out_ds.close()


if __name__ == "__main__":
    start_time = datetime.now()
    print(start_time, "Commencing DCC feature detection and segmentation", flush=True)
    if not save_path.exists():
        save_path.mkdir()

    print("Start date:", start_date.isoformat(), flush=True)
    print("Start date:", end_date.isoformat(), flush=True)
    print("Output save path:", save_path, flush=True)

    main()

    print(
        datetime.now(),
        "Finished successfully, time elapsed:",
        datetime.now() - start_time,
        flush=True,
    )


