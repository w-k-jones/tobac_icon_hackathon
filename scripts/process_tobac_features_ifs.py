#!/home/b/b382728/miniconda3/envs/tobac/bin/python
import pathlib
from datetime import datetime, timedelta
import argparse
import warnings

import numpy as np
import xarray as xr
from iris.analysis.cartography import area_weights

import intake
import healpy
import tobac

parser = argparse.ArgumentParser(description="""ICON tracking using tobac""")
parser.add_argument("date", help="Date on which to start process", type=str)
parser.add_argument("-hours", help="Number of hours to process", type=float, default=3)
parser.add_argument("-offset", help="Number of days to offset from start date", default=0, type=int)
parser.add_argument(
    "-s", help="path to save output data", default="/scratch/b/b382728/ifs_tobac_features/", type=str
)

args = parser.parse_args()

start_date = datetime.strptime(args.date, "%Y-%m-%d-%H:%M:%S")
start_date = start_date + timedelta(days=args.offset)
end_date = start_date + timedelta(hours=args.hours)
save_path = pathlib.Path(args.s) / f"{start_date.strftime('%Y/%m/%d')}" 

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
    dataset = cat.IFS['IFS_9-FESOM_5-production'][f'2D_hourly_healpix512_{start_date.strftime("%Y")}'](chunks="auto", consolidated=False).to_dask()

    lon = xr.DataArray(
        np.arange(0.05, 360, 0.1), dims=("lon",), name="lon", attrs=dict(units="degrees", standard_name="longitude")
    )
    lat = xr.DataArray(
        np.arange(59.95, -60, -0.1), dims=("lat",), name="lat", attrs=dict(units="degrees", standard_name="latitude")
    )

    pix = xr.DataArray(
        healpy.ang2pix(512, *np.meshgrid(lon, lat), nest=True, lonlat=True),
        coords=(lat, lon),
    )

    olr = dataset.ttr.drop_vars(["lat", "lon"])\
        .sel(time=slice(start_date, end_date - timedelta(minutes=1)))\
        .compute().isel(value=pix) / -3.6e3
    # The previous step results in an array that is not in contiguous memory order, which slows down all further calculations
    # By ravelling then reshaping, we get it back in contiguous order
    # Note that if we called "compute" after regridding, it would be in the correct order, but doing so is much slower (~60s)
    olr.data = olr.values.ravel().reshape(olr.shape) 
    # IFS has duplicate time coords at restarts (with missing data), so we find and drop these using np.unique
    olr = olr.isel(time=np.unique(olr.time, return_index=True)[1])
    olr.attrs = dataset.ttr.attrs
    olr.attrs["units"] = "W m**-2"

    bt = get_tb(olr)

    dt = 3600  # in seconds
    dxy = 11100  # in meter (for Latitude)

    parameters_features = dict(
        dxy=dxy,
        threshold=[241, 233, 225],
        n_min_threshold=10,
        min_distance=10*dxy,
        target="minimum",
        position_threshold="center",
        PBC_flag="hdim_2",
        statistic={"feature_min_BT": np.nanmin},
    )

    parameters_segments = dict(
        threshold=241, 
        target="minimum", 
        PBC_flag="hdim_2", 
        seed_3D_flag="box", 
        seed_3D_size=11, 
        statistic={"mean_BT": np.nanmean},
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

    features["mean_BT"] = features["feature_min_BT"].to_numpy().astype(float)

    features["time"] = xr.CFTimeIndex(features["time"].to_numpy()).to_datetimeindex()
    
    print(datetime.now(), f"Calculating feature properties", flush=True)
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, olr, statistic=dict(mean_OLR=np.nanmean, min_OLR=np.nanmin), default=np.nan
    )
    print(datetime.now(), f"finished OLR", flush=True)
    del olr

    precip = dataset.tp.drop_vars(["lat", "lon"])\
        .sel(time=slice(start_date, end_date - timedelta(minutes=1)))\
        .compute().isel(value=pix) * 1e3
    precip.data = precip.values.ravel().reshape(precip.shape) 
    # IFS has duplicate time coords at restarts (with missing data), so we find and drop these using np.unique
    precip = precip.isel(time=np.unique(precip.time, return_index=True)[1])
    precip.attrs = dataset.tp.attrs
    precip.attrs["units"] = "mm"
    print(datetime.now(), f"loaded precip", flush=True)

    # Get area array and calculate area of each segment
    segment_slice = segments[0]
    segment_slice.coord("latitude").guess_bounds()
    segment_slice.coord("longitude").guess_bounds()
    area = area_weights(segment_slice, normalize=False)
    area = xr.DataArray(area, coords=dict(lat=precip.lat, lon=precip.lon), dims=["lat", "lon"])

    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, area, statistic=dict(area=np.nansum), default=np.nan
    )
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, precip, statistic=dict(max_precip=np.nanmax), default=np.nan
    )
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, precip * area.values, statistic=dict(total_precip=np.nansum), default=np.nan
    )

    del precip
    
    out_ds = features.set_index(features.feature).to_xarray()

    out_ds = out_ds.rename_vars(
        {
            "time": "time_feature",
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

    out_ds.to_netcdf(
        save_path 
        / f"tobac_{start_date.strftime('%Y%m%d-%H%M%S')}_{end_date.strftime('%Y%m%d-%H%M%S')}_IFS_feature_mask_file.nc"
    )

    out_ds.close()


if __name__ == "__main__":
    start_time = datetime.now()
    print(start_time, "Commencing DCC feature detection and segmentation", flush=True)
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    print("Start date:", start_date.isoformat(), flush=True)
    print("End date:", end_date.isoformat(), flush=True)
    print("Output save path:", save_path, flush=True)

    main()

    print(
        datetime.now(),
        "Finished successfully, time elapsed:",
        datetime.now() - start_time,
        flush=True,
    )


