import pathlib
from datetime import datetime, timedelta
import argparse
import warnings
import logging

import numpy as np
import pandas as pd
import xarray as xr
import tobac

parser = argparse.ArgumentParser(description="""ICON tracking using tobac""")
parser.add_argument("date", help="Date on which to start process", type=str)
parser.add_argument("-files", default=None)
parser.add_argument(
    "-s", help="path to save output data", default="/scratch/b/b382728/tobac_tracks", type=str
)

args = parser.parse_args()

start_date = datetime.strptime(args.date, "%Y-%m-%d-%H:%M:%S")
end_date = datetime(start_date.year + start_date.month//12, start_date.month % 12 + 1, start_date.day, start_date.hour, start_date.minute, start_date.second)

data_path = pathlib.Path(f"/scratch/b/b382728/tobac_features/{start_date.strftime('%Y')}/{start_date.strftime('%m')}")
save_path = pathlib.Path(args.s) / f"{start_date.strftime('%Y')}"
if not save_path.exists():
    save_path.mkdir(parents=True, exist_ok=True)

files = sorted(list(data_path.glob("*/*_ICON_feature_mask_file.nc")))
if args.files is not None:
    files = files[:int(args.files)]
print("Files found:", len(files), flush=True)

rename_vars = {
    "time_feature":"time" ,
    "y":"hdim_1",
    "x":"hdim_2",
}

combined_features = tobac.utils.combine_feature_dataframes(
    [
        xr.open_dataset(f).drop_vars("all_feature_labels").drop_dims(["time","lat","lon"]).rename(rename_vars).to_dataframe().reset_index()
        for f in files
    ]
)

dt = 900  # in seconds
dxy = 11100  # in meter (for Latitude)
    
parameters_tracking = dict(
    d_max=2.5*dxy,
    method_linking="predict",
    adaptive_stop=0.2,
    adaptive_step=0.95,
    stubs=3,
    memory=0,
    PBC_flag="hdim_2",
    min_h2=0,
    max_h2=3600,
)

parameters_merge = dict(
    distance=dxy*10, frame_len=2, PBC_flag="hdim_2", min_h1=0, max_h1=1200, min_h2=0, max_h2=3600,
)

class DisableLogger():
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, exit_type, exit_value, exit_traceback):
       logging.disable(logging.NOTSET)

with DisableLogger():
    tracks = tobac.linking_trackpy(
        combined_features,
        None,
        dt,
        dxy,
        **parameters_tracking,
    )

track_min_threshold = tracks.groupby("cell").threshold_value.min()
valid_cells = track_min_threshold.index[track_min_threshold <= 225]
valid_cells = np.setdiff1d(valid_cells, -1)
wh_in_track = np.isin(tracks.cell, valid_cells)
tracks = tracks[wh_in_track]

merges = tobac.merge_split.merge_split_MEST(tracks, dxy, **parameters_merge)

tracks["track"] = merges.feature_parent_track_id.data.astype(np.int64)

track_start_time = tracks.groupby("track").time.min()
tracks["time_track"] = tracks.time - track_start_time[tracks.track].to_numpy()

track_max_cell_len = tracks.groupby("track").apply(
    lambda df: max(df.groupby("cell").apply(len, include_groups=False))
)

valid_tracks = track_max_cell_len.index[track_max_cell_len >= 5]
wh_in_track = np.isin(tracks.track, valid_tracks)
tracks = tracks[wh_in_track]


from tobac.utils.periodic_boundaries import weighted_circmean

def process_clusters(tracks):
    groupby_order = ["frame", "track"]
    tracks["cluster"] = (tracks.groupby(groupby_order).feature.cumcount()[tracks.sort_values(groupby_order).index]==0).cumsum().sort_index()
    
    gb_clusters = tracks.groupby("cluster")
    
    clusters = gb_clusters.track.first().to_frame().rename(columns=dict(track="cluster_track_id"))
    
    clusters["cluster_time"] = gb_clusters.time.first().to_numpy()
    
    clusters["cluster_longitude"] = gb_clusters.apply(lambda x:weighted_circmean(x.longitude.to_numpy(), x.area.to_numpy(), low=0, high=360), include_groups=False)
    clusters["cluster_latitude"] = gb_clusters.apply(lambda x:np.average(x.latitude.to_numpy(), weights=x.area.to_numpy()), include_groups=False)
    
    clusters["cluster_area"] = gb_clusters.area.sum().to_numpy()
    clusters["cluster_max_precip"] = gb_clusters.max_precip.max().to_numpy()
    clusters["cluster_total_precip"] = gb_clusters.total_precip.sum().to_numpy()
    
    return tracks, clusters

def max_consecutive_true(condition: np.ndarray[bool]) -> int:
    """Return the maximum number of consecutive True values in 'condition'

    Parameters
    ----------
    condition : np.ndarray[bool]
        numpy array of boolean values

    Returns
    -------
    int
        the maximum number of consecutive True values in 'condition'
    """
    if isinstance(condition, pd.Series):
        condition = condition.to_numpy()
    if np.any(condition):
        return np.max(
            np.diff(
                np.where(
                    np.concatenate(
                        ([condition[0]], condition[:-1] != condition[1:], [True])
                    )
                )[0]
            )[::2],
            initial=0,
        )
    else:
        return 0


def is_track_mcs(clusters: pd.DataFrame) -> pd.DataFrame:
    """Test whether each track in features meets the condtions for an MCS

    Parameters
    ----------
    features : pd.Dataframe
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    consecutive_precip_max = clusters.groupby(["cluster_track_id"]).cluster_max_precip.apply(lambda x:max_consecutive_true(x>=10), include_groups=False)
    
    consecutive_area_max = clusters.groupby(["cluster_track_id"]).cluster_area.apply(lambda x:max_consecutive_true(x>=4e10), include_groups=False)
    
    max_total_precip = clusters.groupby(["cluster_track_id"]).cluster_total_precip.max()
    
    is_mcs = np.logical_and.reduce(
        [
            consecutive_precip_max >= 12,
            consecutive_area_max >= 12,
            max_total_precip.to_numpy() >= 2e10,
        ]
    )
    mcs_tracks =  pd.Series(data=is_mcs, index=consecutive_precip_max.index)
    mcs_tracks.index.name="track"
    return mcs_tracks


tracks, clusters = process_clusters(tracks)

mcs_tracks = is_track_mcs(clusters)

out_ds = tracks.set_index(tracks.feature).to_xarray()

out_ds = out_ds.rename_vars(
    {
        "time": "time_feature",
        "hdim_1": "y",
        "hdim_2": "x",
        "cell": "feature_cell_id",
        "track": "feature_track_id",
        "cluster": "feature_cluster_id"  
    }
)

cell_track_id = tracks.groupby("cell").track.first()

out_ds = out_ds.assign_coords(cell_track_id.to_xarray().coords)
out_ds = out_ds.assign_coords(clusters.to_xarray().coords)
out_ds = out_ds.assign_coords(mcs_tracks.to_xarray().coords)

out_ds["cell_track_id"] = cell_track_id.to_xarray()
out_ds = xr.merge([out_ds, clusters.to_xarray()])
out_ds["is_track_mcs"] = mcs_tracks.to_xarray()

out_ds.to_netcdf(
    save_path 
    / f"tobac_{start_date.strftime('%Y%m%d-%H%M%S')}_{end_date.strftime('%Y%m%d-%H%M%S')}_ICON_tracking_file.nc"
)

out_ds.close()