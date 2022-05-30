# %%
from dataclasses import dataclass
from functools import cache
from skyfield.api import Star, load, wgs84, Angle, Timescale
from skyfield.data import hipparcos
from skyfield.vectorlib import VectorFunction
from skyfield.jpllib import SpiceKernel
from skyfield.positionlib import Barycentric
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np
from timezonefinder import TimezoneFinder


# %%
# @dataclass
# class Body:
#     name: str
#     altitude: Angle
#     azimuth: Angle


# %%
def hipparcos_epoch(ts):
    return ts.J(1991.25)


# %%
def hipparcos_stars() -> Star:
    with load.open(hipparcos.URL) as f:
        df = hipparcos.load_dataframe(f)
    df = df[df["ra_degrees"].notnull()]
    stars = Star.from_dataframe(df)
    # HIPs of the stars
    stars.names = df.index.to_numpy()
    return stars


# %%
@cache
def timescale() -> Timescale:
    return load.timescale()


# %%
@cache
def planets() -> SpiceKernel:
    return load("de421.bsp")


def earth() -> VectorFunction:
    return planets()["earth"]


# %%
def observation_position(
    lat: Angle,
    lon: Angle,
    time: datetime,
) -> Barycentric:
    utc_time = utctime(time, lat, lon)
    t = timescale().from_datetime(utc_time)
    # TODO; Specify the elevation for the Lat/Lon.
    pos = earth() + wgs84.latlon(
        lat.degrees, lon.degrees, elevation_m=0
    )
    pos_t = pos.at(t)
    assert isinstance(pos_t, Barycentric)
    return pos_t


# %%
def starry_sky(
    lat: Angle, lon: Angle, time: datetime
) -> tuple[Star, Angle, Angle]:
    pos = observation_position(lat, lon, time)
    stars = hipparcos_stars()
    # TODO; Atmosphere?
    alt, az, _ = pos.observe(stars).apparent().altaz()
    return stars, alt, az


# %%
def find_nearest_n_stars_to_zenith(
    lat: Angle,
    lon: Angle,
    time: datetime,
    n: int,
) -> tuple[np.ndarray, Angle, Angle]:
    stars, alt, az = starry_sky(lat, lon, time)
    ix = np.flip(np.argsort(alt.degrees)[-n:])
    alt, az = alt.degrees, az.degrees
    hip = stars.names
    assert isinstance(alt, np.ndarray)
    assert isinstance(az, np.ndarray)
    assert isinstance(hip, np.ndarray)
    return hip[ix], Angle(degrees=alt[ix]), Angle(degrees=az[ix])


# %%
def utctime(date: datetime, lat: Angle, lon: Angle) -> datetime:
    """Transform a naive date-time to aware.

    Args:
        date (datetime): _description_
        lat (Angle): _description_
        lon (Angle): _description_

    Returns:
        datetime: _description_
    """
    tz = TimezoneFinder().timezone_at(
        lng=lon.degrees, lat=lat.degrees
    )
    assert tz is not None
    return datetime(
        year=date.year,
        month=date.month,
        day=date.day,
        hour=date.hour,
        minute=date.minute,
        tzinfo=ZoneInfo(tz),
    ).astimezone(timezone.utc)
