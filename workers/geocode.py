"""Offline reverse geocoding against GeoNames (cities1000: every place with pop >= 1000).

Nearest-neighbour search runs on 3-D unit-sphere coordinates so distances are
right near the poles and across the antimeridian. The dataset is downloaded
once into data/geonames/ and cached as .npz; after that no network is used.
"""

from __future__ import annotations

import io
import math
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASE = "https://download.geonames.org/export/dump/"
EARTH_KM = 6371.0
# Sections of cities ("Chicago Loop", "Paris 04 Hôtel-de-Ville") and historical,
# abandoned or destroyed places. Without this, photos in Chicago get tagged with
# a neighbourhood and "photos from Chicago" finds nothing.
SKIP_FEATURES = {"PPLX", "PPLH", "PPLQ", "PPLW", "PPLCH"}
ABSORB_MAX_POP = 25_000
ABSORB_RATIO = 20.0


@dataclass(frozen=True)
class Place:
    name: str
    admin1: str | None
    country: str | None
    distance_km: float


def _xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def city_radius_km(population: float) -> float:
    """Rough radius of a place's built-up area: ~1.5 km for a village, ~13 km for
    Chicago or Paris, ~15 km for Tokyo."""
    return 1.5 + 3.5 * math.log10(max(population, 1000.0) / 1000.0)


class ReverseGeocoder:
    """Nearest place, except that a small nearest place inside a much bigger city's
    radius is replaced by the city.

    GeoNames lists hundreds of neighbourhoods ("Hatsudai", "Paris 04 Hôtel-de-Ville")
    as populated places in their own right; plain nearest-neighbour would tag a photo
    in Shibuya with one of those and "photos from Tokyo" would find nothing. Real towns
    beside a bigger one (Evanston, Oak Park, Palo Alto) keep their own name."""

    def __init__(
        self, names: list[str], admin1: list[str], country: list[str], lat, lon, population=None
    ):
        from scipy.spatial import cKDTree

        self.names, self.admin1, self.country = names, admin1, country
        self.population = np.asarray(
            population if population is not None else [1000] * len(names), float
        )
        self.tree = cKDTree(_xyz(np.asarray(lat, float), np.asarray(lon, float)))

    def lookup(self, lat: float, lon: float, max_km: float = 150.0) -> Place | None:
        k = min(50, len(self.names))
        chords, idxs = self.tree.query(_xyz(np.array([lat]), np.array([lon]))[0], k=k)
        chords, idxs = np.atleast_1d(chords), np.atleast_1d(idxs)
        kms = 2 * EARTH_KM * np.arcsin(np.minimum(1.0, chords / 2))
        near_km, near = kms[0], idxs[0]
        km, i = near_km, near
        # Absorb the nearest place into a bigger city whose radius contains the photo when
        # it is a small place (a neighbourhood entry) or a named district of that city
        # ("Paris 15 Vaugirard", "Lyon 03"). Real towns beside a bigger one keep their name.
        near_name, near_pop = str(self.names[near]), self.population[near]
        best = None
        for d, j in zip(kms, idxs, strict=True):
            if j >= len(self.names) or j == near or d > city_radius_km(self.population[j]):
                continue
            pop_j = self.population[j]
            small_nearby = near_pop < ABSORB_MAX_POP and pop_j >= ABSORB_RATIO * max(
                near_pop, 1000.0
            )
            district = near_name.startswith(str(self.names[j]) + " ") and pop_j > near_pop
            if (small_nearby or district) and (best is None or pop_j > self.population[best[1]]):
                best = (d, j)
        if best is not None:
            km, i = best
        if km > max_km:
            return None
        return Place(self.names[i], self.admin1[i] or None, self.country[i] or None, float(km))

    @classmethod
    def from_geonames(cls, directory: Path, download: bool = True) -> ReverseGeocoder:
        cache = directory / "cities1000_v3.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            return cls(
                list(z["names"]),
                list(z["admin1"]),
                list(z["country"]),
                z["lat"],
                z["lon"],
                z["pop"],
            )
        directory.mkdir(parents=True, exist_ok=True)
        cities = directory / "cities1000.txt"
        admin_f = directory / "admin1CodesASCII.txt"
        country_f = directory / "countryInfo.txt"
        if download:
            _fetch(directory)
        admin = _load_admin1(admin_f)
        countries = _load_countries(country_f)
        names, a1, cc, lat, lon, pop = [], [], [], [], [], []
        with cities.open(encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 11 or p[7] in SKIP_FEATURES:
                    continue
                names.append(p[1])
                lat.append(float(p[4]))
                lon.append(float(p[5]))
                a1.append(admin.get(f"{p[8]}.{p[10]}", ""))
                cc.append(countries.get(p[8], p[8]))
                pop.append(float(p[14] or 0) if len(p) > 14 else 0.0)
        np.savez(
            cache,
            names=np.array(names),
            admin1=np.array(a1),
            country=np.array(cc),
            lat=np.array(lat),
            lon=np.array(lon),
            pop=np.array(pop),
        )
        return cls(names, a1, cc, lat, lon, pop)


def _fetch(directory: Path) -> None:
    import httpx

    with httpx.Client(timeout=120, follow_redirects=True) as c:
        if not (directory / "cities1000.txt").exists():
            r = c.get(BASE + "cities1000.zip")
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extract("cities1000.txt", directory)
        for fname in ("admin1CodesASCII.txt", "countryInfo.txt"):
            if not (directory / fname).exists():
                r = c.get(BASE + fname)
                r.raise_for_status()
                (directory / fname).write_bytes(r.content)


def _load_admin1(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if len(p) >= 2:
            out[p[0]] = p[1]
    return out


def _load_countries(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) >= 5:
            out[p[0]] = p[4]
    return out


_geocoder: ReverseGeocoder | None = None
_lock = threading.Lock()


def get_geocoder() -> ReverseGeocoder | None:
    """Process-wide geocoder, or None if GeoNames isn't available (e.g. offline first run)."""
    global _geocoder
    from api.config import get_settings

    with _lock:
        if _geocoder is None:
            try:
                _geocoder = ReverseGeocoder.from_geonames(get_settings().geonames_dir)
            except Exception:
                return None
        return _geocoder


def set_geocoder(g: ReverseGeocoder | None) -> None:
    global _geocoder
    _geocoder = g
