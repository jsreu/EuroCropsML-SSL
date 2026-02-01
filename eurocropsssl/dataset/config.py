from collections import OrderedDict

ERA5_BANDS = [
    "2m_temperature",
    "total_precipitation",
]  # this needs to be based on all values allowed for climate_variables

S2_BANDS = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "8A",
    "09",
    "10",
    "11",
    "12",
]  # order is important

S1_BANDS = ["VV", "VH"]  # order is important

ALL_BANDS = S1_BANDS + S2_BANDS + ERA5_BANDS


def _build_band_groups(bands: list[str]) -> OrderedDict:
    group_definitions = {
        "S1": S1_BANDS,
        "S2_B1": ["01"],
        "S2_RGB": ["02", "03", "04"],
        "S2_Red_Edge": ["05", "06", "07"],
        "S2_NIR_10m": ["08"],
        "S2_NIR_20m": ["8A"],
        "S2_B9": ["09"],
        "S2_SWIR": ["11", "12"],
        "S2_Cirrus": ["10"],
        "ERA5": ERA5_BANDS,
        "NDVI": ["NDVI"],
    }

    band_groups_dict = {}
    # Create band groups with existing bands
    for group_name, group_bands in group_definitions.items():
        # Get indices for all bands in this group that exist
        indices = [bands.index(b) for b in group_bands if b in bands]
        # Only add group if at least one band exists
        if indices:
            band_groups_dict[group_name] = indices

    # Create the OrderedDict
    band_groups = OrderedDict(band_groups_dict)

    return band_groups
