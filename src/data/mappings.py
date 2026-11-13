"""Canonical team and venue names.

This is the single source of name mappings for cleaning, feature building,
and the prediction API, so training and inference always agree. Unknown
names raise an error rather than passing through silently.

Rules (decided in Phase 2, see docs/cleaning_rules.md):
- Teams: only documented renames of the same franchise are merged.
- Venues: labels for the same physical ground are merged, including grounds
  that were renamed. Canonical names use "Ground, City".
"""

from __future__ import annotations

# --- Teams -----------------------------------------------------------------

# Raw name -> canonical franchise name. Names not listed map to themselves.
TEAM_ALIASES: dict[str, str] = {
    "Delhi Daredevils": "Delhi Capitals",              # renamed 2019
    "Kings XI Punjab": "Punjab Kings",                 # renamed 2021
    "Rising Pune Supergiants": "Rising Pune Supergiant",  # 2016 spelling
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",  # renamed 2024
}

# Deliberately NOT merged (different franchises/owners):
#   Deccan Chargers vs Sunrisers Hyderabad
#   Pune Warriors vs Rising Pune Supergiant
#   Gujarat Lions vs Gujarat Titans
CANONICAL_TEAMS: frozenset[str] = frozenset({
    "Chennai Super Kings",
    "Deccan Chargers",
    "Delhi Capitals",
    "Gujarat Lions",
    "Gujarat Titans",
    "Kochi Tuskers Kerala",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Pune Warriors",
    "Punjab Kings",
    "Rajasthan Royals",
    "Rising Pune Supergiant",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
})

# Franchises that played in the 2024 season (the latest in the data).
ACTIVE_TEAMS: frozenset[str] = frozenset({
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
})


class UnknownNameError(ValueError):
    """Raised when a team or venue name has no mapping."""


def normalize_team(name: str) -> str:
    cleaned = " ".join(str(name).split())
    canonical = TEAM_ALIASES.get(cleaned, cleaned)
    if canonical not in CANONICAL_TEAMS:
        raise UnknownNameError(f"Unknown team name: {name!r}")
    return canonical


# --- Venues ----------------------------------------------------------------

# canonical name -> (city, country, raw labels seen in the data)
VENUES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    # India
    "Arun Jaitley Stadium, Delhi": ("Delhi", "India", (
        "Arun Jaitley Stadium", "Arun Jaitley Stadium, Delhi",
        "Feroz Shah Kotla",  # renamed 2019
    )),
    "Barabati Stadium, Cuttack": ("Cuttack", "India", ("Barabati Stadium",)),
    "Barsapara Cricket Stadium, Guwahati": ("Guwahati", "India", (
        "Barsapara Cricket Stadium, Guwahati",
    )),
    "Ekana Cricket Stadium, Lucknow": ("Lucknow", "India", (
        "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    )),
    "Brabourne Stadium, Mumbai": ("Mumbai", "India", (
        "Brabourne Stadium", "Brabourne Stadium, Mumbai",
    )),
    "Dr DY Patil Sports Academy, Navi Mumbai": ("Navi Mumbai", "India", (
        "Dr DY Patil Sports Academy", "Dr DY Patil Sports Academy, Mumbai",
    )),
    "ACA-VDCA Cricket Stadium, Visakhapatnam": ("Visakhapatnam", "India", (
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
    )),
    "Eden Gardens, Kolkata": ("Kolkata", "India", (
        "Eden Gardens", "Eden Gardens, Kolkata",
    )),
    "Green Park, Kanpur": ("Kanpur", "India", ("Green Park",)),
    "HPCA Stadium, Dharamsala": ("Dharamsala", "India", (
        "Himachal Pradesh Cricket Association Stadium",
        "Himachal Pradesh Cricket Association Stadium, Dharamsala",
    )),
    "Holkar Cricket Stadium, Indore": ("Indore", "India", ("Holkar Cricket Stadium",)),
    "JSCA International Stadium Complex, Ranchi": ("Ranchi", "India", (
        "JSCA International Stadium Complex",
    )),
    "M Chinnaswamy Stadium, Bengaluru": ("Bengaluru", "India", (
        "M Chinnaswamy Stadium", "M Chinnaswamy Stadium, Bengaluru",
        "M.Chinnaswamy Stadium",
    )),
    "MA Chidambaram Stadium, Chennai": ("Chennai", "India", (
        "MA Chidambaram Stadium", "MA Chidambaram Stadium, Chepauk",
        "MA Chidambaram Stadium, Chepauk, Chennai",
    )),
    # New ground opened 2024; not the same as the PCA stadium in Mohali.
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur": ("Mullanpur", "India", (
        "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur",
    )),
    "Maharashtra Cricket Association Stadium, Pune": ("Pune", "India", (
        "Maharashtra Cricket Association Stadium",
        "Maharashtra Cricket Association Stadium, Pune",
        "Subrata Roy Sahara Stadium",  # same ground, earlier sponsor name
    )),
    # Same site as the old Sardar Patel Stadium; rebuilt and renamed 2021.
    "Narendra Modi Stadium, Ahmedabad": ("Ahmedabad", "India", (
        "Narendra Modi Stadium, Ahmedabad", "Sardar Patel Stadium, Motera",
    )),
    "Nehru Stadium, Kochi": ("Kochi", "India", ("Nehru Stadium",)),
    "PCA IS Bindra Stadium, Mohali": ("Mohali", "India", (
        "Punjab Cricket Association IS Bindra Stadium",
        "Punjab Cricket Association IS Bindra Stadium, Mohali",
        "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh",
        "Punjab Cricket Association Stadium, Mohali",
    )),
    "Rajiv Gandhi International Stadium, Hyderabad": ("Hyderabad", "India", (
        "Rajiv Gandhi International Stadium",
        "Rajiv Gandhi International Stadium, Uppal",
        "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
    )),
    "Saurashtra Cricket Association Stadium, Rajkot": ("Rajkot", "India", (
        "Saurashtra Cricket Association Stadium",
    )),
    "Sawai Mansingh Stadium, Jaipur": ("Jaipur", "India", (
        "Sawai Mansingh Stadium", "Sawai Mansingh Stadium, Jaipur",
    )),
    "Shaheed Veer Narayan Singh International Stadium, Raipur": ("Raipur", "India", (
        "Shaheed Veer Narayan Singh International Stadium",
    )),
    "Vidarbha Cricket Association Stadium, Nagpur": ("Nagpur", "India", (
        "Vidarbha Cricket Association Stadium, Jamtha",
    )),
    "Wankhede Stadium, Mumbai": ("Mumbai", "India", (
        "Wankhede Stadium", "Wankhede Stadium, Mumbai",
    )),
    # United Arab Emirates (2014 opening phase, 2020, and 2021 second half)
    "Dubai International Cricket Stadium, Dubai": ("Dubai", "UAE", (
        "Dubai International Cricket Stadium",
    )),
    "Sharjah Cricket Stadium, Sharjah": ("Sharjah", "UAE", ("Sharjah Cricket Stadium",)),
    "Zayed Cricket Stadium, Abu Dhabi": ("Abu Dhabi", "UAE", (
        "Sheikh Zayed Stadium",  # renamed
        "Zayed Cricket Stadium, Abu Dhabi",
    )),
    # South Africa (2009 season)
    "Buffalo Park, East London": ("East London", "South Africa", ("Buffalo Park",)),
    "De Beers Diamond Oval, Kimberley": ("Kimberley", "South Africa", ("De Beers Diamond Oval",)),
    "Kingsmead, Durban": ("Durban", "South Africa", ("Kingsmead",)),
    "New Wanderers Stadium, Johannesburg": ("Johannesburg", "South Africa", ("New Wanderers Stadium",)),
    "Newlands, Cape Town": ("Cape Town", "South Africa", ("Newlands",)),
    "OUTsurance Oval, Bloemfontein": ("Bloemfontein", "South Africa", ("OUTsurance Oval",)),
    "St George's Park, Port Elizabeth": ("Port Elizabeth", "South Africa", ("St George's Park",)),
    "SuperSport Park, Centurion": ("Centurion", "South Africa", ("SuperSport Park",)),
}

# raw label (and canonical name) -> canonical name
VENUE_ALIASES: dict[str, str] = {}
for _canonical, (_city, _country, _aliases) in VENUES.items():
    for _label in (_canonical, *_aliases):
        if VENUE_ALIASES.setdefault(_label, _canonical) != _canonical:
            raise RuntimeError(f"Venue label {_label!r} maps to two venues")


def normalize_venue(name: str) -> str:
    cleaned = " ".join(str(name).split())
    try:
        return VENUE_ALIASES[cleaned]
    except KeyError:
        raise UnknownNameError(f"Unknown venue name: {name!r}") from None


def venue_city(canonical: str) -> str:
    return VENUES[canonical][0]


def venue_country(canonical: str) -> str:
    return VENUES[canonical][1]


# --- Match stage -----------------------------------------------------------

PLAYOFF_TYPES: frozenset[str] = frozenset({
    "Qualifier 1", "Qualifier 2", "Eliminator", "Final",
    "Semi Final", "Elimination Final", "3rd Place Play-Off",
})
LEAGUE_TYPES: frozenset[str] = frozenset({"League"})
