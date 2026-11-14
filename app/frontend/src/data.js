// These descriptive figures were calculated from data/raw/matches.csv (1,095 rows)
// and data/processed/matches_clean.csv (1,090 decided matches). They are historical
// observations, not model output. Team and venue names follow src/data/mappings.py;
// tests/test_frontend_data.py fails if the lists here drift from it.
export const dataset = {
  matches: 1095,
  decidedMatches: 1090,
  firstSeason: 2008,
  lastSeason: 2024,
  seasons: 17,
  rawVenueLabels: 58,
  grounds: 36,
  franchises: 15,
  noResults: 5,
}

// Only documented franchise renames are joined. Deccan Chargers / Sunrisers
// Hyderabad, Pune Warriors / Rising Pune Supergiant, and Gujarat Lions /
// Gujarat Titans remain separate historical identities.
export const franchiseAliases = {
  'Delhi Daredevils': 'Delhi Capitals',
  'Kings XI Punjab': 'Punjab Kings',
  'Rising Pune Supergiants': 'Rising Pune Supergiant',
  'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
}

// Played excludes the five no-result matches. Ties with a recorded winner count.
export const teams = [
  { name: 'Chennai Super Kings', played: 237, wins: 138 },
  { name: 'Deccan Chargers', played: 75, wins: 29 },
  { name: 'Delhi Capitals', played: 250, wins: 115 },
  { name: 'Gujarat Lions', played: 30, wins: 13 },
  { name: 'Gujarat Titans', played: 45, wins: 28 },
  { name: 'Kochi Tuskers Kerala', played: 14, wins: 6 },
  { name: 'Kolkata Knight Riders', played: 251, wins: 131 },
  { name: 'Lucknow Super Giants', played: 43, wins: 24 },
  { name: 'Mumbai Indians', played: 261, wins: 144 },
  { name: 'Pune Warriors', played: 45, wins: 12 },
  { name: 'Punjab Kings', played: 246, wins: 112 },
  { name: 'Rajasthan Royals', played: 219, wins: 112 },
  { name: 'Rising Pune Supergiant', played: 30, wins: 15 },
  { name: 'Royal Challengers Bengaluru', played: 252, wins: 123 },
  { name: 'Sunrisers Hyderabad', played: 182, wins: 88 },
]

// The 36 grounds after merging label variants and renamed grounds
// (e.g. Feroz Shah Kotla -> Arun Jaitley Stadium). Names are "Ground, City".
export const venues = [
  'ACA-VDCA Cricket Stadium, Visakhapatnam',
  'Arun Jaitley Stadium, Delhi',
  'Barabati Stadium, Cuttack',
  'Barsapara Cricket Stadium, Guwahati',
  'Brabourne Stadium, Mumbai',
  'Buffalo Park, East London',
  'De Beers Diamond Oval, Kimberley',
  'Dr DY Patil Sports Academy, Navi Mumbai',
  'Dubai International Cricket Stadium, Dubai',
  'Eden Gardens, Kolkata',
  'Ekana Cricket Stadium, Lucknow',
  'Green Park, Kanpur',
  'HPCA Stadium, Dharamsala',
  'Holkar Cricket Stadium, Indore',
  'JSCA International Stadium Complex, Ranchi',
  'Kingsmead, Durban',
  'M Chinnaswamy Stadium, Bengaluru',
  'MA Chidambaram Stadium, Chennai',
  'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur',
  'Maharashtra Cricket Association Stadium, Pune',
  'Narendra Modi Stadium, Ahmedabad',
  'Nehru Stadium, Kochi',
  'New Wanderers Stadium, Johannesburg',
  'Newlands, Cape Town',
  'OUTsurance Oval, Bloemfontein',
  'PCA IS Bindra Stadium, Mohali',
  'Rajiv Gandhi International Stadium, Hyderabad',
  'Saurashtra Cricket Association Stadium, Rajkot',
  'Sawai Mansingh Stadium, Jaipur',
  'Shaheed Veer Narayan Singh International Stadium, Raipur',
  'Sharjah Cricket Stadium, Sharjah',
  "St George's Park, Port Elizabeth",
  'SuperSport Park, Centurion',
  'Vidarbha Cricket Association Stadium, Nagpur',
  'Wankhede Stadium, Mumbai',
  'Zayed Cricket Stadium, Abu Dhabi',
]

// Decided matches per ground, all label variants merged.
export const leadingVenues = [
  { name: 'Wankhede Stadium, Mumbai', matches: 118 },
  { name: 'Eden Gardens, Kolkata', matches: 93 },
  { name: 'M Chinnaswamy Stadium, Bengaluru', matches: 91 },
  { name: 'Arun Jaitley Stadium, Delhi', matches: 89 },
]

export const tossStats = {
  decided: 1090,
  tossWinnerWon: 554,
  field: { matches: 700, tossWinnerWon: 377 },
  bat: { matches: 390, tossWinnerWon: 177 },
}

export const seasons = Array.from({ length: 17 }, (_, index) => 2024 - index)
