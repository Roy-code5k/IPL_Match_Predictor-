// These descriptive figures were calculated from data/raw/matches.csv (1,095 rows).
// They are historical observations, not model output. Raw venue labels are kept
// distinct until the data-cleaning pipeline defines an approved venue mapping.
export const dataset = {
  matches: 1095,
  decidedMatches: 1090,
  firstSeason: 2008,
  lastSeason: 2024,
  seasons: 17,
  rawVenueLabels: 58,
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

export const venues = [
  'Arun Jaitley Stadium',
  'Arun Jaitley Stadium, Delhi',
  'Barabati Stadium',
  'Barsapara Cricket Stadium, Guwahati',
  'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow',
  'Brabourne Stadium',
  'Brabourne Stadium, Mumbai',
  'Buffalo Park',
  'De Beers Diamond Oval',
  'Dr DY Patil Sports Academy',
  'Dr DY Patil Sports Academy, Mumbai',
  'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
  'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam',
  'Dubai International Cricket Stadium',
  'Eden Gardens',
  'Eden Gardens, Kolkata',
  'Feroz Shah Kotla',
  'Green Park',
  'Himachal Pradesh Cricket Association Stadium',
  'Himachal Pradesh Cricket Association Stadium, Dharamsala',
  'Holkar Cricket Stadium',
  'JSCA International Stadium Complex',
  'Kingsmead',
  'M Chinnaswamy Stadium',
  'M Chinnaswamy Stadium, Bengaluru',
  'M.Chinnaswamy Stadium',
  'MA Chidambaram Stadium',
  'MA Chidambaram Stadium, Chepauk',
  'MA Chidambaram Stadium, Chepauk, Chennai',
  'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur',
  'Maharashtra Cricket Association Stadium',
  'Maharashtra Cricket Association Stadium, Pune',
  'Narendra Modi Stadium, Ahmedabad',
  'Nehru Stadium',
  'New Wanderers Stadium',
  'Newlands',
  'OUTsurance Oval',
  'Punjab Cricket Association IS Bindra Stadium',
  'Punjab Cricket Association IS Bindra Stadium, Mohali',
  'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh',
  'Punjab Cricket Association Stadium, Mohali',
  'Rajiv Gandhi International Stadium',
  'Rajiv Gandhi International Stadium, Uppal',
  'Rajiv Gandhi International Stadium, Uppal, Hyderabad',
  'Sardar Patel Stadium, Motera',
  'Saurashtra Cricket Association Stadium',
  'Sawai Mansingh Stadium',
  'Sawai Mansingh Stadium, Jaipur',
  'Shaheed Veer Narayan Singh International Stadium',
  'Sharjah Cricket Stadium',
  'Sheikh Zayed Stadium',
  "St George's Park",
  'Subrata Roy Sahara Stadium',
  'SuperSport Park',
  'Vidarbha Cricket Association Stadium, Jamtha',
  'Wankhede Stadium',
  'Wankhede Stadium, Mumbai',
  'Zayed Cricket Stadium, Abu Dhabi',
]

export const leadingVenues = [
  { name: 'Eden Gardens', matches: 77 },
  { name: 'Wankhede Stadium', matches: 73 },
  { name: 'M Chinnaswamy Stadium', matches: 65 },
  { name: 'Feroz Shah Kotla', matches: 60 },
]

export const tossStats = {
  decided: 1090,
  tossWinnerWon: 554,
  field: { matches: 700, tossWinnerWon: 377 },
  bat: { matches: 390, tossWinnerWon: 177 },
}

export const seasons = Array.from({ length: 17 }, (_, index) => 2024 - index)
