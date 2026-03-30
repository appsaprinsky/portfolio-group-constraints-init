# functions/group_constraints.py

class GroupConstraints:
    def __init__(self, tickers):
        self.tickers = tickers
        self.groups = self._build_groups()
        self.limits = self._build_limits()

    def _build_groups(self):
        # -----------------------------
        # 30+ OVERLAPPING GROUPS
        # -----------------------------
        return {

            # --- BIG TECH ---
            "big_tech": ["AAPL","MSFT","GOOG","AMZN","META","NVDA"],
            "ai_cloud": ["MSFT","GOOG","AMZN","NVDA","CRM","NOW"],
            "semiconductors": ["NVDA","AMD","INTC","QCOM","TXN","AVGO","MU"],

            # --- FINANCIALS ---
            "banks": ["JPM","BAC","WFC","C","GS","MS"],
            "payments": ["V","MA","PYPL","AXP","COF"],
            "asset_mgmt": ["BLK","SCHW","BK"],

            # --- ENERGY ---
            "oil_majors": ["XOM","CVX","BP"],
            "energy_services": ["SLB","COP"],
            "energy_total": ["XOM","CVX","COP","SLB","BP"],

            # --- CONSUMER ---
            "consumer_staples": ["KO","PEP","PG","WMT","COST"],
            "consumer_discretionary": ["AMZN","TSLA","NKE","MCD","SBUX","HD","LOW","TGT"],
            "retail": ["WMT","COST","TGT","HD","LOW"],

            # --- TECH EXT ---
            "enterprise_software": ["ORCL","IBM","ADBE","CRM","NOW","SAP"],
            "cybersecurity": ["PANW","CRWD","CSCO"],
            "data_platforms": ["SNOW","DDOG","CRM"],

            # --- CHINA ---
            "china": ["BABA","JD","PDD"],
            "asia_mix": ["BABA","JD","PDD","SONY"],

            # --- INDUSTRIAL ---
            "industrials": ["GE","BA","CAT","DE"],
            "transport": ["UPS","FDX"],
            "autos": ["TSLA","F","GM"],

            # --- HEALTHCARE ---
            "big_pharma": ["JNJ","PFE","MRK","ABBV"],
            "healthcare_mix": ["UNH","TMO","MDT","CVS"],

            # --- UTILITIES ---
            "utilities": ["NEE","DUK","SO"],

            # --- DEFENSE ---
            "defense": ["LMT","RTX","NOC"],

            # --- REAL ESTATE ---
            "reits": ["PLD","AMT","CCI"],

            # --- FINTECH ---
            "fintech_growth": ["PYPL","SQ","SHOP"],  # SQ may be filtered out automatically

            # --- GROWTH / HIGH BETA ---
            "high_growth": ["TSLA","NVDA","SNOW","DDOG","SHOP","ZM","DOCU"],
            "unprofitable_tech": ["SNOW","DDOG","ZM","DOCU"],

            # --- PLATFORM ECONOMY ---
            "platforms": ["AMZN","GOOG","META","UBER"],
            "mobility": ["UBER","LYFT"],

            # --- DIVERSIFICATION STYLE ---
            "dividend_like": ["KO","PEP","PG","JNJ","XOM","CVX"],
            "quality": ["MSFT","AAPL","JNJ","PG","V","MA"],

            # --- BROAD OVERLAPS ---
            "mega_caps": ["AAPL","MSFT","GOOG","AMZN","NVDA","META","TSLA"],
            "us_core": [t for t in self.tickers if t not in ["BABA","JD","PDD","SONY","SAP","ASML"]],
        }

    def _build_limits(self):
        # -----------------------------
        # ALL CONSTRAINTS <=
        # -----------------------------
        return {

            "big_tech": 0.25,
            "ai_cloud": 0.20,
            "semiconductors": 0.18,

            "banks": 0.20,
            "payments": 0.15,
            "asset_mgmt": 0.12,

            "oil_majors": 0.12,
            "energy_services": 0.10,
            "energy_total": 0.20,

            "consumer_staples": 0.20,
            "consumer_discretionary": 0.25,
            "retail": 0.15,

            "enterprise_software": 0.20,
            "cybersecurity": 0.12,
            "data_platforms": 0.10,

            "china": 0.10,
            "asia_mix": 0.15,

            "industrials": 0.18,
            "transport": 0.10,
            "autos": 0.12,

            "big_pharma": 0.18,
            "healthcare_mix": 0.20,

            "utilities": 0.10,
            "defense": 0.12,
            "reits": 0.10,

            "fintech_growth": 0.10,
            "high_growth": 0.20,
            "unprofitable_tech": 0.08,

            "platforms": 0.25,
            "mobility": 0.08,

            "dividend_like": 0.20,
            "quality": 0.25,

            "mega_caps": 0.35,
            "us_core": 0.90,
        }

    def get_valid_groups(self, valid_tickers):
        # filter groups to only existing tickers in data
        filtered = {}
        for g, assets in self.groups.items():
            valid_assets = [a for a in assets if a in valid_tickers]
            if valid_assets:
                filtered[g] = valid_assets
        return filtered, self.limits