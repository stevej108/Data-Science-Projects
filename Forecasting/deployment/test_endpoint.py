import os

import requests
from dotenv import load_dotenv


load_dotenv()


DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")


if not DATABRICKS_HOST:
    raise ValueError(
        "DATABRICKS_HOST was not found."
    )

if not DATABRICKS_TOKEN:
    raise ValueError(
        "DATABRICKS_TOKEN was not found."
    )

ENDPOINT_NAME = "marketing-traffic-forecast"

url = (
    f"{DATABRICKS_HOST}/serving-endpoints/"
    f"{ENDPOINT_NAME}/invocations"
)


headers = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}


payload = {
    "dataframe_split": {
    "columns": [
      "Traffic_Lag_1",
      "Rolling_Traffic_3",
      "Transactions_Lag_1",
      "Transactions_Lag_3",
      "Transactions_Lag_6",
      "Store_Count",
      "DMA_Code",
      "Total_Selling_SqFt",
      "Avg_Selling_SqFt",
      "DMA_Population",
      "TradeAreaPopulation",
      "Stores_per_Million",
      "CX_Composite",
      "Comp_Sales_Index",
      "Avg_Competitors",
      "Competitor_Density",
      "Pct_A_Facility",
      "Year",
      "Month_Number",
      "Quarter",
      "Holiday_Season"
    ],
    "data": [
      [
        619582.659410845,
        570408.2799398267,
        309507.40563056234,
        270587.66262309847,
        261606.4795630454,
        32,
        0,
        906394.3522188094,
        26544.26104109138,
        8025235.566338746,
        38490474.36641819,
        4.279561587506839,
        3.451050389886326,
        0.8871995505600715,
        0.6043806731186437,
        0.018707550011118033,
        0.5383814368494656,
        1964,
        4,
        2,
        0
      ],
      [
        291154.3795740823,
        268014.39848632744,
        113015.18863118377,
        105637.09946428856,
        117356.67786809227,
        10,
        1,
        278993.4052188885,
        25772.294847287023,
        2995147.686008721,
        5978190.126905992,
        3.18344596286196,
        3.322841544045147,
        0.9847287154821629,
        0.5869986719787209,
        0.05787322589690227,
        0.6744824320662547,
        2120,
        4,
        2,
        0
      ],
      [
        525148.0476175756,
        471683.9816918473,
        269619.95119126193,
        249123.73636567374,
        219699.25652155437,
        24,
        2,
        765494.0953188853,
        31818.86277024229,
        6998182.791080294,
        8588317.335454084,
        3.425226417600204,
        3.1869324307874165,
        1.322458632081787,
        0.4086641245382043,
        0.016641833398498194,
        0.5641036220231253,
        1951,
        4,
        2,
        0
      ],
      [
        52987.950327445906,
        46232.758967567126,
        27233.7690407993,
        22277.889843398523,
        26053.94090014555,
        5,
        3,
        169252.14922287653,
        33220.19808383336,
        1604018.294296739,
        1422018.987048676,
        3.2735965327525074,
        3.668471908599945,
        0.8047762551394078,
        0.4073625312647573,
        0.07868845754441019,
        0.6060837167416354,
        1998,
        4,
        2,
        0
      ],
      [
        319532.8283751787,
        312144.203364825,
        143464.45817846514,
        122900.1019379787,
        99150.03601033239,
        15,
        4,
        420088.45126498456,
        29265.214657141598,
        3703114.923174136,
        20927175.286329765,
        4.337068493008084,
        3.2224234258296462,
        0.8489889588522638,
        0.7573958994405673,
        0.049916952171127814,
        0.6078614392321463,
        1934,
        4,
        2,
        0
      ]
    ]
  }
}
    


response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=30,
)

print("Status:", response.status_code)
print("Response:", response.text)

response.raise_for_status()


prediction = response.json()

print(prediction)