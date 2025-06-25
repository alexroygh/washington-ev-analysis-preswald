from preswald import connect, get_df, table, text, plotly
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Utility Functions
def abbr_explain(*keys):
    """Return formatted explanations for abbreviations."""
    abbr = {
        'BEV': 'Battery Electric Vehicle',
        'PHEV': 'Plug-in Hybrid Electric Vehicle',
        'MSRP': "Manufacturer's Suggested Retail Price",
        'CAFV': 'Clean Alternative Fuel Vehicle',
        'VIN': 'Vehicle Identification Number',
        'DOL': 'Department of Licensing',
        'EV': 'Electric Vehicle',
        'WA': 'Washington',
    }
    return '\n'.join([f"**{k}**: {abbr[k]}" for k in keys if k in abbr])

def parse_point(val):
    """Parse 'POINT (lng lat)' into (longitude, latitude)."""
    try:
        if pd.isna(val):
            return np.nan, np.nan
        val = val.strip()
        if val.startswith("POINT (") and val.endswith(")"):
            coords = val[7:-1].split()
            if len(coords) == 2:
                return float(coords[0]), float(coords[1])
        return np.nan, np.nan
    except Exception:
        return np.nan, np.nan

# Data Preparation
def load_and_prepare_data():
    """Load and preprocess the EV dataset."""
    connect()
    df = get_df("electric_vehicles")
    column_map = {
        'VIN (1-10)': 'vin_1_10',
        'County': 'county',
        'City': 'city',
        'State': 'state',
        'Postal Code': 'zip_code',
        'Model Year': 'model_year',
        'Make': 'make',
        'Model': 'model',
        'Electric Vehicle Type': 'ev_type',
        'Clean Alternative Fuel Vehicle (CAFV) Eligibility': 'cafv_type',
        'Electric Range': 'electric_range',
        'Base MSRP': 'base_msrp',
        'Legislative District': 'legislative_district',
        'DOL Vehicle ID': 'dol_vehicle_id',
        'Vehicle Location': 'geocoded_column',
        'Electric Utility': 'electric_utility',
        '2020 Census Tract': '_2020_census_tract',
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    for col in ["electric_range", "base_msrp", "model_year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Parse geocoded location
    if "geocoded_column" in df.columns:
        coords = df["geocoded_column"].apply(parse_point)
        df["longitude"] = coords.apply(lambda x: x[0])
        df["latitude"] = coords.apply(lambda x: x[1])
    return df

# Visualization Functions
def plot_ev_map(df):
    """Show a map of EV registrations across Washington."""
    if "longitude" in df.columns and "latitude" in df.columns:
        map_df = df.dropna(subset=["longitude", "latitude"])
        if not map_df.empty:
            text("## Washington State EV Registration Map")
            text("**How to read:** This map shows the locations of registered EVs across Washington State. Each point represents a registered EV.\n**Insight:** Clusters reveal urban hotspots and geographic trends in EV adoption.\n" + abbr_explain('EV', 'WA'))
            fig_map = px.scatter_mapbox(
                map_df.sample(n=min(5000, len(map_df)), random_state=42),
                lat="latitude",
                lon="longitude",
                hover_name="city" if "city" in map_df.columns else None,
                hover_data={"make": True, "model": True, "model_year": True, "county": True},
                color_discrete_sequence=["#F89613"],
                zoom=5.5,
                height=600,
                title="EV Registrations Across Washington State"
            )
            fig_map.update_layout(mapbox_style="open-street-map")
            fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            plotly(fig_map)
        else:
            text("No valid geolocation data available for mapping EV registrations.")
    else:
        text("Geocoded location data not available for mapping.")

def plot_top_makes_models(df):
    """Bar chart of top makes and models."""
    top_models = df.groupby(["make", "model"]).size().reset_index(name="count").sort_values("count", ascending=False).head(15)
    if not top_models.empty:
        text("## Top Makes and Models")
        text("**How to read:** This bar chart shows the most common electric vehicle (EV) makes and models registered in Washington (WA).\n**Insight:** It reveals which brands and models are most popular among EV owners.\n" + abbr_explain('EV', 'WA'))
        top_models["make_model"] = top_models["make"] + ' ' + top_models["model"]
        fig_top_models = px.bar(
            top_models,
            x="count",
            y="make_model",
            orientation="h",
            title="Top 15 EV Makes and Models",
            labels={"count": "Number of Vehicles", "make_model": "Make and Model"}
        )
        plotly(fig_top_models)
    else:
        text("No data available for top makes and models.")

def plot_bev_phev_share(df):
    """Pie chart of BEV vs PHEV share."""
    if "ev_type" in df.columns:
        ev_type_counts = df["ev_type"].value_counts().reset_index()
        ev_type_counts.columns = ["ev_type", "count"]
        if not ev_type_counts.empty:
            text("## BEV vs PHEV Share")
            text("**How to read:** This pie chart shows the proportion of Battery Electric Vehicles (BEVs) vs Plug-in Hybrid Electric Vehicles (PHEVs) in the state.\n**Insight:** It highlights the market split between fully electric and plug-in hybrid vehicles.\n" + abbr_explain('BEV', 'PHEV', 'EV'))
            fig_ev_type = px.pie(
                ev_type_counts,
                names="ev_type",
                values="count",
                title="BEV vs PHEV Share",
                labels={"ev_type": "EV Type (BEV/PHEV)", "count": "Number of Vehicles"}
            )
            plotly(fig_ev_type)
        else:
            text("No data available for BEV vs PHEV share.")
    else:
        text("EV type data not available.")

def plot_electric_range_distribution(df):
    """Histogram of electric range."""
    if "electric_range" in df.columns:
        text("## Electric Range Distribution")
        text("**How to read:** This histogram shows how far EVs can travel on electric power alone.\n**Insight:** It reveals the most common electric ranges and the spread of EV capabilities.\n" + abbr_explain('EV'))
        range_df = df.dropna(subset=["electric_range"])
        if not range_df.empty:
            fig_range = px.histogram(
                range_df,
                x="electric_range",
                nbins=40,
                title="Distribution of Electric Range (miles)",
                labels={"electric_range": "Electric Range (miles)", "count": "Number of Vehicles"}
            )
            plotly(fig_range)
        else:
            text("No data available for electric range distribution.")
    else:
        text("Electric range data not available.")

def plot_top_cities(df):
    """Bar chart of top cities for EVs."""
    if "city" in df.columns:
        top_cities = df["city"].value_counts().reset_index()
        top_cities.columns = ["city", "count"]
        if not top_cities.empty:
            text("## Top Cities for EVs")
            text("**How to read:** This bar chart shows which cities have the most registered EVs.\n**Insight:** It highlights geographic hotspots for EV adoption.\n" + abbr_explain('EV'))
            fig_cities = px.bar(
                top_cities.head(15),
                x="city",
                y="count",
                title="Top 15 Cities for EVs",
                labels={"city": "City", "count": "Number of Vehicles"}
            )
            plotly(fig_cities)
        else:
            text("No data available for top cities.")
    else:
        text("City data not available.")

def plot_top_counties(df):
    """Bar chart of top counties for EVs."""
    if "county" in df.columns:
        top_counties = df["county"].value_counts().reset_index()
        top_counties.columns = ["county", "count"]
        if not top_counties.empty:
            text("## Top Counties for EVs")
            text("**How to read:** This bar chart shows which counties have the most registered EVs.\n**Insight:** It highlights geographic hotspots for EV adoption.\n" + abbr_explain('EV'))
            fig_counties = px.bar(
                top_counties.head(15),
                x="county",
                y="count",
                title="Top 15 Counties for EVs",
                labels={"county": "County", "count": "Number of Vehicles"}
            )
            plotly(fig_counties)
        else:
            text("No data available for top counties.")
    else:
        text("County data not available.")

def plot_msrp_by_ev_type(df):
    """Boxplot of MSRP by EV type."""
    if "base_msrp" in df.columns and "ev_type" in df.columns:
        text("## MSRP by EV Type")
        text("**How to read:** This boxplot shows the distribution of Manufacturer's Suggested Retail Price (MSRP) for BEVs and PHEVs.\n**Insight:** It reveals price differences between fully electric and plug-in hybrid vehicles.\n" + abbr_explain('MSRP', 'BEV', 'PHEV'))
        msrp_df = df.dropna(subset=["base_msrp", "ev_type"])
        if not msrp_df.empty:
            fig_msrp = px.box(
                msrp_df,
                x="ev_type",
                y="base_msrp",
                points="all",
                title="MSRP by EV Type",
                labels={"ev_type": "EV Type (BEV/PHEV)", "base_msrp": "Base MSRP (USD)"}
            )
            plotly(fig_msrp)
        else:
            text("No MSRP data available for BEV or PHEV vehicles.")
    else:
        text("MSRP or EV type data not available.")

def plot_yearly_trend(df):
    """Line chart of yearly trend of EV registrations."""
    if "model_year" in df.columns:
        text("## Yearly Trend of EV Registrations")
        text("**How to read:** This line chart shows how many EVs were registered each year.\n**Insight:** It reveals growth trends in EV adoption over time.\n" + abbr_explain('EV'))
        year_trend = df["model_year"].value_counts().reset_index()
        year_trend.columns = ["model_year", "count"]
        year_trend = year_trend.sort_values("model_year")
        if not year_trend.empty:
            fig_year = px.line(
                year_trend,
                x="model_year",
                y="count",
                markers=True,
                title="EV Registrations by Model Year",
                labels={"model_year": "Model Year", "count": "Number of Registrations"}
            )
            plotly(fig_year)
        else:
            text("No registration data available by model year.")
    else:
        text("Model year data not available.")

df = load_and_prepare_data()
text("# Washington State Electric Vehicle Population Explorer")
plot_ev_map(df)
plot_top_makes_models(df)
plot_bev_phev_share(df)
plot_electric_range_distribution(df)
plot_top_cities(df)
plot_top_counties(df)
plot_msrp_by_ev_type(df)
plot_yearly_trend(df)


