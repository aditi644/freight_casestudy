"""Core logic: weekly cost per tonne-km per route, plus the two graded baselines.

Definitions follow the brief exactly:
  cost_per_tonne_km = sum(freight_cost) / sum(quantity_tonnes * distance_km)   (per route, per week)
  week_of           = Monday of the shipment's Mon-Sun week
  vs own history    = trailing 8-week mean of the route's weekly cost_per_tonne_km, using only
                      weeks strictly before the current week (fewer weeks -> use all available)
  vs similar routes = mean cost_per_tonne_km, same week, of all OTHER routes sharing route_type
"""
import pandas as pd

HISTORY_WEEKS = 8

# Flag rule (our choice; the brief leaves the threshold open). See README.
OWN_RISE_PCT = 8.0     # rising >= 8% vs own trailing average
PEER_GAP_PCT = 20.0    # or >= 20% above similar-length routes (and still rising vs own history)


def load_shipments(path):
    d = pd.read_csv(path, parse_dates=["shipment_date"])
    d["route"] = d["origin"] + "-" + d["destination"]
    d["week_of"] = d["shipment_date"] - pd.to_timedelta(d["shipment_date"].dt.weekday, unit="D")
    d["tonne_km"] = d["quantity_tonnes"] * d["distance_km"]
    return d


def weekly_metrics(d):
    w = (d.groupby(["route", "route_type", "week_of"])
          .agg(cost=("freight_cost_inr", "sum"), tonne_km=("tonne_km", "sum"))
          .reset_index())
    w["cpt"] = w["cost"] / w["tonne_km"]
    w = w.sort_values(["route", "week_of"]).reset_index(drop=True)

    # Own history: reindex to full calendar weeks so a missing week is a gap, not silently skipped.
    parts = []
    for route, g in w.groupby("route"):
        full = pd.date_range(g["week_of"].min(), g["week_of"].max(), freq="7D")
        s = g.set_index("week_of")["cpt"].reindex(full)
        hist = s.shift(1).rolling(HISTORY_WEEKS, min_periods=1).mean()
        n_prior = s.shift(1).rolling(HISTORY_WEEKS, min_periods=1).count()
        parts.append(pd.DataFrame({"route": route, "week_of": full,
                                   "hist": hist.values, "hist_weeks": n_prior.values}))
    w = w.merge(pd.concat(parts), on=["route", "week_of"], how="left")

    # Peer average: other routes, same route_type, same week (self excluded).
    grp = w.groupby(["route_type", "week_of"])["cpt"]
    n_peers = grp.transform("count") - 1
    w["peer"] = (grp.transform("sum") - w["cpt"]) / n_peers.where(n_peers > 0)

    w["own_pct"] = (w["cpt"] / w["hist"] - 1) * 100
    w["peer_pct"] = (w["cpt"] / w["peer"] - 1) * 100
    return w


def add_flags(w):
    """Mark candidate rows: cost rising AND out of the ordinary vs own history or vs peers."""
    rising = w["own_pct"] > 0
    w["own_leg"] = w["own_pct"] >= OWN_RISE_PCT
    w["peer_leg"] = rising & (w["peer_pct"] >= PEER_GAP_PCT)
    w["candidate"] = rising & (w["own_leg"] | w["peer_leg"])
    return w
