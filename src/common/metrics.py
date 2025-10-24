def merge_step_metrics(agg: dict, info: dict) -> dict:
    for k, v in info.get("metrics", {}).items():
        agg[k] = agg.get(k, 0.0) + float(v)
    return agg
