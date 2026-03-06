from datetime import timedelta
from feast import FeatureView, Field, FileSource
from feast.types import Float64, Int64

from route import route

route_stats_source = FileSource(
    path="data/route_features.parquet",
    timestamp_field="event_timestamp",
)

route_stats_fv = FeatureView(
    name="route_stats",
    entities=[route],
    ttl=timedelta(days=365 * 10), 
    schema=[
        Field(name="route_avg_duration", dtype=Float64),
        Field(name="route_avg_distance", dtype=Float64),
        Field(name="route_avg_fare", dtype=Float64),
        Field(name="route_trip_count", dtype=Int64),
    ],
    source=route_stats_source,
)