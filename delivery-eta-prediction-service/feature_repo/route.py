from feast import Entity, ValueType

route = Entity(
    name="route_id",
    join_keys=["route_id"],
    value_type=ValueType.STRING,
    description="Unique identifier for a pickup-dropoff route combination",
)
