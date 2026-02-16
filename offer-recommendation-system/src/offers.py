# Static configuration for offers
# In a real production system, this would likely come from a database or dynamic config service.

OFFERS = [
    {'offer_id': 'OFF001', 'name': 'Free Oil Change', 'type': 'Free Service', 'value': 49.99},
    {'offer_id': 'OFF002', 'name': '20% Off Any Service', 'type': 'Discount', 'value': 0.20},
    {'offer_id': 'OFF003', 'name': 'Loyalty Points 2X', 'type': 'Points', 'value': 2.0},
    {'offer_id': 'OFF004', 'name': '$25 Off Next Visit', 'type': 'Credit', 'value': 25.00},
    {'offer_id': 'OFF005', 'name': 'Free Tire Rotation', 'type': 'Free Service', 'value': 29.99},
    {'offer_id': 'OFF006', 'name': 'Winter Package Deal', 'type': 'Bundle', 'value': 50.00},
    {'offer_id': 'OFF007', 'name': 'Refer a Friend $50', 'type': 'Referral', 'value': 50.00},
    {'offer_id': 'OFF008', 'name': 'Birthday Special 30%', 'type': 'Discount', 'value': 0.30},
]

OFFER_TYPES = ['Bundle', 'Credit', 'Discount', 'Free Service', 'Points', 'Referral']
