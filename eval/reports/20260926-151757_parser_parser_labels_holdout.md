| parser | field acc. | exact match | semantic F1 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| rules | 0.946 | 0.960 | 0.973 | 0.1 | 0.2 |
| llm_raw | 0.458 | 0.080 | 0.757 | 2055.8 | 2347.1 |
| llm | 0.829 | 0.800 | 0.757 | 2010.3 | 2305.7 |
| hybrid | 0.921 | 0.920 | 0.773 | 1985.2 | 2317.2 |
| auto | 0.946 | 0.960 | 0.973 | 0.1 | 1553.3 |

| field | support | rules P / R | llm_raw P / R | llm P / R | hybrid P / R | auto P / R |
|---|---:|---:|---:|---:|---:|---:|
| aperture_max | 1 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| aperture_min | 1 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| camera | 2 | 1.00 / 1.00 | 0.67 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| date_from | 7 | 0.86 / 0.86 | 0.75 / 0.86 | 0.75 / 0.86 | 0.86 / 0.86 | 0.86 / 0.86 |
| date_to | 7 | 0.86 / 0.86 | 0.71 / 0.71 | 0.71 / 0.71 | 0.86 / 0.86 | 0.86 / 0.86 |
| focal_max | 1 | 1.00 / 1.00 | 0.50 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| focal_min | 1 | 1.00 / 1.00 | 0.50 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| iso_max | 1 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| iso_min | 1 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| lens | 2 | 1.00 / 1.00 | 0.67 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| media | 1 | 1.00 / 1.00 | 0.04 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| months | 1 | 1.00 / 1.00 | 0.50 / 1.00 | 0.50 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| orientation | 1 | 1.00 / 1.00 | 0.50 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| people | 3 | 1.00 / 1.00 | 0.60 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| place | 7 | 1.00 / 1.00 | 0.67 / 0.86 | 0.88 / 1.00 | 0.88 / 1.00 | 1.00 / 1.00 |

**rules: sample failures**

- `christmas lights 2023`: wrong ['date_to', 'date_from']; got {'date_from': '2023-01-01', 'date_to': '2023-12-31'}

**llm_raw: sample failures**

- `kids building a sandcastle`: wrong ['people', 'media']; got {'people': ['Rohan', 'Priya'], 'media': 'photo'}
- `neon signs in Tokyo`: wrong ['media']; got {'place': 'Tokyo', 'media': 'photo'}
- `Priya and Mom at dinner`: wrong ['media']; got {'people': ['Priya', 'Mom'], 'media': 'photo'}
- `foggy bridge shots from San Francisco`: wrong ['media']; got {'place': 'San Francisco', 'media': 'photo'}
- `night sky with the Sigma`: wrong ['media']; got {'lens': 'Sigma 35mm F1.4 DG HSM', 'media': 'photo'}
- `portraits at f/1.4`: wrong ['focal_max', 'focal_min', 'media']; got {'focal_min': 28.0, 'focal_max': 28.0, 'aperture_min': 1.35, 'aperture_max': 1.45, 'media': 'photo'}
- `photos from March 2026`: wrong ['months', 'media']; got {'date_from': '2026-03-01', 'date_to': '2026-03-31', 'months': [3], 'media': 'photo'}
- `snow in Kyoto last winter`: wrong ['date_to', 'place', 'media']; got {'date_from': '2025-12-01', 'place': 'Kyoto, Japan', 'media': 'photo'}

**llm: sample failures**

- `photos from March 2026`: wrong ['months']; got {'date_from': '2026-03-01', 'date_to': '2026-03-31', 'months': [3]}
- `snow in Kyoto last winter`: wrong ['date_to']; got {'date_from': '2025-12-01', 'place': 'Kyoto'}
- `the drone over the lake`: wrong ['place']; got {'place': 'South Lake Tahoe', 'camera': 'dji'}
- `Ohio in the fall`: wrong ['date_to', 'date_from']; got {'date_from': '2026-09-23', 'date_to': '2026-11-30', 'months': [9, 10, 11], 'place': 'Ohio'}
- `christmas lights 2023`: wrong ['date_to', 'date_from']; got {'date_from': '2023-12-01', 'date_to': '2023-12-25'}

**hybrid: sample failures**

- `the drone over the lake`: wrong ['place']; got {'place': 'South Lake Tahoe', 'camera': 'dji'}
- `christmas lights 2023`: wrong ['date_to', 'date_from']; got {'date_from': '2023-01-01', 'date_to': '2023-12-31'}

**auto: sample failures**

- `christmas lights 2023`: wrong ['date_to', 'date_from']; got {'date_from': '2023-01-01', 'date_to': '2023-12-31'}
