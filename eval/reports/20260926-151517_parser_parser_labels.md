| parser | field acc. | exact match | semantic F1 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| rules | 0.976 | 0.968 | 0.946 | 0.1 | 0.2 |
| llm_raw | 0.364 | 0.081 | 0.766 | 1807.9 | 2007.1 |
| llm | 0.720 | 0.774 | 0.782 | 2002.6 | 2255.7 |
| hybrid | 0.988 | 0.984 | 0.815 | 1935.4 | 2233.0 |
| auto | 0.976 | 0.968 | 0.922 | 0.1 | 1888.9 |

| field | support | rules P / R | llm_raw P / R | llm P / R | hybrid P / R | auto P / R |
|---|---:|---:|---:|---:|---:|---:|
| aperture_max | 1 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| aperture_min | 1 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| camera | 5 | 1.00 / 1.00 | 0.71 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| date_from | 21 | 1.00 / 1.00 | 0.58 / 0.67 | 0.61 / 0.67 | 1.00 / 1.00 | 1.00 / 1.00 |
| date_to | 21 | 1.00 / 1.00 | 0.59 / 0.62 | 0.62 / 0.62 | 1.00 / 1.00 | 1.00 / 1.00 |
| focal_max | 2 | 1.00 / 1.00 | 0.67 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| focal_min | 1 | 1.00 / 1.00 | 0.33 / 1.00 | 0.50 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| iso_max | 1 | 1.00 / 1.00 | 0.50 / 1.00 | 0.50 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| iso_min | 2 | 1.00 / 1.00 | 0.50 / 0.50 | 0.50 / 0.50 | 1.00 / 1.00 | 1.00 / 1.00 |
| lens | 5 | 1.00 / 1.00 | 0.71 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| media | 1 | 1.00 / 1.00 | 0.02 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| months | 2 | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 0.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| orientation | 1 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| people | 6 | 1.00 / 1.00 | 0.71 / 0.83 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| place | 15 | 0.93 / 0.87 | 0.38 / 0.53 | 1.00 / 1.00 | 0.93 / 0.93 | 0.93 / 0.87 |

**rules: sample failures**

- `trip to Golden Colorado`: wrong ['place']; got {'place': 'Colorado'}
- `lake tahoe kayaking`: wrong ['place']; got {}

**llm_raw: sample failures**

- `dog on a beach`: wrong ['media']; got {'media': 'photo'}
- `golden hour on the beach`: wrong ['place', 'media']; got {'place': 'beach', 'media': 'photo'}
- `moody night street`: wrong ['media']; got {'media': 'photo'}
- `red car`: wrong ['media']; got {'media': 'photo'}
- `photos from Chicago in 2025`: wrong ['media']; got {'date_from': '2025-01-01', 'date_to': '2025-12-31', 'place': 'Chicago', 'media': 'photo'}
- `portraits with the 85mm`: wrong ['focal_min', 'focal_max', 'media']; got {'lens': '85mm', 'focal_min': 85.0, 'focal_max': 85.0, 'media': 'photo'}
- `street shots at night with the 35mm`: wrong ['media', 'camera']; got {'camera': 'Fujifilm X100V', 'lens': 'Sigma 35mm F1.4 DG HSM', 'media': 'photo'}
- `me and Rohan hiking`: wrong ['media']; got {'people': ['me', 'Rohan'], 'media': 'photo'}

**llm: sample failures**

- `sunset last summer`: wrong ['date_from', 'date_to']; got {'date_from': '2025-06-01', 'date_to': '2025-08-31'}
- `high iso night shots`: wrong ['iso_max', 'iso_min']; got {'iso_min': 800, 'iso_max': 6400}
- `fall foliage in October`: wrong ['date_from', 'date_to', 'months']; got {'date_from': '2026-10-01', 'date_to': '2026-10-31'}
- `beach photos from summer`: wrong ['date_from', 'date_to', 'months']; got {'date_from': '2026-06-01', 'date_to': '2026-08-31'}
- `christmas 2024`: wrong ['date_from', 'date_to']; got {'date_from': '2024-12-01', 'date_to': '2024-12-25'}
- `food last month`: wrong ['date_from', 'date_to', 'months']; got {'date_from': '2026-08-26', 'date_to': '2026-09-25', 'months': [8]}
- `this week`: wrong ['date_from', 'date_to']; got {'date_from': '2026-09-26', 'date_to': '2026-10-02'}
- `mountains since 2023`: wrong ['date_to']; got {'date_from': '2023-01-01'}

**hybrid: sample failures**

- `trip to Golden Colorado`: wrong ['place']; got {'place': 'Colorado'}

**auto: sample failures**

- `trip to Golden Colorado`: wrong ['place']; got {'place': 'Colorado'}
- `lake tahoe kayaking`: wrong ['place']; got {}
