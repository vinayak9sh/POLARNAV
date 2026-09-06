\# POLARNAV



\## Predictive Decision Support for Antarctic Maritime Navigation



POLARNAV is an Antarctic maritime navigation decision-support prototype that combines:



\- Sea-ice forecasting

\- Iceberg trajectory prediction

\- Integrated environmental risk assessment

\- Least-cost route planning

\- Dynamic route replanning

\- Interactive map-based visualization



The project is designed as a modular ML + navigation system for Antarctic maritime decision support.



\---



\## Architecture



```text

Module 1

Sea-Ice Forecasting

&#x20;       |

&#x20;       v

Module 2

Iceberg Trajectory Prediction

&#x20;       |

&#x20;       v

Module 3

Integrated Risk / Decision Engine

&#x20;       |

&#x20;       v

Navigation Context Cache

&#x20;       |

&#x20;       v

Module 4

Navigation + Dynamic Replanning

&#x20;       |

&#x20;       +-------------------+

&#x20;       |                   |

&#x20;       v                   v

&#x20;    FastAPI            React + Leaflet

&#x20;     Backend              Frontend

