# 0005. The lab and the service run the same scheduler

## Decision
`platform/build.py` builds the scheduler from profiles. The lab calls it for every learned design; the HTTP
service (`service.py`) calls it when a site's profiles arrive and again after a restart, with the same seed, so it
gives the same placement. The service is the standard library's threaded HTTP server: no framework, one JSON state
file written atomically, an hour's observations applied once.

## Consequences
What the README reports is what the service does. The service keeps no training progress across a restart: queued
jobs are resubmitted with their full token counts and the controller's next reports correct them.
