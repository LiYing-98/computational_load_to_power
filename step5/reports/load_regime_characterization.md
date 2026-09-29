# Load regime characterization

## Finding

The 565 stable GPU-side text-LLM inference runs are best described as a
**high-load, near-saturation steady-state serving regime**. Median observed
average-batch occupancy is 0.9983 of configured `max_num_seqs`;
the 5th–95th percentile interval is 0.9422–0.9999.

## Evidence

- Stable runs: 565; all have an observed average batch-size diagnostic.
- Occupancy p25/p50/p75: 0.9970 / 0.9983 / 0.9994.
- Minimum/maximum occupancy: 0.8540 / 1.0000.
- Observed request throughput spans 0.0227–37.4114 requests/s.

## Boundary

Average batch size and request throughput are post-run, **diagnostic-only**
fields. They are not included in B0–B5 and are never formal model inputs. The
evidence supports prediction for high-load steady-state operation; it does not
identify an arbitrary request-arrival response curve `P(lambda)`.
