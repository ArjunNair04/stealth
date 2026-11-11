# A guide to Stealth

The shortest description is: train in PyTorch, run the saved model in C++, and
check that both give the same answer. We measure execution speed separately from
prediction quality. Improving one does not establish the other.

## Start with a working example

Build and run the native demo using the [development guide](development.md).
The [model walkthrough](forecast-v1.md) explains the calculation, and the
[Python guide](python-stream-v1.md) shows how to call it from NumPy.
For a trained example, follow the [training/export guide](training-v1.md) and read
what happened in the [bike-rental study](bike-demand-results-v1.md).

The main pieces have distinct jobs:

| Piece | Job |
| --- | --- |
| Native layers and forecaster | Calculate a prediction from a fixed window of numbers |
| Model file | Store weights, dimensions and preprocessing in a checked format |
| Session | Hold working memory so repeated calls can reuse it |
| Stream | Keep the most recent observations and move the window forward |
| Feature index and replay bridge | Assemble only observations available at the decision time |
| Training scripts | Fit the model in PyTorch and export its weights |
| Benchmark scripts | Compare answers first, then measure execution time |

## Read the results

The [Linux batching report](native-batching-v1.md) is the latest completed native
speed comparison. It includes the batch-eight improvement and the batch-sixteen
loss to eager PyTorch. The [runtime comparison guide](runtime-comparison.md)
explains the next test against compiled PyTorch and ONNX; those acceptance timings
have not been collected.

The bike example is a useful learning demonstration. The
[ETTm1 study](correction-results-v1.md) and [BTC study](btc-return-results-v1.md)
show why we keep simple baselines: a neural network can run correctly and still
make worse predictions. These reports preserve the negative results.

## Understand the intended use

A daily factor should not appear in a morning forecast if it was only published
that evening. The [feature timing contract](signal-timeline-v1.md) and
[native feature index](native-signal-index-v1.md) make that distinction explicit.
The [model bundle](model-bundle-v1.md) binds feature order, preprocessing and weights
so a replay can reproduce the original predictions.

These are offline building blocks. We still need real feed adapters, durable
recovery, sustained-load testing and a useful financial experiment. The
[direction page](direction.md) explains the order; the [work queue](tasks.md)
records current tasks and links to earlier milestones.

## Review or contribute

Start with [testing](testing.md), [numerical rules](numeric-contract.md) and
[architecture](architecture.md). A kernel change needs both an accuracy check and
a matched performance comparison. A new prediction idea needs its own frozen
experiment and evaluation period.

The [baseline review](review-baseline.md) maps the earlier implementation PRs.
Dated result reports describe the code and machine measured at that time; they
should not be read as promises about every current build or workload.
