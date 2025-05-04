"""
bitbrush_demo.py

Demonstrates and benchmarks the BitBrush utility using matplotlib
and compares pure-Python vs NumPy backends.
"""

import time
import matplotlib.pyplot as plt
from bitbrush import BitBrush
import numpy as np

def benchmark_operation(func, *args) -> float:
    """
    Measures execution time for a given BitBrush operation.

    Args:
        func (callable): The bound method or generator function.
        *args: Arguments to pass to the function.

    Returns:
        float: Execution time in milliseconds.
    """
    start = time.perf_counter()
    result = func(*args)
    # if numpy array, just materialize length
    if hasattr(result, '__iter__') and not hasattr(result, '__next__'):
        # assume numpy array
        _ = result.shape
    else:
        for _ in result:
            pass
    end = time.perf_counter()
    return (end - start) * 1000


def plot_benchmark(results: dict):
    """
    Plots the benchmark results using matplotlib.

    Args:
        results (dict): Mapping of operation names to execution times in ms.
    """
    names = list(results.keys())
    times = list(results.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, times, edgecolor='black')
    plt.title("BitBrush: Python vs NumPy Benchmark")
    plt.ylabel("Time (ms)")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{t:.1f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def main():
    brush_py = BitBrush(width=32, backend="python")
    brush_np = BitBrush(width=32, backend="numpy")

    ops = [
        ("sweep_ones", brush_py.sweep_ones, brush_np.sweep_ones),
        ("sweep_zeros", brush_py.sweep_zeros, brush_np.sweep_zeros),
        ("toggle_sparse", lambda: brush_py.toggle_sparse(3), lambda: brush_np.toggle_sparse(3)),
        ("scan_patterns", brush_py.scan_patterns, brush_np.scan_patterns),
    ]

    results = {}
    for name, py_op, np_op in ops:
        results[f"{name} (py)"] = benchmark_operation(py_op)
        results[f"{name} (np)"] = benchmark_operation(np_op)

    # mirror_mask: compare 1000 values
    results["mirror_mask (py)"] = benchmark_operation(lambda: (brush_py.mirror_mask(x) for x in range(1000)))
    results["mirror_mask (np)"] = benchmark_operation(lambda: brush_np.mirror_mask(np.arange(1000, dtype=np.uint64)))

    plot_benchmark(results)


if __name__ == "__main__":
    main()
