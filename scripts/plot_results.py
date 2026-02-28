#!/usr/bin/env python3

import json
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

def main():
    if len(sys.argv) < 2:
        print("Usage: plot_results.py <benchmark_json_file>")
        sys.exit(1)

    json_file = sys.argv[1]

    with open(json_file, 'r') as f:
        data = json.load(f)

    benchmarks = data.get('benchmarks', [])
    if not benchmarks:
        print("No benchmark data found.")
        sys.exit(1)

    results = defaultdict(lambda: defaultdict(dict))
    
    for b in benchmarks:
        name = b['name']
        if name.endswith("_BigO") or name.endswith("_RMS"):
            continue

        parts = name.split('/')
        if len(parts) != 2:
            continue
            
        base_name = parts[0]
        size = int(parts[1])
        cpu_time = b['cpu_time']
        
        if 'Multiplication' in base_name:
            method = base_name.replace('BM_Multiplication_', '')
            results['Multiplication'][method][size] = cpu_time
        elif 'Daxpy' in base_name:
            method = base_name.replace('BM_Daxpy_', '')
            results['Daxpy'][method][size] = cpu_time
        elif 'FFT' in base_name:
            method = base_name.replace('BM_FFT_', '')
            results['FFT'][method][size] = cpu_time

    # Standard clean matplotlib visual style (no excessive theming)
    plt.style.use('bmh')

    # Create 1x3 subplot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Multiplication
    for method, points in results['Multiplication'].items():
        sizes = sorted(points.keys())
        times = [points[s] for s in sizes]
        ax1.plot(sizes, times, marker='o', label=method)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Vector Size (Elements)')
    ax1.set_ylabel('CPU Time (ns)')
    ax1.set_title('Complex Multiplication')
    ax1.legend()

    # Plot 2: DAXPY
    for method, points in results['Daxpy'].items():
        sizes = sorted(points.keys())
        times = [points[s] for s in sizes]
        ax2.plot(sizes, times, marker='^', label=method)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.set_xlabel('Vector Size (Elements)')
    ax2.set_title('DAXPY (y = a*x+y)')
    ax2.legend()
    
    # Plot 3: FFT
    for method, points in results['FFT'].items():
        sizes = sorted(points.keys())
        times = [points[s] for s in sizes]
        ax3.plot(sizes, times, marker='s', label=method)
    ax3.set_xscale('log', base=2)
    ax3.set_yscale('log')
    ax3.set_xlabel('Vector Size (Elements)')
    ax3.set_title('1D Radix-2 FFT')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150)
    print("Generated benchmark_results.png in current directory.")

if __name__ == "__main__":
    main()
