# Low-K fused distance benchmarks

The optional SM90a configuration builds the complete low-K TF32 WGMMA path alongside the packed
warp-MMA, CUTLASS FastF32, and unfused FP32 implementations. All variants consume ordinary resident
FP32 inputs and precomputed norms and return top-1 assignments directly.

Configure and build on a Hopper machine with:

```sh
cmake -S cpp -B cpp/build-sm90a \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90a-real \
  -DBUILD_TESTS=ON \
  -DBUILD_CUVS_DISTANCE_BENCH=ON \
  -DCUVS_ENABLE_SM90A_WGMMA=ON
cmake --build cpp/build-sm90a --target \
  LOW_K_WGMMA_SM90A_TEST CUVS_LOW_K_DISTANCE_BENCH -j2
```

Run correctness before benchmarking:

```sh
cpp/build-sm90a/gtests/LOW_K_WGMMA_SM90A_TEST
cpp/build-sm90a/bench/CUVS_LOW_K_DISTANCE_BENCH \
  --benchmark_filter='(wgmma_tf32|packed_tf32|cutlass_fast_f32)/'
```

The WGMMA source requires the exact SM90a feature target. It intentionally has no generic SM90 PTX
fallback and will fail compilation if its device body is instantiated for another architecture.
