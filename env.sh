export MUJOCO_GL=osmesa
export JAX_ENABLE_COMPILATION_CACHE=1
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_triton_gemm_any=true"EOF
