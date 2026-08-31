# CUDA setup and verification

The dedicated WSL2 environment is:
 /home/marcu/.venvs/ascento-cuda

It uses JAX 0.6.2, jaxlib 0.6.2, and matching CUDA 12 plugin/PJRT
packages. The host GPU is an NVIDIA GeForce RTX 3060 with driver 591.86.

Verify the backend from PowerShell:

    wsl.exe -d Ubuntu -- env JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=cuda /home/marcu/.venvs/ascento-cuda/bin/python -c "import jax; print(jax.default_backend(), jax.devices())"

Run the Section 40 Step 6 nominal balance validation:

    wsl.exe -d Ubuntu -- env JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=cuda PYTHONPATH=/mnt/c/Users/marcu/Downloads/Ascento_MuJoCo_Playground /home/marcu/.venvs/ascento-cuda/bin/python /mnt/c/Users/marcu/Downloads/Ascento_MuJoCo_Playground/evaluation/benchmark_balance.py --artifact /mnt/c/Users/marcu/Downloads/Ascento_MuJoCo_Playground/training/artifacts_cuda_smoke

No later-section training or behaviors are included.

Save the best nominal balance run as MP4:

    wsl.exe -d Ubuntu -- env MUJOCO_GL=egl JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=cuda ASCENTO_JAX_PLATFORM=cuda PYTHONPATH=/mnt/c/Users/marcu/Downloads/Ascento_MuJoCo_Playground /home/marcu/.venvs/ascento-cuda/bin/python /mnt/c/Users/marcu/Downloads/Ascento_MuJoCo_Playground/evaluation/benchmark_balance.py --artifact /mnt/c/Users/marcu/Downloads/Ascento_MuJoCo_Playground/training/artifacts_cuda_smoke --episodes 32 --save-mp4 /mnt/c/Users/marcu/Downloads/Ascento_MuJoCo_Playground/training/artifacts_cuda_smoke/best_balance.mp4

The evaluator chooses the highest-return nominal episode. MP4 export is
optional and does not change PPO training.
