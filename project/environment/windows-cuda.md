# Windows CUDA

Follow [the platform guide](../docs/PLATFORM_GUIDE.md#windows-with-nvidia-cuda-cmd).
The Python pipeline has no POSIX process dependency. The supplied CMD wrappers
support quoted paths and disable delayed expansion. Actual Windows/CUDA and
special-character-path execution remain pending on a Windows machine.
