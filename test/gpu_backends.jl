using KernelAbstractions

"""
    supports_float64(ArrayType)

Whether `ArrayType`'s device can hold `Float64`, decided by trying it.

Double precision is a property of the part and its driver, not of the vendor
library: Intel's data-center GPUs support it, consumer integrated graphics
generally do not, and `oneArray(::Matrix{Float64})` throws rather than falling
back. So this asks the device instead of assuming anything about the backend.
"""
function supports_float64(ArrayType)
    try
        # invokelatest for the same reason as the functional() call below: the
        # backend package is loaded dynamically, so its constructors are too
        # new for this world age and a direct call raises MethodError.
        Base.invokelatest(ArrayType, zeros(Float64, 1, 1))
        return true
    catch e
        get(ENV, "NEXTLA_GPU_DEBUG", "0") == "1" &&
            @warn "$ArrayType: no Float64 support — $(sprint(showerror, e))"
        return false
    end
end

# The backends asked for with the runner flag --backends (e.g. "CPU,CUDA"), as
# lowercase names; empty means every available backend. test/runtests.jl
# defines TEST_BACKENDS_WANTED in each test's module.
_wanted_backends() =
    isdefined(@__MODULE__, :TEST_BACKENDS_WANTED) && !isempty(strip(TEST_BACKENDS_WANTED)) ?
        lowercase.(strip.(split(TEST_BACKENDS_WANTED, ','))) : String[]

"""
    available_gpu_backends()

Returns a vector of (name, ArrayType, synchronize_fn) for GPU backends that are
loadable and functional. Uses try/catch to skip backends that are not installed.

Synchronize_fn is a function that takes a GPU array and synchronizes its backend:
    sync(arr) = KernelAbstractions.synchronize(KernelAbstractions.get_backend(arr))
"""
function available_gpu_backends()
    backends = Tuple{String, Type, Function}[]
    backend_defs = [
        ("CUDA", :CUDA, :CuArray, :functional),
        ("AMDGPU", :AMDGPU, :ROCArray, :functional),
        ("oneAPI", :oneAPI, :oneArray, :functional),
        ("Metal", :Metal, :MtlArray, :functional),
    ]
    debug = get(ENV, "NEXTLA_GPU_DEBUG", "0") == "1"
    wanted = _wanted_backends()
    for (name, pkg_sym, array_sym, func_sym) in backend_defs
        # A backend not asked for is not even loaded, so --backends=CPU never
        # loads a GPU package.
        isempty(wanted) || lowercase(name) in wanted || continue
        try
            # Load package - fails if not installed; require returns the module
            pkg = Base.require(Main, pkg_sym)
            # Check if backend is functional (GPU available)
            # invokelatest avoids "method too new" world-age errors when loading packages dynamically
            if Base.invokelatest(getproperty(pkg, func_sym))
                AT = getproperty(pkg, array_sym)
                # This suite is written in double precision throughout, and
                # NextLA does not currently run on a device without it. Leave
                # such backends out rather than erroring on every case; see
                # KNOWN_ISSUES.md for what supporting them would involve.
                if !supports_float64(AT)
                    debug && @warn "$name: no Float64 on this device, backend skipped"
                    continue
                end
                sync_fn = function(arr)
                    KernelAbstractions.synchronize(KernelAbstractions.get_backend(arr))
                end
                push!(backends, (name, AT, sync_fn))
            elseif debug
                @warn "$name: functional() returned false (no GPU or driver issue)"
            end
        catch e
            debug && @warn "$name: $(sprint(showerror, e))"
            continue
        end
    end
    return backends
end

"""
    available_backends()

Returns a vector of (name, ArrayType, synchronize_fn) for all available backends,
including CPU first, then any functional GPU backends. Ensures tests always run
on at least CPU when no GPU is available.
"""
function available_backends()
    backends = Tuple{String, Type, Function}[]
    wanted = _wanted_backends()
    # CPU always available (unless --backends leaves it out)
    (isempty(wanted) || "cpu" in wanted) && push!(backends, ("CPU", Array, _ -> nothing))
    # Add GPU backends
    for (name, AT, sync) in available_gpu_backends()
        push!(backends, (name, AT, sync))
    end
    # A backend asked for but not usable is an error, so a GPU run cannot pass
    # on the CPU alone.
    absent = setdiff(wanted, lowercase.(first.(backends)))
    isempty(absent) || error("--backends=$TEST_BACKENDS_WANTED: not usable here: ", join(absent, ", "),
        ". A GPU backend needs its package in the test environment (see test/README.md, ",
        "\"GPU backends\") and a working device; NEXTLA_GPU_DEBUG=1 prints why it was skipped.")
    return backends
end

