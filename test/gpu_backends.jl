using KernelAbstractions

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
    for (name, pkg_sym, array_sym, func_sym) in backend_defs
        try
            # Load package - fails if not installed; require returns the module
            pkg = Base.require(Main, pkg_sym)
            # Check if backend is functional (GPU available)
            # invokelatest avoids "method too new" world-age errors when loading packages dynamically
            if Base.invokelatest(getproperty(pkg, func_sym))
                AT = getproperty(pkg, array_sym)
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
    # CPU always available
    push!(backends, ("CPU", Array, _ -> nothing))
    # Add GPU backends
    for (name, AT, sync) in available_gpu_backends()
        push!(backends, (name, AT, sync))
    end
    return backends
end

