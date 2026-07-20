using CUDA, LinearAlgebra, Printf

# ==============================================================================
# 1. EXTRACTION & SANITIZATION ENGINE
# ==============================================================================
function extract_and_sanitize(filepath::String, level::Int)
    raw_text = read(filepath, String)
    
    # Extract only code inside markdown julia blocks
    blocks = [m.match for m in eachmatch(r"```(?:julia)?\s*([\s\S]*?)```", raw_text)]
    if isempty(blocks)
        # Strip markdown prose before Julia execution
        lines = split(raw_text, '\n')
        code_str = join(
            filter(line -> 
                startswith(strip(line), ("using ", "import ", "const ", "function ", "struct ", "mutable struct ", "macro ", "julia"))
                || occursin("=", line)
                || occursin("end", strip(line)),
                lines
            ),
            "\n"
        )
    else
        code_str = join(blocks, "\n\n")
    end
    
    # Strip all LLM-generated include(...) statements to prevent path errors
    clean_code = replace(code_str, r"^include\(.*?\)"m => "# [Stripped LLM include]")
    
    # For L3 & L5 (Tier C), strip hallucinated struct definitions so the sandbox 
    # strictly binds to your ground-truth src/fullmixedprec.jl
    if level in (3, 5)
        clean_code = replace(clean_code, r"(struct\s+FullMixedPrec[\s\S]*?end)"m => s"# [Stripped echoed struct to use ground truth: \1]")
    end
    
    return clean_code
end

# ==============================================================================
# 2. STATIC RUBRIC EVALUATOR
# ==============================================================================
function evaluate_rubric(code_str::String)
    rubric = Dict{Symbol, Bool}()
    rubric[:used_custom_trsm] = occursin("unified_rectrxm!", code_str)
    rubric[:correct_unit_diag_flag] = occursin(r"('L'|\"L\")\s*,\s*('L'|\"L\")\s*,\s*('N'|\"N\")\s*,\s*('U'|\"U\")", code_str)
    rubric[:zero_allocation_views] = occursin("@views", code_str) || occursin("view(", code_str)
    rubric[:no_scalar_loops] = !occursin(r"for\s+i\s*in\s*1:n,\s*j\s*in\s*1:n", code_str)
    return rubric
end

# ==============================================================================
# 3. TIER-AWARE DEPENDENCY INJECTOR
# ==============================================================================
function inject_dependencies!(sandbox::Module, level::Int)
    # Bind `include` inside the anonymous sandbox so internal file includes work
    Core.eval(sandbox, :(include(x) = Base.include($sandbox, x)))
    Core.eval(sandbox, quote
        using CUDA
        using LinearAlgebra
        using GPUArrays
        using GPUArraysCore
    end)
    
    # All levels require the base-case CUSOLVER wrappers
    Base.include(sandbox, "src/wrappers.jl")
    
    if level == 1
        println("  [Tier A: Baseline Flat Matrix Environment Injected]")
        return
    end
    
    # Levels 2-5 rely on your unified TRSM/TRMM solve kernels
    Base.include(sandbox, "src/rectrxm.jl")
    
    if level in (3, 5)
        # TIER C: Inject ground-truth data struct and recursive GEMM kernels
        println("  [Tier C: Injecting ground-truth src/matmul.jl and src/fullmixedprec.jl]")
        Base.include(sandbox, "src/matmul.jl")
        Base.include(sandbox, "src/fullmixedprec.jl")
    elseif level in (2, 4)
        # TIER B: Do NOT inject fullmixedprec.jl or matmul.jl!
        # Forces the sandbox to evaluate the LLM-generated struct and GEMM implementations.
        println("  [Tier B: Isolating sandbox to test LLM-generated struct & GEMM kernels]")
    end
end

# ==============================================================================
# 4. UNIVERSAL RECONSTRUCTION INJECTOR
# ==============================================================================
function ensure_reconstruct_matrix!(sandbox::Module)
    if isdefined(sandbox, :reconstruct_matrix)
        return # Already loaded via src/fullmixedprec.jl in Tier C
    end
    
    println("  [Injecting universal reconstruct_matrix fallback for numerical validation...]")
    
    # Dynamically evaluate ground-truth un-flattening logic against custom LLM structs
    Core.eval(sandbox, quote
        function reconstruct_matrix(A)
            if A.BaseCase !== nothing
                return copy(A.BaseCase)
            end
            
            C11 = reconstruct_matrix(A.A11)
            C22 = reconstruct_matrix(A.A22)
            C21 = A.A21
            C12 = A.A12
            
            n1, m1 = size(C11)
            n2, m2 = size(C22)
            n = n1 + n2

            # Promote off-diagonal Float16 blocks to the base precision of C11
            T_Base = eltype(C11)
            C_full = similar(C21, T_Base, n, n)
            
            C_full[1:n1, 1:m1] .= C11
            C_full[n1+1:n, 1:m1] .= C21
            C_full[n1+1:n, m1+1:n] .= C22
            C_full[1:n1, m1+1:n] .= C12

            return C_full
        end
    end)
end

# ==============================================================================
# 5. EXECUTION & VALIDATION SANDBOX
# ==============================================================================
function run_codex_test(filepath::String, level::Int; N::Int=1024)
    println("\n==========================================================")
    println("Evaluating: $filepath (Level $level)")
    println("==========================================================")
    
    clean_code = extract_and_sanitize(filepath, level)
    rubric = evaluate_rubric(clean_code)
    
    println("--- Static Rubric Checks ---")
    for (k, v) in rubric
        println(@sprintf("  %-25s : %s", string(k), v ? "PASS" : "FAIL"))
    end
    
    sandbox = Module()
    inject_dependencies!(sandbox, level)
    
    try
        # Base.include_string evaluates directly into the module and maps line numbers to the file
        Base.include_string(sandbox, clean_code, filepath)
        println("--- Compilation: SUCCESS ---")
    catch e
        println("--- Compilation: FAILED ---")
        println("Error: ", e)
        return
    end
    
    # Diagonally dominant matrix guarantees numerical stability for non-pivoting LU
    A_cpu = rand(Float32, N, N) + N * I
    A_gpu = CuArray(A_cpu)
    
    # Identify entry point symbol invented by the LLM
    possible_names = [:lu_nopiv_recursive_mixed!, :getrf_recursive!, :lu_recursive!, :lu_nopiv_recursive!]
    target_sym = nothing
    for name in possible_names
        if isdefined(sandbox, name)
            target_sym = name
            break
        end
    end
    
    if target_sym === nothing
        println("--- Execution: FAILED (No recognized entry point found) ---")
        return
    end
    llm_func = getfield(sandbox, target_sym)
    println("--- Executing Entry Point: $target_sym ---")
    
    try
        if level > 1 && isdefined(sandbox, :FullMixedPrec)
            StructType = getfield(sandbox, :FullMixedPrec)
            println("  -> Constructing hierarchical FullMixedPrec container...")
            A_test = StructType(copy(A_gpu); precisions=[Float16, Float32])
            
            # 1. Execute LLM factorization kernel
            llm_func(A_test)
            
            # 2. Guarantee reconstruct_matrix exists in sandbox (injects for L2/L4)
            ensure_reconstruct_matrix!(sandbox)
            
            # 3. Unpack back to flat matrix for validation
            A_result = sandbox.reconstruct_matrix(A_test)
        else
            println("  -> Executing flat CuMatrix path...")
            A_result = copy(A_gpu)
            llm_func(A_result)
        end
        
        # Verify Relative Residual: ||A - LU||_F / ||A||_F
        L = UnitLowerTriangular(Array(A_result))
        U = UpperTriangular(Array(A_result))
        rel_error = norm(A_cpu - (L * U)) / norm(A_cpu)
        
        println(@sprintf("--- Numerical Verification: Relative Error = %.2e ---", rel_error))
        if rel_error < 1e-3
            println("  -> VERDICT: NUMERICALLY SOUND")
        else
            println("  -> VERDICT: HIGH RESIDUAL / INSTABILITY DETECTED")
        end
        
    catch e
        println("--- Execution: RUNTIME GPU ERROR ---")
        println("Error: ", e)
    finally
        # Force H200 memory cleanup between evaluations
        A_gpu = nothing
        A_cpu = nothing
        GC.gc(true)
        CUDA.reclaim()
    end
end

# ==============================================================================
# 6. BATCH EXECUTION RUNNER
# ==============================================================================
output_dir = "llm_outputs"
if isdir(output_dir)
    files = sort(readdir(output_dir, join=true))
    for f in files
        if endswith(f, ".md")
            # Automatically extract level integer from filename (e.g., codex3lu.md -> 3)
            m = match(r"codex(\d+)lu\.md$", basename(f))
            if m !== nothing
                level = parse(Int, m.captures[1])
                run_codex_test(f, level; N=1024)
            else
                println("Skipping unrecognized file format: $f")
            end
        end
    end
else
    println("Directory '$output_dir' not found. Please ensure llm_outputs is in the repository root.")
end