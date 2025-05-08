using CUDA, LinearAlgebra, Test

TILE = 32
TOL  = 1e-14

# --------------- problem inventory (N,R,M) ---------------------------------
shapes = [
    (16,   16,   16),      # tiny square
    (30,   15,   20),      # very skinny-tall
    (32,   48,   24),      # tall × skinny   (not a power of 2)
    (64,   32,   48),
    (96,   96,   96),      # “odd” square
    (128,   8,   64),
    (256,  64,   32),
    (300, 128,  200),
    (512, 512,  512),
    (1024,512,  256)
]

alphas   = [1.0,  2.0, -0.5]
transVal = ('N','T')

@testset "matmul! — C += α * op(A) * op(B)" begin
    for (N,R,M) in shapes
        # -------------------------------------------------------------------
        # create *physical* matrices that are large enough for any transpose
        # -------------------------------------------------------------------
        A0 = rand(Float64, N, R)      #  N×R   (before transpose)
        B0 = rand(Float64, R, M)      #  R×M

        for tA in transVal, tB in transVal, α in alphas
            # ---------------------------------------------------------------
            # determine the logical matrices after the transpose flags
            # ---------------------------------------------------------------
            A = tA == 'N' ?  A0      :  Transpose(A0)
            B = tB == 'N' ?  B0      :  Transpose(B0)

            # dimensions after transpose
            rowsA, colsA = size(A)
            rowsB, colsB = size(B)

            # the kernel is written for **result size N×M**, so we require:
            legal =  (colsA == rowsB) && (rowsA == N) && (colsB == M)
            legal || continue                     # skip incompatible combo

            @testset "N=$N R=$R M=$M   α=$α  tA=$tA tB=$tB" begin
                # ---------------- GPU buffers --------------------------------
                Ag = CuArray(parent(A))
                Bg = CuArray(parent(B))


                Arows = tA == 'N' ? size(A0, 1) : size(A0, 2)
                Acols = tA == 'N' ? size(A0, 2) : size(A0, 1)
                Bcols = tB == 'N' ? size(B0, 2) : size(B0, 1)

                N, R, M = Arows, Acols, Bcols  # logical matrix sizes

                
                Cg = CuArray(zeros(Float64, N, M))

                # ---------------- kernel launch ------------------------------
                backend = get_backend(Ag)
                matmul!(backend, (TILE, TILE))(
                    Cg, Ag, Bg, N, R, M,
                    α, tA, tB;
                    ndrange = (ceil(Int, N / TILE) * TILE,
                               ceil(Int, M / TILE) * TILE))

                CUDA.synchronize()

                # ---------------- reference & check --------------------------
                Cref = α * (A * B)
                denom = max(norm(Cref), eps())     # avoid divide-by-zero
                err   = norm(Matrix(Cg) - Cref) / denom

                @test err < TOL
            end
        end
    end
end
