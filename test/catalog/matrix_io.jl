using MatrixMarket, SparseArrays, LinearAlgebra

function load_input(path, expected_n)
    raw = MatrixMarket.mmread(path)
    size(raw) == (expected_n, expected_n) || error("Dimension does not match catalog: $(size(raw)) vs $expected_n")
    A = sparse(Float64.(raw))
    # Symmetric headers may already have been expanded by MatrixMarket.
    # A general header with exactly one stored triangle needs mirroring once.
    if !issymmetric(A)
        if istril(A)
            A = sparse(Symmetric(A, :L))
        elseif istriu(A)
            A = sparse(Symmetric(A, :U))
        else
            error("Input has inconsistent triangles; refusing to average or shift it")
        end
    end
    dropzeros!(A)
    all(isfinite, nonzeros(A)) || error("Nonfinite input")
    A
end
