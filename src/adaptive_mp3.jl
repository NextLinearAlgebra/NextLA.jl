export adaptive_precision_LT
# computes the approximated working precsion of each level of the 
#   matrix.
# Will be used to store the matrix blocks in a lower precision 
# Assumption is the matrix is lower triangular: Will be genrealized

using LinearAlgebra # for the inbuilt Frobenius norm function

# Function to select the adaptive precision at each level of the triangular matrix
# A : Lower Triangular matrix
# n_min : minimal diagonal block size, proxy for the number of levels
# U : vector of the available precision levels expressed as the unit roundoff errors
    # 1: q52
    # 2: bf16
    # 3: f16
    # 4: f32
    # 5: f64

    # used to compute the machine precision levels
    # HAS TO BE SORTED IN ASCENDING ORDER
# returns the adaptive precision u_k for each level of the matrix blocks k

# also returns the precision matrix blocks for debugging
function adaptive_precision_LT(A, U = [4,5], n_min = 4, epsilon = 1e-8)
    
    # creating the array of roundoff errors (working precision)
    u16 = eps(Float16)/2
    u32 = eps(Float32)/2
    u64 = eps(Float64)/2

    # Hardcoding the rest for convenience (will change later, or create another module)
    q52 = 1.25e-1 # quarter precision
    bf16 = 3.91e-3 # bfloat16


    U_all = Float64[q52, bf16, u16, u32, u64] # Full precision set
    U_string = ["q52", "bf16", "f16", "f32", "f64"]

    U_set = [] # will hold the user specified precision levels
    U_str = []

    for i in U
        push!(U_set, U_all[i])
        push!(U_str, U_string[i])
    end

    u_approx = [] # hold the approximated precision levels

    norm_A = norm(A)

    N = size(A, 1)
    @assert N == size(A, 2) "Matrix must be triangular in shape (square dimensions)"

    vertices = [[(1, 1), (N, N)]]  # [(top-left, bottom-right)] coordinates

    half_length = N
    k = 0 # the current level

    while ((half_length/2) >= n_min)
        #partition the submatrices
        new_vertices_diag = []
        frob_norms_level = []
        half_length = Int(ceil(half_length/2))

        for vertex in vertices

            # diagonal submatrices
            top_left_vertices = [[vertex[1][1], vertex[1][2]], [vertex[1][1] + half_length-1,vertex[1][2] + half_length-1]]
            bottom_right_vertices = [[vertex[1][1] + half_length, vertex[1][2] + half_length], [vertex[2][1] ,vertex[2][2]]]

            # off diagonal submatrices
            off_diag_left = [[vertex[1][1] + half_length, vertex[1][2]], [vertex[2][1], vertex[1][2] + half_length -1]]
            # off_diag_right = [[vertex[1][1], vertex[1][2]+ half_length], [vertex[1][1]+half_length-1, vertex[2][2]]]
            

            off_diag_norm_left = norm(@view(A[off_diag_left[1][1] : off_diag_left[2][1], off_diag_left[1][2] : off_diag_left[2][2]]))
            # off_diag_norm_right = norm(@view(A[off_diag_right[1][1] : off_diag_right[2][1], off_diag_right[1][2] : off_diag_right[2][2]]))


            #result[off_diag_left[1][1] : off_diag_left[2][1], off_diag_left[1][2] : off_diag_left[2][2]] .= off_diag_norm_left
            #result[off_diag_right[1][1] : off_diag_right[2][1], off_diag_right[1][2] : off_diag_right[2][2]] .= off_diag_norm_right

            push!(frob_norms_level, off_diag_norm_left)
            
            push!(new_vertices_diag, top_left_vertices, bottom_right_vertices)
        end

        # find the maximum norm at the level
        # then change the elements on this level to the maximum norm in the result matrix 
        level_norm = maximum(frob_norms_level)
        
        E_k = level_norm/norm_A
        u_work = epsilon/((2^((k+1)/2))*E_k)

        i = 1
        while ((u_work < U_set[i]) && (i < size(U, 1)))
            i += 1
        end

        push!(u_approx, U_str[i])


        vertices = new_vertices_diag
        k = k+1
        
    end

    return u_approx

end