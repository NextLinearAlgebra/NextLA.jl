IndexGPUArray{T} = Union{AbstractGPUArray{T},
                               SubArray{T, <:Any, <:AbstractGPUArray},
                               LinearAlgebra.Adjoint{T}, 
                               SubArray{T, <:Any, <:LinearAlgebra.Adjoint{T, <:AnyGPUArray }}}


function LinearAlgebra.triu!(A::IndexGPUArray{T}, d::Integer = 0) where T
    @kernel cpu=false  inbounds=true unsafe_indices=false function triu_kernel!(_A, _d)
      I = @index(Global, Cartesian)
      i, j = Tuple(I)
      if j < i + _d
        _A[i, j] = zero(T)
      end
    end
    triu_kernel!(get_backend(A))(A, d; ndrange = size(A))
    return A
  end

function LinearAlgebra.tril!(A::IndexGPUArray{T}, d::Integer = 0) where T
    @kernel cpu=false  inbounds=true unsafe_indices=false function tril_kernel!(_A, _d)
      I = @index(Global, Cartesian)
      i, j = Tuple(I)
      if j + _d >  i 
        _A[i, j] = zero(T)
      end
    end
    tril_kernel!(get_backend(A))(A, d; ndrange = size(A))
    return A
  end

  function myAminB!(A::IndexGPUArray{T}, B::IndexGPUArray{T}) where T
	  @assert size(A) == size(B)
  @kernel cpu=false  inbounds=true unsafe_indices=false function myAminB_kernel!(_A, _B)
      I = @index(Global, Cartesian)
      i, j = Tuple(I)

      for i in (1:32).+(i-1)*32
      	if (i<=size(_A,1) && j<=size(_A,2))
      		_A[i, j] = _A[i,j]-_B[i,j]
      	end
      end
    end
    myAminB_kernel!(get_backend(A),128)(A, B; ndrange = (cld(size(A,1),32),size(A,2)))
    return A
  end


  function Base.fill!(A::IndexGPUArray{T}, x) where T
    isempty(A) && return A

    @kernel cpu=false  inbounds=true unsafe_indices=false function fill_kernel!(a, val)
        idx = @index(Global, Linear)
        a[idx] = val
    end

    # ndims check for 0D support
    kernel = fill_kernel!(get_backend(A))
    kernel(A, x; ndrange = ndims(A) > 0 ? size(A) : (1,))
    A
end
