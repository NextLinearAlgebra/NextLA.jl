"""
	zunmqr(side, trans, m, n, k, ib, A, lda, T, ldt, C, ldc, work, ldwork)

Overwrites the general m-by-n tile C with
					side = 'L'		side = 'R'
	trans = 'N'		   Q*C				C*Q
	trans = 'C'		   Q^H*C			C*Q^H

where Q is a unitary matrix defined as the product of k elementary reflectors
	Q = H(1) H(2) ... H(k)
as returned by zgeqrt. Q is of order m if side = 'L" and of order n if side = 'R'

# Arguments
- 'side': 
	- = 'L': apply Q or Q^H from the left
	- = 'R': apply Q or Q^H from the right
- 'trans':
	- = 'N': no transpose, apply Q
	- = 'C': conjugate transpose, apply Q^H
- 'm': the number of rows of the tile C. m >= 0
- 'n': the number of columns of the tile C. n >= 0
- 'k': the number of elementary refelctors whose product defines the matrix Q
	- if side = 'L', m >= k >= 0
	- if side = 'R', n >= k >= 0
- 'ib': the inner blocking size. ib >= 0
- 'A': dimension (lda, k)
	the i-th column must contain the vector which defines the 
	elementary reflector H(i) for i = 1,2,...,k,
	as returned by zgeqrt in the first k columns of its array argument A
- 'lda': the leading dimension of array A
	if side = 'L', lda >= max(1,m)
	if side = 'R', lda >= max(1,n)
- 'T': the ib-by-k triangular factor T of the block reflector
	T is upper triangular by block (economic storage)
	The rest of the array is not referenced
- 'ldt': the elding dimension of the array T. ldt >= ib
- 'C': 
	On entry the m-by-n tile C
	On exit, C is overwritten by Q*C or Q^H*C or C*Q^H or C*Q.
-'work': auxillary workspace of array work
	ldwork-by-n if side = 'L'
	ldwork by ib if side = 'R'
- 'ldwork': the leading dimension of array work
	ldwork >= max(1,ib) if side = 'L'
	ldwork >= max(1,m) if side = 'R'
"""
function zunmqr(side, trans, m, n, k, ib, A, lda, T, ldt, C, ldc, work, ldwork)
	if side != 'L' && side != 'R'
        throw(ArgumentError("illegal value of side"))
		return -1
	end

	if side == 'L'
		nq = m
		nw = n
	else
		nq = n
		nw = m
	end

	if trans != 'N' && trans != 'C' && trans != 'T'
        throw(ArgumentError("illegal value of trans"))
		return -2
	end

	if m < 0
        throw(ArgumentError("illegal value of m"))
		return -3
	end

	if n < 0
        throw(ArgumentError("illegal value of n"))
		return -4
	end

	if k < 0 || k > nq
        throw(ArgumentError("illegal value of k"))
		return -5
	end

	if ib < 0 
        throw(ArgumentError("illegal value of ib"))
		return -6
	end

	if lda < max(1, nq) && nq > 0
        throw(ArgumentError("illegal value of lda"))
		return -8
	end

	if ldt < max(1,ib)
        throw(ArgumentError("illegal value of ldt"))
		return -10
	end

	if ldc < max(1,m) && m > 0
        throw(ArgumentError("illegal value of ldc"))
		return -12
	end

	if ldwork < max(1,nw) && nw > 0
        throw(ArgumentError("illegal value of ldwork"))
		return -14
	end

	# quick return 
	if m == 0 || n == 0 || k == 0
		return
	end

	if ((side == 'L' && trans != 'N') || (side == 'R' && trans == 'N'))
		i1 = 1
		i3 = ib
		ibstop = k
	else
		i1 = div((k-1),ib)*ib + 1
		i3 = -ib
		ibstop = 1
	end
	
	ic = 1
	jc = 1
	ni = n
	mi = m

	if side == 'L'
		wwork = ones(eltype(A), n, ib)
		ldw = n
	else
		wwork = ones(eltype(A), m, ib)
		ldw = m
	end

	for i in i1 : i3 : ibstop
		kb = min(ib, k-i+1)

		if side == 'L'
			# apply to C[i:m, 1:n]
			mi = m - i + 1
			ic = i
		else
			# apply to C[1:m, i:n]
			ni = n-i + 1
			jc = i
		end

        cv = @view C[ic:m, jc:n]

        zlarfb(side, trans, 'F', 'C', mi, ni, kb, (@view A[i:lda, i:i+kb-1]), lda-i+1, (@view T[1:kb, i:i+kb-1]), kb, cv, ldc, (@view wwork[:, 1:kb]), ldw)
	end
end
