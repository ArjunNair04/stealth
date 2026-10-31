    1af8:	4b 8d 04 0a          	lea    (%r10,%r9,1),%rax
    1afc:	49 8d 34 b8          	lea    (%r8,%rdi,4),%rsi
    1b00:	31 c9                	xor    %ecx,%ecx
    1b02:	48 89 84 24 f0 00 00 	mov    %rax,0xf0(%rsp)
    1b09:	00
    1b0a:	48 39 e9             	cmp    %rbp,%rcx
    1b0d:	0f 84 99 00 00 00    	je     1bac <forward_batch+0x5bc>
    1b13:	48 8b 84 24 f0 00 00 	mov    0xf0(%rsp),%rax
    1b1a:	00
    1b1b:	31 d2                	xor    %edx,%edx
    1b1d:	66 0f ef c0          	pxor   %xmm0,%xmm0
    1b21:	66 0f ef c9          	pxor   %xmm1,%xmm1
    1b25:	f3 0f 5a 0e          	cvtss2sd (%rsi),%xmm1
    1b29:	49 f7 f7             	div    %r15
    1b2c:	48 89 d0             	mov    %rdx,%rax
    1b2f:	49 8b 96 28 02 00 00 	mov    0x228(%r14),%rdx
    1b36:	48 0f af c5          	imul   %rbp,%rax
    1b3a:	48 01 c8             	add    %rcx,%rax
    1b3d:	f3 0f 5a 04 82       	cvtss2sd (%rdx,%rax,4),%xmm0
    1b42:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
    1b46:	66 0f 28 c8          	movapd %xmm0,%xmm1
    1b4a:	66 0f 54 0d 00 00 00 	andpd  0x0(%rip),%xmm1        # 1b52 <forward_batch+0x562>
    1b51:	00
    1b52:	66 0f 2e d9          	ucomisd %xmm1,%xmm3
    1b56:	72 20                	jb     1b78 <forward_batch+0x588>
    1b58:	66 0f 2f 0d 00 00 00 	comisd 0x0(%rip),%xmm1        # 1b60 <forward_batch+0x570>
    1b5f:	00
    1b60:	77 16                	ja     1b78 <forward_batch+0x588>
    1b62:	f2 0f 5a c0          	cvtsd2ss %xmm0,%xmm0
    1b66:	48 83 c1 01          	add    $0x1,%rcx
    1b6a:	f3 0f 11 06          	movss  %xmm0,(%rsi)
    1b6e:	48 83 c6 04          	add    $0x4,%rsi
    1b72:	eb 96                	jmp    1b0a <forward_batch+0x51a>
    1b74:	0f 1f 40 00          	nopl   0x0(%rax)
