    1ad0:	31 ff                	xor    %edi,%edi
    1ad2:	31 f6                	xor    %esi,%esi
    1ad4:	4c 8b 8b 28 02 00 00 	mov    0x228(%rbx),%r9
    1adb:	f2 0f 10 1d 00 00 00 	movsd  0x0(%rip),%xmm3        # 1ae3 <forward_batch+0x4f3>
    1ae2:	00
    1ae3:	4c 89 d8             	mov    %r11,%rax
    1ae6:	4c 8b 94 24 f8 00 00 	mov    0xf8(%rsp),%r10
    1aed:	00
    1aee:	f3 0f 7e 15 00 00 00 	movq   0x0(%rip),%xmm2        # 1af6 <forward_batch+0x506>
    1af5:	00
    1af6:	49 f7 f6             	div    %r14
    1af9:	4c 8b 44 24 60       	mov    0x60(%rsp),%r8
    1afe:	48 8b 84 24 80 00 00 	mov    0x80(%rsp),%rax
    1b05:	00
    1b06:	4c 0f af c2          	imul   %rdx,%r8
    1b0a:	48 8d 0c 38          	lea    (%rax,%rdi,1),%rcx
    1b0e:	31 c0                	xor    %eax,%eax
    1b10:	4d 01 c8             	add    %r9,%r8
    1b13:	48 39 e8             	cmp    %rbp,%rax
    1b16:	74 7c                	je     1b94 <forward_batch+0x5a4>
    1b18:	66 0f ef c0          	pxor   %xmm0,%xmm0
    1b1c:	66 0f ef c9          	pxor   %xmm1,%xmm1
    1b20:	f3 0f 5a 09          	cvtss2sd (%rcx),%xmm1
    1b24:	f3 41 0f 5a 04 80    	cvtss2sd (%r8,%rax,4),%xmm0
    1b2a:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
    1b2e:	66 0f 28 c8          	movapd %xmm0,%xmm1
    1b32:	66 0f 54 0d 00 00 00 	andpd  0x0(%rip),%xmm1        # 1b3a <forward_batch+0x54a>
    1b39:	00
    1b3a:	66 0f 2e d9          	ucomisd %xmm1,%xmm3
    1b3e:	72 20                	jb     1b60 <forward_batch+0x570>
    1b40:	66 0f 2f 0d 00 00 00 	comisd 0x0(%rip),%xmm1        # 1b48 <forward_batch+0x558>
    1b47:	00
    1b48:	77 16                	ja     1b60 <forward_batch+0x570>
    1b4a:	f2 0f 5a c0          	cvtsd2ss %xmm0,%xmm0
    1b4e:	48 83 c0 01          	add    $0x1,%rax
    1b52:	f3 0f 11 01          	movss  %xmm0,(%rcx)
    1b56:	48 83 c1 04          	add    $0x4,%rcx
    1b5a:	eb b7                	jmp    1b13 <forward_batch+0x523>
    1b5c:	0f 1f 40 00          	nopl   0x0(%rax)
    1b60:	41 bc 05 00 00 00    	mov    $0x5,%r12d
    1b66:	e9 39 fb ff ff       	jmpq   16a4 <forward_batch+0xb4>
    1b6b:	48 8b 54 24 20       	mov    0x20(%rsp),%rdx
    1b70:	4c 8d 04 95 00 00 00 	lea    0x0(,%rdx,4),%r8
    1b77:	00
    1b78:	4c 89 d2             	mov    %r10,%rdx
    1b7b:	48 2b 54 24 08       	sub    0x8(%rsp),%rdx
    1b80:	49 39 d0             	cmp    %rdx,%r8
    1b83:	0f 86 29 fc ff ff    	jbe    17b2 <forward_batch+0x1c2>
    1b89:	41 bc 03 00 00 00    	mov    $0x3,%r12d
    1b8f:	e9 10 fb ff ff       	jmpq   16a4 <forward_batch+0xb4>
    1b94:	48 83 c2 01          	add    $0x1,%rdx
    1b98:	49 39 d6             	cmp    %rdx,%r14
    1b9b:	75 02                	jne    1b9f <forward_batch+0x5af>
    1b9d:	31 d2                	xor    %edx,%edx
    1b9f:	48 8d 46 01          	lea    0x1(%rsi),%rax
    1ba3:	48 03 7c 24 60       	add    0x60(%rsp),%rdi
    1ba8:	49 39 f2             	cmp    %rsi,%r10
    1bab:	0f 85 fe 03 00 00    	jne    1faf <forward_batch+0x9bf>
    1bb1:	49 83 c3 20          	add    $0x20,%r11
    1bb5:	48 8b bc 24 b8 00 00 	mov    0xb8(%rsp),%rdi

# Row-loop continuation
    1faf:	48 89 c6             	mov    %rax,%rsi
    1fb2:	e9 42 fb ff ff       	jmpq   1af9 <forward_batch+0x509>
