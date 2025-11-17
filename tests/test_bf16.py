import torch
import random

import deep_gemm
from deep_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes
)
from generators import (
    get_arch_major,
    enumerate_normal, enumerate_m_grouped_contiguous, enumerate_m_grouped_masked, generate_normal,
    generate_m_grouped_contiguous, generate_m_grouped_masked,
    enumerate_k_grouped_contiguous_bf16, generate_k_grouped_contiguous_bf16
)


def test_gemm() -> None:
    print('Testing GEMM:')
    for kernel_type, m, n, k, major_a, major_b, accumulate, out_dtype in enumerate_normal(torch.bfloat16):
        # TODO: support accumulation for SM90 BF16 GEMM 
        if get_arch_major() == 9 and accumulate:
            continue

        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'

        for test_alias in (False, True):
            a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_bf16=True)
            func_name = f'bf16_gemm_{major_opt.lower() if test_alias else "nt"}'
            if test_alias:
                a = a if major_a.is_k_major() else a.T
                b = b if major_b.is_k_major() else b.T
                assert a.is_contiguous() and b.is_contiguous()
            getattr(deep_gemm, func_name)(a, b, d, c=c)
            diff = calc_diff(d, ref_d)
            assert diff < 0.0001, (f'{m=}, {n=}, {k=}, {major_opt=}, {accumulate=}, {out_dtype=}, '
                                   f'{diff:.5f}, alias={test_alias}')
        a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_bf16=True)

        t = bench_kineto(lambda: deep_gemm.bf16_gemm_nt(a, b, d, c=c), 'bf16_gemm', suppress_kineto_output=True)
        cublas_t, split_k_t = bench_kineto(lambda: deep_gemm.cublaslt_gemm_nt(a, b, d, c=c), ('nvjet', 'reduce'), suppress_kineto_output=True)
        print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, layout={major_opt}, {out_opt}, {acc_opt}): '
              f'{t * 1e6:5.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s | '
              f'{(cublas_t + split_k_t) / t:.2f}x cuBLAS')
    print()


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing m-grouped contiguous GEMM:')

    for _, num_groups, expected_m_per_group, n, k, major_a, major_b in enumerate_m_grouped_contiguous(torch.bfloat16):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'

        for test_alias in (False, True):
            m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b, use_bf16=True)
            func_name = f"m_grouped_bf16_gemm_{(major_opt.lower() if test_alias else 'nt')}_contiguous"
            if test_alias:
                assert major_a.is_k_major()
                b = b if major_b.is_k_major() else b.mT
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(a, b, d, m_indices)
            d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
            diff = calc_diff(d, ref_d)
            assert diff < 0.001, f'{m=}, {n=}, {k=}, {major_opt}, {kernel_opt}, {diff:.5f}, alias={test_alias}'
        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b, use_bf16=True)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices)

        t = bench_kineto(test_func, 'bf16_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}, layout={major_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for _, num_groups, max_m, expected_m_per_group, n, k in enumerate_m_grouped_masked(torch.bfloat16):
        # Test correctness
        for i in range(10):
            a, b, masked_m, d, ref_d = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k, use_bf16=True)
            deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group)
            for j in range(num_groups):
                diff = calc_diff(d[j, :masked_m[j].item()], ref_d[j, :masked_m[j].item()])
                assert diff < 0.001, f'{m=}, {n=}, {k=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'

        # Construct full cases
        a, b, masked_m, d, ref_d = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k, use_bf16=True)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group)

        # Test performance with fixed shapes
        valid_m = masked_m.sum().item()
        t = bench_kineto(test_func, 'bf16_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)) / 1e9 / t:4.0f} GB/s')
    print()


def test_k_grouped_gemm_contiguous() -> None:
    print('Testing k-grouped contiguous GEMM:')

    for num_groups, m, n, ks, expected_k_per_group in enumerate_k_grouped_contiguous_bf16():
        # Test without accumulation (BF16 output)
        for accumulate in (False, True):
            k, a, b, c, d, ref_d = generate_k_grouped_contiguous_bf16(num_groups, m, n, ks, accumulate)
            ks_tensor = torch.tensor(ks, dtype=torch.int, device='cuda')

            # Debug: verify c and d are the same object when accumulate=True
            if accumulate:
                assert c is not None, "c should not be None when accumulate=True"
                assert d is c, f"d must be the same object as c when accumulate=True, but d is c: {d is c}"
                assert c.data_ptr() == d.data_ptr(), f"c and d must have same data_ptr"

            deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks, ks_tensor, c)

            diff = calc_diff(d, ref_d)
            out_dtype = 'FP32' if accumulate else 'BF16'
            assert diff < 0.001, f'{m=}, {n=}, k={k}, {num_groups=}, {out_dtype=}, {diff:.5f}'

        # Benchmark with FP32 accumulation
        k, a, b, c, d, ref_d = generate_k_grouped_contiguous_bf16(num_groups, m, n, ks, accumulate=True)
        ks_tensor = torch.tensor(ks, dtype=torch.int, device='cuda')

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks, ks_tensor, c)

        t = bench_kineto(test_func, 'bf16_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=:3}, m={m:5}, n={n:5}, k={k:6}, expected_k={expected_k_per_group:5}): '
              f'{t * 1e6:5.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_cublaslt_gemm() -> None:
    print('Testing cuBLASLt GEMM:')
    for kernel_type, m, n, k, major_a, major_b, accumulate, out_dtype in enumerate_normal(dtype=torch.bfloat16):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'

        a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_bf16=True)
        deep_gemm.cublaslt_gemm_nt(a, b, d, c=c)
        diff = calc_diff(d, ref_d)
        assert diff < 5e-7, f'{diff=}, ({m=}, {n=}, {k=}, {major_opt=}, {accumulate=}, {out_dtype=})'

        t = bench_kineto(lambda: deep_gemm.cublaslt_gemm_nt(a, b, d, c=c), 'nvjet', suppress_kineto_output=True,)
        print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, layout={major_opt}, {out_opt}, {acc_opt}): '
              f'{t * 1e6:5.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm()
    # TODO: support SM100
    if get_arch_major() == 9:
        test_m_grouped_gemm_contiguous()
        test_m_grouped_gemm_masked()

    # K-grouped GEMM supports both SM90 and SM100
    test_k_grouped_gemm_contiguous()

    test_cublaslt_gemm()
