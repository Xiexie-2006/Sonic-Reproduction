import os
import sys
import time
import subprocess


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = sys.executable

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CORE_TESTS = [
    {
        "name": "OT Extension",
        "server1": "run/test_ot_extension_server1.py",
        "server0": "run/test_ot_extension_server0.py",
        "expect": "PASSED"
    },
    {
        "name": "OT Boolean Triple",
        "server1": "run/test_ot_bit_triple_server1.py",
        "server0": "run/test_ot_bit_triple_server0.py",
        "expect": "PASSED"
    },
    {
        "name": "OT Arithmetic Triple",
        "server1": "run/test_ot_arith_triple_server1.py",
        "server0": "run/test_ot_arith_triple_server0.py",
        "expect": "PASSED"
    },
    {
        "name": "SReLU",
        "server1": "run/test_srelu_server1.py",
        "server0": "run/test_srelu_server0.py",
        "expect": "SReLU test PASSED"
    },
    {
        "name": "Secret Linear",
        "server1": "run/test_secret_linear_server1.py",
        "server0": "run/test_secret_linear_server0.py",
        "expect": "Secret Linear test PASSED"
    },
    {
        "name": "Secret MLP",
        "server1": "run/test_secret_mlp_server1.py",
        "server0": "run/test_secret_mlp_server0.py",
        "expect": "Secret MLP test PASSED"
    },
    {
        "name": "Fixed-point Trunc MLP",
        "server1": "run/test_fixed_trunc_mlp_server1.py",
        "server0": "run/test_fixed_trunc_mlp_server0.py",
        "expect": "Fixed-point Trunc MLP test PASSED"
    },
    {
        "name": "PyTorch MLP",
        "server1": "run/test_torch_mlp_server1.py",
        "server0": "run/test_torch_mlp_server0.py",
        "expect": "PyTorch MLP vs MPC MLP test PASSED"
    },
    {
        "name": "Fixed-point Conv2D",
        "server1": "run/test_fixed_conv2d_server1.py",
        "server0": "run/test_fixed_conv2d_server0.py",
        "expect": "Fixed-point Secret Conv2D test PASSED"
    },
    {
        "name": "Fixed-point CNN",
        "server1": "run/test_fixed_cnn_server1.py",
        "server0": "run/test_fixed_cnn_server0.py",
        "expect": "Fixed-point Secret CNN test PASSED"
    },
    {
        "name": "PyTorch CNN",
        "server1": "run/test_torch_cnn_server1.py",
        "server0": "run/test_torch_cnn_server0.py",
        "expect": "PyTorch CNN vs MPC CNN test PASSED"
    },
]


EXTENDED_TESTS = [
    {
        "name": "CNN Accuracy",
        "server1": "run/test_cnn_accuracy_server1.py",
        "server0": "run/test_cnn_accuracy_server0.py",
        "expect": "CNN accuracy pipeline test PASSED"
    },
    {
        "name": "Dataset CNN Accuracy",
        "pre_script": "run/make_toy_dataset.py",
        "server1": "run/test_cnn_accuracy_dataset_server1.py",
        "server0": "run/test_cnn_accuracy_dataset_server0.py",
        "expect": "CNN dataset accuracy test PASSED"
    },
    {
        "name": "Secure MaxPool2D",
        "server1": "run/test_maxpool2d_server1.py",
        "server0": "run/test_maxpool2d_server0.py",
        "expect": "Secure MaxPool2D test PASSED"
    },
    {
        "name": "Fixed-point CNN with MaxPool2D",
        "server1": "run/test_fixed_cnn_pool_server1.py",
        "server0": "run/test_fixed_cnn_pool_server0.py",
        "expect": "Fixed-point Secret CNN with MaxPool2D test PASSED"
    },
    {
        "name": "Secure AvgPool2D",
        "server1": "run/test_avgpool2d_server1.py",
        "server0": "run/test_avgpool2d_server0.py",
        "expect": "Secure AvgPool2D test PASSED"
    },
    {
        "name": "Secure Argmax",
        "server1": "run/test_secure_argmax_server1.py",
        "server0": "run/test_secure_argmax_server0.py",
        "expect": "Secure Argmax test PASSED"
    },
    {
        "name": "CNN Secure Argmax Accuracy",
        "pre_script": "run/make_toy_dataset.py",
        "server1": "run/test_cnn_secure_argmax_server1.py",
        "server0": "run/test_cnn_secure_argmax_server0.py",
        "expect": "CNN secure argmax accuracy test PASSED"
    },
    {
        "name": "Secure BatchNorm / SBN",
        "server1": "run/test_sbn_server1.py",
        "server0": "run/test_sbn_server0.py",
        "expect": "Secure BatchNorm / SBN test PASSED"
    },
    {
        "name": "Fixed-point CNN with SBN",
        "server1": "run/test_fixed_cnn_sbn_server1.py",
        "server0": "run/test_fixed_cnn_sbn_server0.py",
        "expect": "Fixed-point Secret CNN with SBN test PASSED"
    },
    {
        "name": "Sonic Optimized CNN SBN-MaxPool-SReLU",
        "server1": "run/test_fixed_cnn_sbn_pool_opt_server1.py",
        "server0": "run/test_fixed_cnn_sbn_pool_opt_server0.py",
        "expect": "Sonic optimized CNN SBN-MaxPool-SReLU test PASSED"
    },
]


PAPER_MODEL_TESTS = [
    {
        "name": "Sonic M1 Paper Model MPC",
        "server1": "run/test_m1_mpc_server1.py",
        "server0": "run/test_m1_mpc_server0.py",
        "expect_any": [
            "Sonic M1 MPC functional test PASSED"
        ]
    },
    {
        "name": "Sonic M2 Paper Model MPC",
        "server1": "run/test_m2_mpc_server1.py",
        "server0": "run/test_m2_mpc_server0.py",
        "expect_any": [
            "Sonic M2 MPC functional test PASSED"
        ]
    },
    {
        "name": "Sonic C1 Paper Model MPC",
        "server1": "run/test_c1_mpc_server1.py",
        "server0": "run/test_c1_mpc_server0.py",
        "expect_any": [
            "Sonic C1 MPC functional test PASSED",
            "Sonic C1 MPC structure/logits test PASSED"
        ]
    },
    {
        "name": "Sonic C2 Paper Model MPC",
        "server1": "run/test_c2_mpc_server1.py",
        "server0": "run/test_c2_mpc_server0.py",
        "expect_any": [
            "Sonic C2 MPC functional test PASSED",
            "Sonic C2 MPC structure/logits test PASSED"
        ]
    },
]


BENCHMARK_TESTS = [
    {
        "name": "MLP Batch Benchmark",
        "server1": "run/benchmark_batch_server1.py",
        "server0": "run/benchmark_batch_server0.py",
        "expect": "Benchmark CSV saved"
    },
    {
        "name": "CNN Batch Benchmark",
        "server1": "run/benchmark_torch_cnn_server1.py",
        "server0": "run/benchmark_torch_cnn_server0.py",
        "expect": "CNN benchmark CSV saved"
    },
]


SUMMARY_TESTS = [
    {
        "name": "MLP Benchmark Markdown Summary",
        "script": "run/make_benchmark_summary.py",
        "expect": "Markdown summary saved"
    },
    {
        "name": "CNN Benchmark Markdown Summary",
        "script": "run/make_cnn_benchmark_summary.py",
        "expect": "CNN benchmark markdown summary saved"
    },
]


def make_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    old_pythonpath = env.get("PYTHONPATH", "")

    if old_pythonpath:
        env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + old_pythonpath
    else:
        env["PYTHONPATH"] = PROJECT_ROOT

    return env


def abs_path(relative_path):
    return os.path.join(PROJECT_ROOT, relative_path)


def terminate_process(proc):
    if proc is None:
        return

    if proc.poll() is not None:
        return

    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def check_expected(test, stdout):
    if "expect_any" in test:
        return any(item in stdout for item in test["expect_any"])

    if "expect" in test and test["expect"]:
        return test["expect"] in stdout

    return True


def run_single_script(test, timeout=180):
    name = test["name"]
    script_path = abs_path(test["script"])

    print("\n" + "=" * 80)
    print(f"[RUN] {name}")
    print("=" * 80)

    try:
        result = subprocess.run(
            [PYTHON, script_path],
            cwd=PROJECT_ROOT,
            env=make_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print("[STDERR]")
            print(result.stderr)

        ok = result.returncode == 0 and check_expected(test, result.stdout)

        if ok:
            print(f"[PASS] {name}")
        else:
            print(f"[FAIL] {name}")

        return ok

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {name}")
        return False

    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        return False


def run_pre_script(script_path):
    test = {
        "name": f"Pre-script: {script_path}",
        "script": script_path,
        "expect": ""
    }
    return run_single_script(test, timeout=120)


def run_pair_test(test, timeout=420):
    name = test["name"]
    server1_path = abs_path(test["server1"])
    server0_path = abs_path(test["server0"])

    print("\n" + "=" * 80)
    print(f"[RUN] {name}")
    print("=" * 80)

    if test.get("pre_script"):
        pre_ok = run_pre_script(test["pre_script"])
        if not pre_ok:
            print(f"[FAIL] {name}: pre_script failed")
            return False

    server1 = None

    try:
        server1 = subprocess.Popen(
            [PYTHON, server1_path],
            cwd=PROJECT_ROOT,
            env=make_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        time.sleep(1.5)

        result = subprocess.run(
            [PYTHON, server0_path],
            cwd=PROJECT_ROOT,
            env=make_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print("[SERVER0 STDERR]")
            print(result.stderr)

        ok = result.returncode == 0 and check_expected(test, result.stdout)

        if ok:
            print(f"[PASS] {name}")
        else:
            print(f"[FAIL] {name}")

            if server1 is not None:
                try:
                    s1_out, s1_err = server1.communicate(timeout=1)
                    if s1_out:
                        print("[SERVER1 STDOUT]")
                        print(s1_out)
                    if s1_err:
                        print("[SERVER1 STDERR]")
                        print(s1_err)
                except Exception:
                    pass

        return ok

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {name}")
        return False

    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        return False

    finally:
        terminate_process(server1)
        time.sleep(0.8)


def run_pair_suite(tests, timeout):
    passed = 0
    failed = 0

    for test in tests:
        ok = run_pair_test(test, timeout=timeout)

        if ok:
            passed += 1
        else:
            failed += 1

    return passed, failed


def run_single_suite(tests, timeout):
    passed = 0
    failed = 0

    for test in tests:
        ok = run_single_script(test, timeout=timeout)

        if ok:
            passed += 1
        else:
            failed += 1

    return passed, failed


def print_usage():
    print("Usage:")
    print("  python run/run_all_tests.py core")
    print("  python run/run_all_tests.py extended")
    print("  python run/run_all_tests.py functional")
    print("  python run/run_all_tests.py paper")
    print("  python run/run_all_tests.py final")
    print("  python run/run_all_tests.py benchmark")
    print("  python run/run_all_tests.py summary")
    print("  python run/run_all_tests.py all")
    print("")
    print("Modes:")
    print("  core       : original core protocol / MLP / CNN tests")
    print("  extended   : SBN / Pooling / Argmax / dataset accuracy tests")
    print("  functional : core + extended")
    print("  paper      : M1 / M2 / C1 / C2 paper model MPC tests")
    print("  final      : functional + paper")
    print("  benchmark  : MLP and CNN benchmark")
    print("  summary    : generate markdown summaries")
    print("  all        : functional + paper + benchmark + summary")


def main():
    mode = "functional"

    if len(sys.argv) >= 2:
        mode = sys.argv[1].lower()

    valid_modes = [
        "core",
        "quick",
        "extended",
        "functional",
        "paper",
        "final",
        "benchmark",
        "summary",
        "all"
    ]

    if mode not in valid_modes:
        print_usage()
        return

    if mode == "quick":
        mode = "core"

    total_passed = 0
    total_failed = 0

    if mode in ["core", "functional", "final", "all"]:
        passed, failed = run_pair_suite(CORE_TESTS, timeout=420)
        total_passed += passed
        total_failed += failed

    if mode in ["extended", "functional", "final", "all"]:
        passed, failed = run_pair_suite(EXTENDED_TESTS, timeout=600)
        total_passed += passed
        total_failed += failed

    if mode in ["paper", "final", "all"]:
        passed, failed = run_pair_suite(PAPER_MODEL_TESTS, timeout=2400)
        total_passed += passed
        total_failed += failed

    if mode in ["benchmark", "all"]:
        passed, failed = run_pair_suite(BENCHMARK_TESTS, timeout=1200)
        total_passed += passed
        total_failed += failed

    if mode in ["summary", "all"]:
        passed, failed = run_single_suite(SUMMARY_TESTS, timeout=180)
        total_passed += passed
        total_failed += failed

    print("\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    print(f"MODE   : {mode}")
    print(f"PASSED : {total_passed}")
    print(f"FAILED : {total_failed}")

    if total_failed == 0:
        print("ALL SELECTED TESTS PASSED ✅")
    else:
        print("SOME TESTS FAILED ❌")

    print("=" * 80)


if __name__ == "__main__":
    main()