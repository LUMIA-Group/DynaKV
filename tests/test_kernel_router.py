import unittest

from dynakv.kernels.router import KernelPolicy


class KernelPolicyTest(unittest.TestCase):
    def test_default_kernel_boundary(self):
        policy = KernelPolicy()
        self.assertEqual(policy.select(0), "short_fused")
        self.assertEqual(policy.select(499), "short_fused")
        self.assertEqual(policy.select(500), "long_two_stage")

    def test_custom_kernel_boundary(self):
        policy = KernelPolicy(switch_tokens=128)
        self.assertEqual(policy.select(127), "short_fused")
        self.assertEqual(policy.select(128), "long_two_stage")

    def test_negative_cache_length_is_rejected(self):
        with self.assertRaises(ValueError):
            KernelPolicy().select(-1)


if __name__ == "__main__":
    unittest.main()
