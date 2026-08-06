import unittest

from experiment_configs import TABLE_III_CONFIGS, TRAINING_METHODS


class ExperimentConfigTests(unittest.TestCase):
    """防止后续改动意外删掉或耦合论文的消融配置。"""
    def test_all_training_methods_exist(self):
        self.assertEqual(set(TRAINING_METHODS), {"nominal", "reward_only", "filter_only", "dual"})

    def test_table_iii_has_twelve_unique_configs(self):
        self.assertEqual(len(TABLE_III_CONFIGS), 12)
        self.assertEqual(len(set(TABLE_III_CONFIGS)), 12)

    def test_runtime_filter_is_independent_of_training_filter(self):
        self.assertFalse(TABLE_III_CONFIGS["dual_no_runtime_filter"].use_runtime_filter)
        self.assertTrue(TABLE_III_CONFIGS["filter_only_runtime_filter"].use_runtime_filter)


if __name__ == "__main__":
    unittest.main()
