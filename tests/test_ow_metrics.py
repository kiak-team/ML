"""code/ow_metrics.py 단위 테스트. 실행: python -I -m unittest discover -s tests -v  (레포 루트에서)"""
import math, os, sys, unittest

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))
from ow_metrics import (apply_mon_threshold, apply_threshold, fpr_threshold, mon_score,  # noqa: E402
                        ow_counts, ow_metrics, unmon_confusion)


class TestCounts(unittest.TestCase):
    def test_all_correct(self):
        y = [0, 1, 2, -1, -1]
        r = ow_metrics(y, y)
        self.assertEqual((r['TP'], r['FP'], r['FN'], r['wrong_site']), (3, 0, 0, 0))
        self.assertEqual((r['TPR'], r['FPR'], r['precision'], r['recall']), (1.0, 0.0, 1.0, 1.0))
        self.assertAlmostEqual(r['macro_f1'], 1.0)

    def test_mon_to_other_mon_is_not_tp(self):
        # 사이트 0을 사이트 1로: monitored로 판정했지만 사이트가 틀렸다 → TP 아님, wrong_site
        r = ow_metrics([0, 1], [1, 1])
        self.assertEqual(r['TP'], 1)
        self.assertEqual(r['wrong_site'], 1)
        self.assertEqual(r['FN'], 0)
        self.assertAlmostEqual(r['TPR'], 0.5)
        self.assertAlmostEqual(r['det_TPR'], 1.0)      # 탐지만 보면 둘 다 잡았다
        self.assertAlmostEqual(r['precision'], 0.5)    # wrong_site는 precision 분모에 들어간다

    def test_unmon_to_any_mon_is_fp(self):
        r = ow_metrics([-1, -1, -1, -1], [3, 7, -1, -1])
        self.assertEqual((r['FP'], r['TN']), (2, 2))
        self.assertAlmostEqual(r['FPR'], 0.5)
        self.assertEqual(r['TP'], 0)
        self.assertTrue(math.isnan(r['TPR']))          # mon이 없으면 TPR 정의 안 됨
        self.assertAlmostEqual(r['precision'], 0.0)    # 판정 2개가 전부 오탐

    def test_mon_to_unmon_is_fn_and_not_in_precision(self):
        r = ow_metrics([0, 1, -1], [-1, 1, -1])
        self.assertEqual((r['TP'], r['FN'], r['FP']), (1, 1, 0))
        self.assertAlmostEqual(r['TPR'], 0.5)
        self.assertAlmostEqual(r['precision'], 1.0)    # -1로 판정한 것은 분모에 없다

    def test_mixed_hand_computed(self):
        #        mon: TP  TP  wrong FN | unmon: FP  TN  TN  FP  TN  TN
        y_true = [0,  1,  2,    3,      -1, -1, -1, -1, -1, -1]
        y_pred = [0,  1,  0,   -1,       2, -1, -1,  0, -1, -1]
        c = ow_counts(y_true, y_pred)
        self.assertEqual(c, {'n_mon': 4, 'n_unmon': 6, 'TP': 2, 'wrong_site': 1, 'FN': 1, 'FP': 2, 'TN': 4})
        r = ow_metrics(y_true, y_pred)
        self.assertAlmostEqual(r['TPR'], 2 / 4)
        self.assertAlmostEqual(r['FPR'], 2 / 6)
        self.assertAlmostEqual(r['precision'], 2 / (2 + 1 + 2))
        self.assertAlmostEqual(r['wrong_site_rate'], 1 / 4)

    def test_accuracy_score_is_misleading(self):
        # unmon이 많으면 accuracy는 높지만 monitored 사이트는 하나도 못 맞힌다
        y_true = [0, 1, 2] + [-1] * 97
        y_pred = [1, 2, 0] + [-1] * 97
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 0.97)
        r = ow_metrics(y_true, y_pred)
        self.assertEqual(r['TPR'], 0.0)
        self.assertEqual(r['precision'], 0.0)

    def test_nothing_flagged_precision_nan(self):
        r = ow_metrics([0, -1], [-1, -1])
        self.assertTrue(math.isnan(r['precision']))
        self.assertEqual(r['TPR'], 0.0)

    def test_binary_labels(self):
        r = ow_metrics([1, 1, -1, -1], [1, -1, 1, -1])
        self.assertEqual((r['TP'], r['wrong_site'], r['FN'], r['FP'], r['TN']), (1, 0, 1, 1, 1))
        self.assertAlmostEqual(r['TPR'], 0.5)
        self.assertAlmostEqual(r['FPR'], 0.5)
        self.assertAlmostEqual(r['precision'], 0.5)


class TestThreshold(unittest.TestCase):
    classes = np.array([-1, 0, 1])
    proba = np.array([[0.1, 0.8, 0.1],    # 0 (0.8)
                      [0.3, 0.3, 0.4],    # 1 (0.4)
                      [0.6, 0.2, 0.2],    # -1 (argmax가 unmon)
                      [0.2, 0.45, 0.35]])  # 0 (0.45)

    def test_zero_threshold_is_argmax(self):
        np.testing.assert_array_equal(apply_threshold(self.proba, self.classes, 0.0), [0, 1, -1, 0])

    def test_below_threshold_becomes_unmon(self):
        np.testing.assert_array_equal(apply_threshold(self.proba, self.classes, 0.45), [0, -1, -1, 0])
        np.testing.assert_array_equal(apply_threshold(self.proba, self.classes, 0.5), [0, -1, -1, -1])

    def test_argmax_unmon_stays_unmon(self):
        self.assertEqual(apply_threshold(self.proba, self.classes, 0.0)[2], -1)

    def test_classes_order_independent(self):
        perm = [2, 0, 1]
        np.testing.assert_array_equal(apply_threshold(self.proba[:, perm], self.classes[perm], 0.45), [0, -1, -1, 0])


class TestMonThreshold(unittest.TestCase):
    classes = TestThreshold.classes
    proba = TestThreshold.proba

    def test_mon_score(self):
        s, site = mon_score(self.proba, self.classes)
        np.testing.assert_allclose(s, [0.8, 0.4, 0.2, 0.45])
        np.testing.assert_array_equal(site, [0, 1, 0, 0])

    def test_low_threshold_flags_even_if_unmon_is_argmax(self):
        # 3번째 행은 argmax가 -1(0.6)이지만 monitored 최댓값 0.2 >= 0.15라 사이트 0으로 판정한다
        np.testing.assert_array_equal(apply_mon_threshold(self.proba, self.classes, 0.15), [0, 1, 0, 0])
        np.testing.assert_array_equal(apply_threshold(self.proba, self.classes, 0.15), [0, 1, -1, 0])

    def test_rules_agree_at_half_and_above(self):
        for t in (0.5, 0.7, 0.9):
            np.testing.assert_array_equal(apply_mon_threshold(self.proba, self.classes, t),
                                          apply_threshold(self.proba, self.classes, t))

    def test_fpr_threshold(self):
        u = np.arange(100) / 100          # unmon 점수 0.00 ~ 0.99
        t = fpr_threshold(u, 0.05)
        self.assertAlmostEqual(np.mean(u >= t), 0.05)   # 0.95~0.99의 5개 → 정확히 5%, 넘지 않는다
        self.assertAlmostEqual(np.mean(u >= 0.94), 0.06)  # 그보다 조금이라도 낮추면 넘는다


class TestConfusion(unittest.TestCase):
    def test_unmon_row_col(self):
        y_true = [0, 1, 2, 3, -1, -1, -1, -1]
        y_pred = [0, -1, 0, -1, 2, 2, 0, -1]
        cm, labels, s = unmon_confusion(y_true, y_pred)
        self.assertEqual(labels[0], -1)
        self.assertEqual(cm.sum(), 8)
        self.assertEqual(s['unmon_as_unmon'], 1)
        self.assertEqual(s['unmon_as_mon'], 3)
        self.assertEqual(s['mon_as_unmon'], 2)
        self.assertEqual(s['top_FP_sites'][0], (2, 2))


if __name__ == '__main__':
    unittest.main()
