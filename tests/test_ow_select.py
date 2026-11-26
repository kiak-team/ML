"""code/ow_models.py의 E9 피처 선택 함수 단위 테스트. 실행: python -I -m unittest discover -s tests"""
import os, sys, unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))
from check_features import eta2  # noqa: E402  (A팀 구현)
from ow_models import CorrReduce, EtaSelect, corr_reduce_idx, eta_squared  # noqa: E402


class TestEta(unittest.TestCase):
    def test_matches_team_a_eta2(self):
        rng = np.random.default_rng(0)
        y = rng.integers(-1, 6, 300)
        X = rng.normal(size=(300, 7)) + (y[:, None] == 2) * np.arange(7)
        X[:, 3] = 5.0  # 상수 열 → 0
        np.testing.assert_allclose(eta_squared(X, y), [eta2(X[:, j], y) for j in range(7)], atol=1e-12)

    def test_select_top_and_bottom(self):
        y = np.repeat([0, 1], 50)
        X = np.column_stack([y + 0.01 * np.arange(100) % 1, np.arange(100) % 7, np.ones(100), y * 2.0])
        top = EtaSelect(2).fit(X, y)
        self.assertEqual(sorted(top.idx_.tolist()), [0, 3])
        bot = EtaSelect(1, bottom=True).fit(X, y)
        self.assertEqual(bot.idx_.tolist(), [1])          # 상수 열(2)은 하위 후보에서 뺀다
        self.assertEqual(top.transform(X).shape, (100, 2))

    def test_mon_only_ignores_unmon_rows(self):
        # 열 0: unmon만 다르고 사이트끼리는 같다 → 96클래스 η²는 높지만 사이트 η²는 0
        y = np.array([-1] * 50 + [0] * 25 + [1] * 25)
        X = np.column_stack([np.r_[np.full(50, 10.0), np.zeros(50)], np.r_[np.zeros(75), np.ones(25)]])
        self.assertEqual(EtaSelect(1).fit(X, y).idx_.tolist(), [0])
        self.assertEqual(EtaSelect(1, mon_only=True).fit(X, y).idx_.tolist(), [1])


class TestCorrReduce(unittest.TestCase):
    def test_drops_later_of_correlated_pair(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=200), rng.normal(size=200)
        X = np.column_stack([a, b, 2 * a + 1e-3 * rng.normal(size=200), -b, np.zeros(200)])
        self.assertEqual(corr_reduce_idx(X, 0.95).tolist(), [0, 1, 4])  # 2(≈a), 3(=-b) 제거, 상수 열은 남김
        self.assertEqual(CorrReduce(0.95).fit(X).transform(X).shape, (200, 3))


if __name__ == '__main__':
    unittest.main()
