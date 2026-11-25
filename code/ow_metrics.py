"""open-world 평가 지표 (B 파트, C팀과 공유). 단위 테스트: tests/test_ow_metrics.py

multi-class open-world에서 sklearn의 accuracy_score를 그대로 쓰면 안 된다. 정의:

  mon 샘플 (y_true != -1)
    TP          예측 == 정답 사이트                     (monitored로 판정했고 AND 사이트까지 맞힘)
    wrong_site  예측이 다른 monitored 사이트           → TP가 아니다. 따로 센다
    FN          예측 == -1                             (놓침)
  unmon 샘플 (y_true == -1)
    FP          예측이 monitored 사이트 중 아무거나    (오탐)
    TN          예측 == -1

  TPR       = TP / mon 수
  FPR       = FP / unmon 수
  precision = TP / monitored로 판정한 수 = TP / (TP + wrong_site + FP)
              (wrong_site도 "monitored로 판정했지만 틀린 것"이므로 분모에 들어간다.)
              monitored로 판정한 것이 0개이면 정의되지 않으므로 NaN
  recall    = TPR (positive = 정답 사이트까지 맞힌 것. 같은 정의라 값이 같다)
  macro_f1  = (N+1)개 클래스 전체의 macro F1 (-1 포함, 보조 지표)

  보조: det_TPR = (TP + wrong_site) / mon 수  — 사이트는 틀려도 "monitored"로는 잡은 비율
        wrong_site_rate = wrong_site / mon 수

binary(1 / -1)도 같은 함수로 계산한다. 이때 wrong_site는 항상 0이다.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

UNMON = -1


def ow_counts(y_true, y_pred, unmon=UNMON):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    assert y_true.shape == y_pred.shape
    mon, flag = y_true != unmon, y_pred != unmon
    return {
        'n_mon': int(mon.sum()), 'n_unmon': int((~mon).sum()),
        'TP': int((mon & (y_pred == y_true)).sum()),
        'wrong_site': int((mon & flag & (y_pred != y_true)).sum()),
        'FN': int((mon & ~flag).sum()),
        'FP': int((~mon & flag).sum()),
        'TN': int((~mon & ~flag).sum()),
    }


def _div(a, b):
    return a / b if b else float('nan')


def ow_metrics(y_true, y_pred, unmon=UNMON, labels=None):
    """위 정의대로 TPR / FPR / precision / recall / macro_f1과 개수를 dict로."""
    c = ow_counts(y_true, y_pred, unmon)
    flagged = c['TP'] + c['wrong_site'] + c['FP']
    if labels is None:
        labels = np.union1d(np.unique(y_true), np.unique(y_pred))
    r = dict(c)
    r['TPR'] = _div(c['TP'], c['n_mon'])
    r['FPR'] = _div(c['FP'], c['n_unmon'])
    r['precision'] = _div(c['TP'], flagged)
    r['recall'] = r['TPR']
    r['det_TPR'] = _div(c['TP'] + c['wrong_site'], c['n_mon'])
    r['wrong_site_rate'] = _div(c['wrong_site'], c['n_mon'])
    r['macro_f1'] = float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))
    return r


def apply_threshold(proba, classes, t, unmon=UNMON):
    """predict_proba의 최댓값이 t 미만이면 -1, 아니면 argmax 클래스. t=0이면 argmax와 같다.
    argmax가 -1(unmon)이면 t와 상관없이 -1이다."""
    proba, classes = np.asarray(proba), np.asarray(classes)
    pred = classes[proba.argmax(axis=1)].copy()
    pred[proba.max(axis=1) < t] = unmon
    return pred


def unmon_confusion(y_true, y_pred, unmon=UNMON):
    """confusion matrix와 -1 클래스의 행/열 요약. labels 순서 = [-1, 사이트 오름차순]."""
    labels = np.union1d(np.unique(y_true), np.unique(y_pred))
    labels = np.concatenate([[unmon], labels[labels != unmon]])
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row, col = cm[0], cm[:, 0]  # 행 = 실제 unmon의 예측 분포, 열 = -1로 예측된 것의 실제 분포
    top = np.argsort(row[1:])[::-1][:5]
    return cm, labels, {
        'unmon_as_unmon': int(row[0]), 'unmon_as_mon': int(row[1:].sum()),
        'mon_as_unmon': int(col[1:].sum()),
        'top_FP_sites': [(int(labels[1 + i]), int(row[1 + i])) for i in top if row[1 + i] > 0],
    }
