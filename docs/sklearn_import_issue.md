> 실행 시점이 포함된 진단 기록이며 바이트 재현성 검증 대상이 아님.

# sklearn import 이슈 기록 (Windows 애플리케이션 제어 정책, `DLL load failed`)

> 이 문서는 `python env_tools.py --sklearn-issue`로 생성한다(시점이 들어가므로 실행마다 바이트가 달라지며 재현성 비교 대상이 아니다). 홈·프로젝트 경로는 `<USER>`, `<PROJECT>`로 마스킹했다. **이 파이프라인의 결과 수치는 sklearn에 의존하지 않는다**(Ridge/GroupKFold/표준화=`ridge_lib`, GBM=`gbt_lib`, MLP=torch).

## 재현 시도 방법
- 시점: 2026-10-16T15:34:38 (로컬 시각). Python 3.14.7, OS Windows-11-10.0.26200-SP0.
- 방법: 서로 다른 새 인터프리터(`python -X utf8 -c <import 문>`)에서 아래 8개 import를 각각 시도하고 3회 반복했다(총 24회).
- 설치된 scikit-learn 배포 버전(메타데이터 조회, import 없음): 1.9.1

## 결과 요약
| import | 시도 1 | 시도 2 | 시도 3 |
|---|---|---|---|
| `import sklearn` | 성공 | 성공 | 성공 |
| `sklearn.linear_model.Ridge` | 성공 | 성공 | 성공 |
| `sklearn.model_selection.GroupKFold` | 성공 | 성공 | 성공 |
| `sklearn.preprocessing.StandardScaler` | 성공 | 성공 | 성공 |
| `sklearn.tree.DecisionTreeRegressor` | **실패** | 성공 | 성공 |
| `sklearn.ensemble.HistGradientBoostingRegressor` | 성공 | 성공 | 성공 |
| `sklearn.neural_network.MLPRegressor` | 성공 | 성공 | 성공 |
| `sklearn.linear_model.QuantileRegressor` | 성공 | 성공 | 성공 |

- 총 24회 중 실패 1회.

## 재현된 오류의 traceback 전문 (경로 마스킹)

### 시도 1: `sklearn.tree.DecisionTreeRegressor`
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    from sklearn.tree import DecisionTreeRegressor
  File "<USER>\AppData\Local\Programs\Python\Python314\Lib\site-packages\sklearn\tree\__init__.py", line 6, in <module>
    from sklearn.tree._classes import (
    ...<5 lines>...
    )
  File "<USER>\AppData\Local\Programs\Python\Python314\Lib\site-packages\sklearn\tree\_classes.py", line 28, in <module>
    from sklearn.tree import _criterion, _splitter
  File "sklearn/tree/_criterion.pyx", line 1, in init sklearn.tree._criterion
  File "sklearn/tree/_splitter.pyx", line 1, in init sklearn.tree._splitter
  File "sklearn/tree/_tree.pyx", line 1, in init sklearn.tree._tree
  File "<USER>\AppData\Local\Programs\Python\Python314\Lib\site-packages\sklearn\neighbors\__init__.py", line 7, in <module>
    from sklearn.neighbors._base import (
    ...<3 lines>...
    )
  File "<USER>\AppData\Local\Programs\Python\Python314\Lib\site-packages\sklearn\neighbors\_base.py", line 23, in <module>
    from sklearn.neighbors._kd_tree import KDTree
ImportError: DLL load failed while importing _kd_tree: 애플리케이션 제어 정책에서 이 파일을 차단했습니다.
```

- 관찰: 실패하는 모듈 이름(예: `_minpack`, `_pava_pybind` 등 scipy/sklearn 컴파일 확장)이 실행마다 달라지는 간헐 현상이 이전 작업에서 반복 확인되었다. 원인은 Windows Application Control 정책 때문으로 **추정**되지만 정책 설정을 확인하지 못했다(미확인). 정책 우회는 시도하지 않았다.
