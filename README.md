# DS-_Matching

## В проекте решается задача разработки алгоритма, который для всех товаров из `validation.csv` предложит несколько вариантов наиболее похожих товаров из `base.csv`.

При этом:
- `base.csv` - анонимизированный набор товаров. Каждый товар представлен как уникальный id (0-base, 1-base, 2-base) и вектор признаков размерностью 72.
- `validation.csv` - датасет с товарами (уникальный id и вектор признаков), для которых надо найти наиболее близкие товары из base.csv

Для подбора оптимальных параметров приближенного поиска использовался обучающий датасет `target.csv`. 
Каждая строчка - один товар, для которого известен уникальный id (0-query, 1-query, …), вектор признаков и id товара из base.csv, который максимально похож на него (по мнению экспертов).

При решении задачи была использована библиотека для приближённого поиска ближайших соседей **`FAISS`**. 
Продемонстрировано обучение **CatBoostClassifier** в качестве ранжирующей модели.

Сохранённые индекс и модель позволяют реализовать развёртывание решения в виде микросервиса.

Ссылка на датасет: https://disk.yandex.ru/d/BBEphK0EHSJ5Jw
