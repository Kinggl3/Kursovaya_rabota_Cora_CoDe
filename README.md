# Выявление сообществ в графе цитирования научных публикаций

Курсовая работа по дисциплине «Машинное обучение в семантическом и сетевом анализе»  
Финансовый университет при Правительстве РФ, кафедра искусственного интеллекта  
Студент группы ПМ23-3 Певин Н. | Научный руководитель: к.п.н., доцент Тюжина И.В.

## Задача

Выявление тематических сообществ в графе цитирования научных публикаций на датасете [Cora](https://github.com/tkipf/pygcn/tree/master/data/cora). Граф содержит 2708 статей (узлы), 5278 связей цитирования (рёбра) и 1433 бинарных признака (bag-of-words). Каждая статья отнесена к одному из 7 тематических классов.

Задача решается как **unsupervised community detection** - модель должна найти группы плотно связанных узлов без доступа к меткам классов. Метки используются только для оценки качества (NMI, ARI, Modularity, Conductance).

## Реализованные модели

| Модель | Тип | Описание |
|---|---|---|
| Louvain | классика | Жадная максимизация модулярности (Blondel et al., 2008) |
| k-means | классика | Кластеризация на TF-IDF+SVD признаках |
| GCN | supervised GNN | Graph Convolutional Network (Kipf & Welling, 2017) |
| DMoN | unsupervised GNN | Deep Modularity Networks (Tsitsulin et al., 2023) |
| DMoN + GAT | unsupervised GNN | DMoN с attention-backbone (Veličković et al., 2018) |

## Как запустить

### Вариант 1: Google Colab (Kaggle, аналоги) (рекомендуется)

1. Откройте файл `Курсовая__1черновик__ПМ233_Певин.ipynb` в Google Colab
2. Выберите Runtime → Change runtime type → GPU (T4)
3. Запустите все ячейки по порядку (Runtime → Run all)

Датасет Cora скачивается автоматически из репозитория pygcn при первом запуске.

### Вариант 2: локальный запуск

```bash
git clone https://github.com/<Kinggl3>/cora-community-detection.git
cd cora-community-detection
pip install -r requirements.txt
jupyter notebook
```

Откройте ноутбук и запустите ячейки по порядку.

## Гиперпараметры, которые можно менять

### GCN (supervised)
- `hidden_dim` - размер скрытого слоя (по умолчанию 64)
- `lr` - learning rate (по умолчанию 0.01)
- `dropout` - вероятность dropout (по умолчанию 0.5)
- `weight_decay` - L2-регуляризация (по умолчанию 5e-4)

### DMoN / DMoN+GAT (unsupervised)
- `hidden_dim` — размер скрытого слоя GCN/GAT-энкодера (по умолчанию 64)
- `num_clusters` — число целевых сообществ (по умолчанию 7)
- `lr` — learning rate (по умолчанию 0.001)
- `dropout` — вероятность dropout (по умолчанию 0.5)
- `heads` — число attention heads в GAT (по умолчанию 8)

### TF-IDF + SVD
- `n_components` — число компонент SVD (по умолчанию 256)

## Структура репозитория

```
├── README.md                        — этот файл
├── requirements.txt                 — зависимости Python
├── Курсовая__1черновик__ПМ233_Певин.ipynb  — основной ноутбук с кодом
└── Пояснительная_записка_ПМ233_Певин.docx  — пояснительная записка
```

## Методические указания

[Методические указания по выполнению курсовой работы (PDF)](https://www.fa.ru/upload/medialibrary/9a0/b0gmbar3qe534ykhrjbngy92qhkdjspl/Metodicheskie_ukazaniya_po_vypolneniya_KR_MOvSiSA.pdf)

## Источники

1. Tsitsulin, A., Palowitch, J., Perozzi, B., Müller, E. Graph Clustering with Graph Neural Networks // Journal of Machine Learning Research. — 2023. — Vol. 24, No. 127. — P. 1–21.
2. Kipf, T. N., Welling, M. Semi-Supervised Classification with Graph Convolutional Networks // ICLR. — 2017.
3. Veličković, P. et al. Graph Attention Networks // ICLR. — 2018.
4. Blondel, V. D. et al. Fast unfolding of communities in large networks // Journal of Statistical Mechanics. — 2008.
5. Newman, M. E. J. Modularity and community structure in networks // PNAS. — 2006. — Vol. 103, No. 23. — P. 8577–8582.
6. Moradan, A. et al. UCoDe: Unified Community Detection with Graph Convolutional Networks // Machine Learning, Springer. — 2023.
7. Sen, P. et al. Collective classification in networked data // AI Magazine. — 2008. — Vol. 29, No. 3. — P. 93–106.
8. Deerwester, S. et al. Indexing by latent semantic analysis // Journal of the American Society for Information Science. — 1990. — Vol. 41, No. 6. — P. 391–407.
9. Fey, M., Lenssen, J. E. Fast Graph Representation Learning with PyTorch Geometric // ICLR Workshop. — 2019.
10. Yang, Z., Cohen, W. W., Salakhutdinov, R. Revisiting Semi-Supervised Learning with Graph Embeddings // ICML. — 2016.
