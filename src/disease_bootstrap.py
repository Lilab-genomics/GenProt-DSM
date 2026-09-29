import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)


# =============================================================================
# 1. 文件路径
# =============================================================================

TRAIN_FILE = (
    r"F:\GenProt-DSM_Resubmit\dataset\data"
    r"\VarGeneDisjointTrain.csv"
)

TEST_FILE = (
    r"F:\GenProt-DSM_Resubmit\dataset\data"
    r"\VarGeneDisjointTest.csv"
)

OUTPUT_FILE = (
    r"F:\GenProt-DSM_Resubmit\dataset\data"
    r"\disease_stratified_bootstrap_results.csv"
)

PAIR_OUTPUT_FILE = (
    r"F:\GenProt-DSM_Resubmit\dataset\data"
    r"\disease_matched_pairs.csv"
)

PAPER_OUTPUT_FILE = (
    r"F:\GenProt-DSM_Resubmit\dataset\data"
    r"\disease_stratified_bootstrap_table.csv"
)


# =============================================================================
# 2. 参数
# =============================================================================

LABEL_COL = "label"
DISEASE_COL = "disease type"
SCORE_COL = "prob_ensemble"

CHROM_COL = "Chrom"
POS_COL = "Position"

# GenProt-DSM 最终固定阈值
THRESHOLD = 0.5543934838088572

# Bootstrap 次数
N_BOOTSTRAP = 1000

# 随机种子
RANDOM_SEED = 2026


# =============================================================================
# 3. 读取数据
# =============================================================================

train_df = pd.read_csv(
    TRAIN_FILE,
    low_memory=False
)

test_df = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

print("=" * 100)
print("Disease-stratified paired bootstrap evaluation")
print("=" * 100)

print(
    f"Train rows: {len(train_df):,}"
)

print(
    f"Test rows : {len(test_df):,}"
)


# =============================================================================
# 4. 检查必要列
# =============================================================================

required_test_columns = [
    LABEL_COL,
    DISEASE_COL,
    SCORE_COL,
    CHROM_COL,
    POS_COL
]

for col in required_test_columns:

    if col not in test_df.columns:

        raise ValueError(
            f"VarGeneDisjointTest.csv 缺少必要列：{col}\n"
            f"实际列名：{list(test_df.columns)}"
        )


# =============================================================================
# 5. 标准化标签、染色体、位置和分数
# =============================================================================

def normalize_chromosome(x):
    """
    标准化染色体名称：

    chr1 -> 1
    ChrX -> X
    1.0  -> 1
    """

    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    if x.lower().startswith("chr"):
        x = x[3:]

    try:

        f = float(x)

        if f.is_integer():
            x = str(int(f))

    except Exception:
        pass

    return x.upper()


test_df[LABEL_COL] = pd.to_numeric(
    test_df[LABEL_COL],
    errors="coerce"
)

test_df[SCORE_COL] = pd.to_numeric(
    test_df[SCORE_COL],
    errors="coerce"
)

test_df[POS_COL] = pd.to_numeric(
    test_df[POS_COL],
    errors="coerce"
)

test_df["_chrom_norm"] = (
    test_df[CHROM_COL]
    .apply(normalize_chromosome)
)


# =============================================================================
# 6. 提取正样本和负样本
# =============================================================================

positive_df = test_df[
    test_df[LABEL_COL] == 1
].copy()

negative_df = test_df[
    test_df[LABEL_COL] == 0
].copy()


# 正样本必须具有疾病类型
positive_df = positive_df[
    positive_df[DISEASE_COL].notna()
].copy()

positive_df = positive_df[
    positive_df[DISEASE_COL]
    .astype(str)
    .str.strip()
    .ne("")
].copy()


print("\nTest data:")

print(
    f"Positive with disease type = "
    f"{len(positive_df):,}"
)

print(
    f"Negative                  = "
    f"{len(negative_df):,}"
)


# =============================================================================
# 7. 显示 Train/Test 中的疾病类型
#
# train.csv 不参与性能计算。
# 仅用于检查疾病类别名称是否一致。
# =============================================================================

if DISEASE_COL in train_df.columns:

    train_labels = pd.to_numeric(
        train_df[LABEL_COL],
        errors="coerce"
    )

    train_diseases = sorted(
        train_df.loc[
            (
                train_labels == 1
            )
            &
            train_df[DISEASE_COL].notna(),
            DISEASE_COL
        ]
        .astype(str)
        .str.strip()
        .unique()
    )

    print("\nDisease types found in TRAIN:")

    for x in train_diseases:
        print(
            "  ",
            x
        )


test_diseases = sorted(
    positive_df[DISEASE_COL]
    .astype(str)
    .str.strip()
    .unique()
)

print("\nDisease types found in TEST:")

for x in test_diseases:
    print(
        "  ",
        x
    )


print(
    f"\nNumber of disease types in TEST = "
    f"{len(test_diseases)}"
)


if len(test_diseases) != 8:

    print(
        "\n[WARNING] VarGeneDisjointTest.csv "
        "中检测到的疾病类型不是 8 类。"
    )

    print(
        "请检查 disease type 是否存在拼写差异、"
        "空格或类别名称不统一。"
    )


# =============================================================================
# 8. 给每个正样本寻找同染色体最近的负样本
#
# 规则：
# 1. 必须位于同一染色体
# 2. 优先选择 genomic position 最近的负样本
# 3. 每个负样本仅允许使用一次
# 4. 使用 greedy matching
# =============================================================================

def build_matched_pairs(
    positive_df,
    negative_df
):

    pair_rows = []

    used_negative_indices = set()

    # 保证匹配过程可重复
    positive_sorted = (
        positive_df
        .sort_values(
            by=[
                "_chrom_norm",
                POS_COL
            ]
        )
    )


    for pos_index, pos_row in positive_sorted.iterrows():

        chrom = pos_row[
            "_chrom_norm"
        ]

        position = pos_row[
            POS_COL
        ]

        disease = str(
            pos_row[
                DISEASE_COL
            ]
        ).strip()

        score_pos = pos_row[
            SCORE_COL
        ]


        # ---------------------------------------------------------------------
        # 同染色体 + 尚未使用的负样本
        # ---------------------------------------------------------------------

        candidates = negative_df[
            (
                negative_df["_chrom_norm"]
                == chrom
            )
            &
            (
                ~negative_df.index.isin(
                    used_negative_indices
                )
            )
        ].copy()


        if len(candidates) == 0:

            raise RuntimeError(
                f"\n无法给正样本找到同染色体负样本：\n"
                f"positive row = {pos_index}\n"
                f"chromosome   = {chrom}\n"
                f"position     = {position}"
            )


        # ---------------------------------------------------------------------
        # 如果有 genomic position，则寻找最近负样本
        # ---------------------------------------------------------------------

        if pd.notna(position):

            candidates["_distance"] = (
                candidates[POS_COL]
                - position
            ).abs()


            finite_candidates = candidates[
                candidates["_distance"].notna()
            ]


            if len(finite_candidates) > 0:

                nearest_index = (
                    finite_candidates
                    .sort_values(
                        by=[
                            "_distance"
                        ]
                    )
                    .index[0]
                )

            else:

                # 同染色体候选样本中没有有效 Position
                nearest_index = (
                    candidates.index[0]
                )

        else:

            nearest_index = (
                candidates.index[0]
            )


        neg_row = negative_df.loc[
            nearest_index
        ]

        used_negative_indices.add(
            nearest_index
        )


        # ---------------------------------------------------------------------
        # 计算 genomic distance
        # ---------------------------------------------------------------------

        if (
            pd.notna(position)
            and
            pd.notna(
                neg_row[
                    POS_COL
                ]
            )
        ):

            distance = abs(
                position
                -
                neg_row[
                    POS_COL
                ]
            )

        else:

            distance = np.nan


        # ---------------------------------------------------------------------
        # 保存 matched pair
        # ---------------------------------------------------------------------

        pair_rows.append(
            {
                "disease type":
                    disease,

                "positive_original_index":
                    int(pos_index),

                "negative_original_index":
                    int(nearest_index),

                "chromosome":
                    chrom,

                "positive_position":
                    position,

                "negative_position":
                    neg_row[
                        POS_COL
                    ],

                "distance":
                    distance,

                "positive_score":
                    score_pos,

                "negative_score":
                    neg_row[
                        SCORE_COL
                    ],
            }
        )


    pair_df = pd.DataFrame(
        pair_rows
    )

    return pair_df


pair_df = build_matched_pairs(
    positive_df,
    negative_df
)


print(
    f"\nSuccessfully matched pairs: "
    f"{len(pair_df):,}"
)


# =============================================================================
# 9. 检查预测分数
# =============================================================================

missing_positive_scores = int(
    pair_df[
        "positive_score"
    ]
    .isna()
    .sum()
)

missing_negative_scores = int(
    pair_df[
        "negative_score"
    ]
    .isna()
    .sum()
)


print(
    f"Missing positive scores: "
    f"{missing_positive_scores}"
)

print(
    f"Missing negative scores: "
    f"{missing_negative_scores}"
)


if (
    missing_positive_scores > 0
    or
    missing_negative_scores > 0
):

    print(
        "\n[WARNING] 有 matched pair 缺少 prob_ensemble。"
    )

    print(
        "这些 matched pairs 将从性能评价中删除。"
    )

    pair_df = pair_df.dropna(
        subset=[
            "positive_score",
            "negative_score"
        ]
    ).copy()


# =============================================================================
# 10. 保存 matched pairs
# =============================================================================

pair_df.to_csv(
    PAIR_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)


print(
    f"\nMatched-pair audit saved:\n"
    f"{PAIR_OUTPUT_FILE}"
)


# =============================================================================
# 11. 将 matched pairs 转换为 y_true / y_score
# =============================================================================

def pairs_to_arrays(
    disease_pairs
):

    positive_scores = (
        disease_pairs[
            "positive_score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    negative_scores = (
        disease_pairs[
            "negative_score"
        ]
        .to_numpy(
            dtype=float
        )
    )


    # 每个 pair:
    #
    # positive label = 1
    # negative label = 0

    y_true = np.concatenate(
        [
            np.ones(
                len(
                    positive_scores
                ),
                dtype=int
            ),

            np.zeros(
                len(
                    negative_scores
                ),
                dtype=int
            )
        ]
    )


    y_score = np.concatenate(
        [
            positive_scores,
            negative_scores
        ]
    )


    return (
        y_true,
        y_score
    )


# =============================================================================
# 12. 指标计算
# =============================================================================

def calculate_metrics(
    y_true,
    y_score,
    threshold
):

    auc = roc_auc_score(
        y_true,
        y_score
    )

    aupr = average_precision_score(
        y_true,
        y_score
    )


    y_pred = (
        y_score
        >= threshold
    ).astype(int)


    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )


    return (
        float(auc),
        float(aupr),
        float(mcc)
    )


# =============================================================================
# 13. 计算 percentile 95% CI
# =============================================================================

def get_ci(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(
            values
        )
    ]


    lower = np.percentile(
        values,
        2.5
    )

    upper = np.percentile(
        values,
        97.5
    )


    return (
        float(lower),
        float(upper)
    )


# =============================================================================
# 14. Disease-stratified paired bootstrap
#
# 对每一个疾病类别：
#
# 假设有 n 个 matched pairs：
#
#   pair 1 = pathogenic + matched benign
#   pair 2 = pathogenic + matched benign
#   ...
#
# 每次 bootstrap：
#
#   从 n 个 matched pairs 中有放回抽取 n 个 pair
#
# 因而始终保持：
#
#   n pathogenic
#   n benign
#
# 同时保留 pathogenic-benign 配对关系。
# =============================================================================

def paired_bootstrap(
    disease_pairs,
    threshold,
    n_bootstrap,
    seed
):

    rng = np.random.default_rng(
        seed
    )


    n_pairs = len(
        disease_pairs
    )


    auc_values = []

    aupr_values = []

    mcc_values = []


    for _ in range(
        n_bootstrap
    ):

        sampled_pair_indices = (
            rng.integers(
                low=0,
                high=n_pairs,
                size=n_pairs
            )
        )


        boot_pairs = (
            disease_pairs
            .iloc[
                sampled_pair_indices
            ]
        )


        y_true, y_score = (
            pairs_to_arrays(
                boot_pairs
            )
        )


        (
            auc,
            aupr,
            mcc
        ) = calculate_metrics(
            y_true,
            y_score,
            threshold
        )


        auc_values.append(
            auc
        )

        aupr_values.append(
            aupr
        )

        mcc_values.append(
            mcc
        )


    return (
        np.asarray(
            auc_values
        ),

        np.asarray(
            aupr_values
        ),

        np.asarray(
            mcc_values
        )
    )


# =============================================================================
# 15. 对各疾病类别分别计算
# =============================================================================

results = []


disease_list = sorted(
    pair_df[
        "disease type"
    ]
    .astype(str)
    .unique()
)


print(
    "\n"
    + "=" * 100
)

print(
    "Disease-specific performance"
)

print(
    "=" * 100
)


for disease_index, disease in enumerate(
    disease_list
):

    disease_pairs = pair_df[
        pair_df[
            "disease type"
        ]
        == disease
    ].copy()


    n_pairs = len(
        disease_pairs
    )


    # -------------------------------------------------------------------------
    # N = pathogenic + matched benign
    # -------------------------------------------------------------------------

    N = (
        2
        * n_pairs
    )


    print(
        "\n"
        + "-" * 100
    )

    print(
        f"Disease: {disease}"
    )

    print(
        f"N = {N}"
    )

    print(
        f"Matched pairs = {n_pairs}"
    )


    # -------------------------------------------------------------------------
    # Point estimate
    # -------------------------------------------------------------------------

    y_true, y_score = (
        pairs_to_arrays(
            disease_pairs
        )
    )


    (
        auc_point,
        aupr_point,
        mcc_point
    ) = calculate_metrics(
        y_true,
        y_score,
        THRESHOLD
    )


    # -------------------------------------------------------------------------
    # 1000 次 paired bootstrap
    # -------------------------------------------------------------------------

    (
        auc_boot,
        aupr_boot,
        mcc_boot
    ) = paired_bootstrap(
        disease_pairs=disease_pairs,
        threshold=THRESHOLD,
        n_bootstrap=N_BOOTSTRAP,
        seed=(
            RANDOM_SEED
            + disease_index
        )
    )


    # -------------------------------------------------------------------------
    # 95% CI
    # -------------------------------------------------------------------------

    (
        auc_low,
        auc_high
    ) = get_ci(
        auc_boot
    )


    (
        aupr_low,
        aupr_high
    ) = get_ci(
        aupr_boot
    )


    (
        mcc_low,
        mcc_high
    ) = get_ci(
        mcc_boot
    )


    # -------------------------------------------------------------------------
    # 保存
    # -------------------------------------------------------------------------

    results.append(
        {
            "Disease":
                disease,

            "N":
                N,

            "AUROC":
                auc_point,

            "AUROC_CI_lower":
                auc_low,

            "AUROC_CI_upper":
                auc_high,

            "AUPR":
                aupr_point,

            "AUPR_CI_lower":
                aupr_low,

            "AUPR_CI_upper":
                aupr_high,

            "MCC":
                mcc_point,

            "MCC_CI_lower":
                mcc_low,

            "MCC_CI_upper":
                mcc_high,

            "Threshold":
                THRESHOLD,

            "Bootstrap":
                N_BOOTSTRAP,
        }
    )


    print(
        f"AUROC = "
        f"{auc_point:.4f} "
        f"(95% CI "
        f"{auc_low:.4f}–"
        f"{auc_high:.4f})"
    )


    print(
        f"AUPR  = "
        f"{aupr_point:.4f} "
        f"(95% CI "
        f"{aupr_low:.4f}–"
        f"{aupr_high:.4f})"
    )


    print(
        f"MCC   = "
        f"{mcc_point:.4f} "
        f"(95% CI "
        f"{mcc_low:.4f}–"
        f"{mcc_high:.4f})"
    )


# =============================================================================
# 16. 保存详细结果
#
# 所有浮点数统一保留 4 位小数
# =============================================================================

result_df = pd.DataFrame(
    results
)


result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)


print(
    "\n"
    + "=" * 100
)

print(
    f"Detailed results saved:\n"
    f"{OUTPUT_FILE}"
)

print(
    "=" * 100
)


# =============================================================================
# 17. 生成论文表格格式
#
# 格式：
#
# 0.9123 (0.8635–0.9532)
# =============================================================================

def format_ci(
    value,
    lower,
    upper
):

    return (
        f"{value:.4f} "
        f"({lower:.4f}–{upper:.4f})"
    )


paper_df = pd.DataFrame()


paper_df[
    "Disease"
] = result_df[
    "Disease"
]


paper_df[
    "N"
] = result_df[
    "N"
]


paper_df[
    "AUROC (95% CI)"
] = result_df.apply(
    lambda x: format_ci(
        x[
            "AUROC"
        ],
        x[
            "AUROC_CI_lower"
        ],
        x[
            "AUROC_CI_upper"
        ]
    ),
    axis=1
)


paper_df[
    "AUPR (95% CI)"
] = result_df.apply(
    lambda x: format_ci(
        x[
            "AUPR"
        ],
        x[
            "AUPR_CI_lower"
        ],
        x[
            "AUPR_CI_upper"
        ]
    ),
    axis=1
)


paper_df[
    "MCC (95% CI)"
] = result_df.apply(
    lambda x: format_ci(
        x[
            "MCC"
        ],
        x[
            "MCC_CI_lower"
        ],
        x[
            "MCC_CI_upper"
        ]
    ),
    axis=1
)


# =============================================================================
# 18. 保存论文表格
# =============================================================================

paper_df.to_csv(
    PAPER_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    f"\nPaper-format table saved:\n"
    f"{PAPER_OUTPUT_FILE}"
)


print(
    "\nFinal table:"
)

print(
    paper_df.to_string(
        index=False
    )
)