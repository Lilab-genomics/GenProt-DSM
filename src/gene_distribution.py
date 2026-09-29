import os
import numpy as np
import pandas as pd


# =============================================================================
# 1. 文件路径
# =============================================================================

INPUT_FILE = (
    r"F:\GenProt-DSM_Resubmit\结果"
    r"\gene_distribution（基因分布统计）"
    r"\gene_partition_summary.csv"
)

OUTPUT_DIR = os.path.dirname(INPUT_FILE)

OUTPUT_DISJOINT = os.path.join(
    OUTPUT_DIR,
    "gene_disjoint_summary.csv"
)

OUTPUT_DISTRIBUTION = os.path.join(
    OUTPUT_DIR,
    "gene_variant_distribution_summary.csv"
)

OUTPUT_BINS = os.path.join(
    OUTPUT_DIR,
    "gene_variant_count_bins.csv"
)


# =============================================================================
# 2. 读取数据
# =============================================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 80)
print("Gene distribution statistics")
print("=" * 80)

print(f"\nInput file:\n{INPUT_FILE}")
print(f"\nOriginal rows: {len(df)}")

print("\nColumns:")
print(df.columns.tolist())


# =============================================================================
# 3. 检查必要列
# =============================================================================

required_columns = [
    "Gene",
    "Train_variant_count",
    "Test_variant_count",
    "Total_variant_count",
    "Partition",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# =============================================================================
# 4. 基础清洗
# =============================================================================

df = df.copy()

# Gene 清洗
df["Gene"] = (
    df["Gene"]
    .astype(str)
    .str.strip()
)

# 去除空 Gene
invalid_gene = (
    df["Gene"].isna()
    | df["Gene"].eq("")
    | df["Gene"].str.lower().eq("nan")
)

if invalid_gene.any():
    print(
        f"\n[WARNING] Removing {invalid_gene.sum()} rows "
        f"with missing Gene."
    )
    df = df.loc[~invalid_gene].copy()


# 数值列转换
count_columns = [
    "Train_variant_count",
    "Test_variant_count",
    "Total_variant_count",
]

for col in count_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    if df[col].isna().any():
        raise ValueError(
            f"Column '{col}' contains "
            f"{df[col].isna().sum()} non-numeric/missing values."
        )

    if (df[col] < 0).any():
        raise ValueError(
            f"Column '{col}' contains negative counts."
        )

    df[col] = df[col].astype(int)


# =============================================================================
# 5. 检查 Gene 是否重复
# =============================================================================

duplicate_gene_mask = df["Gene"].duplicated(
    keep=False
)

if duplicate_gene_mask.any():

    duplicate_genes = (
        df.loc[duplicate_gene_mask, "Gene"]
        .unique()
        .tolist()
    )

    print("\n[WARNING] Duplicate genes detected:")
    print(duplicate_genes)

    print(
        "\nDuplicate rows will be aggregated "
        "by summing Train/Test variant counts."
    )

    df = (
        df.groupby(
            "Gene",
            as_index=False
        )
        .agg(
            Train_variant_count=(
                "Train_variant_count",
                "sum"
            ),
            Test_variant_count=(
                "Test_variant_count",
                "sum"
            ),
        )
    )

    # 重算总数
    df["Total_variant_count"] = (
        df["Train_variant_count"]
        + df["Test_variant_count"]
    )

else:
    print("\n[OK] No duplicated Gene entries.")


# =============================================================================
# 6. 检查 Total_variant_count
# =============================================================================

df["Calculated_total"] = (
    df["Train_variant_count"]
    + df["Test_variant_count"]
)

total_mismatch = (
    df["Calculated_total"]
    != df["Total_variant_count"]
)

if total_mismatch.any():

    print(
        f"\n[WARNING] {total_mismatch.sum()} genes have "
        f"Total_variant_count != "
        f"Train_variant_count + Test_variant_count."
    )

    print(
        "\nFor all following statistics, "
        "Total_variant_count will be recalculated "
        "from Train + Test."
    )

    df["Total_variant_count"] = (
        df["Calculated_total"]
    )

else:
    print(
        "\n[OK] Total_variant_count is consistent "
        "with Train + Test for all genes."
    )


# =============================================================================
# 7. 根据真实 count 重新确定 gene partition
# =============================================================================

df["In_train"] = (
    df["Train_variant_count"] > 0
)

df["In_test"] = (
    df["Test_variant_count"] > 0
)


def determine_partition(row):

    if row["In_train"] and row["In_test"]:
        return "Shared"

    elif row["In_train"]:
        return "Train"

    elif row["In_test"]:
        return "Test"

    else:
        return "Neither"


df["Calculated_partition"] = df.apply(
    determine_partition,
    axis=1
)


# =============================================================================
# 8. TABLE 1:
#    Gene-disjoint integrity summary
# =============================================================================

train_genes = set(
    df.loc[df["In_train"], "Gene"]
)

test_genes = set(
    df.loc[df["In_test"], "Gene"]
)

shared_genes = (
    train_genes
    & test_genes
)

train_only_genes = (
    train_genes
    - test_genes
)

test_only_genes = (
    test_genes
    - train_genes
)

all_genes = (
    train_genes
    | test_genes
)


train_variant_total = int(
    df["Train_variant_count"].sum()
)

test_variant_total = int(
    df["Test_variant_count"].sum()
)

total_variant_count = (
    train_variant_total
    + test_variant_total
)


if len(all_genes) > 0:
    gene_overlap_fraction = (
        len(shared_genes)
        / len(all_genes)
    )
else:
    gene_overlap_fraction = np.nan


disjoint_summary = pd.DataFrame(
    [
        {
            "Train_unique_genes":
                len(train_genes),

            "Test_unique_genes":
                len(test_genes),

            "Shared_genes":
                len(shared_genes),

            "Train_only_genes":
                len(train_only_genes),

            "Test_only_genes":
                len(test_only_genes),

            "Total_unique_genes":
                len(all_genes),

            "Gene_overlap_fraction":
                gene_overlap_fraction,

            "Train_variant_count":
                train_variant_total,

            "Test_variant_count":
                test_variant_total,

            "Total_variant_count":
                total_variant_count,

            "Gene_disjoint":
                len(shared_genes) == 0,
        }
    ]
)


disjoint_summary.to_csv(
    OUTPUT_DISJOINT,
    index=False,
    encoding="utf-8-sig"
)


# =============================================================================
# 9. TABLE 2:
#    Per-gene variant distribution summary
# =============================================================================

def calculate_distribution(
    values,
    partition_name
):

    values = pd.Series(
        values
    ).dropna()

    if len(values) == 0:
        return {
            "Partition":
                partition_name,

            "N_genes":
                0,

            "N_variants":
                0,

            "Mean_variants_per_gene":
                np.nan,

            "Median_variants_per_gene":
                np.nan,

            "Std_variants_per_gene":
                np.nan,

            "Q1":
                np.nan,

            "Q3":
                np.nan,

            "IQR":
                np.nan,

            "Min":
                np.nan,

            "Max":
                np.nan,
        }

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    return {
        "Partition":
            partition_name,

        "N_genes":
            int(len(values)),

        "N_variants":
            int(values.sum()),

        "Mean_variants_per_gene":
            float(values.mean()),

        "Median_variants_per_gene":
            float(values.median()),

        "Std_variants_per_gene":
            float(values.std(ddof=1))
            if len(values) > 1
            else 0.0,

        "Q1":
            float(q1),

        "Q3":
            float(q3),

        "IQR":
            float(q3 - q1),

        "Min":
            int(values.min()),

        "Max":
            int(values.max()),
    }


train_values = df.loc[
    df["Train_variant_count"] > 0,
    "Train_variant_count"
]

test_values = df.loc[
    df["Test_variant_count"] > 0,
    "Test_variant_count"
]

total_values = df.loc[
    df["Total_variant_count"] > 0,
    "Total_variant_count"
]


distribution_summary = pd.DataFrame(
    [
        calculate_distribution(
            train_values,
            "Train"
        ),

        calculate_distribution(
            test_values,
            "Test"
        ),

        calculate_distribution(
            total_values,
            "Overall"
        ),
    ]
)


distribution_summary.to_csv(
    OUTPUT_DISTRIBUTION,
    index=False,
    encoding="utf-8-sig"
)


# =============================================================================
# 10. TABLE 3:
#     Gene variant-count bins
# =============================================================================

# 分组：
# 1
# 2–5
# 6–10
# 11–20
# 21–50
# >50

bin_edges = [
    0,
    1,
    5,
    10,
    20,
    50,
    np.inf,
]

bin_labels = [
    "1 variant",
    "2–5 variants",
    "6–10 variants",
    "11–20 variants",
    "21–50 variants",
    ">50 variants",
]


def calculate_bins(
    values,
    partition_name
):

    temp = pd.DataFrame(
        {
            "Variant_count":
                values.astype(int)
        }
    )

    temp["Variant_count_bin"] = pd.cut(
        temp["Variant_count"],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        right=True,
    )

    total_genes = len(temp)

    total_variants = (
        temp["Variant_count"].sum()
    )

    rows = []

    for category in bin_labels:

        subset = temp.loc[
            temp["Variant_count_bin"]
            .astype(str)
            .eq(category)
        ]

        n_genes = len(subset)

        n_variants = int(
            subset["Variant_count"].sum()
        )

        rows.append(
            {
                "Partition":
                    partition_name,

                "Variant_count_bin":
                    category,

                "N_genes":
                    n_genes,

                "Fraction_of_genes":
                    (
                        n_genes / total_genes
                        if total_genes > 0
                        else np.nan
                    ),

                "N_variants":
                    n_variants,

                "Fraction_of_variants":
                    (
                        n_variants / total_variants
                        if total_variants > 0
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(rows)


train_bins = calculate_bins(
    train_values,
    "Train"
)

test_bins = calculate_bins(
    test_values,
    "Test"
)

overall_bins = calculate_bins(
    total_values,
    "Overall"
)


bin_summary = pd.concat(
    [
        train_bins,
        test_bins,
        overall_bins,
    ],
    ignore_index=True
)


bin_summary.to_csv(
    OUTPUT_BINS,
    index=False,
    encoding="utf-8-sig"
)


# =============================================================================
# 11. 输出控制台结果
# =============================================================================

print("\n" + "=" * 80)
print("1. Gene-disjoint summary")
print("=" * 80)

print(
    disjoint_summary.to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("2. Per-gene variant distribution")
print("=" * 80)

print(
    distribution_summary.to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("3. Gene variant-count bins")
print("=" * 80)

print(
    bin_summary.to_string(
        index=False
    )
)


# =============================================================================
# 12. Gene-disjoint 审计
# =============================================================================

print("\n" + "=" * 80)
print("Gene-disjoint audit")
print("=" * 80)

if len(shared_genes) == 0:

    print(
        "[PASS] Train and test sets are strictly gene-disjoint."
    )

    print(
        "Shared genes = 0"
    )

else:

    print(
        "[WARNING] Train/test gene overlap detected!"
    )

    print(
        f"Shared genes ({len(shared_genes)}):"
    )

    for gene in sorted(shared_genes):
        print(
            f"  {gene}"
        )


# =============================================================================
# 13. 输出路径
# =============================================================================

print("\n" + "=" * 80)
print("Output files")
print("=" * 80)

print(
    f"\n1. Gene-disjoint summary:\n"
    f"   {OUTPUT_DISJOINT}"
)

print(
    f"\n2. Variant distribution summary:\n"
    f"   {OUTPUT_DISTRIBUTION}"
)

print(
    f"\n3. Variant-count bins:\n"
    f"   {OUTPUT_BINS}"
)

print("\nDone.")