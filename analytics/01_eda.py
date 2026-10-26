import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "titanic.csv")

try:
    df = sns.load_dataset("titanic")
    print("Dataset loaded successfully via sns.load_dataset('titanic')!")
except Exception:
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        print(f"Dataset loaded from offline fallback {CSV_PATH}!")
    else:
        raise FileNotFoundError(f"Cannot load dataset from Seaborn or {CSV_PATH}")

print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

df.to_csv(CSV_PATH, index=False)

print(f"\nTitanic dataset saved to {CSV_PATH}")


print("\n===== DATASET INFO =====")
df.info()

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe(include="all").T)

print("\n===== DATASET SHAPE =====")
print(df.shape)

missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100

missing_report = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percent
})

print("\n===== MISSING VALUES =====")
print(missing_report[missing_report["Missing Count"] > 0])

print("\nMissing-Value Threshold Rule Justifications:")
print("- age (19.87% missing): Falls within 5%–30% threshold -> Imputed using median.")
print("- embarked & embark_town (0.22% missing each): Below 5% threshold -> Dropped affected rows (2 rows).")
print("- deck (77.22% missing): Above 30% threshold -> Dropped column because >77% missing data makes imputation unreliable.")

print("\n===== HANDLING MISSING VALUES =====")

df["age"] = df["age"].fillna(df["age"].median())

df = df.dropna(subset=["embarked", "embark_town"])

df = df.drop(columns=["deck"])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nShape after cleaning:", df.shape)


plt.figure(figsize=(8, 5))
plt.hist(df["age"], bins=20, edgecolor="black")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution")
plt.savefig(os.path.join(BASE_DIR, "eda_age_distribution.png"), dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.hist(df["fare"], bins=20, edgecolor="black")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.title("Fare Distribution")
plt.savefig(os.path.join(BASE_DIR, "eda_fare_distribution.png"), dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.boxplot(df["age"])
plt.ylabel("Age")
plt.title("Age Box Plot")
plt.savefig(os.path.join(BASE_DIR, "eda_age_boxplot.png"), dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.boxplot(df["fare"])
plt.ylabel("Fare")
plt.title("Fare Box Plot")
plt.savefig(os.path.join(BASE_DIR, "eda_fare_boxplot.png"), dpi=150)
plt.close()


def find_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(f"\n===== {column.upper()} OUTLIERS =====")
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)
    print("Number of Outliers:", len(outliers))


find_outliers("age")
find_outliers("fare")


print("\n===== FARE STATISTICS & SKEWNESS =====")

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]
fare_skewness = df["fare"].skew()

print(f"Mean    : {fare_mean:.4f}")
print(f"Median  : {fare_median:.4f}")
print(f"Mode    : {fare_mode:.4f}")
print(f"Skewness: {fare_skewness:.4f}")
print(f"Ordering: Mode ({fare_mode:.2f}) < Median ({fare_median:.2f}) < Mean ({fare_mean:.2f})")
print("Conclusion: The fare distribution is strongly right-skewed (positive skewness), characterized by a long right tail of luxury ticket fares.")


print("\n===== BIVARIATE ANALYSIS (BOOLEAN MASKING) =====")

# (a) By Sex using boolean masking
female_mask = df["sex"] == "female"
male_mask = df["sex"] == "male"
survival_female = df[female_mask]["survived"].mean() * 100
survival_male = df[male_mask]["survived"].mean() * 100

print("\n(a) Survival Rate by Sex:")
print(f"    Female Survival Rate: {survival_female:.2f}%")
print(f"    Male Survival Rate  : {survival_male:.2f}%")

# (b) By Pclass using boolean masking
print("\n(b) Survival Rate by Passenger Class:")
for pc in [1, 2, 3]:
    pclass_mask = df["pclass"] == pc
    rate = df[pclass_mask]["survived"].mean() * 100
    print(f"    Class {pc} Survival Rate: {rate:.2f}%")

# (c) By Sex and Pclass combined using boolean masking (& combinations)
print("\n(c) Survival Rate by Sex and Passenger Class Combined:")
for s in ["female", "male"]:
    for pc in [1, 2, 3]:
        mask = (df["sex"] == s) & (df["pclass"] == pc)
        rate = df[mask]["survived"].mean() * 100
        print(f"    Sex: {s.capitalize():<6} | Class: {pc} | Survival Rate: {rate:.2f}%")


print("\n===== CORRELATION MATRIX (6x6 NUMERIC ONLY) =====")

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = df[correlation_columns].corr()

print(correlation_matrix.round(4))


plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Titanic Correlation Matrix (6x6)")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "eda_correlation_heatmap.png"), dpi=150)
plt.close()

corr_pairs = correlation_matrix.where(
    ~correlation_matrix.eq(1)
).stack()

corr_pairs = corr_pairs.abs().sort_values(ascending=False).drop_duplicates()

print("\n===== TOP 2 STRONGEST CORRELATIONS =====")
print(corr_pairs.head(2))

print("\nInterpretation of Top 2 Correlations:")
print("1. pclass vs. fare (|r| = 0.5482, r = -0.5482): Strong negative correlation reflecting that 1st class passengers paid substantially higher ticket fares than 2nd and 3rd class.")
print("2. sibsp vs. parch (|r| = 0.4145, r = +0.4145): Moderate positive correlation showing that passengers traveling with siblings/spouses frequently traveled with parents/children as part of larger family groups.")

# Chart 1: Survival by Sex
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="sex", y="survived")
plt.title("Chart 1: Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.xlabel("Sex")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "eda_survival_by_sex.png"), dpi=150)
plt.close()

# Chart 2: Survival by Passenger Class
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="pclass", y="survived")
plt.title("Chart 2: Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "eda_survival_by_pclass.png"), dpi=150)
plt.close()

# Chart 3: Survival by Sex and Passenger Class
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="pclass", y="survived", hue="sex")
plt.title("Chart 3: Survival Rate by Sex and Passenger Class")
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "eda_survival_by_sex_pclass.png"), dpi=150)
plt.close()

# Chart 4: Age vs Fare by Survival
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="age", y="fare", hue="survived", alpha=0.8)
plt.title("Chart 4: Age vs Fare by Survival Status")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "eda_age_fare_scatter.png"), dpi=150)
plt.close()


from sklearn.preprocessing import StandardScaler

print("\n===== BEFORE STANDARDIZATION =====")

print("Age:")
print("Mean:", df["age"].mean())
print("Standard Deviation:", df["age"].std())

print("\nFare:")
print("Mean:", df["fare"].mean())
print("Standard Deviation:", df["fare"].std())


scaler = StandardScaler()

df[["age_scaled", "fare_scaled"]] = scaler.fit_transform(
    df[["age", "fare"]]
)

print("\n===== AFTER STANDARDIZATION =====")

print("Age scaled:")
print("Mean:", df["age_scaled"].mean())
print("Standard Deviation:", df["age_scaled"].std())

print("\nFare scaled:")
print("Mean:", df["fare_scaled"].mean())
print("Standard Deviation:", df["fare_scaled"].std())


print("\n===== CHART INTERPRETATIONS =====")

print("""
1. Survival Rate by Sex:
Female passengers had a substantially higher survival rate than male passengers.
This shows a strong difference in survival outcomes between the two sex groups.
""")

print("""
2. Survival Rate by Passenger Class:
First-class passengers had the highest survival rate, followed by second-class passengers.
Third-class passengers had the lowest survival rate, showing differences in survival across passenger classes.
""")

print("""
3. Survival Rate by Sex and Passenger Class:
Female passengers had higher survival rates than male passengers within every passenger class.
The difference was especially large in first and second class, while third-class females also had a lower survival rate than females in the higher classes.
""")

print("""
4. Age vs Fare by Survival:
The scatter plot shows the relationship between passenger age and fare while distinguishing survival status.
Higher fares are concentrated among some passengers, while survival outcomes are distributed across different age and fare values.
""")