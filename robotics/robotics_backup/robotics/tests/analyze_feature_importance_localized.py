from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "failure_prediction"
MODEL_DIR = DATA_DIR / "models"

MODELS = {
    "Random Forest": MODEL_DIR / "random_forest_localized.joblib",
    "XGBoost": MODEL_DIR / "xgboost_localized.joblib",
}


def main():

    print("=" * 70)
    print("LOCALIZED FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    all_importances = []

    for model_name, model_path in MODELS.items():

        print("\n" + "=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found:\n{model_path}"
            )

        model = joblib.load(model_path)

        # ---------------------------------------------------------
        # Extract the actual classifier from a possible pipeline
        # ---------------------------------------------------------

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

        elif hasattr(model, "named_steps"):
            classifier = model.named_steps.get("classifier")

            if hasattr(classifier, "feature_importances_"):
                importances = classifier.feature_importances_
            else:
                print(
                    "This model does not provide tree feature "
                    "importance. Skipping."
                )
                continue

        else:
            print(
                "This model does not provide tree feature "
                "importance. Skipping."
            )
            continue

        feature_names = list(
            pd.read_csv(
                DATA_DIR / "X_train_localized.csv",
                nrows=1
            ).columns
        )

        if len(importances) != len(feature_names):
            raise RuntimeError(
                f"Feature count mismatch: "
                f"{len(importances)} importances vs "
                f"{len(feature_names)} feature names."
            )

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        })

        importance_df = importance_df.sort_values(
            "importance",
            ascending=False
        ).reset_index(drop=True)

        importance_df["percentage"] = (
            importance_df["importance"] * 100
        )

        importance_df["model"] = model_name

        all_importances.append(importance_df)

        print("\nFeature importance:")
        print(
            importance_df[
                ["feature", "percentage"]
            ].to_string(
                index=False,
                formatters={
                    "percentage": lambda x: f"{x:.2f}%"
                }
            )
        )

        print(
            f"\nImportance sum: "
            f"{importance_df['importance'].sum():.6f}"
        )

        # ---------------------------------------------------------
        # Save CSV
        # ---------------------------------------------------------

        filename = (
            "feature_importance_"
            + model_name.lower().replace(" ", "_")
            + "_localized.csv"
        )

        output_path = DATA_DIR / filename

        importance_df[
            ["feature", "importance", "percentage"]
        ].to_csv(
            output_path,
            index=False
        )

        print(f"Saved: {output_path}")

    # -------------------------------------------------------------
    # Compare the two new localized features directly
    # -------------------------------------------------------------

    if all_importances:

        combined = pd.concat(
            all_importances,
            ignore_index=True
        )

        print("\n" + "=" * 70)
        print("LOCALIZED FEATURES")
        print("=" * 70)

        localized_names = [
            "number_of_low_clearance_runs",
            "longest_low_clearance_run",
        ]

        localized = combined[
            combined["feature"].isin(localized_names)
        ]

        print(
            localized[
                ["model", "feature", "percentage"]
            ].to_string(
                index=False,
                formatters={
                    "percentage": lambda x: f"{x:.2f}%"
                }
            )
        )

        localized_output = (
            DATA_DIR / "localized_feature_importance.csv"
        )

        localized[
            ["model", "feature", "importance", "percentage"]
        ].to_csv(
            localized_output,
            index=False
        )

        print(
            f"\nSaved localized-feature comparison: "
            f"{localized_output}"
        )

    # -------------------------------------------------------------
    # Plot RF and XGBoost importance
    # -------------------------------------------------------------

    for importance_df in all_importances:

        model_name = importance_df["model"].iloc[0]

        plot_df = importance_df.sort_values(
            "importance",
            ascending=True
        )

        plt.figure(figsize=(10, 7))

        plt.barh(
            plot_df["feature"],
            plot_df["importance"]
        )

        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title(
            f"{model_name} — Localized Feature Importance"
        )

        plt.tight_layout()

        filename = (
            "feature_importance_"
            + model_name.lower().replace(" ", "_")
            + "_localized.png"
        )

        output_path = DATA_DIR / filename

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved plot: {output_path}")

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()