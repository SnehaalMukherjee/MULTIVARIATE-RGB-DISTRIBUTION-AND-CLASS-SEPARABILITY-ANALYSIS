from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


DATA_PATH = (
    Path.home()
    / "Downloads"
    / "intel_dataset"
    / "seg_train"
    / "seg_train"
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def print_heading(title):
    print("\n" + "=" * 88)
    print(title.center(88))
    print("=" * 88)


def print_table(headers, rows):
    if not rows:
        print("No data available.")
        return

    rows = [[str(value) for value in row] for row in rows]
    headers = [str(value) for value in headers]

    widths = [
        max(
            len(headers[i]),
            max(len(row[i]) for row in rows)
        )
        for i in range(len(headers))
    ]

    border = (
        "+"
        + "+".join("-" * (width + 2) for width in widths)
        + "+"
    )

    def format_row(row):
        return (
            "| "
            + " | ".join(
                str(value).ljust(widths[i])
                for i, value in enumerate(row)
            )
            + " |"
        )

    print(border)
    print(format_row(headers))
    print(border)

    for row in rows:
        print(format_row(row))

    print(border)


def calculate_rgb_mean(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((128, 128))
        pixels = np.asarray(image, dtype=np.float64)

    return pixels.mean(axis=(0, 1))


def load_dataset():
    if not DATA_PATH.is_dir():
        raise SystemExit(
            "\nERROR: Dataset folder was not found.\n"
            f"Expected location:\n{DATA_PATH}\n"
            "Check that the Intel dataset is extracted correctly."
        )

    records = []

    print_heading("MULTIVARIATE RGB DISTRIBUTION ANALYSIS")
    print("Dataset: Intel Image Classification")
    print("Feature extraction: Mean RGB intensity")
    print("\nLoading images. Please wait...")

    for class_folder in sorted(
        path for path in DATA_PATH.iterdir()
        if path.is_dir()
    ):
        print(f"Processing class: {class_folder.name}")

        for image_path in sorted(class_folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                red, green, blue = calculate_rgb_mean(image_path)

                records.append({
                    "image": image_path.name,
                    "class": class_folder.name,
                    "mean_R": red,
                    "mean_G": green,
                    "mean_B": blue
                })

            except Exception as error:
                print(
                    f"Warning: Could not process "
                    f"{image_path.name}: {error}"
                )

    if not records:
        raise SystemExit(
            "\nERROR: No valid images were found in the dataset."
        )

    return pd.DataFrame(records)


def calculate_class_statistics(data):
    class_names = sorted(data["class"].unique())

    class_means = {}
    class_covariances = {}
    class_features = {}
    statistics_rows = []

    for class_name in class_names:
        values = data.loc[
            data["class"] == class_name,
            ["mean_R", "mean_G", "mean_B"]
        ].to_numpy()

        mean_rgb = values.mean(axis=0)

        std_rgb = (
            values.std(axis=0, ddof=1)
            if len(values) > 1
            else np.zeros(3)
        )

        covariance = (
            np.cov(values, rowvar=False)
            if len(values) > 1
            else np.zeros((3, 3))
        )

        class_means[class_name] = mean_rgb
        class_covariances[class_name] = covariance
        class_features[class_name] = values

        statistics_rows.append([
            class_name,
            len(values),
            f"{mean_rgb[0]:.2f}",
            f"{mean_rgb[1]:.2f}",
            f"{mean_rgb[2]:.2f}",
            f"{std_rgb[0]:.2f}",
            f"{std_rgb[1]:.2f}",
            f"{std_rgb[2]:.2f}"
        ])

    print_heading("SECTION 1: CLASS-WISE RGB STATISTICS")

    print_table(
        [
            "Class",
            "Images",
            "Mean R",
            "Mean G",
            "Mean B",
            "Std R",
            "Std G",
            "Std B"
        ],
        statistics_rows
    )

    return (
        class_names,
        class_means,
        class_covariances,
        class_features
    )


def calculate_separability(
    class_names,
    class_means,
    class_features
):
    results = []

    for i, class_a in enumerate(class_names):
        for class_b in class_names[i + 1:]:
            values_a = class_features[class_a]
            values_b = class_features[class_b]

            difference = (
                class_means[class_a]
                - class_means[class_b]
            )

            euclidean_distance = np.linalg.norm(difference)

            covariance_a = np.cov(values_a, rowvar=False)
            covariance_b = np.cov(values_b, rowvar=False)

            pooled_covariance = (
                (len(values_a) - 1) * covariance_a
                + (len(values_b) - 1) * covariance_b
            ) / (len(values_a) + len(values_b) - 2)

            inverse_covariance = np.linalg.pinv(
                pooled_covariance + np.eye(3) * 1e-6
            )

            mahalanobis_squared = float(
                difference.T
                @ inverse_covariance
                @ difference
            )

            mahalanobis_distance = np.sqrt(
                max(mahalanobis_squared, 0.0)
            )

            results.append({
                "class_A": class_a,
                "class_B": class_b,
                "euclidean": float(euclidean_distance),
                "mahalanobis": float(mahalanobis_distance)
            })

    print_heading("SECTION 2: PAIRWISE CLASS SEPARABILITY")

    table_rows = [
        [
            result["class_A"],
            result["class_B"],
            f'{result["euclidean"]:.2f}',
            f'{result["mahalanobis"]:.2f}'
        ]
        for result in results
    ]

    print_table(
        [
            "Class A",
            "Class B",
            "Euclidean Distance",
            "Mahalanobis Distance"
        ],
        table_rows
    )

    return results


def plot_mean_rgb(class_names, class_means):
    positions = np.arange(len(class_names))
    width = 0.25

    red = [class_means[name][0] for name in class_names]
    green = [class_means[name][1] for name in class_names]
    blue = [class_means[name][2] for name in class_names]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(positions - width, red, width, label="Red")
    ax.bar(positions, green, width, label="Green")
    ax.bar(positions + width, blue, width, label="Blue")

    ax.set_xticks(positions)
    ax.set_xticklabels(class_names, rotation=15, ha="right")
    ax.set_ylabel("Mean intensity (0–255)")
    ax.set_xlabel("Image class")
    ax.set_title("Class-Wise Mean RGB Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()


def plot_rgb_scatter(data, class_names):
    channel_pairs = [
        ("mean_R", "mean_G"),
        ("mean_R", "mean_B"),
        ("mean_G", "mean_B")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (x_channel, y_channel) in zip(
        axes, channel_pairs
    ):
        for class_name in class_names:
            subset = data[data["class"] == class_name]

            ax.scatter(
                subset[x_channel],
                subset[y_channel],
                label=class_name,
                alpha=0.35,
                s=9
            )

        ax.set_xlabel(x_channel)
        ax.set_ylabel(y_channel)
        ax.grid(True, alpha=0.25)

    axes[-1].legend(fontsize=7)
    fig.suptitle("RGB Feature Distribution by Class")
    fig.tight_layout()


def plot_euclidean(results):
    sorted_results = sorted(
        results,
        key=lambda item: item["euclidean"]
    )

    labels = [
        f'{item["class_A"]} vs {item["class_B"]}'
        for item in sorted_results
    ]

    distances = [
        item["euclidean"]
        for item in sorted_results
    ]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(labels, distances)
    ax.set_xlabel("Euclidean distance")
    ax.set_title("Pairwise Euclidean Class Separability")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()


def plot_mahalanobis(results):
    sorted_results = sorted(
        results,
        key=lambda item: item["mahalanobis"]
    )

    labels = [
        f'{item["class_A"]} vs {item["class_B"]}'
        for item in sorted_results
    ]

    distances = [
        item["mahalanobis"]
        for item in sorted_results
    ]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(labels, distances)
    ax.set_xlabel("Mahalanobis distance")
    ax.set_title("Pairwise Mahalanobis Class Separability")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()


def plot_rgb_profiles(class_names, class_means):
    fig, ax = plt.subplots(figsize=(10, 6))

    for class_name in class_names:
        ax.plot(
            ["Red", "Green", "Blue"],
            class_means[class_name],
            marker="o",
            label=class_name
        )

    ax.set_ylabel("Mean intensity (0–255)")
    ax.set_title("RGB Profiles Across Image Classes")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()


def main():
    data = load_dataset()

    (
        class_names,
        class_means,
        class_covariances,
        class_features
    ) = calculate_class_statistics(data)

    results = calculate_separability(
        class_names,
        class_means,
        class_features
    )

    print_heading("SECTION 3: PROJECT SUMMARY")

    summary_rows = [
        ["Dataset", "Intel Image Classification"],
        ["Total images analysed", len(data)],
        ["Number of classes", len(class_names)],
        ["Features", "Mean Red, Green and Blue intensity"],
        ["Distance measures", "Euclidean and Mahalanobis"],
        ["Graphs", 5],
        ["File output", "No files saved"]
    ]

    print_table(
        ["Parameter", "Value"],
        summary_rows
    )

    max_euclidean = max(
        results,
        key=lambda item: item["euclidean"]
    )

    min_euclidean = min(
        results,
        key=lambda item: item["euclidean"]
    )

    max_mahalanobis = max(
        results,
        key=lambda item: item["mahalanobis"]
    )

    min_mahalanobis = min(
        results,
        key=lambda item: item["mahalanobis"]
    )

    print_heading("SECTION 4: KEY OBSERVATIONS")

    observations = [
        [
            "Largest Euclidean separation",
            f'{max_euclidean["class_A"]} vs '
            f'{max_euclidean["class_B"]}',
            f'{max_euclidean["euclidean"]:.2f}'
        ],
        [
            "Smallest Euclidean separation",
            f'{min_euclidean["class_A"]} vs '
            f'{min_euclidean["class_B"]}',
            f'{min_euclidean["euclidean"]:.2f}'
        ],
        [
            "Largest Mahalanobis separation",
            f'{max_mahalanobis["class_A"]} vs '
            f'{max_mahalanobis["class_B"]}',
            f'{max_mahalanobis["mahalanobis"]:.2f}'
        ],
        [
            "Smallest Mahalanobis separation",
            f'{min_mahalanobis["class_A"]} vs '
            f'{min_mahalanobis["class_B"]}',
            f'{min_mahalanobis["mahalanobis"]:.2f}'
        ]
    ]

    print_table(
        ["Observation", "Class Pair", "Distance"],
        observations
    )

    print(
        "\nInterpretation: Larger distances indicate greater separation "
        "between class mean RGB features. These distances do not represent "
        "classification accuracy."
    )

    plot_mean_rgb(class_names, class_means)
    plot_rgb_scatter(data, class_names)
    plot_euclidean(results)
    plot_mahalanobis(results)
    plot_rgb_profiles(class_names, class_means)

    print_heading("ANALYSIS COMPLETED SUCCESSFULLY")
    print("Five graphs are ready to display.")
    print("No output files have been saved.")
    print("Close the graph windows to finish.")

    plt.show()


if __name__ == "__main__":
    main()