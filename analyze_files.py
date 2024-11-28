import os
from collections import defaultdict


def analyze_folder_numbers(folder_path):
    """
    Analyzes files in a folder to count the proportion of high/low numbers
    in the first two digits of filenames.
    """
    # Dictionary to store counts for each number
    number_counts = defaultdict(int)
    total_numeric_files = 0

    # Get all jpg files in the folder
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]

    for filename in files:
        # Try to get first two characters and convert to number
        try:
            if filename[:2].isdigit():
                number = int(filename[:2])
                number_counts[number] += 1
                total_numeric_files += 1
        except ValueError:
            continue

    if total_numeric_files == 0:
        return {
            "total_files": len(files),
            "files_with_numbers": 0,
            "low_numbers": 0,
            "high_numbers": 0,
            "low_percentage": 0,
            "high_percentage": 0,
            "number_distribution": {},
        }

    # Consider numbers >= 50 as high numbers
    high_threshold = 50
    high_numbers = sum(
        count for num, count in number_counts.items() if num >= high_threshold
    )
    low_numbers = sum(
        count for num, count in number_counts.items() if num < high_threshold
    )

    return {
        "total_files": len(files),
        "files_with_numbers": total_numeric_files,
        "low_numbers": low_numbers,
        "high_numbers": high_numbers,
        "low_percentage": (
            (low_numbers / total_numeric_files) * 100 if total_numeric_files > 0 else 0
        ),
        "high_percentage": (
            (high_numbers / total_numeric_files) * 100 if total_numeric_files > 0 else 0
        ),
        "number_distribution": dict(number_counts),
    }


def main():
    # Analyze both yes and no folders
    folders = ["yes", "no"]

    for folder in folders:
        if not os.path.exists(folder):
            print(f"Folder '{folder}' not found!")
            continue

        print(f"\nAnalyzing {folder} folder:")
        print("-" * 40)

        results = analyze_folder_numbers(folder)

        print(f"Total files: {results['total_files']}")
        print(f"Files with numeric prefix: {results['files_with_numbers']}")
        print("\nDistribution:")
        print(
            f"Low numbers (0-49): {results['low_numbers']} files ({results['low_percentage']:.1f}%)"
        )
        print(
            f"High numbers (50-99): {results['high_numbers']} files ({results['high_percentage']:.1f}%)"
        )

        print("\nDetailed number distribution:")
        sorted_dist = sorted(results["number_distribution"].items())
        for num, count in sorted_dist:
            print(f"Number {num:02d}: {count} files")


if __name__ == "__main__":
    main()
