from pathlib import Path

def print_directory_tree(root: Path, indent: str = '') -> None:
    """
    Recursively prints the directory tree structure, skipping directories that start with a dot.

    Args:
        root (Path): The root directory path as a Path object.
        indent (str): The indentation for the current directory level.
    """
    try:
        # Get the list of files and directories in the current directory
        items = list(root.iterdir())
    except PermissionError:
        # Skip directories where permission is denied
        return
    except FileNotFoundError:
        # Handle if the user enters an invalid directory
        print(f"Error: The directory '{root}' does not exist.")
        return

    # Loop through each item in the directory
    for i, item in enumerate(items):
        # Skip directories that start with a dot
        if item.is_dir() and item.name.startswith('.'):
            continue

        # Determine the appropriate branch character
        branch = '└── ' if i == len(items) - 1 else '├── '

        # Print the item with the current indentation
        print(f"{indent}{branch}{item.name}")

        # If the item is a directory, recursively print its contents
        if item.is_dir():
            new_indent = indent + ('    ' if i == len(items) - 1 else '│   ')
            print_directory_tree(item, new_indent)

# Ask the user for the starting directory path
user_input: str = input("Enter the directory path to start from: ")

# Convert user input to a Path object and print the directory tree
root_path: Path = Path(user_input)
print(f"\nProject directory tree starting from: {root_path}")
print_directory_tree(root_path)
